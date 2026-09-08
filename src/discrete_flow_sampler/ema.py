"""Exponential moving average of trainable parameters, eval-side only.

The shadow is a passive observer: gradients and optimiser steps act on the
raw parameters; after each step

    shadow <- d_t * shadow + (1 - d_t) * theta_t

and the shadow is swapped in only for evaluation, so training dynamics are
bit-identical with or without it.

Warmup (measured failure, MDNS gate-3 arm 0): a plain shadow (d_t = decay)
starts at the init weights, so after k updates it is decay^k init +
(1 - decay^k) iterates; with decay 0.9999 that is 82% init at k = 2000, and
eval-on-EMA reads a nearly untrained model. warmup=True uses the torch-ema /
diffusers schedule d_t = min(decay, (1+t)/(10+t)): the init weight becomes
prod_{t<=k} (1+t)/(10+t) = 10!(k+1)!/(10+k)! (~1e-17 by k = 200) while d_t
still reaches the requested decay for t >= ~9e4.

`state_dict`/`load_state_dict` carry the shadow and the update counter; both
must persist across preemption resume, or the shadow restarts at the
resume-point weights and the warmup schedule restarts mid-run.
"""

import torch


class ExponentialMovingAverage:
    """Shadow copy of the trainables; see module docstring for the maths."""

    def __init__(self, parameters, decay, warmup=False):
        self.decay = decay
        self.warmup = warmup
        self.updates = 0
        self.parameters = list(parameters)
        self.shadow = [p.detach().clone() for p in self.parameters]

    def effective_decay(self, step):
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + step) / (10 + step))

    def update(self):
        self.updates += 1
        decay = self.effective_decay(self.updates)
        with torch.no_grad():
            for shadow, parameter in zip(self.shadow, self.parameters):
                shadow.mul_(decay).add_(parameter, alpha=1 - decay)

    def swap_in(self):
        with torch.no_grad():
            self._backup = [p.detach().clone() for p in self.parameters]
            for parameter, shadow in zip(self.parameters, self.shadow):
                parameter.copy_(shadow)

    def swap_out(self):
        with torch.no_grad():
            for parameter, backup in zip(self.parameters, self._backup):
                parameter.copy_(backup)

    def state_dict(self):
        # Clone because .cpu() can alias live CPU tensors; snapshots must
        # remain unchanged by later updates.
        return {
            "updates": self.updates,
            "shadow": [tensor.detach().cpu().clone() for tensor in self.shadow],
        }

    def load_state_dict(self, state):
        self.updates = int(state["updates"])
        self.shadow = [
            saved.to(parameter.device).clone()
            for saved, parameter in zip(state["shadow"], self.parameters)
        ]


class CTGridEMA:
    """Per-slot EMA of the c_t grid across outer cycles.

    The swap Kolmogorov residual regresses xi_t toward c_t, the stand-in for
    dt log Z_t (DNFS Eq. (8), Alg. 1; `swap_training.py`). c_t is
    re-estimated every outer cycle from `outer_batch` rollout states and its
    noise enters the gradient through (xi - c) * grad(xi): at d256-naive the
    per-slot standard error is ~ sqrt(110/128) ~= 0.93 nats, 125x noisier
    than the d64 record's CV-stabilised target.

    Lemma 1 (the discrete Stein identity) gives E[xi_t] = dt log Z_t for any
    admissible rate matrix, but only under the annealing target p_t (the law
    in the ratio `target.swap_log_ratio`), not under the model's own law;
    the licence to substitute any q_t (App. A.3) is proved only at
    optimality. What justifies smoothing is the objective's shape instead:
    with c detached, E_q[(xi - c)^2] = Var_q[xi] + Delta^2 with
    Delta = E_q[xi] - c, so only the offset Delta reaches the gradient (see
    `c_t_offset_rms` in swap_kolmogorov.py). If xi becomes constant, Lemma 1
    pins it to dt log Z_t regardless of c, provided the sampled law covers
    p_t, the assumption d256 breaks. Averaging c_t across cycles aligns it
    with the mixture of up to `replay_buffer_cycles` past models the buffer
    holds, where the raw estimate is the newest rollout's mean alone; the
    EMA is a Delta correction rather than variance reduction.

    The first grid after construction or `reset()` seeds the state; later
    grids are smoothed with the configured halflife. Reset at every
    curriculum sigma transition so estimates from different targets do not
    mix. Checkpoints hold the state on CPU; the next `update()` moves it to
    the incoming grid's device.
    """

    def __init__(self, n_grid: int, halflife_cycles: float):
        if halflife_cycles <= 0:
            raise ValueError(
                f"CTGridEMA requires halflife > 0 cycles, got "
                f"{halflife_cycles}; the trainer constructs it only when "
                f"the knob is on (see is_enabled)."
            )
        self.halflife_cycles = float(halflife_cycles)
        self.alpha = 1.0 - 0.5 ** (1.0 / self.halflife_cycles)
        self.n_grid = n_grid
        self.state = None  # (n_grid,) tensor after the first update

    @staticmethod
    def is_enabled(halflife_cycles: float) -> bool:
        """Enable smoothing only for a positive halflife; zero disables it."""
        return halflife_cycles > 0

    def update(self, c_t_grid):
        """Fold one outer cycle's raw grid in; return the smoothed grid.

        Adopts the incoming grid's device: there are no model parameters
        from which to infer one when restoring CPU checkpoint state.
        """
        if self.state is None:
            self.state = c_t_grid.clone()
        else:
            if self.state.device != c_t_grid.device:
                self.state = self.state.to(c_t_grid.device)
            self.state.mul_(1 - self.alpha).add_(c_t_grid, alpha=self.alpha)
        return self.state

    def reset(self):
        """Clear the state so the next grid seeds a new curriculum stage."""
        self.state = None

    def state_dict(self):
        # Clone on CPU for the same aliasing reason as the parameter EMA.
        return {
            "halflife_cycles": self.halflife_cycles,
            "state": (
                None if self.state is None else self.state.detach().cpu().clone()
            ),
        }

    def load_state_dict(self, state):
        saved = state["state"]
        self.state = None if saved is None else saved.clone()
