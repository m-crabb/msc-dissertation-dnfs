"""Exponential moving average of trainable parameters, eval-side only.

The shadow is a passive observer of training: gradients and optimiser steps
act on the raw parameters; the shadow is updated AFTER each step as

    shadow <- d_t * shadow + (1 - d_t) * theta_t

and is swapped in only for evaluation. Training dynamics with and without
the shadow are bit-identical.

Why the warmup schedule exists (measured failure, MDNS gate-3 arm 0): a
plain shadow (d_t = decay always) starts AT the init weights, so after k
updates it is decay^k init + (1 - decay^k) training iterates — with decay
0.9999 that is 82% init at k = 2000, and eval-on-EMA reads a nearly
untrained model however well training went. warmup=True applies the
standard bias-correction schedule d_t = min(decay, (1+t)/(10+t)) (the
torch-ema/diffusers default): the init weight becomes
prod_{t<=k} (1+t)/(10+t) = 10!(k+1)!/(10+k)! (~1e-17 by k = 200) while d_t
still reaches the requested decay for t >= ~9e4, so warmup and plain agree
asymptotically.

Checkpoint contract: `state_dict`/`load_state_dict` carry the shadow AND
the update counter. The counter must persist across preemption resume —
re-initialising the shadow at the resume-point weights would re-create the
init-contamination failure through the back door, and resetting the counter
alone would restart the warmup schedule mid-run.
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

    Why this exists: the swap Kolmogorov residual regresses xi_t toward
    c_t, which stands in for dt log Z_t (DNFS Alg. 1; `swap_training.py`).
    c_t is re-estimated every outer cycle from `outer_batch` rollout states,
    and its noise enters the loss gradient multiplicatively through
    (xi - c) * grad(xi): at d256-naive the per-slot standard error is
    ~ sqrt(110/128) ~= 0.93 nats (run-support case D2), 125x noisier than
    the d64 record's CV-stabilised target.

    DNFS Eq. (8) defines c_t as dt log Z_t, and its
    Lemma 1 (the discrete Stein identity) delivers E[xi_t] = dt log Z_t
    for ANY admissible rate matrix at ANY training stage -- but only under
    p_t, the ANNEALING TARGET, because Lemma 1 needs the expectation taken
    under the same law that appears in the ratio p(y)/p(x), and in this
    code that ratio is `target.swap_log_ratio`. Under the model's own law
    the identity does NOT hold; the paper's licence to swap in any q_t
    (App. A.3) is proved only AT OPTIMALITY, where the residual vanishes
    pointwise and every distribution integrates it to zero.

    What justifies smoothing is therefore not the identity but the
    objective's shape. With c detached, one slot's loss decomposes as

        E_q[(xi - c)^2] = Var_q[xi] + (E_q[xi] - c)^2 = Var_q[xi] + Delta^2

    so c's VALUE is irrelevant -- only Delta, its offset from the mean of
    xi over the distribution the LOSS averages over, ever reaches the
    gradient, and it reaches it as a rank-one term 2*Delta*E_q[grad xi]
    that shifts xi uniformly instead of narrowing it. At Delta = 0 the
    detached gradient equals the exact variance gradient identically.
    The fixed point is safe from either side: if xi becomes constant,
    Lemma 1 pins that constant to dt log Z_t regardless of c -- PROVIDED
    the sampled law covers p_t, which is exactly the assumption d256
    breaks.

    So this EMA is best read as an approximate Delta-CORRECTION rather
    than as variance reduction: averaging c_t across cycles aligns it with
    the mixture of up to `replay_buffer_cycles` past models that the
    buffer actually holds, whereas the raw estimate is the newest
    rollout's mean alone. Delta itself is still unlogged; the residual's
    MEAN (not just its mean square) would measure it for free, since
    `residual_swap` already computes xi - c on the inner batch.

    The first grid after construction or `reset()` seeds the state directly.
    Constant inputs remain unchanged; changes are smoothed with the configured
    halflife. Reset at every curriculum sigma transition to avoid mixing
    estimates from different targets. Checkpoints preserve the state on CPU;
    the next `update()` moves it to the incoming grid's device.
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

        Adopt the incoming grid's device when restoring CPU checkpoint state.
        This class has no model parameters from which to infer a device at
        load time.
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
