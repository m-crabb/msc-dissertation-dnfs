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
        # Clone: on CPU, .cpu()/.to() are no-ops returning the SAME tensor,
        # so an un-cloned snapshot would alias the live shadow and be
        # silently mutated by every later update().
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
    """Per-slot EMA of the c_t grid across outer cycles (M2, 2026-08-14).

    Why this exists: the swap Kolmogorov residual regresses xi_t toward
    c_t, which stands in for dt log Z_t (DNFS Alg. 1; `swap_training.py`).
    c_t is re-estimated every outer cycle from `outer_batch` rollout states,
    and its noise enters the loss gradient multiplicatively through
    (xi - c) * grad(xi): at d256-naive the per-slot standard error is
    ~ sqrt(110/128) ~= 0.93 nats (run-support case D2), 125x noisier than
    the d64 record's CV-stabilised target. The Eq.-8 identity
    E[xi_t] = dt log Z_t holds for the model's OWN law at any training
    stage, so consecutive cycles estimate the same slowly-drifting quantity
    and an EMA over cycles is pure variance reduction at an UNCHANGED fixed
    point — the across-cycle complement of the Stein control variate's
    within-cycle reduction.

    Contracts the trainer relies on:

    * First-cycle passthrough: the state is seeded AT the first raw grid
      after construction (or `reset`), never at zeros — the
      init-contamination failure the parameter EMA's warmup schedule
      exists to prevent is avoided by construction.
    * Fixed-point invariance: a constant integrand sequence is reproduced
      exactly; a step change is tracked with the advertised halflife.
    * `reset()` on every curriculum sigma transition: c_t is a function of
      sigma, so smoothing must never mix estimates across a boundary.
    * Checkpoint contract: `state_dict`/`load_state_dict` carry the state
      (or its absence) so a preempted run's continuation is bit-exact —
      the same contract test_swap_training_resume.py pins for the base
      trainer state.
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
        """Knob semantics: 0.0 (the default, and every archived run's
        implicit value) is OFF — byte-identical archived behaviour."""
        return halflife_cycles > 0

    def update(self, c_t_grid):
        """Fold one outer cycle's raw grid in; return the smoothed grid."""
        if self.state is None:
            self.state = c_t_grid.clone()
        else:
            self.state.mul_(1 - self.alpha).add_(c_t_grid, alpha=self.alpha)
        return self.state

    def reset(self):
        """Drop the state (curriculum sigma transition): the next cycle
        passes through raw, re-seeding at the new sigma's c_t."""
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
