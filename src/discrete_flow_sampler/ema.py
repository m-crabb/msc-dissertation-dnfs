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
