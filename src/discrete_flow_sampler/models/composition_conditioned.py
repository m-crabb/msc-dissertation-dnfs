"""Adapter presenting a composition-conditioned model through `(x, t)`.

The samplers, the ∂_t log Z_t estimator and the residual losses all call the
rate model as `model(x, t)`. An amortised model needs a third input, the
target composition c. Rather than thread that argument through every one of
those call sites — which would churn signatures shared with the
fixed-composition route — the trainer binds c into this adapter and hands the
adapter over wherever a model is expected.

The adapter is deliberately signature-agnostic: it forwards `(x, t, c)` to
whatever it wraps, so it serves a single-site flip rate matrix and a
pair-swap head alike, both of which are called as `f(x, t)` today.

It is NOT an `nn.Module`. It holds a reference, so the trainer keeps building
the optimiser and the checkpoints from the underlying model — wrapping in a
Module would prefix every `state_dict` key and break checkpoint
compatibility with the eval scripts.
"""
from torch import Tensor

from discrete_flow_sampler.composition import expand_b_major

# Attributes the samplers and residual losses read off the model to pick a
# code path. Mirrored onto the adapter so it is a drop-in substitute.
_FORWARDED_ATTRIBUTES = ("is_locally_equivariant", "vocab_size", "d", "D")


class CompositionConditioned:
    """Bind a composition to `model`, exposing the plain `(x, t)` signature.

    Args:
        model: callable `(x, t, c) -> rates`, e.g. a rate matrix or swap head
            built with composition conditioning enabled.
        composition: (n_blocks,) tensor of target compositions. Expanded to
            one entry per row on every call by the shared b-major rule, so a
            single-entry tensor serves a whole outer cycle at one composition
            and an N-entry tensor serves a mixed inner batch.
    """

    def __init__(self, model, composition: Tensor):
        if composition.ndim != 1:
            raise ValueError(
                f"composition must be 1-D (one entry per block), got shape "
                f"{tuple(composition.shape)}"
            )
        self.model = model
        self.composition = composition
        for name in _FORWARDED_ATTRIBUTES:
            if hasattr(model, name):
                setattr(self, name, getattr(model, name))

    def __call__(self, x: Tensor, t: Tensor):
        return self.model(x, t, expand_b_major(self.composition, x.shape[0]))

    def parameters(self):
        return self.model.parameters()

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
