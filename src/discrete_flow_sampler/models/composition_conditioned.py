"""Adapter presenting a composition-conditioned model through `(x, t)`.

The trainer binds target composition c for samplers, the ∂_t log Z_t
estimator and residual losses that call `model(x, t)`. The adapter forwards
`(x, t, c)` to either a single-site flip rate matrix or a pair-swap head.

This is not an `nn.Module`: optimisers and checkpoints use the underlying
model, preserving parameter names without a wrapper's `state_dict` prefix
and keeping eval-script checkpoint compatibility.
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
