"""Rate-matrix model interface. All implementations satisfy this Protocol.

The training loop and CTMC sampler depend only on the Protocol below, so
swapping parameterisations (Stage 1/2 MLP -> Stage 3 LeT) is a plug-in.

Output convention for binary spins (S = 2): for each site i, there is only
one alternative state (the flip). So a rate matrix only needs to emit one
non-negative rate per site -- shape (B, D**2) -- rather than the general
(B, D**2, S - 1). This collapses the per-site dimension that would exist
for higher-S problems.

Stage 1 / 2: MLPRateMatrix in `mlp.py`.
Stage 3:    LocallyEquivariantTransformer (added in stage-3 plan).
"""
from __future__ import annotations

from typing import Protocol

from torch import Tensor


class RateMatrix(Protocol):
    """A rate-matrix model takes (state x, time t) and returns flip rates per
    site. Implementations must satisfy this signature; the training loop and
    CTMC sampler are written against this interface, not against any concrete
    class.
    """

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:  # noqa: D401
        """x: (B, D**2) in {-1, +1}; t: (B,) in [0, 1]. Returns (B, D**2) >= 0."""
        ...


__all__ = ["RateMatrix"]
