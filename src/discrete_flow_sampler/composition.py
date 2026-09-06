"""Composition draws and batch alignment for amortised samplers.

Shared by soft-penalty and fixed-composition targets and the composition-
conditioned model adapter. Expanded batches repeat each original row
contiguously (b-major order).
"""

import torch
from torch import Tensor


def expand_b_major(values: Tensor, batch_size: int) -> Tensor:
    """Expand a per-block conditioning vector to one entry per batch row.

    Repeat each value `k` times so row `b*k + j` retains its parent row's
    composition, matching the state and time expansion. Reject batches that
    cannot be divided into equal blocks to avoid misaligning compositions.

    Args:
        values: (n_blocks,) per-block conditioning values.
        batch_size: number of rows to expand to; must be an integer multiple
            of `n_blocks`.

    Returns:
        (batch_size,) tensor. Returned unchanged when already aligned.
    """
    n_blocks = values.shape[0]
    if batch_size % n_blocks != 0:
        raise ValueError(
            f"batch of {batch_size} rows is not an integer multiple of the "
            f"{n_blocks} conditioning blocks, so the b-major expansion rule "
            "cannot align them."
        )
    if batch_size == n_blocks:
        return values
    return values.repeat_interleave(batch_size // n_blocks)


def draw_composition(
    centre: float,
    half_width: float,
    values: tuple[float, ...] | None = None,
    *,
    quantise_to: int | None = None,
    generator: torch.Generator | None = None,
) -> float:
    """Draw one target composition for an outer cycle.

    The outer batch shares one composition so its `∂_t log Z_t` estimate
    uses a single normaliser. Inner batches may mix compositions from
    different replay cycles.

    Sample uniformly from `values` when supplied; otherwise draw uniformly
    from [centre - half_width, centre + half_width]. Quantise if requested,
    then clamp the result to [0, 1].

    Args:
        quantise_to: when set, round the draw onto the lattice of
            realisable compositions {n / quantise_to}. Fixed-composition
            targets require an integer site count N_A = c·d; soft targets
            accept any real c and can leave this unset.
    """
    if values is not None:
        index = int(torch.randint(len(values), (1,), generator=generator).item())
        composition = float(values[index])
    else:
        offset = (2.0 * torch.rand((), generator=generator).item() - 1.0) * half_width
        composition = centre + offset

    if quantise_to is not None:
        composition = round(composition * quantise_to) / quantise_to
    return float(min(1.0, max(0.0, composition)))
