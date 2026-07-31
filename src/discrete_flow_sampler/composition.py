"""Composition conditioning: shared helpers for the amortised sampler.

An amortised sampler is conditioned on the target composition c, so one
trained model serves many compositions instead of one specialist per
composition. Everything here is deliberately target-agnostic: the same
helpers serve the soft-penalty leg (c enters a penalty term) and the
fixed-composition leg (c selects the slice).

The b-major rule is stated once here and used by both the target
(`IsingTarget._row_composition`) and the model adapter
(`models.composition_conditioned`).
"""
import torch
from torch import Tensor


def expand_b_major(values: Tensor, batch_size: int) -> Tensor:
    """Expand a per-block conditioning vector to one entry per batch row.

    Every batch-expanding call site in this codebase builds its expanded
    batch **b-major** — each original row repeated `k` times contiguously —
    and rides the time vector along with `t.repeat_interleave(k)`:

        `_neighbours._log_p_tilde_at_neighbours`   k = d · S
        `kolmogorov.residual_general`              k = d
        `ctmc._compute_xi_t_general`               k = d

    A composition vector must follow the identical rule, so that row
    (b·k + j) carries the composition of its parent row b. Getting this
    wrong never raises — it silently trains row b against another row's
    target composition, and shows up only as a quietly worse ESS. Hence the
    divisibility check is a hard error rather than a broadcast: a ragged
    batch means the caller expanded some other way and the row-to-block map
    is simply unknown.

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

    One composition is drawn per outer cycle rather than per row. That keeps
    the `∂_t log Z_t` estimate averaged over the full outer batch at a single
    composition — mixing compositions inside one average would combine
    incompatible normalisers, since Z_t depends on c — while inner batches
    still span several compositions by drawing across replay cycles.

    Two regimes:
      * `values` set  -> uniform over that finite set: amortising over the
        compositions we already have specialists for.
      * otherwise     -> uniform on [centre - half_width, centre + half_width],
        clamped into [0, 1].

    The range is centred rather than given as (lo, hi) because zero-bias
    Ising is invariant under the joint map (x -> -x, c -> 1-c); a symmetric
    draw respects a symmetry the target actually has, and it lets the window
    be widened during training through the single scalar `half_width`.

    Args:
        quantise_to: when set, round the draw onto the lattice of
            realisable compositions {n / quantise_to}. The fixed-composition
            leg needs this — there N_A = c·d must be an integer or no exact
            slice exists — and it is a no-op for the soft leg, which accepts
            any real c.
    """
    if values is not None:
        index = int(torch.randint(len(values), (1,), generator=generator).item())
        composition = float(values[index])
    else:
        offset = (
            2.0 * torch.rand((), generator=generator).item() - 1.0
        ) * half_width
        composition = centre + offset

    if quantise_to is not None:
        composition = round(composition * quantise_to) / quantise_to
    return float(min(1.0, max(0.0, composition)))
