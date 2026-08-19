"""Sample montages: what the draws themselves look like, across the thesis.

Every other figure in this project reduces configurations to a scalar --
energy, composition, phi, ESS. Those scalars are what the arguments are
made of, but they hide two things a reader legitimately wants to check by
eye:

  1. that the samples are *configurations of the right kind* (domains at
     sigma_c, exact 50/50 composition under swap dynamics), not noise that
     happens to reproduce one marginal, and
  2. what the 16x16 estimator channel actually costs: plain importance
     sampling puts a third of the weight on one draw, and the SMC fix that
     removes that heavy tail pays for it in duplicated lineages.

Three montages, all drawn from artefacts already on local disk. Nothing
here re-runs a sampler or a chain; the script loads saved tensors and
draws them.

  headline_8x8.png   -- DNFS draws at (sigma_c, 8x8, c = 0.5) over the
      certified non-local Kawasaki reference chains at the same cell. The
      DNFS row is weight-resampled (see below) so both rows are samples of
      the same measure and the comparison is like-for-like.
  d256_plain_vs_smc.png -- the 16x16 checkpoint sampled two ways: plain IS
      and SMC with adaptive resampling at tau = 0.9. Each column is the
      most similar PAIR of draws inside one 512-particle eval population.
      WHAT TO LOOK FOR: on the left the closest pair still differs in ~58
      of 256 sites and the two tiles look unrelated; on the right the
      closest pairs are identical or one swap apart, so the column reads as
      the same lattice twice. That is resampling duplicating lineages.
  spine_row.png      -- one block per constraint regime at sigma = 0.10:
      unconstrained (chapter 3), soft composition penalty (chapter 4),
      hard fixed-composition swap dynamics (chapter 5). The narrative
      spine of the thesis in one row.

Why duplication is measured by Hamming distance, not by exact repeats
---------------------------------------------------------------------
The obvious degeneracy check is "how many distinct configurations are
left", and the eval records exactly that (`n_unique_samples`, a
`torch.unique(samples, dim=0)` over rows). At d = 256 that statistic is
useless in both directions: a particle cloned at a resampling event keeps
evolving, and ONE accepted swap after the event makes it a different row
while it is still the same lineage. Exact-row counts therefore sit near
the population size whether or not resampling created clones, and no
conclusion about degeneracy may be drawn from them here.

The ancestry-aware substitute used in this figure is the nearest-sibling
Hamming distance within each independent eval population (populations are
independent because eval draws run in `eval_sample_chunk`-sized batches,
and a lineage can only be cloned inside its own batch). Plain IS supplies
the null: independent draws from this model never land closer than ~52
sites apart, so any pair below that is resampling-induced, not chance.

Why the neural rows are weight-resampled, not raw
-------------------------------------------------
A plain-IS eval returns proposal draws x_i with log importance weights
log w_i; the target expectation is the *weighted* average, so the raw
draws are NOT a sample from pi. Showing raw draws as if they were would
be the same category error the chapters spend their time avoiding. Tiles
for the headline and spine montages are therefore selected by systematic
resampling on the archived weights, reusing
`samplers.resampling.systematic_resample_indices` -- the identical
resampler the SMC eval path uses, so the figures and the SMC machinery
cannot drift apart. Systematic rather than multinomial because its
per-particle multiplicity is pinned to {floor(N w_i), ceil(N w_i)}: the
picture then shows the weights, not resampling noise on top of them.

The SMC draws in the second montage are shown AS SAVED, with no further
resampling: they are already the output of the resampling sampler and
their post-event weights are near-uniform by construction, so reweighting
them for display would double-count the correction.

Which tiles get shown
---------------------
`systematic_resample_indices` returns ancestors in ascending order, so
the first K of them are the K lowest-index survivors -- a biased window.
Tiles are instead taken at a fixed stride N/K through that ancestor
array, which is exactly a K-point systematic resample of the same
population (stride N/K on positions (u+k)/N gives K positions spaced
1/K apart). Multiplicities among the shown tiles are therefore
proportional to the weights, and no draw is hand-picked for effect.

The pair montage uses the opposite, and equally rule-bound, selection:
the most similar disjoint pairs under the same statistic on BOTH sides.
Selecting the extreme of one statistic is only fair if the same extreme
is taken from the comparator, which is why the plain-IS block is built by
the identical rule rather than from typical draws.

Colour
------
Spins use the house spin map everywhere (figure_style.SPIN_CMAP: indigo =
spin -1, gold = spin +1, the pair from background.tex fig:ising-phases),
which is luminance-separated and so survives greyscale print unchanged.
Colour identity is carried by the tile FRAME instead, taking
the house role hues -- our sampler blue, classical MCMC amber, the
hard-constraint delta red where the point is the constraint or the clone
structure -- so no figure needs the reader to distinguish two
mid-luminance fills.

Run with (add --dry-run to check data selection without rendering):
  pixi run -e dev python experiments/constrained_hard_03/analysis_sample_montages.py

CPU-only: the heaviest step is a 512x512 dot product per eval population.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from discrete_flow_sampler.diagnostics.figure_style import (
    CLASSICAL_HUE,
    HARD_DELTA_HUE,
    MUTED,
    NEURAL_COMPARATOR_HUE,
    SAMPLER_HUE,
    SPIN_CMAP,
    use_house_style,
)
from discrete_flow_sampler.samplers.resampling import systematic_resample_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "03_hard" / "sample_montages"

# Spin fills: the house spin map (indigo = -1 / down, gold = +1 / up),
# established at background.tex fig:ising-phases. The first cut of this
# script invented a grey/ink pair, which reads as a different system next
# to every other lattice figure in the thesis.
SPIN_COLOUR_MAP = SPIN_CMAP

# --- run selection --------------------------------------------------------
# Every default below is a run the writeup already cites, so a reader can
# match a montage to a number they have already seen. Alternatives were
# rejected for the reasons given inline.

# The sigma_c 8x8 headline run: same run dir the probe figures use as their
# sigma_c default. Its `eval/` holds metrics only -- the draws live in the
# eight replicate evals -- so the montage reads replicate s101, the first
# of the pooled eight.
HEADLINE_8X8_RUN = (
    RESULTS_ROOT / "03_hard"
    / "H2_d64_c50_s223_letf_ma_100k_curr_seed42_20260722-124315"
)
HEADLINE_8X8_EVAL_SUBDIR = "eval_replicate_s101"

# The certified reference for that same cell: eight non-local Kawasaki
# chains, 1e6 sweeps each, R-hat <= 1.01 on all four observables. Their
# snapshots are the only on-disk configurations certified against this
# target, which is why the reference row is chains and not, say, a longer
# neural run.
HEADLINE_8X8_REFERENCE_DIR = RESULTS_ROOT / "kawasaki_probe" / "reference" / "sc"

# The 16x16 recipe run whose landing was judged, and the one the SMC
# tau-sweep was run on -- i.e. the archive's live 16x16 arm rather than an
# abandoned rescue attempt.
D256_RUN = (
    RESULTS_ROOT / "03_hard"
    / "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_seed43"
      "_20260818-d256-recipe-s43"
)
# Plain-IS side: `eval` (not `eval_ema`) is the frozen protocol, the only
# draw set comparable to the published ESS tables, and the SMC evals were
# run off the same raw checkpoint weights -- so this is the like-for-like
# comparator, not merely the more dramatic one.
D256_PLAIN_EVAL_SUBDIR = "eval"
# SMC side: tau = 0.9 rather than 0.5. Firing earlier and more often kills
# fewer lineages per event and leaves the survivors longer to decorrelate,
# which is why it is the better operating point for observables despite a
# lower pooled ESS (that drop is a log-Z banking artefact, not a
# regression).
D256_SMC_EVAL_SUBDIR = "eval_smc_tau0.9"

# The spine, matched at sigma = 0.10 rather than matched at lattice size.
# No unconstrained 8x8 run exists locally at sigma = 0.10 (the only 8x8
# unconstrained run is at sigma_c), so the row had to give up one match or
# the other. Coupling was kept because sigma is what sets domain texture:
# a row whose unconstrained panel sat at sigma_c would invite the reader to
# read a coupling difference as a constraint effect, which is precisely the
# misreading this figure exists to prevent. The lattice-size difference is
# labelled on the panel instead.
SPINE_RUNS = (
    (
        "unconstrained",
        RESULTS_ROOT / "01_baseline" / "stage_4_d10_budget_seed42_20260609-120825",
        "eval",
    ),
    (
        "soft penalty",
        RESULTS_ROOT / "02_constrained_soft"
        / "S2_d8_c05_l50_letf_ne64_seed42_20260812-walkback",
        "eval",
    ),
    (
        "hard swap",
        RESULTS_ROOT / "03_hard"
        / "H2_d64_c50_s010_letf_ma_50k_seed42_20260812-floor",
        "eval",
    ),
)

N_TILES_HEADLINE_PER_ROW = 8
# Four pairs per side: enough that a lucky coincidence cannot carry the
# picture, few enough that a 16x16 lattice still prints large enough to
# recognise as the SAME lattice rather than merely a similar one.
N_PAIRS_D256_PER_BLOCK = 4
N_TILES_SPINE_PER_REGIME = 4

# Near-clone threshold quoted alongside the figure, in sites (four swaps).
# Matches the ancestry-aware analysis the SMC sweep was judged on.
NEAR_CLONE_HAMMING = 8

# One fixed jitter for every systematic resample in this script, so reruns
# reproduce the montages tile for tile.
RESAMPLE_JITTER = 0.5


@dataclass(frozen=True)
class DrawSet:
    """One population of lattice configurations plus what identifies it.

    `spins` is (n_draws, n_sites) in {-1, +1}; site index = row * side + col
    (the row-major flattening the Ising coupling matrix is built with, so a
    reshape to (side, side) is the physical lattice, on a torus).
    `log_weights` is None for chain snapshots, which are already
    target-distributed and must NOT be reweighted. `population_size` is the
    eval's sampling chunk: draws in different chunks were generated by
    independent sampler runs, so no clone relationship can cross that
    boundary and every sibling statistic is computed within it.
    """

    label: str
    source: Path
    spins: np.ndarray
    log_weights: np.ndarray | None
    lattice_side: int
    sigma: float
    target_composition: float | None
    penalty_strength: float | None
    ess_fraction: float | None
    population_size: int

    @property
    def n_draws(self) -> int:
        return int(self.spins.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.spins.shape[1])

    @property
    def n_populations(self) -> int:
        return -(-self.n_draws // self.population_size)

    def lattice(self, draw_index: int) -> np.ndarray:
        """One draw as a (side, side) array of 0/1 for the two-level colour map."""
        spin_vector = self.spins[draw_index]
        return (spin_vector > 0).astype(np.float32).reshape(
            self.lattice_side, self.lattice_side
        )


def require(path: Path, what: str) -> Path:
    """Fail loudly on a missing artefact instead of silently regenerating it.

    Regenerating any of these means re-running a sampler or an MCMC chain,
    which is off-limits on the laptop this script is written for; a missing
    file is a stop condition, not a task.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing {what}: {path}")
    return path


def load_neural_draws(run_dir: Path, eval_subdir: str, label: str) -> DrawSet:
    """Eval draws + log importance weights + the run's Ising and eval settings."""
    eval_dir = require(run_dir / eval_subdir, f"eval dir for {label}")
    spins = torch.load(
        require(eval_dir / "samples.pt", f"samples for {label}"),
        map_location="cpu",
        weights_only=True,
    ).float().numpy()
    log_weights = torch.load(
        require(eval_dir / "log_weights.pt", f"log weights for {label}"),
        map_location="cpu",
        weights_only=True,
    ).double().numpy()
    config = json.loads(
        require(run_dir / "config.json", f"config for {label}").read_text()
    )
    ising_config = config["ising"]
    metrics_path = eval_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    chunk = config.get("eval", {}).get("eval_sample_chunk") or len(spins)
    return DrawSet(
        label=label,
        source=eval_dir,
        spins=spins,
        log_weights=log_weights,
        lattice_side=int(ising_config["D"]),
        sigma=float(ising_config["sigma"]),
        target_composition=ising_config.get("target_composition"),
        penalty_strength=ising_config.get("composition_penalty_strength"),
        ess_fraction=metrics.get("ess_fraction"),
        population_size=int(chunk),
    )


def load_kawasaki_reference_draws(reference_dir: Path, label: str) -> DrawSet:
    """The FINAL snapshot of each certified reference chain, one draw per chain.

    One draw per chain rather than many draws from one chain: consecutive
    snapshots of a Kawasaki chain at sigma_c are correlated over hundreds of
    sweeps, so a montage built from a single chain would show the same
    slowly-relaxing domain pattern eight times and read as degeneracy that
    is really autocorrelation. The last snapshot is the furthest point from
    each chain's phase-separated initialisation, i.e. unambiguously past any
    burn-in rule.
    """
    chain_dirs = sorted(reference_dir.glob("chain_*"))
    if not chain_dirs:
        raise FileNotFoundError(f"no reference chains under {reference_dir}")
    final_snapshots = []
    for chain_dir in chain_dirs:
        snapshots = np.load(
            require(chain_dir / "snapshots.npz", f"snapshots for {chain_dir.name}")
        )
        final_snapshots.append(snapshots["spins"][-1].astype(np.float32))
    meta = json.loads((chain_dirs[0] / "meta.json").read_text())
    spins = np.stack(final_snapshots)
    return DrawSet(
        label=label,
        source=reference_dir,
        spins=spins,
        log_weights=None,
        lattice_side=int(meta["lattice_side"]),
        sigma=float(meta["sigma"]),
        target_composition=0.5,  # canonical swap dynamics: fixed by construction
        penalty_strength=None,
        ess_fraction=None,
        population_size=len(spins),
    )


def normalised_weights(log_weights: np.ndarray) -> np.ndarray:
    """Self-normalised weights, max-shifted so the largest exponent is exp(0).

    log w is extensive in the lattice size and reaches the hundreds at
    d = 256; without the shift the exponentials overflow float64 before the
    normalisation can cancel the offset.
    """
    shifted = log_weights - log_weights.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def resampled_ancestors(draws: DrawSet) -> tuple[np.ndarray, int, float]:
    """Ancestor index per slot of a full N-particle systematic resample.

    Returns (ancestors, n_distinct_ancestors, largest_normalised_weight).
    The distinct count here is over ANCESTOR INDICES -- lineage identity,
    which is exact by construction -- and must not be confused with a count
    of distinct configurations: at d = 256 the latter saturates because a
    clone stops being an identical row after a single swap.
    """
    ancestors = systematic_resample_indices(
        torch.from_numpy(draws.log_weights), uniform=RESAMPLE_JITTER
    ).numpy()
    n_distinct = int(np.unique(ancestors).size)
    return ancestors, n_distinct, float(normalised_weights(draws.log_weights).max())


def tiles_from_resample(ancestors: np.ndarray, n_tiles: int) -> np.ndarray:
    """`n_tiles` source indices, taken at a fixed stride through the ancestors.

    A stride of N/K through the ascending ancestor array is itself a K-point
    systematic resample (see module docstring), so tile multiplicities track
    the weights. Taking the first K instead would show only the low-index end
    of the population.
    """
    stride = max(1, len(ancestors) // n_tiles)
    return ancestors[::stride][:n_tiles]


def population_slices(draws: DrawSet) -> list[slice]:
    """The independent eval populations, as slices into the draw array."""
    return [
        slice(start, min(start + draws.population_size, draws.n_draws))
        for start in range(0, draws.n_draws, draws.population_size)
    ]


def within_population_hamming(draws: DrawSet, block: slice) -> np.ndarray:
    """Pairwise Hamming distances inside one population, diagonal masked out.

    For spins in {-1, +1} the number of AGREEING sites is (d + x.y)/2, so
    the Hamming distance is (d - x.y)/2 and the whole matrix is one dot
    product -- no O(d) python loop over pairs. Self-distance is set to d
    (the maximum) so a draw is never its own nearest sibling.
    """
    block_spins = draws.spins[block]
    distances = (draws.n_sites - block_spins @ block_spins.T) / 2.0
    np.fill_diagonal(distances, draws.n_sites)
    return distances


def sibling_statistics(draws: DrawSet) -> dict[str, float]:
    """Nearest-sibling summary: the ancestry-aware degeneracy read.

    `min_nearest`: closest any two draws in one population come. Under plain
    IS this is the null distance for independent draws from this model, so
    it is the number every SMC clone claim is measured against.
    `n_pairs_one_swap` / `fraction_with_near_clone`: how much of the
    population sits in a cluster tight enough that resampling, not chance,
    must have produced it.
    """
    nearest = np.empty(draws.n_draws)
    n_pairs_one_swap = 0
    for block in population_slices(draws):
        distances = within_population_hamming(draws, block)
        nearest[block] = distances.min(axis=1)
        upper_triangle = distances[np.triu_indices(distances.shape[0], k=1)]
        n_pairs_one_swap += int((upper_triangle <= 2).sum())
    return {
        "min_nearest": float(nearest.min()),
        "median_nearest": float(np.median(nearest)),
        "n_pairs_one_swap": n_pairs_one_swap,
        "fraction_with_near_clone": float((nearest <= NEAR_CLONE_HAMMING).mean()),
    }


def closest_disjoint_pairs(draws: DrawSet, n_pairs: int) -> list[tuple[int, int, int]]:
    """The `n_pairs` most similar within-population pairs, sharing no draw.

    Disjoint because a repeated draw across columns would show the same
    lineage twice and overstate how many independent clone events the
    figure is evidencing.
    """
    candidates: list[tuple[float, int, int]] = []
    for block in population_slices(draws):
        distances = within_population_hamming(draws, block)
        rows, columns = np.triu_indices(distances.shape[0], k=1)
        pair_distances = distances[rows, columns]
        keep = np.argsort(pair_distances, kind="stable")[: 4 * n_pairs]
        candidates.extend(
            (float(pair_distances[k]), block.start + int(rows[k]),
             block.start + int(columns[k]))
            for k in keep
        )
    candidates.sort()
    chosen: list[tuple[int, int, int]] = []
    used: set[int] = set()
    for distance, first, second in candidates:
        if first in used or second in used:
            continue
        chosen.append((first, second, int(distance)))
        used.update((first, second))
        if len(chosen) == n_pairs:
            break
    return chosen


def draw_lattice_tile(
    axis,
    draws: DrawSet,
    draw_index: int,
    frame_colour: str,
    frame_width: float = 1.1,
    annotation: str | None = None,
) -> None:
    """One lattice as an image tile, framed in its role colour."""
    axis.imshow(
        draws.lattice(draw_index),
        cmap=SPIN_COLOUR_MAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_colour)
        spine.set_linewidth(frame_width)
    if annotation is not None:
        axis.set_xlabel(annotation, fontsize=6.5, color=MUTED, labelpad=2)


def build_headline_8x8(
    neural: DrawSet, reference: DrawSet, n_tiles: int
) -> tuple[plt.Figure, str]:
    """Two rows at the same cell: weight-resampled DNFS over certified chains."""
    ancestors, n_distinct_ancestors, _ = resampled_ancestors(neural)
    tile_sources = tiles_from_resample(ancestors, n_tiles)
    n_reference_tiles = min(n_tiles, reference.n_draws)

    figure, axes = plt.subplots(2, n_tiles, figsize=(7.0, 2.35))
    for column, source_index in enumerate(tile_sources):
        draw_lattice_tile(axes[0, column], neural, int(source_index), SAMPLER_HUE)
    for column in range(n_tiles):
        axis = axes[1, column]
        if column < n_reference_tiles:
            draw_lattice_tile(axis, reference, column, CLASSICAL_HUE)
        else:
            axis.set_visible(False)
    axes[0, 0].set_ylabel("DNFS", fontsize=8, color=SAMPLER_HUE)
    axes[1, 0].set_ylabel("Kawasaki", fontsize=8, color=CLASSICAL_HUE)
    figure.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.02,
                           wspace=0.12, hspace=0.08)

    caption = (
        f"Fixed-composition draws on the {neural.lattice_side}x"
        f"{neural.lattice_side} periodic lattice at $\\sigma_c$ = "
        f"{neural.sigma:g}, $c$ = {neural.target_composition:g}; dark tiles "
        "are spin $+1$, pale tiles spin $-1$. Top (blue frames): "
        f"{n_tiles} DNFS draws selected by systematic resampling on their "
        f"importance weights, ESS/N = {neural.ess_fraction:.2f}. Bottom "
        f"(amber frames): one draw from each of the {n_reference_tiles} "
        "certified non-local Kawasaki reference chains. At this size the "
        "weighted and unweighted pictures are the same picture."
    )
    return figure, caption


def build_d256_plain_vs_smc(
    plain: DrawSet, smc: DrawSet, n_pairs: int
) -> tuple[plt.Figure, str]:
    """Closest within-population draw pairs, plain IS beside SMC.

    What a reader should look for: each column is the two most similar draws
    inside one independent eval population. On the plain-IS side the closest
    pair anywhere still differs in roughly a quarter of the lattice, and the
    two tiles read as unrelated configurations. On the SMC side the closest
    pairs are identical or one swap apart -- the column is visibly the same
    lattice twice. That difference is the cost of the resampling fix, and it
    is invisible to any exact-duplicate count at this lattice size (see the
    module docstring).
    """
    plain_pairs = closest_disjoint_pairs(plain, n_pairs)
    smc_pairs = closest_disjoint_pairs(smc, n_pairs)
    smc_statistics = sibling_statistics(smc)
    plain_statistics = sibling_statistics(plain)

    figure = plt.figure(figsize=(7.0, 2.6))
    left_block, right_block = figure.subfigures(1, 2, wspace=0.06)
    for block, draws, pairs, hue, title in (
        (left_block, plain, plain_pairs, SAMPLER_HUE, "plain importance sampling"),
        (right_block, smc, smc_pairs, HARD_DELTA_HUE, "SMC, $\\tau$ = 0.9"),
    ):
        axes = block.subplots(2, n_pairs)
        for column, (first, second, distance) in enumerate(pairs):
            draw_lattice_tile(axes[0, column], draws, first, hue)
            draw_lattice_tile(
                axes[1, column], draws, second, hue,
                annotation=f"H = {distance}",
            )
        block.suptitle(title, fontsize=8, color=hue)
        block.subplots_adjust(top=0.88, bottom=0.08, hspace=0.10, wspace=0.12)

    caption = (
        f"{plain.lattice_side}x{plain.lattice_side} fixed-composition draws at "
        f"$\\sigma_c$ = {plain.sigma:g}, $c$ = {plain.target_composition:g}; dark "
        "tiles are spin $+1$, pale tiles spin $-1$. Each column is the closest "
        "pair of draws, by Hamming distance $H$ (sites differing, printed "
        f"beneath), within one {plain.population_size}-particle eval population: "
        "left (blue frames) plain importance sampling, right (red frames) SMC "
        "resampling at $\\tau$ = 0.9 from the same checkpoint. Pooled over the "
        f"{plain.n_populations} populations, plain IS has no pair closer than "
        f"{plain_statistics['min_nearest']:.0f} of {plain.n_sites} sites, while "
        f"SMC has {smc_statistics['n_pairs_one_swap']} pairs within one swap and "
        f"{smc_statistics['fraction_with_near_clone']:.0%} of draws with a "
        f"sibling inside {NEAR_CLONE_HAMMING} sites -- SMC buys its weight "
        "uniformity with near-clone lineages."
    )
    return figure, caption


def build_spine_row(
    regimes: list[DrawSet], n_tiles: int
) -> tuple[plt.Figure, str]:
    """One weight-resampled block per constraint regime, unconstrained -> hard."""
    regime_hues = (SAMPLER_HUE, NEURAL_COMPARATOR_HUE, HARD_DELTA_HUE)
    figure = plt.figure(figsize=(7.0, 2.5))
    blocks = figure.subfigures(1, len(regimes), wspace=0.05)

    composition_summaries = []
    for block, draws, hue in zip(blocks, regimes, regime_hues):
        ancestors, _, _ = resampled_ancestors(draws)
        tile_sources = tiles_from_resample(ancestors, n_tiles)
        axes = block.subplots(2, n_tiles // 2)
        for axis, source_index in zip(axes.flat, tile_sources):
            draw_lattice_tile(axis, draws, int(source_index), hue)
        up_fraction = (draws.spins > 0).mean(axis=1)
        composition_summaries.append((up_fraction.mean(), up_fraction.std()))
        block.suptitle(
            f"{draws.label}\n{draws.lattice_side}x{draws.lattice_side}, "
            f"$c$ = {up_fraction.mean():.3f} $\\pm$ {up_fraction.std():.3f}",
            fontsize=8, color=hue,
        )
        block.subplots_adjust(top=0.84, bottom=0.02, hspace=0.10, wspace=0.10)

    sigmas = {f"{draws.sigma:g}" for draws in regimes}
    sigma_clause = (
        f"$\\sigma$ = {sigmas.pop()}" if len(sigmas) == 1
        else "per-panel $\\sigma$ as titled"
    )
    penalty = regimes[1].penalty_strength
    sizes = ", ".join(
        f"{draws.lattice_side}x{draws.lattice_side}" for draws in regimes
    )
    composition_clause = ", ".join(
        f"{mean:.3f} $\\pm$ {std:.3f}" for mean, std in composition_summaries
    )
    caption = (
        f"Weight-resampled DNFS draws at {sigma_clause}, {n_tiles} per "
        "constraint regime: unconstrained (blue), soft composition penalty "
        f"$\\lambda$ = {penalty:g} (green), hard fixed-composition swap "
        "dynamics (red); dark tiles are spin $+1$, pale tiles spin $-1$. "
        f"Up-spin fraction over each run's {regimes[0].n_draws} eval draws: "
        f"{composition_clause}. Panels are {sizes} -- no unconstrained "
        f"{regimes[2].lattice_side}x{regimes[2].lattice_side} run exists at "
        "this coupling. Only the swap sampler puts every draw exactly on the "
        "composition manifold."
    )
    return figure, caption


def resolve_draw_sets(args: argparse.Namespace) -> dict[str, object]:
    """Load every draw set the three montages need, failing on the first gap."""
    return {
        "headline_neural": load_neural_draws(
            args.headline_run, args.headline_eval_subdir, "DNFS 8x8 sigma_c"
        ),
        "headline_reference": load_kawasaki_reference_draws(
            args.headline_reference_dir, "Kawasaki reference 8x8 sigma_c"
        ),
        "d256_plain": load_neural_draws(
            args.d256_run, args.d256_plain_eval_subdir, "DNFS 16x16 plain IS"
        ),
        "d256_smc": load_neural_draws(
            args.d256_run, args.d256_smc_eval_subdir, "DNFS 16x16 SMC tau=0.9"
        ),
        "spine": [
            load_neural_draws(run_dir, eval_subdir, label)
            for label, run_dir, eval_subdir in SPINE_RUNS
        ],
    }


def describe_draw_set(
    draws: DrawSet, tile_note: str, with_siblings: bool,
    with_resample: bool = True,
) -> None:
    """One draw set's provenance, shape and selection statistics.

    `with_resample` is off for the pair montage: its tiles are chosen by
    distance, not by resampling, and printing a hypothetical resample of
    an already-resampled SMC population would suggest the figure does
    something it does not.
    """
    lines = [
        f"  {draws.label}",
        f"    source     : {draws.source.relative_to(REPO_ROOT)}",
        f"    spins      : {draws.spins.shape} -> {tile_note} of "
        f"{draws.lattice_side}x{draws.lattice_side}",
        f"    sigma      : {draws.sigma:g}   target c: {draws.target_composition}"
        f"   populations: {draws.n_populations} x {draws.population_size}",
    ]
    if draws.log_weights is None:
        lines.append("    weights    : unweighted (chain snapshots)")
    else:
        ess = "n/a" if draws.ess_fraction is None else f"{draws.ess_fraction:.4f}"
        top_weight = float(normalised_weights(draws.log_weights).max())
        summary = f"ESS/N {ess}, top weight {top_weight:.3f}"
        if with_resample:
            _, n_distinct, _ = resampled_ancestors(draws)
            summary += (
                f", systematic resample keeps {n_distinct} distinct ancestors "
                f"of {draws.n_draws}"
            )
        lines.append(f"    weights    : {summary}")
    if with_siblings:
        statistics = sibling_statistics(draws)
        lines.append(
            f"    siblings   : nearest-sibling Hamming median "
            f"{statistics['median_nearest']:.0f}, min "
            f"{statistics['min_nearest']:.0f}; "
            f"{statistics['n_pairs_one_swap']} pairs within one swap; "
            f"{statistics['fraction_with_near_clone']:.1%} with a sibling "
            f"inside {NEAR_CLONE_HAMMING} sites"
        )
        lines.append(
            "    pairs shown: "
            + ", ".join(
                f"({first}, {second}) H={distance}"
                for first, second, distance in closest_disjoint_pairs(
                    draws, N_PAIRS_D256_PER_BLOCK
                )
            )
        )
    print("\n".join(lines))


def report_dry_run(draw_sets: dict[str, object]) -> None:
    """What would be drawn, and from where -- the cheap data-selection check."""
    print("montage 1: headline_8x8.png")
    describe_draw_set(
        draw_sets["headline_neural"], f"{N_TILES_HEADLINE_PER_ROW} tiles", False
    )
    describe_draw_set(
        draw_sets["headline_reference"], f"{N_TILES_HEADLINE_PER_ROW} tiles", False
    )
    print("\nmontage 2: d256_plain_vs_smc.png")
    for key in ("d256_plain", "d256_smc"):
        describe_draw_set(
            draw_sets[key], f"{N_PAIRS_D256_PER_BLOCK} closest pairs", True,
            with_resample=False,
        )
    print("\nmontage 3: spine_row.png")
    for draws in draw_sets["spine"]:
        describe_draw_set(draws, f"{N_TILES_SPINE_PER_REGIME} tiles", False)
    print(f"\nfigures would be written to {OUTPUT_DIR / 'figures'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-run", type=Path, default=HEADLINE_8X8_RUN)
    parser.add_argument("--headline-eval-subdir", default=HEADLINE_8X8_EVAL_SUBDIR)
    parser.add_argument(
        "--headline-reference-dir", type=Path, default=HEADLINE_8X8_REFERENCE_DIR
    )
    parser.add_argument("--d256-run", type=Path, default=D256_RUN)
    parser.add_argument("--d256-plain-eval-subdir", default=D256_PLAIN_EVAL_SUBDIR)
    parser.add_argument("--d256-smc-eval-subdir", default=D256_SMC_EVAL_SUBDIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report resolved run dirs and tensor shapes, render nothing",
    )
    args = parser.parse_args(argv)

    draw_sets = resolve_draw_sets(args)
    if args.dry_run:
        report_dry_run(draw_sets)
        return

    use_house_style()
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    captions: dict[str, str] = {}
    builders = (
        (
            "headline_8x8",
            lambda: build_headline_8x8(
                draw_sets["headline_neural"], draw_sets["headline_reference"],
                N_TILES_HEADLINE_PER_ROW,
            ),
        ),
        (
            "d256_plain_vs_smc",
            lambda: build_d256_plain_vs_smc(
                draw_sets["d256_plain"], draw_sets["d256_smc"],
                N_PAIRS_D256_PER_BLOCK,
            ),
        ),
        (
            "spine_row",
            lambda: build_spine_row(draw_sets["spine"], N_TILES_SPINE_PER_REGIME),
        ),
    )
    for name, builder in builders:
        figure, caption = builder()
        output_path = figures_dir / f"{name}.png"
        figure.savefig(output_path)
        plt.close(figure)
        captions[name] = caption
        print(f"[montage] wrote {output_path}")
        print(f"[caption] {caption}\n")

    # Captions live beside the figures and are rebuilt from the same numbers
    # the figures were drawn from, so a caption can never quote a stale ESS.
    captions_path = args.out_dir / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2) + "\n")
    print(f"[montage] wrote {captions_path}")


if __name__ == "__main__":
    main()
