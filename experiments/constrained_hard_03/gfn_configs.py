"""Config cells for the GFlowNet comparator (hard chapter).

A deliberately flat dataclass, SEPARATE from HardStageCfg: the GFN has no
time grid, no Euler steps, no swap head and no CTMC knobs, so inheriting the
swap-stack config would carry dead fields that a recipe-parity audit then has
to explain away. The two families meet at the artefact level instead — run
dirs, eval/metrics.json schema and composition observables are shared, so
the table scripts ingest GFN rows unchanged.

Fairness protocol (design doc 2026-08-30-gfn-comparator): the s92
correctness wave ran hidden 128 / 3 layers, which the s93 parity audit
measured at 597.6k params against 79.5k-101k for the wave-2 d16 heads —
the `_par` cells are the judging wave at measured parameter parity
(hidden 64 / 2 layers / 4 heads = 101.4k params ~= the ma head's 101.0k;
batch 128). sigma_stages is the
beta-annealing knob (the VAN-line criticality mitigation); the 4x4 cells
train flat, mirroring the wave-2 house convention that the floor/4x4 rung
measures the operating point, not the curriculum.
"""

from dataclasses import dataclass, replace

from discrete_flow_sampler.targets.ising import SIGMA_C

GFN_OBJECTIVES = ("tb", "fldb")


@dataclass(frozen=True)
class GFNCellCfg:
    name: str
    objective: str  # "tb" (trajectory balance) | "fldb" (forward-looking DB)
    sigma: float
    D: int = 4
    target_composition: float = 0.5
    # Policy network (see models/raster_gfn_policy.py for the O(d^2)-vs-O(d^3) note).
    hidden_dim: int = 128
    n_layers: int = 3
    n_heads: int = 4
    with_flow_head: bool = False  # forced True for fldb in the builder below
    # Training.
    n_steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    # Separate Adam lr for the TB arm's scalar log_z (None = share
    # learning_rate). Adam moves a scalar at ~lr/step under a consistent
    # gradient, so at the flat 1e-3 the s92 wave's log Z (init 0) climbed at
    # its measured speed limit (+7e-4/step) and sat 2.4 nats below the exact
    # slice value 10.81 at 10k steps — it arithmetically could not arrive.
    # ~100x on log_z alone is the Malkin et al. / torchgfn convention.
    log_z_learning_rate: float | None = None
    epsilon: float = 0.05  # uniform behaviour mix; off-policy, TB-tolerated
    sigma_stages: tuple[float, ...] = ()  # annealing ladder; () = train flat
    # Eval (house protocol: 5000 draws, chunked).
    n_eval_samples: int = 5000
    eval_sample_chunk: int = 512
    # Bookkeeping.
    log_every: int = 100
    checkpoint_every: int = 2000
    wandb_project: str = "dnfs-constraints"

    def __post_init__(self):
        if self.objective not in GFN_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {GFN_OBJECTIVES}, got {self.objective!r}"
            )
        if self.sigma_stages and self.sigma_stages[-1] != self.sigma:
            raise ValueError(
                "sigma_stages must end at the cell's own sigma "
                f"(got stages {self.sigma_stages} for sigma={self.sigma})"
            )


def _gfn_d16_cell(objective: str, sigma_label: str, sigma: float) -> GFNCellCfg:
    """4x4 rung, both arms, both house couplings; flat (no annealing)."""
    cell = GFNCellCfg(
        name=f"GFN_d16_c50_{sigma_label}_{objective}_10k",
        objective=objective,
        sigma=sigma,
    )
    if objective == "fldb":
        cell = replace(cell, with_flow_head=True)
    return cell


def _gfn_d16_parity_cell(objective: str, sigma_label: str, sigma: float) -> GFNCellCfg:
    """s93 judging wave: parameter parity with the wave-2 heads and the
    split log_z lr on the TB arm. The plain cells above are the s92
    correctness wave, kept as archived configs (never retro-flipped).

    Parity is measured in PARAMETERS, not copied hyperparameters: the house
    d16 sizing (hidden 32 / 2 layers) gives this policy only 26.1k params
    because the swap cells' 79.5k-101k live in a backbone+head stack the
    policy doesn't have. hidden 64 / 2 layers = 101,378 params, within 0.4%
    of the masked-attention head (100,960) — the flagship comparator row."""
    cell = replace(
        _gfn_d16_cell(objective, sigma_label, sigma),
        hidden_dim=64,
        n_layers=2,
        batch_size=128,
    )
    cell = replace(cell, name=f"{cell.name}_par")
    if objective == "tb":
        cell = replace(cell, log_z_learning_rate=0.1)
    return cell


GFN_CONFIGS = {
    cell.name: cell
    for objective in GFN_OBJECTIVES
    for cell in (
        _gfn_d16_cell(objective, "s010", 0.10),
        _gfn_d16_cell(objective, "s220", SIGMA_C),
        _gfn_d16_parity_cell(objective, "s010", 0.10),
        _gfn_d16_parity_cell(objective, "s220", SIGMA_C),
    )
}
