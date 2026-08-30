"""Config cells for the GFlowNet comparator (hard chapter).

A deliberately flat dataclass, SEPARATE from HardStageCfg: the GFN has no
time grid, no Euler steps, no swap head and no CTMC knobs, so inheriting the
swap-stack config would carry dead fields that a recipe-parity audit then has
to explain away. The two families meet at the artefact level instead — run
dirs, eval/metrics.json schema and composition observables are shared, so
the table scripts ingest GFN rows unchanged.

Fairness protocol (design doc 2026-08-30-gfn-comparator): hidden 128 /
3 layers / 4 heads mirrors the leTF house sizing at the small rungs; exact
parameter/FLOP matching against the swap-head cells is done at sweep time,
not hard-coded here. sigma_stages is the beta-annealing knob (the VAN-line
criticality mitigation); the 4x4 cells train flat, mirroring the wave-2
house convention that the floor/4x4 rung measures the operating point, not
the curriculum.
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


GFN_CONFIGS = {
    cell.name: cell
    for objective in GFN_OBJECTIVES
    for cell in (
        _gfn_d16_cell(objective, "s010", 0.10),
        _gfn_d16_cell(objective, "s220", SIGMA_C),
    )
}
