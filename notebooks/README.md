# Exploratory notebooks and reference work

[Project overview](../README.md) · [Data guide](../data/README.md)

These files record familiarisation and early cross-checks. The dissertation's
training and report entrypoints are mapped in the
[experiment guide](../experiments/README.md).

| File | Purpose |
| --- | --- |
| [icet_agpb_tutorial.ipynb](icet_agpb_tutorial.ipynb) | Walkthrough adapted from the icet getting-started tutorial: Ag–Pd cluster-expansion fitting, structure enumeration, SGC/VCSGC sampling and diagnostics |
| [mixing_energy.ce](mixing_energy.ce) | Retained Ag–Pd cluster expansion produced by that notebook |
| `*.png` | Retained notebook plots of fitted energies, ECIs, mixing energies, free-energy derivatives, acceptance and ESS |
| [stage_2_d10_constrained_gibbs_reference.py](stage_2_d10_constrained_gibbs_reference.py) | Historical long Gibbs-reference producer for soft-constrained Stage 2; saves samples and a mixing plot under `results/02_constrained_soft/` |

The notebook filename contains `agpb`, but its actual species are **Ag–Pd**.
The filename is retained for existing references. Open it through `pixi run
jupyter` from the repository root; its kernel should run in `notebooks/` so the
relative database and output paths resolve. Running all cells fits a model,
enumerates structures, runs Monte Carlo and can overwrite the retained notebook
outputs. Saved plots do not include the untracked Monte Carlo trajectories.

Upstream tutorial links and licensing scope are recorded in
[third-party notices](../THIRD_PARTY_NOTICES.md).
