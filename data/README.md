# Retained data

[Project overview](../README.md) · [Alloy experiments](../experiments/alloy_ce/README.md)

This directory holds small inputs needed by the alloy targets and the exploratory
icet notebook. Production trajectories, trained neural checkpoints and reference
chains live under the untracked `results/` directory.

| File | Contents and provenance |
| --- | --- |
| [ce/cuau_fcc_2x2x4.json](ce/cuau_fcc_2x2x4.json) | 16-site FCC Cu–Au spin-product expansion exported from the MetaDNS CLEASE model |
| [ce/cuau_fcc_4x4x4.json](ce/cuau_fcc_4x4x4.json) | 64-site FCC export of the same model |
| [ce/square_cuau_4x4.json](ce/square_cuau_4x4.json) | 16-site periodic evaluation of the earlier EMT-fitted square-grid toy expansion |
| [ce/square_cuau_8x8.json](ce/square_cuau_8x8.json) | 64-site periodic evaluation of the same toy expansion |
| [reference_data.db](reference_data.db) | ASE database with 625 Ag–Pd reference structures used by the [icet tutorial notebook](../notebooks/icet_agpb_tutorial.ipynb) |

The FCC JSON metadata credits the MetaDNS Cu–Au expansion (Du et al., 2026) and
the original ECI fit by Damewood et al. (2022). The exporter is
[export_binary_expansion.py](../experiments/alloy_ce/export_binary_expansion.py).
Recreating those exports requires the upstream ECI/structure files and CLEASE;
the committed JSON can be consumed directly by the PyTorch target.

All four exports use `+1 = Au`, `-1 = Cu`. FCC energies are total periodic-cell
energies in eV. The square exports use
`ClusterExpansionCalculator.calculate_total` on periodic cells without vacancy
padding; they differ from the open-boundary fitting geometry described in the
[toy fitting guide](../icet-ce/README.md). Each JSON retains its source, units,
geometry, fit residual and reference energies.

The Ag–Pd tutorial database is separate from the Cu–Au dissertation application.
See [third-party notices](../THIRD_PARTY_NOTICES.md) for upstream sources and
licensing scope.
