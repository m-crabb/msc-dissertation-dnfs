# Toy cluster expansion on a square grid

A deliberately non-physical Cu/Au cluster expansion, intended as a cheap energy
function for a neural sampler. Atoms sit on a finite, open-boundary square grid,
reference energies come from ASE's EMT potential, and only pair and triplet
terms are included.

```bash
python fit_cluster_expansion.py   # -> cluster_expansion.ce, cluster_expansion.png
python predict_energy.py          # occupation matrix (1 = Au, 0 = Cu) -> energy
```

`build()` surrounds the grid with vacancy sites (`X`) because icet always applies
periodic boundary conditions and silently ignores `pbc=False`.

The fit targets energy per alloy atom, so `GRID` need not match between the two
scripts, but accuracy is best at the size it was trained on: ~2 meV/atom at
`GRID = 10`, ~45 meV/atom at `GRID = 6`.

## Units

The fitted parameters map a cluster vector to **eV per alloy atom**, so a total
energy needs an explicit scale factor. `ClusterExpansion.predict` returns the
intensive value (multiply by `matrix.size`), while
`ClusterExpansionCalculator.calculate_total` has already scaled by the number of
sites in the padded cell (multiply by `matrix.size / len(atoms)`).

## Fast path

`predict_energy.py` uses `mchammer`'s `ClusterExpansionCalculator`, which builds
the orbits once and then evaluates from an occupation array. Per energy on a
10x10 grid:

| | time |
|---|---|
| EMT reference | 5.5 ms |
| `ClusterExpansion.predict` | 0.62 ms |
| `ClusterExpansionCalculator.calculate_total` | 0.026 ms |
