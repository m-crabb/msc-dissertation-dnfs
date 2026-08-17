"""Fit a Cu/Au cluster expansion on a finite, non-periodic square grid.

Reference energies come from ASE's EMT potential. The model is deliberately
simple and non-physical: a flat square lattice with pair and triplet terms only.
"""
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from icet import ClusterExpansion, ClusterSpace, StructureContainer
from trainstation import Optimizer

GRID = 10
N_STRUCTURES = 1000


def build(matrix):
    """Occupation matrix (1 = Au, 0 = Cu) -> lattice padded with vacancies.

    icet always applies periodic boundary conditions, so open boundaries come
    from surrounding the patch with vacancies ('X'), not from setting pbc=False.
    """
    n = len(matrix)
    atoms = Atoms('Cu', cell=[2.5, 2.5, 20], pbc=True).repeat((n + 2, n + 2, 1))
    for atom in atoms:
        i, j = round(atom.position[0] / 2.5), round(atom.position[1] / 2.5)
        atom.symbol = ('Au' if matrix[i][j] else 'Cu') if i < n and j < n else 'X'
    return atoms


def emt_energy(atoms):
    """EMT energy of the isolated patch, with the vacancies stripped out."""
    patch = atoms[[a.index for a in atoms if a.symbol != 'X']]
    patch.pbc = False
    patch.calc = EMT()
    return patch.get_potential_energy()


if __name__ == '__main__':
    cluster_space = ClusterSpace(Atoms('Cu', cell=[2.5, 2.5, 20], pbc=True),
                                 [3.6, 3.6], ['Cu', 'Au', 'X'])
    print(cluster_space)

    rng = np.random.default_rng(42)
    container = StructureContainer(cluster_space)
    for _ in range(N_STRUCTURES):
        matrix = (rng.random((GRID, GRID)) < rng.uniform()).astype(int)
        atoms = build(matrix)
        # per alloy atom, so the model can be applied to other grid sizes
        energy = emt_energy(atoms) / matrix.size
        container.add_structure(atoms, properties={'energy': energy})

    # ridge keeps the ECIs finite; the padding is identical in every structure,
    # which leaves some cluster vector columns constant and hence collinear
    fit_data = container.get_fit_data(key='energy')
    opt = Optimizer(fit_data, fit_method='ridge', train_size=0.9)
    opt.train()
    print(opt)

    ce = ClusterExpansion(cluster_space, opt.parameters)
    ce.write('cluster_expansion.ce')

    cluster_vectors, energies = fit_data
    predicted = cluster_vectors.dot(opt.parameters)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    for idx, name, rmse, color in [(opt.train_set, 'train', opt.rmse_train, '#2a78d6'),
                                   (opt.test_set, 'test', opt.rmse_test, '#eb6834')]:
        ax1.plot(energies[idx], predicted[idx], 'o', ms=5, mfc=color, mec='white',
                 mew=0.5, ls='none', label=f'{name}  RMSE {1000 * rmse:.1f} meV/atom')
    limits = [energies.min(), energies.max()]
    ax1.plot(limits, limits, '-', color='0.7', lw=1, zorder=0)
    ax1.set_xlabel('EMT energy (eV/atom)')
    ax1.set_ylabel('cluster expansion energy (eV/atom)')
    ax1.legend(frameon=False)

    df = ce.to_dataframe()
    for order, name, color in [(2, 'pairs', '#2a78d6'), (3, 'triplets', '#eb6834')]:
        rows = df[df.order == order]
        ax2.plot(rows.radius, rows.eci, 'o', ms=7, mfc=color, mec='white', mew=0.5,
                 ls='none', label=name)
    ax2.axhline(0, color='0.7', lw=1, zorder=0)
    ax2.set_xlabel('cluster radius (Å)')
    ax2.set_ylabel('effective cluster interaction (eV/atom)')
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig('cluster_expansion.png', dpi=200)
