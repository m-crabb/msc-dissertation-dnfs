"""icet/mchammer plumbing spike for the F(c) campaign (2026-06-13).

Resolves the go/no-go for benchmarking the soft-constrained 2D Ising against
literal mchammer vcSGC, and pins the lambda <-> kappa mapping. Run with
`pixi run -e dev python -m scripts.icet_vcsgc_spike`. Zero GPU.

Findings (verified here):
  1. The 2D torus embeds in icet as a single-layer cell with a vacuum gap in z
     and a pair cutoff in (1.0, sqrt(2)); this yields exactly one NN pair orbit
     (4 neighbours per site), no spurious z-image or 2nd-NN bonds.
  2. Our target log p(x) = 2*sigma*sum_<ij> x_i x_j + bias*sum_i x_i (the
     `IsingTarget` convention, A counts each undirected edge twice) is
     reproduced bit-for-bit by a binary CE with
        ECI_zero  = 0
        ECI_point = -bias            (0 here)
        ECI_pair  = -4*sigma
     and the mchammer temperature trick T = 1/kB, so exp(-E/kB T) = exp(-E)
     with E = -log p.
  3. vcSGC mapping: our penalty -lambda*d*(c-c_t)^2 (d = N sites) matches
     mchammer's +kappa*N*(c + phi/2)^2 at
        kappa = lambda,   phi_1 = -2*c_target   (species 1 = the 'up'/Au spin),
     giving <c> -> c_target with std(c) = 1/sqrt(2*kappa*N) = 1/sqrt(2*lambda*d).
     lambda=50 <-> kappa=50 (the docstring's "kappa ~ 200" is only an example of
     a sharply peaked value, not a required match).
  4. The data container logs free_energy_derivative_Au = -(1/N) dF/dc_1, so the
     D=10 reference F(c) is the thermodynamic integral of that over the phi
     ladder, no extra TI machinery to build.
"""

import numpy as np
from ase import Atoms
from ase.units import kB
from icet import ClusterSpace, ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import VCSGCEnsemble


def ising_cluster_expansion(sigma: float, bias: float = 0.0):
    """Binary CE whose total energy equals -log p of the IsingTarget."""
    prim = Atoms(
        "Au", positions=[(0, 0, 0)], cell=[[1, 0, 0], [0, 1, 0], [0, 0, 10]], pbc=True
    )
    cs = ClusterSpace(prim, cutoffs=[1.1], chemical_symbols=["Au", "Ag"])
    ce = ClusterExpansion(cs, parameters=[0.0, -bias, -4.0 * sigma])
    return prim, cs, ce


def run_vcsgc(D: int, sigma: float, c_target: float, lam: float, n_steps: int,
              seed: int = 0):
    """One vcSGC chain at kappa=lam, phi=-2*c_target. Returns (mean_c, std_c)."""
    prim, cs, ce = ising_cluster_expansion(sigma)
    sc = prim.repeat((D, D, 1))
    N = len(sc)
    n_up = int(round(c_target * N))
    syms = ["Au"] * n_up + ["Ag"] * (N - n_up)
    np.random.default_rng(seed).shuffle(syms)
    sc.set_chemical_symbols(syms)
    calc = ClusterExpansionCalculator(sc, ce)
    ens = VCSGCEnsemble(
        sc, calc, temperature=1.0 / kB, kappa=lam, phis={"Au": -2.0 * c_target},
        ensemble_data_write_interval=100,
    )
    ens.run(n_steps)
    df = ens.data_container.data
    c = df["Au_count"].values[len(df) // 3:] / N
    return c.mean(), c.std()


if __name__ == "__main__":
    D, sigma, lam = 10, 0.1, 50.0
    print(f"{'c_t':>5} {'<c>':>8} {'std(c)':>8} {'1/sqrt(2*lam*d)':>16}")
    for ct in (0.50, 0.65, 0.80):
        mc, sc_ = run_vcsgc(D, sigma, ct, lam, n_steps=300_000)
        print(f"{ct:>5} {mc:>8.4f} {sc_:>8.4f} {1/np.sqrt(2*lam*D*D):>16.4f}")
