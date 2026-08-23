"""Tests for the literal icet/mchammer Ising baselines.

Written before the implementation. What correct looks like:

1. The cluster-expansion embedding IS the target: total CE energy equals
   -log p̃(x) of the unconstrained ``IsingTarget`` for every configuration,
   because the whole natural-units construction (temperature 1, k_B 1) rests
   on exp(-E) = p̃(x). A mis-set cutoff that silently grabbed second-neighbour
   or z-image bonds would break this on random configurations.
2. The VC-SGC mapping is the algebraic identity
   kappa * N * (c + phi/2)^2 = lambda * d * (c - c_target)^2
   at kappa = lambda, phi = -2 * c_target (N = d). Getting a sign or factor
   wrong here silently samples the wrong ensemble while still "running fine".
3. The canonical ensemble is the hard constraint: swap moves must leave the
   composition bit-exact at every step, with no penalty term anywhere.
4. Every runner reports the timing currency the DNFS comparison needs:
   wall-clock of the MC loop alone, tau_int, ESS, and seconds per effective
   sample. A baseline without these cannot enter the cost-vs-quality grid.
"""

import numpy as np
import pytest
import torch

from discrete_flow_sampler.mcmc.mchammer_ising import (
    atoms_to_spins,
    ising_cluster_expansion,
    ising_supercell,
    run_canonical,
    run_vcsgc,
    spins_to_symbols,
    vcsgc_parameters,
)
from discrete_flow_sampler.targets.ising import IsingTarget

D_SMALL = 4


def _random_spins(n_configs: int, d: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 2, (n_configs, d), generator=generator).float() * 2.0 - 1.0
    )


class TestClusterExpansionEmbedding:
    @pytest.mark.parametrize("sigma,bias", [(0.1, 0.0), (0.223, 0.0), (0.1, 0.3)])
    def test_total_energy_is_minus_log_prob(self, sigma, bias):
        """E_CE(x) = -log p̃(x): the identity the whole embedding rests on."""
        target = IsingTarget(D=D_SMALL, sigma=sigma, bias=bias)
        prim, cluster_space, expansion = ising_cluster_expansion(sigma, bias)
        supercell = ising_supercell(prim, D_SMALL)
        n_sites = len(supercell)
        assert n_sites == D_SMALL * D_SMALL

        spins = _random_spins(8, n_sites)
        for row in spins:
            atoms = supercell.copy()
            atoms.set_chemical_symbols(spins_to_symbols(row.numpy()))
            # ce.predict is per-atom; the target identity is on totals.
            # Tolerance is float32 resolution (IsingTarget computes in
            # float32); a single wrong bond would be off by O(4*sigma) ~ 0.4.
            total_energy = expansion.predict(atoms) * n_sites
            log_prob = target.log_prob(row.unsqueeze(0)).item()
            assert total_energy == pytest.approx(-log_prob, abs=1e-5)

    def test_spin_symbol_round_trip(self):
        spins = np.array([1.0, -1.0, -1.0, 1.0])
        assert np.array_equal(
            atoms_to_spins(spins_to_symbols(spins)), spins
        )


class TestVcsgcMapping:
    def test_kappa_and_phi(self):
        params = vcsgc_parameters(penalty_strength=50.0, target_composition=0.3)
        assert params["kappa"] == 50.0
        assert params["phis"] == {"Au": -0.6}

    def test_penalty_identity(self):
        """kappa*N*(c + phi/2)^2 == lambda*d*(c - c_t)^2 for all c."""
        penalty_strength, target_composition, n_sites = 50.0, 0.3, 100
        params = vcsgc_parameters(penalty_strength, target_composition)
        phi = params["phis"]["Au"]
        for composition in np.linspace(0.0, 1.0, 11):
            mchammer_exponent = (
                params["kappa"] * n_sites * (composition + phi / 2.0) ** 2
            )
            ours = penalty_strength * n_sites * (composition - target_composition) ** 2
            assert mchammer_exponent == pytest.approx(ours, abs=1e-12)


class TestRunners:
    TIMING_KEYS = {
        "wall_seconds_run",
        "wall_seconds_setup",
        "steps_per_second",
        "n_steps",
        "data_write_interval",
    }
    OBSERVABLE_KEYS = {
        "mean",
        "std",
        "tau_int_frames",
        "tau_int_steps",
        "ess",
        "seconds_per_effective_sample",
    }

    def test_vcsgc_summary_schema_and_composition(self):
        summary = run_vcsgc(
            D=D_SMALL,
            sigma=0.1,
            penalty_strength=50.0,
            target_composition=0.5,
            n_steps=4000,
            seed=0,
            data_write_interval=10,
        )
        assert summary["ensemble"] == "vcsgc"
        assert self.TIMING_KEYS <= summary.keys()
        assert summary["wall_seconds_run"] > 0
        for observable in ("composition", "potential"):
            stats = summary["observables"][observable]
            assert self.OBSERVABLE_KEYS <= stats.keys()
            assert stats["ess"] > 0
            assert stats["seconds_per_effective_sample"] == pytest.approx(
                summary["wall_seconds_run"] / stats["ess"]
            )
        # std(c) = 1/sqrt(2*lambda*d) = 0.025 at these settings, so the mean
        # over hundreds of frames sits well inside 0.05 of the request.
        assert summary["observables"]["composition"]["mean"] == pytest.approx(
            0.5, abs=0.05
        )

    def test_vcsgc_recorded_spins_match_the_unrecorded_chain(self):
        """record_spins must not change the chain: chunked driving reproduces
        the one-shot run frame for frame, and the spin frames must agree with
        mchammer's own composition trace (the check that atom order is read
        back consistently, as for the canonical probe)."""
        kwargs = dict(D=D_SMALL, sigma=0.1, penalty_strength=50.0,
                      target_composition=0.5, n_steps=4000, seed=0,
                      data_write_interval=10)
        plain = run_vcsgc(**kwargs)
        recorded = run_vcsgc(**kwargs, record_spins=True)
        spins = recorded["traces"]["spins"]
        n_frames = len(recorded["traces"]["composition"])
        assert spins.shape == (n_frames, D_SMALL * D_SMALL)
        assert spins.dtype == np.int8 and set(np.unique(spins)) <= {-1, 1}
        np.testing.assert_array_equal(
            recorded["traces"]["composition"], plain["traces"]["composition"])
        np.testing.assert_allclose(
            (spins > 0).mean(axis=1), recorded["traces"]["composition"])
        assert "spins" not in plain["traces"]

    def test_canonical_fixes_composition_exactly(self):
        summary = run_canonical(
            D=D_SMALL,
            sigma=0.223,
            target_composition=0.5,
            n_steps=2000,
            seed=0,
            data_write_interval=10,
        )
        assert summary["ensemble"] == "canonical"
        assert self.TIMING_KEYS <= summary.keys()
        # Hard constraint: the realised composition is the quantised target on
        # every frame, not on average.
        assert summary["composition_realised"] == pytest.approx(0.5, abs=1e-12)
        assert summary["composition_is_constant"] is True
        stats = summary["observables"]["potential"]
        assert self.OBSERVABLE_KEYS <= stats.keys()
        assert stats["ess"] > 0

    def test_canonical_quantises_composition_to_lattice(self):
        """c=0.53 on 16 sites has no integer occupation; the runner must land
        on round(c*d)/d rather than silently running something else."""
        summary = run_canonical(
            D=D_SMALL,
            sigma=0.1,
            target_composition=0.53,
            n_steps=500,
            seed=0,
            data_write_interval=10,
        )
        assert summary["composition_realised"] == pytest.approx(
            round(0.53 * 16) / 16, abs=1e-12
        )
