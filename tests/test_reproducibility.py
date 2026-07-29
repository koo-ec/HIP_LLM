"""Reproducibility and numerical-convergence tests (specification items 26-30)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TOL_CDF_SUP, TOL_MEDIAN, TOL_QUANTILE
from hip_llm.envelopes import cdf_envelope, quantile_envelope, summarise_envelope
from hip_llm.grids import sample_configurations
from hip_llm.numerics import child_generator, spawn_generators
from hip_llm.posterior import run_domain, run_model
from hip_llm.schemas import GlobalSettings, config_hash


# --- 26. identical seeds reproduce byte-identical output -------------------- #
def test_identical_seeds_give_byte_identical_configuration_sets(wide_interval):
    a = sample_configurations(wide_interval, K=160, seed=123)
    b = sample_configurations(wide_interval, K=160, seed=123)
    assert np.array_equal(
        np.array([c.as_array() for c in a]), np.array([c.as_array() for c in b])
    )
    c = sample_configurations(wide_interval, K=160, seed=124)
    assert not np.array_equal(
        np.array([x.as_array() for x in a]), np.array([x.as_array() for x in c])
    )


def test_identical_seeds_give_byte_identical_posteriors(toy_domain, fast_settings, wide_interval):
    a = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    b = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    assert np.array_equal(a.theta, b.theta)
    assert np.array_equal(a.p, b.p)
    assert np.array_equal(a.mu, b.mu) and np.array_equal(a.nu, b.nu)


def test_identical_seeds_give_byte_identical_llm_output(toy_model, fast_settings, wide_interval):
    _, a = run_model(toy_model, [wide_interval] * 2, fast_settings)
    _, b = run_model(toy_model, [wide_interval] * 2, fast_settings)
    assert np.array_equal(a.p_L, b.p_L)
    assert np.array_equal(a.pair_index, b.pair_index)


def test_rng_streams_are_addressable_and_independent():
    """SeedSequence spawning: same address -> same stream; different -> independent."""
    assert np.array_equal(
        child_generator(7, (0, 1, 5)).normal(size=64), child_generator(7, (0, 1, 5)).normal(size=64)
    )
    a = child_generator(7, (0, 1, 5)).normal(size=200_000)
    b = child_generator(7, (0, 2, 5)).normal(size=200_000)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.01

    gens = spawn_generators(7, 4)
    draws = np.array([g.normal(size=100_000) for g in gens])
    corr = np.corrcoef(draws)
    assert np.all(np.abs(corr[~np.eye(4, dtype=bool)]) < 0.02)


def test_no_reliance_on_global_numpy_random_state(toy_domain, fast_settings, wide_interval):
    """Perturbing the legacy global state must not change any result."""
    np.random.seed(0)
    a = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    np.random.seed(12345)
    np.random.random(1000)
    b = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    assert np.array_equal(a.theta, b.theta)


def test_domain_index_changes_the_stream_but_not_the_configurations(toy_domain, fast_settings,
                                                                    wide_interval):
    a = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    b = run_domain(toy_domain, wide_interval, fast_settings, domain_index=1)
    assert [c.as_array().tolist() for c in a.configs] == [c.as_array().tolist() for c in b.configs]
    assert not np.array_equal(a.theta, b.theta)


# --- 27. convergence in S, G and K ------------------------------------------ #
def _settings(S: int, n_mu: int, n_nu: int, K: int) -> GlobalSettings:
    return GlobalSettings(
        n_mu=n_mu, n_nu=n_nu, cdf_points_T=201, S=S, K_per_domain=K,
        max_llm_configuration_pairs=64, seed_global=7, seed_configs=123, seed_pairs=999,
        config_sampling="uniform_random", nu_grid_scheme="log", mu_grid_scheme="midpoint",
    )


@pytest.mark.slow
def test_convergence_in_S(toy_domain, wide_interval):
    """Increasing S must converge; the S=1000 -> S=8000 gap sets the MC error scale."""
    medians = {}
    for S in (1000, 2000, 4000, 8000):
        ds = run_domain(toy_domain, wide_interval, _settings(S, 40, 50, 16), domain_index=0)
        medians[S] = np.mean(quantile_envelope(ds.p, 0.5))
    assert abs(medians[8000] - medians[4000]) <= abs(medians[2000] - medians[1000]) + 2e-3
    assert abs(medians[8000] - medians[1000]) < TOL_MEDIAN


@pytest.mark.slow
def test_convergence_in_G(toy_domain, wide_interval):
    medians, envelopes = {}, {}
    for n in (20, 40, 80, 120):
        ds = run_domain(toy_domain, wide_interval, _settings(4000, n, n, 16), domain_index=0)
        s = summarise_envelope(ds.p, "p")
        medians[n] = 0.5 * (s.median_lower + s.median_upper)
        envelopes[n] = s.envelope_area
    assert abs(medians[120] - medians[80]) < TOL_MEDIAN
    assert abs(medians[120] - medians[40]) < TOL_MEDIAN
    assert abs(envelopes[120] - envelopes[80]) < 0.01


@pytest.mark.slow
def test_convergence_in_K(toy_domain, wide_interval):
    """The envelope must widen monotonically-ish in K and then stabilise."""
    areas = {}
    for K in (10, 40, 160, 320):
        ds = run_domain(toy_domain, wide_interval, _settings(3000, 40, 50, K), domain_index=0)
        areas[K] = cdf_envelope(ds.p, quantity="p").area
    assert areas[10] <= areas[40] + 1e-3 <= areas[160] + 2e-3
    assert abs(areas[320] - areas[160]) < 0.01


@pytest.mark.slow
def test_all_nu_schemes_agree_within_the_declared_tolerances(toy_domain, wide_interval):
    """The unresolved nu-axis choice must not move results beyond tolerance."""
    results = {}
    for scheme in ("log", "linear", "gamma_quantile"):
        st = GlobalSettings(
            n_mu=40, n_nu=50, cdf_points_T=201, S=4000, K_per_domain=40,
            max_llm_configuration_pairs=64, seed_global=7, seed_configs=123, seed_pairs=999,
            config_sampling="uniform_random", nu_grid_scheme=scheme, mu_grid_scheme="midpoint",
        )
        ds = run_domain(toy_domain, wide_interval, st, domain_index=0)
        results[scheme] = (summarise_envelope(ds.p, "p"), cdf_envelope(ds.p, quantity="p"))

    schemes = list(results)
    for i in range(len(schemes)):
        for j in range(i + 1, len(schemes)):
            a, b = results[schemes[i]][0], results[schemes[j]][0]
            assert abs(a.median_lower - b.median_lower) < TOL_MEDIAN
            assert abs(a.median_upper - b.median_upper) < TOL_MEDIAN
            assert abs(a.q05_lower - b.q05_lower) < TOL_QUANTILE
            assert abs(a.q95_upper - b.q95_upper) < TOL_QUANTILE
            ea, eb = results[schemes[i]][1], results[schemes[j]][1]
            assert float(np.max(np.abs(ea.lower - eb.lower))) < TOL_CDF_SUP
            assert float(np.max(np.abs(ea.upper - eb.upper))) < TOL_CDF_SUP


@pytest.mark.slow
def test_seed_sensitivity_stays_within_tolerance(toy_domain, wide_interval):
    """Repeated seeded runs must be stable, as the acceptance criteria require."""
    medians = []
    for seed in (7, 8, 9, 10, 11):
        st = GlobalSettings(
            n_mu=40, n_nu=50, cdf_points_T=201, S=3000, K_per_domain=40,
            max_llm_configuration_pairs=64, seed_global=seed, seed_configs=123, seed_pairs=999,
            config_sampling="uniform_random", nu_grid_scheme="log", mu_grid_scheme="midpoint",
        )
        ds = run_domain(toy_domain, wide_interval, st, domain_index=0)
        medians.append(np.mean(quantile_envelope(ds.p, 0.5)))
    assert max(medians) - min(medians) < TOL_MEDIAN


# --- 30. saved samples regenerate the plotted statistics -------------------- #
def test_saved_samples_regenerate_the_plotted_statistics(tmp_path, toy_domain, fast_settings,
                                                         wide_interval):
    ds = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    original = summarise_envelope(ds.p, "p_1")
    env = cdf_envelope(ds.p, quantity="p_1")

    path = tmp_path / "samples.npz"
    np.savez_compressed(path, p=ds.p, theta=ds.theta, mu=ds.mu, nu=ds.nu)

    loaded = np.load(path)
    assert np.array_equal(loaded["p"], ds.p)
    restored = summarise_envelope(loaded["p"], "p_1")
    assert restored.as_row() == original.as_row()
    env2 = cdf_envelope(loaded["p"], quantity="p_1")
    assert np.array_equal(env.lower, env2.lower) and np.array_equal(env.upper, env2.upper)


def test_config_hash_is_stable_and_discriminating(fast_settings):
    assert fast_settings.hash() == fast_settings.hash()
    other = GlobalSettings(**{**fast_settings.__dict__, "S": fast_settings.S + 1})
    assert other.hash() != fast_settings.hash()
    assert config_hash({"a": 1, "b": [2, 3]}) == config_hash({"b": [2, 3], "a": 1})


def test_config_hash_handles_numpy_and_dataclasses(wide_interval):
    h1 = config_hash({"x": np.array([1.0, 2.0]), "iv": wide_interval})
    h2 = config_hash({"x": np.array([1.0, 2.0]), "iv": wide_interval})
    assert h1 == h2 and len(h1) == 64
