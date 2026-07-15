"""Pure-function physics checks for the SANDI-style GM compartment kernels."""
import numpy as np

from symdwi.gm_compartments import gpd_sphere_attenuation, isotropic_stick_attenuation


def test_gpd_sphere_b0_is_unity():
    atten = gpd_sphere_attenuation(
        np.array([0.0]), big_delta_ms=22.0, small_delta_ms=13.0,
        d_is_mm2_s=3.0e-3, r_s_um=6.0,
    )
    assert np.allclose(atten, 1.0, atol=1e-12)


def test_gpd_sphere_monotonic_decay_in_b():
    bvals = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0])
    atten = gpd_sphere_attenuation(
        bvals, big_delta_ms=22.0, small_delta_ms=13.0,
        d_is_mm2_s=3.0e-3, r_s_um=6.0,
    )
    assert np.all(np.diff(atten) < 0.0)
    assert np.all((atten >= 0.0) & (atten <= 1.0))


def test_gpd_sphere_long_delta_plateaus():
    # As Delta grows, attenuation must plateau (successive differences shrink),
    # not diverge, since a restricted sphere compartment is bounded.
    b = np.array([2000.0])
    deltas = [10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]
    vals = [
        gpd_sphere_attenuation(b, Delta, 10.0, 3.0e-3, 6.0)[0]
        for Delta in deltas
    ]
    diffs = np.abs(np.diff(vals))
    assert diffs[-1] < diffs[0]
    assert diffs[-1] < 5e-3  # near-converged by the last step


def test_isotropic_stick_b0_is_unity():
    atten = isotropic_stick_attenuation(np.array([0.0]), d_in_mm2_s=1.7e-3)
    assert np.allclose(atten, 1.0, atol=1e-12)


def test_isotropic_stick_matches_numeric_orientation_average():
    # Closed form must match a brute-force Monte Carlo orientation average.
    rng = np.random.default_rng(0)
    n = 2_000_000
    cos_theta = 2.0 * rng.random(n) - 1.0
    b, d_in = 1500.0, 1.7e-3
    mc = np.mean(np.exp(-b * d_in * cos_theta ** 2))
    closed = isotropic_stick_attenuation(np.array([b]), d_in)[0]
    assert abs(mc - closed) < 1e-3
