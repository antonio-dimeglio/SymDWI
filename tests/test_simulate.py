"""Core simulation invariants."""
import numpy as np
import pytest

import symdwi


def _bundle():
    pts = np.array([[15, 15, 5], [15, 15, 18], [15, 15, 31]], float)
    return symdwi.Bundle(pts, n_streamlines=150, radius=4.0, dispersion=0.3, seed=0)


def _sim(snr=None, seed=None, return_groundtruth=False, params=None):
    b = _bundle()
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=3, seed=1)
    params = params or symdwi.DWIParameters(f_intra=0.7, f_extra=0.3, te_ms=None,
                                            axon_radius_um=1.0)
    return symdwi.simulate_dwi(
        [b], bvals, bvecs, params, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, snr=snr, seed=seed, sphere="repulsion100",
        return_groundtruth=return_groundtruth,
    )


def test_fractions_must_sum_to_one():
    symdwi.DWIParameters(f_intra=0.6, f_extra=0.2, f_csf=0.2)  # ok
    with pytest.raises(AssertionError):
        symdwi.DWIParameters(f_intra=0.5, f_extra=0.2, f_csf=0.0)  # sums to 0.7


def test_shapes_and_affine():
    bvals, _ = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=3, seed=1)
    signal, affine = _sim()
    assert signal.shape == (30, 30, 36, len(bvals))
    assert affine.shape == (4, 4)
    assert np.allclose(np.diag(affine)[:3], 1.0)  # isotropic 1mm


def test_b0_is_unit_signal_in_fiber_voxels():
    # With te_ms=None all T2 weights are 1, and compartment fractions sum to 1,
    # so the unweighted (b0) signal must be exactly 1 in every fiber voxel.
    signal, _, gt = _sim(return_groundtruth=True)
    m = gt["wm_mask"] > 0
    assert m.sum() > 0
    b0 = signal[..., 0][m]
    assert np.allclose(b0, 1.0, atol=1e-6)


def test_noise_is_reproducible_with_seed():
    s_a, _ = _sim(snr=20, seed=123)
    s_b, _ = _sim(snr=20, seed=123)
    s_c, _ = _sim(snr=20, seed=124)
    assert np.array_equal(s_a, s_b)          # same seed -> identical
    assert not np.array_equal(s_a, s_c)      # different seed -> different


def test_noiseless_signal_in_unit_range():
    signal, _ = _sim()
    assert signal.min() >= 0.0
    assert signal.max() <= 1.0 + 1e-6


def test_tissue_masks_reject_unknown_key():
    b = _bundle()
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=3, seed=1)
    params = symdwi.DWIParameters()
    bad = {"wmm": np.ones((30, 30, 36))}
    with pytest.raises(ValueError):
        symdwi.simulate_dwi([b], bvals, bvecs, params, origin=np.zeros(3),
                            dims=(30, 30, 36), tissue_masks=bad)
