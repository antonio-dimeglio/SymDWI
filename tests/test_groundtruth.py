"""Per-voxel ground-truth extraction (mask, fractions, peaks)."""
import numpy as np

import symdwi


def _angle(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))


def test_single_bundle_has_one_peak_aligned_with_axis():
    # Straight bundle along +z.
    pts = np.array([[15, 15, 4], [15, 15, 18], [15, 15, 32]], float)
    b = symdwi.Bundle(pts, n_streamlines=200, radius=4.0, dispersion=0.2, seed=0)
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=2, seed=1)
    params = symdwi.DWIParameters(f_intra=0.7, f_extra=0.3, te_ms=80.0, axon_radius_um=1.0)
    _, _, gt = symdwi.simulate_dwi(
        [b], bvals, bvecs, params, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, snr=None, sphere="repulsion724", return_groundtruth=True,
    )

    m = gt["wm_mask"] > 0
    assert m.sum() > 0
    # mask and fiber fraction agree on support
    assert np.all(gt["fiber_fraction"][m] > 0)
    assert np.all(gt["fiber_fraction"][~m] == 0)

    # essentially no voxel should have a spurious 2nd peak for a single bundle
    second = np.linalg.norm(gt["peaks"][..., 1, :], axis=-1)[m] > 0
    assert second.mean() < 0.02

    # dominant peak aligns with +z
    p0 = gt["peaks"][..., 0, :][m]
    p0 = p0[np.linalg.norm(p0, axis=1) > 0]
    ang = np.array([_angle(v, [0, 0, 1.0]) for v in p0])
    assert ang.mean() < 10.0


def test_crossing_produces_two_peaks_somewhere():
    # Two near-orthogonal bundles crossing in the x-y plane.
    cc = symdwi.Bundle(np.array([[4, 15, 18], [15, 15, 18], [26, 15, 18]], float),
                       n_streamlines=250, radius=4.0, dispersion=0.3, seed=0)
    cst = symdwi.Bundle(np.array([[15, 4, 18], [15, 15, 18], [15, 26, 18]], float),
                        n_streamlines=250, radius=4.0, dispersion=0.3, seed=1)
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(2000, 40)], n_b0=2, seed=1)
    params = symdwi.DWIParameters(f_intra=0.7, f_extra=0.3, te_ms=80.0, axon_radius_um=1.0)
    _, _, gt = symdwi.simulate_dwi(
        [cc, cst], bvals, bvecs, params, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, snr=None, sphere="repulsion724", return_groundtruth=True,
    )
    m = gt["wm_mask"] > 0
    has_second = np.linalg.norm(gt["peaks"][..., 1, :], axis=-1)[m] > 0
    # the crossing region must yield genuine multi-peak voxels
    assert has_second.sum() > 0
