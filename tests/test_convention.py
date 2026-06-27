"""Gradient-convention regression tests.
"""
import numpy as np

import symdwi
from symdwi import world_bvecs_to_fsl, make_affine


def _angle_field(a, b):
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    a = np.divide(a, na, out=np.zeros_like(a), where=na > 0)
    b = np.divide(b, nb, out=np.zeros_like(b), where=nb > 0)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), -1, 1)))


def test_identity_affine_flips_x_only():
    aff = make_affine(np.zeros(3), 1.0)
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, np.sqrt(0.5)], [0, 1, 0]], float)
    out = world_bvecs_to_fsl(bvecs, aff)
    assert np.allclose(out[:, 0], -bvecs[:, 0])
    assert np.allclose(out[:, 1:], bvecs[:, 1:])


def test_convention_is_an_involution_for_axis_aligned_affine():
    aff = make_affine(np.array([10.0, -5.0, 3.0]), 2.0)
    rng = np.random.default_rng(0)
    bvecs = rng.normal(size=(20, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    twice = world_bvecs_to_fsl(world_bvecs_to_fsl(bvecs, aff), aff)
    assert np.allclose(twice, bvecs)


def test_save_dwi_writes_fsl_convention(tmp_path):
    pts = np.array([[10, 10, 18], [15, 15, 18], [20, 20, 18]], float)
    b = symdwi.Bundle(pts, n_streamlines=120, radius=4.0, dispersion=0.2, seed=0)
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=2, seed=1)
    params = symdwi.DWIParameters(te_ms=80.0, axon_radius_um=1.0)
    signal, affine = symdwi.simulate_dwi(
        [b], bvals, bvecs, params, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, snr=None, sphere="repulsion100",
    )
    symdwi.save_dwi(signal, affine, bvals, bvecs, tmp_path)
    on_disk = np.loadtxt(tmp_path / "bvecs.txt").T
    expected = world_bvecs_to_fsl(bvecs, affine)
    assert np.allclose(on_disk, expected, atol=1e-5)
    # and it must differ from the raw world bvecs (i.e. the flip really happened)
    assert not np.allclose(on_disk, bvecs, atol=1e-5)


def test_oblique_bundle_reconstructs_to_groundtruth_via_fsl_bvecs(tmp_path):
    """End-to-end: an FSL-aware consumer of the saved files recovers the true
    orientation of an OBLIQUE bundle. If save_dwi drops the radiological flip,
    the consumer's gradients mirror in x and the DTI peak lands ~90 deg off.
    """
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    pts = np.array([[6, 6, 18], [14, 14, 18], [22, 22, 18]], float)
    b = symdwi.Bundle(pts, n_streamlines=300, radius=5.0, dispersion=0.2, seed=0)
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 40)], n_b0=4, seed=1)
    params = symdwi.DWIParameters(f_intra=0.7, f_extra=0.3, te_ms=80.0, axon_radius_um=1.0)
    signal, affine, gt = symdwi.simulate_dwi(
        [b], bvals, bvecs, params, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, snr=None, sphere="repulsion724", return_groundtruth=True,
    )
    symdwi.save_dwi(signal, affine, bvals, bvecs, tmp_path)

    fsl_on_disk = np.loadtxt(tmp_path / "bvecs.txt").T
    assert np.linalg.det(affine[:3, :3]) > 0
    consumer_world = fsl_on_disk.copy()
    consumer_world[:, 0] *= -1.0

    gtab = gradient_table(bvals, bvecs=consumer_world)
    peaks = TensorModel(gtab).fit(signal).evecs[..., 0]

    m = gt["wm_mask"] > 0
    ref = np.zeros((*m.shape, 3))
    ref[..., 0] = ref[..., 1] = np.sqrt(0.5)
    ang = _angle_field(peaks, ref)[m]
    assert ang.mean() < 15.0
    assert np.median(ang) < 90.0 - 20
