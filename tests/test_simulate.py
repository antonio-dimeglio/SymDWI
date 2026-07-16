"""Core simulation invariants."""
import json

import numpy as np
import pytest

import symdwi


def _geometry(seed=0):
    pts = np.array([[15, 15, 5], [15, 15, 18], [15, 15, 31]], float)
    return symdwi.BundleGeometry(pts, n_streamlines=150, radius=4.0, dispersion=0.3, seed=seed)


def _bundle(tissue=None, seed=0):
    return symdwi.Bundle(_geometry(seed=seed), tissue=tissue)


def _sim(snr=None, seed=None, return_groundtruth=False, scan=None, tissue=None, bundles=None):
    bundles = bundles if bundles is not None else [_bundle(tissue=tissue)]
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=3, seed=1)
    scan = scan or symdwi.ScanParameters(te_ms=None)
    return symdwi.simulate_dwi(
        bundles, bvals, bvecs, scan, tissue=tissue, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, snr=snr, seed=seed, sphere="repulsion100",
        return_groundtruth=return_groundtruth,
    )


def test_f_csf_split_bounds_check():
    # f_csf_split must be accepted only inside [0, 1].
    symdwi.TissueParameters(f_csf_split=0.2)
    symdwi.TissueParameters(f_csf_split=0.0)
    symdwi.TissueParameters(f_csf_split=1.0)
    with pytest.raises(AssertionError):
        symdwi.TissueParameters(f_csf_split=1.5)
    with pytest.raises(AssertionError):
        symdwi.TissueParameters(f_csf_split=-0.1)


def test_no_f_intra_field_exists():
    # f_intra was removed; it must not exist as an attribute or accepted kwarg.
    assert not hasattr(symdwi.TissueParameters(), "f_intra")
    with pytest.raises(TypeError):
        symdwi.TissueParameters(f_intra=0.7)


def test_f_myelin_bounds_check():
    # f_myelin must be accepted only inside [0, 1].
    symdwi.TissueParameters(f_myelin=0.2)
    symdwi.TissueParameters(f_myelin=0.0)
    symdwi.TissueParameters(f_myelin=1.0)
    with pytest.raises(AssertionError):
        symdwi.TissueParameters(f_myelin=1.5)
    with pytest.raises(AssertionError):
        symdwi.TissueParameters(f_myelin=-0.1)


def test_gm_f_in_f_ec_bounds_check():
    # f_in, f_ec must each be in [0, 1] and sum to at most 1.
    symdwi.GMParameters(f_in=0.5, f_ec=0.35)
    symdwi.GMParameters(f_in=0.6, f_ec=0.4)
    with pytest.raises(AssertionError):
        symdwi.GMParameters(f_in=1.5)
    with pytest.raises(AssertionError):
        symdwi.GMParameters(f_in=-0.1)
    with pytest.raises(AssertionError):
        symdwi.GMParameters(f_in=0.9, f_ec=0.9)


def test_gm_f_is_is_derived_not_settable():
    # f_is is a read-only property computed as 1 - f_in - f_ec.
    gm = symdwi.GMParameters(f_in=0.5, f_ec=0.35)
    assert gm.f_is == pytest.approx(0.15)
    with pytest.raises(TypeError):
        symdwi.GMParameters(f_is=0.1)


def test_gm_delta_small_delta_bounds_check():
    # small_delta_ms must be positive and <= big_delta_ms.
    symdwi.GMParameters(big_delta_ms=22.0, small_delta_ms=13.0)
    symdwi.GMParameters(big_delta_ms=10.0, small_delta_ms=10.0)
    with pytest.raises(AssertionError):
        symdwi.GMParameters(big_delta_ms=10.0, small_delta_ms=20.0)
    with pytest.raises(AssertionError):
        symdwi.GMParameters(big_delta_ms=0.0)
    with pytest.raises(AssertionError):
        symdwi.GMParameters(small_delta_ms=0.0)


def test_gm_parameters_defaults_are_documented_ranges():
    # Defaults must stay inside SANDI's cited sensitivity-analysis ranges.
    gm = symdwi.GMParameters()
    assert 0.15 <= gm.f_in <= 0.85
    assert 0.15 <= gm.f_ec <= 0.85
    assert 0.5e-3 <= gm.d_in <= 2.5e-3
    assert 0.5e-3 <= gm.d_ec <= 2.5e-3
    assert 2.0 <= gm.r_s <= 10.0


def test_generate_bvals_bvecs_shapes_and_b0_rows():
    # Output shapes match n_b0 + total directions; b0 rows are zero vectors.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 8), (2500, 12)], n_b0=2, seed=0)
    assert bvals.shape == (22,)
    assert bvecs.shape == (22, 3)
    assert np.array_equal(bvals[:2], [0, 0])
    assert np.array_equal(bvecs[:2], np.zeros((2, 3)))
    assert np.all(bvals[2:10] == 1000)
    assert np.all(bvals[10:] == 2500)


def test_generate_bvals_bvecs_directions_are_unit_norm():
    # Non-b0 directions must be unit vectors after charge dispersion.
    _, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 16)], n_b0=1, seed=0, iters=50)
    norms = np.linalg.norm(bvecs[1:], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_generate_bvals_bvecs_seed_reproducibility():
    # Same seed reproduces directions exactly; a different seed does not.
    _, a = symdwi.generate_bvals_bvecs(shells=[(1000, 10)], n_b0=1, seed=7, iters=20)
    _, b = symdwi.generate_bvals_bvecs(shells=[(1000, 10)], n_b0=1, seed=7, iters=20)
    _, c = symdwi.generate_bvals_bvecs(shells=[(1000, 10)], n_b0=1, seed=8, iters=20)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generate_bvals_bvecs_no_shells_is_just_b0():
    # An empty shell list with n_b0>0 must not error, and yield only b0 rows.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[], n_b0=3, seed=0)
    assert bvals.shape == (3,)
    assert np.array_equal(bvals, np.zeros(3))
    assert np.array_equal(bvecs, np.zeros((3, 3)))


def test_make_affine_basic_construction():
    # Diagonal holds voxel_size, translation holds origin, bottom row is identity.
    aff = symdwi.make_affine(np.array([1.0, -2.0, 3.5]), 2.0)
    assert np.allclose(np.diag(aff)[:3], 2.0)
    assert np.allclose(aff[:3, 3], [1.0, -2.0, 3.5])
    assert np.array_equal(aff[3], [0, 0, 0, 1])


def test_world_bvecs_to_fsl_flips_x_for_identity_affine():
    # Identity (positive-determinant) affine negates only the x component.
    aff = symdwi.make_affine(np.zeros(3), 1.0)
    bvecs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.5, 0.5, np.sqrt(0.5)]])
    out = symdwi.world_bvecs_to_fsl(bvecs, aff)
    assert np.allclose(out[:, 0], -bvecs[:, 0])
    assert np.allclose(out[:, 1:], bvecs[:, 1:])


def test_world_bvecs_to_fsl_zero_bvec_row_stays_zero():
    # b=0 rows (zero vectors) must map to zero, not be corrupted by normalization.
    aff = symdwi.make_affine(np.zeros(3), 1.0)
    bvecs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = symdwi.world_bvecs_to_fsl(bvecs, aff)
    assert np.array_equal(out[0], [0.0, 0.0, 0.0])


def test_shapes_and_affine():
    # Signal shape matches (*dims, n_gradients); affine is 4x4 with isotropic diagonal.
    bvals, _ = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=3, seed=1)
    signal, affine = _sim()
    assert signal.shape == (30, 30, 36, len(bvals))
    assert affine.shape == (4, 4)
    assert np.allclose(np.diag(affine)[:3], 1.0)


def test_b0_is_unit_signal_in_fiber_voxels():
    # With te_ms=None all T2 weights are 1 and fractions sum to 1, so b0 signal must be exactly 1.
    signal, _, gt = _sim(return_groundtruth=True)
    m = gt["wm_mask"] > 0
    assert m.sum() > 0
    b0 = signal[..., 0][m]
    assert np.allclose(b0, 1.0, atol=1e-6)


def test_noiseless_signal_in_unit_range():
    # Attenuation-only signal (no noise) must stay within [0, 1].
    signal, _ = _sim()
    assert signal.min() >= 0.0
    assert signal.max() <= 1.0 + 1e-6


def test_non_unit_norm_bvecs_are_normalized():
    # A bvec supplied with the wrong magnitude must give the same signal as its unit form.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 10)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None)
    b = _bundle()
    s_unit, _ = symdwi.simulate_dwi(
        [b], bvals, bvecs, scan, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, sphere="repulsion100",
    )
    s_scaled, _ = symdwi.simulate_dwi(
        [b], bvals, bvecs * 3.7, scan, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, sphere="repulsion100",
    )
    assert np.allclose(s_unit, s_scaled, atol=1e-10)


def test_empty_bundle_list_gives_all_zero_signal_without_masks():
    # No bundles and no tissue_masks: nothing to voxelise, output is all zero.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 5)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None)
    signal, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, origin=np.zeros(3), dims=(10, 10, 10), voxel_size=1.0,
    )
    assert np.array_equal(signal, np.zeros_like(signal))


def test_n_jobs_threaded_matches_single_threaded():
    # Threaded voxel processing must be bit-identical to the single-threaded path.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 20)], n_b0=2, seed=1)
    scan = symdwi.ScanParameters(te_ms=None)
    b = _bundle()
    s1, _ = symdwi.simulate_dwi(
        [b], bvals, bvecs, scan, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, sphere="repulsion100", n_jobs=1,
    )
    s4, _ = symdwi.simulate_dwi(
        [b], bvals, bvecs, scan, origin=np.zeros(3), dims=(30, 30, 36),
        voxel_size=1.0, sphere="repulsion100", n_jobs=4,
    )
    assert np.array_equal(s1, s4)


def test_noise_is_reproducible_with_seed():
    # Same noise seed reproduces the exact signal; a different seed does not.
    s_a, _ = _sim(snr=20, seed=123)
    s_b, _ = _sim(snr=20, seed=123)
    s_c, _ = _sim(snr=20, seed=124)
    assert np.array_equal(s_a, s_b)
    assert not np.array_equal(s_a, s_c)


def test_noise_is_zero_in_background_voxels():
    # SNR is relative to each voxel's own S0: a voxel with S0=0 must stay exactly zero.
    signal, _, gt = _sim(snr=5, seed=1, return_groundtruth=True)
    m = gt["wm_mask"] == 0
    assert m.sum() > 0
    assert np.array_equal(signal[m], np.zeros_like(signal[m]))


def test_noise_magnitude_scales_with_per_voxel_s0():
    # Noise std must scale with each voxel's own S0, checked via Monte Carlo at two signal levels.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 5)], n_b0=1, seed=1)
    dims = (10, 10, 10)
    scan_low = symdwi.ScanParameters(te_ms=None, background_csf=0.2)
    scan_high = symdwi.ScanParameters(te_ms=None, background_csf=0.8)

    def b0_std_at_origin(scan, snr, n_draws=400):
        vals = []
        for draw in range(n_draws):
            signal, _ = symdwi.simulate_dwi(
                [], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
                voxel_size=1.0, snr=snr, seed=draw,
            )
            vals.append(signal[0, 0, 0, 0])
        return np.std(vals)

    std_low = b0_std_at_origin(scan_low, snr=5)
    std_high = b0_std_at_origin(scan_high, snr=5)
    ratio = std_high / std_low
    assert 3.0 < ratio < 5.0


def test_snr_requires_b0_volume():
    # snr with no b=0 entries anywhere in bvals must raise, since S0 is undefined.
    bvals = np.array([1000.0, 1000.0])
    bvecs = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    b = _bundle()
    scan = symdwi.ScanParameters(te_ms=None)
    with pytest.raises(ValueError):
        symdwi.simulate_dwi(
            [b], bvals, bvecs, scan, origin=np.zeros(3), dims=(30, 30, 36),
            voxel_size=1.0, snr=10,
        )


def test_tissue_masks_reject_unknown_key():
    # tissue_masks only accepts "wm", "gm", "csf" keys.
    b = _bundle()
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=3, seed=1)
    scan = symdwi.ScanParameters()
    bad = {"wmm": np.ones((30, 30, 36))}
    with pytest.raises(ValueError):
        symdwi.simulate_dwi([b], bvals, bvecs, scan, origin=np.zeros(3),
                            dims=(30, 30, 36), tissue_masks=bad)


def test_tissue_masks_reject_wrong_shape():
    # Each tissue_masks array must match dims exactly.
    b = _bundle()
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 10)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters()
    bad = {"wm": np.ones((5, 5, 5))}
    with pytest.raises(ValueError):
        symdwi.simulate_dwi([b], bvals, bvecs, scan, origin=np.zeros(3),
                            dims=(30, 30, 36), tissue_masks=bad)


def test_tissue_masks_values_are_clipped_to_unit_range():
    # Out-of-range mask values (e.g. from noisy probability maps) must be clipped, not rejected.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 5)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None, d_iso=3.0e-3)
    dims = (10, 10, 10)
    signal_over, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"csf": np.full(dims, 5.0)},
    )
    signal_clamped, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"csf": np.ones(dims)},
    )
    assert np.array_equal(signal_over, signal_clamped)


def test_tissue_masks_fractions_summing_below_one_are_renormalized():
    # Partial-volume masks (wm+gm+csf < 1 at a voxel) must be renormalized to sum to 1, not
    # scaled down as if the missing fraction contributes zero signal.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 5)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None, d_iso=3.0e-3)
    dims = (10, 10, 10)
    signal_half, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"csf": np.full(dims, 0.5)},
    )
    signal_full, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"csf": np.ones(dims)},
    )
    assert np.array_equal(signal_half, signal_full)


def test_tissue_mask_voxels_with_no_bundle_still_get_signal():
    # Voxel discovery isn't limited to streamline-touched voxels: a tissue_masks voxel far from
    # any bundle (e.g. deep GM or CSF) must still get its tissue-appropriate signal.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 10)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None, d_iso=3.0e-3)
    dims = (10, 10, 10)

    gm_signal, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"gm": np.ones(dims)},
    )
    assert not np.allclose(gm_signal, 0.0)

    csf_signal, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"csf": np.ones(dims)},
    )
    expected_csf = np.exp(-bvals * scan.d_iso)
    assert np.allclose(csf_signal[0, 0, 0], expected_csf)


def test_csf_tissue_mask_takes_precedence_over_f_csf_split():
    # A CSF tissue_masks entry must override per-bundle f_csf_split entirely.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 20)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None, d_iso=3.0e-3)
    dims = (30, 30, 36)

    csf_mask = np.zeros(dims)
    csf_mask[:, :, :20] = 0.5

    def run(f_csf_split):
        tissue = symdwi.TissueParameters(f_csf_split=f_csf_split, axon_radius=None)
        b = _bundle(tissue=tissue)
        return symdwi.simulate_dwi(
            [b], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
            voxel_size=1.0, sphere="repulsion100",
            tissue_masks={"wm": 1.0 - csf_mask, "csf": csf_mask},
            return_groundtruth=True,
        )

    signal_low, _, gt = run(0.0)
    signal_high, _, _ = run(0.9)

    m = gt["wm_mask"] > 0
    assert np.allclose(signal_low[m], signal_high[m])

    signal_no_mask_low, _, gt2 = symdwi.simulate_dwi(
        [_bundle(tissue=symdwi.TissueParameters(f_csf_split=0.0, axon_radius=None))],
        bvals, bvecs, scan, origin=np.zeros(3), dims=dims, voxel_size=1.0,
        sphere="repulsion100", return_groundtruth=True,
    )
    signal_no_mask_high, _, _ = symdwi.simulate_dwi(
        [_bundle(tissue=symdwi.TissueParameters(f_csf_split=0.9, axon_radius=None))],
        bvals, bvecs, scan, origin=np.zeros(3), dims=dims, voxel_size=1.0,
        sphere="repulsion100", return_groundtruth=True,
    )
    m2 = gt2["wm_mask"] > 0
    assert not np.allclose(signal_no_mask_low[m2], signal_no_mask_high[m2])


def test_per_bundle_override_reaches_signal():
    # A bundle-level TissueParameters override must drive that bundle's voxels, matched against
    # a hand-computed stick+zeppelin signal, and differ from what the default would give.
    from dipy.data import get_sphere

    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 30)], n_b0=1, seed=1)
    default_tissue = symdwi.TissueParameters(di_axial=1.7e-3, de_axial=1.7e-3,
                                              de_radial=0.5e-3, axon_radius=None)
    override = symdwi.TissueParameters(di_axial=3.0e-3, de_axial=2.0e-3,
                                        de_radial=0.9e-3, axon_radius=None)
    b = _bundle(tissue=override)
    scan = symdwi.ScanParameters(te_ms=None)
    signal, affine, gt = symdwi.simulate_dwi(
        [b], bvals, bvecs, scan, tissue=default_tissue, origin=np.zeros(3),
        dims=(30, 30, 36), voxel_size=1.0, sphere="repulsion100",
        return_groundtruth=True,
    )
    m = gt["wm_mask"] > 0
    idx = tuple(np.argwhere(m)[0])
    vf_intra = gt["fiber_fraction"][idx]
    remaining = 1.0 - vf_intra

    sphere = get_sphere(name="repulsion100")
    from symdwi._voxel import voxelise_bundles
    volume = voxelise_bundles([b], np.zeros(3), (30, 30, 36), 1.0, sphere=sphere)
    odf = volume[idx][0]
    odf = odf / odf.sum()

    dot_sq = (bvecs @ sphere.vertices.T) ** 2
    stick = np.exp(-bvals[:, None] * override.di_axial * dot_sq)
    zeppelin = (
        np.exp(-bvals[:, None] * override.de_radial)
        * np.exp(-bvals[:, None] * (override.de_axial - override.de_radial) * dot_sq)
    )
    expected = vf_intra * (stick @ odf) + remaining * (zeppelin @ odf)

    assert np.allclose(signal[idx], expected, atol=1e-6)

    stick_def = np.exp(-bvals[:, None] * default_tissue.di_axial * dot_sq)
    zeppelin_def = (
        np.exp(-bvals[:, None] * default_tissue.de_radial)
        * np.exp(-bvals[:, None] * (default_tissue.de_axial - default_tissue.de_radial) * dot_sq)
    )
    expected_default = vf_intra * (stick_def @ odf) + remaining * (zeppelin_def @ odf)
    assert not np.allclose(signal[idx], expected_default, atol=1e-3)


def _hand_signal(bvals, bvecs, sphere, odf, tissue, vf_intra):
    """Hand-compute the stick+zeppelin signal for one bundle's voxel contribution."""
    remaining = 1.0 - vf_intra
    vf_extra = remaining * (1.0 - tissue.f_csf_split)
    vf_csf = remaining * tissue.f_csf_split
    dot_sq = (bvecs @ sphere.vertices.T) ** 2
    stick = np.exp(-bvals[:, None] * tissue.di_axial * dot_sq)
    zeppelin = (
        np.exp(-bvals[:, None] * tissue.de_radial)
        * np.exp(-bvals[:, None] * (tissue.de_axial - tissue.de_radial) * dot_sq)
    )
    water = np.ones_like(bvals)
    return vf_intra * (stick @ odf) + vf_extra * (zeppelin @ odf) + vf_csf * water


def test_two_nonoverlapping_bundles_no_cross_contamination():
    # Two spatially separate bundles must each match their own hand-computed signal, with no
    # leakage of the other bundle's tissue parameters into their voxels.
    from dipy.data import get_sphere
    from symdwi._voxel import voxelise_bundles

    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1500, 20)], n_b0=1, seed=1)
    tissue_a = symdwi.TissueParameters(di_axial=1.0e-3, de_axial=1.8e-3,
                                        de_radial=0.4e-3, axon_radius=None)
    tissue_b = symdwi.TissueParameters(di_axial=2.5e-3, de_axial=2.5e-3,
                                        de_radial=1.2e-3, axon_radius=None)
    geom_a = symdwi.BundleGeometry(
        np.array([[5, 5, 5], [5, 5, 15], [5, 5, 25]], float),
        n_streamlines=100, radius=2.0, seed=0,
    )
    geom_b = symdwi.BundleGeometry(
        np.array([[25, 25, 5], [25, 25, 15], [25, 25, 25]], float),
        n_streamlines=100, radius=2.0, seed=1,
    )
    a = symdwi.Bundle(geom_a, tissue=tissue_a)
    b = symdwi.Bundle(geom_b, tissue=tissue_b)
    scan = symdwi.ScanParameters(te_ms=None)
    dims = (30, 30, 30)
    signal, affine, gt = symdwi.simulate_dwi(
        [a, b], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, sphere="repulsion100", return_groundtruth=True,
    )

    ia = (5, 5, 15)
    ib = (25, 25, 15)
    assert gt["wm_mask"][ia] > 0
    assert gt["wm_mask"][ib] > 0

    sphere = get_sphere(name="repulsion100")
    volume = voxelise_bundles([a, b], np.zeros(3), dims, 1.0, sphere=sphere)

    odf_a = volume[ia][0]
    odf_a = odf_a / odf_a.sum()
    expected_a = _hand_signal(bvals, bvecs, sphere, odf_a, tissue_a, gt["fiber_fraction"][ia])

    odf_b = volume[ib][1]
    odf_b = odf_b / odf_b.sum()
    expected_b = _hand_signal(bvals, bvecs, sphere, odf_b, tissue_b, gt["fiber_fraction"][ib])

    assert np.allclose(signal[ia], expected_a, atol=1e-6)
    assert np.allclose(signal[ib], expected_b, atol=1e-6)
    assert not np.allclose(signal[ia], expected_b, atol=1e-3)
    assert not np.allclose(signal[ib], expected_a, atol=1e-3)


def test_crossing_voxel_is_density_weighted_mixture():
    # A genuine two-bundle crossing voxel's signal must equal the density-weighted mixture of
    # each bundle's own hand-computed signal, not one bundle overriding the other.
    from dipy.data import get_sphere
    from symdwi._voxel import voxelise_bundles

    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1500, 20)], n_b0=1, seed=1)
    tissue_a = symdwi.TissueParameters(di_axial=1.0e-3, de_axial=1.8e-3,
                                        de_radial=0.4e-3, axon_radius=None)
    tissue_b = symdwi.TissueParameters(di_axial=2.5e-3, de_axial=2.5e-3,
                                        de_radial=1.2e-3, axon_radius=None)
    geom_a = symdwi.BundleGeometry(
        np.array([[4, 15, 15], [15, 15, 15], [26, 15, 15]], float),
        n_streamlines=250, radius=4.0, dispersion=0.0, seed=0,
    )
    geom_b = symdwi.BundleGeometry(
        np.array([[15, 4, 15], [15, 15, 15], [15, 26, 15]], float),
        n_streamlines=250, radius=4.0, dispersion=0.0, seed=1,
    )
    cc = symdwi.Bundle(geom_a, tissue=tissue_a)
    cst = symdwi.Bundle(geom_b, tissue=tissue_b)
    scan = symdwi.ScanParameters(te_ms=None)
    dims = (30, 30, 30)
    signal, affine, gt = symdwi.simulate_dwi(
        [cc, cst], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, sphere="repulsion100", return_groundtruth=True,
    )

    center = (15, 15, 15)
    assert gt["wm_mask"][center] > 0

    sphere = get_sphere(name="repulsion100")
    volume = voxelise_bundles([cc, cst], np.zeros(3), dims, 1.0, sphere=sphere)
    assert len(volume[center]) == 2

    odf_a = volume[center][0]
    odf_a = odf_a / odf_a.sum()
    odf_b = volume[center][1]
    odf_b = odf_b / odf_b.sum()

    fiber_vols = {}
    for key, per_bundle in volume.items():
        for bidx, hist in per_bundle.items():
            r = (tissue_a if bidx == 0 else tissue_b).axon_radius
            count = hist.sum()
            fiber_vols[(key, bidx)] = (
                np.pi * (r * 1e-3) ** 2 * count if r is not None else count
            )
    max_fv = max(fiber_vols.values())
    vf_a = min(1.0, fiber_vols[(center, 0)] / max_fv)
    vf_b = min(1.0, fiber_vols[(center, 1)] / max_fv)

    sig_a = _hand_signal(bvals, bvecs, sphere, odf_a, tissue_a, vf_a)
    sig_b = _hand_signal(bvals, bvecs, sphere, odf_b, tissue_b, vf_b)
    expected = sig_a + sig_b

    assert np.allclose(signal[center], expected, atol=1e-6)


def test_axon_radius_changes_density_not_decay_shape():
    # axon_radius only matters relative to other bundles in the same scene: it shifts relative
    # fiber_fraction, but a bundle simulated alone gives a bit-identical signal regardless of it.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 20), (2000, 20)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None)

    geom_a = symdwi.BundleGeometry(
        np.array([[4, 15, 15], [15, 15, 15], [26, 15, 15]], float),
        n_streamlines=200, radius=4.0, dispersion=0.0, seed=0,
    )
    geom_b = symdwi.BundleGeometry(
        np.array([[15, 4, 15], [15, 15, 15], [15, 26, 15]], float),
        n_streamlines=200, radius=4.0, dispersion=0.0, seed=1,
    )

    def run(r_a, r_b):
        a = symdwi.Bundle(geom_a, tissue=symdwi.TissueParameters(axon_radius=r_a))
        b = symdwi.Bundle(geom_b, tissue=symdwi.TissueParameters(axon_radius=r_b))
        return symdwi.simulate_dwi(
            [a, b], bvals, bvecs, scan, origin=np.zeros(3), dims=(30, 30, 30),
            voxel_size=1.0, sphere="repulsion100", return_groundtruth=True,
        )

    signal_equal, _, gt_equal = run(1.0, 1.0)
    signal_skewed, _, gt_skewed = run(0.5, 5.0)

    center = (15, 15, 15)
    assert gt_equal["wm_mask"][center] > 0

    assert not np.isclose(
        gt_equal["fiber_fraction"][center], gt_skewed["fiber_fraction"][center]
    )

    tissue_small = symdwi.TissueParameters(axon_radius=0.3)
    tissue_large = symdwi.TissueParameters(axon_radius=8.0)
    solo_small, _ = symdwi.simulate_dwi(
        [symdwi.Bundle(geom_a, tissue=tissue_small)], bvals, bvecs, scan,
        origin=np.zeros(3), dims=(30, 30, 30), voxel_size=1.0, sphere="repulsion100",
    )
    solo_large, _ = symdwi.simulate_dwi(
        [symdwi.Bundle(geom_a, tissue=tissue_large)], bvals, bvecs, scan,
        origin=np.zeros(3), dims=(30, 30, 30), voxel_size=1.0, sphere="repulsion100",
    )
    assert np.allclose(solo_small, solo_large, atol=1e-10)


def test_density_driven_fiber_fraction_preserved():
    # Denser streamline packing must yield a strictly higher fiber_fraction than sparse packing.
    tissue = symdwi.TissueParameters(axon_radius=1.0)
    dense = symdwi.Bundle(
        symdwi.BundleGeometry(
            np.array([[5, 5, 5], [5, 5, 15], [5, 5, 25]], float),
            n_streamlines=400, radius=2.0, seed=0,
        ),
        tissue=tissue,
    )
    sparse = symdwi.Bundle(
        symdwi.BundleGeometry(
            np.array([[20, 20, 5], [20, 20, 15], [20, 20, 25]], float),
            n_streamlines=40, radius=2.0, seed=1,
        ),
        tissue=tissue,
    )
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 10)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None)
    _, _, gt = symdwi.simulate_dwi(
        [dense, sparse], bvals, bvecs, scan, origin=np.zeros(3), dims=(30, 30, 30),
        voxel_size=1.0, sphere="repulsion100", return_groundtruth=True,
    )
    dense_frac = gt["fiber_fraction"][5, 5, 15]
    sparse_frac = gt["fiber_fraction"][20, 20, 15]
    assert dense_frac > sparse_frac > 0


def test_myelin_pool_off_matches_prior_two_pool_model():
    # f_myelin=0 must reproduce the signal exactly as if the myelin term didn't exist,
    # regardless of what t2_myelin_ms is set to.
    tissue = symdwi.TissueParameters(f_myelin=0.0, axon_radius=None)
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 20)], n_b0=2, seed=1)
    scan = symdwi.ScanParameters(te_ms=80.0)
    signal, _ = _sim(scan=scan, tissue=tissue)

    tissue_hot_myelin_but_off = symdwi.TissueParameters(
        f_myelin=0.0, t2_myelin_ms=5.0, axon_radius=None,
    )
    signal_alt, _ = _sim(scan=scan, tissue=tissue_hot_myelin_but_off)
    assert np.allclose(signal, signal_alt, atol=1e-10)


def test_myelin_pool_fast_component_recovered_across_te_sweep():
    # A nonzero myelin fraction must show up as a genuine fast-decaying component when fitting
    # a biexponential T2 curve at the scene's purely intra-axonal (vf_intra=1) voxel.
    t2_myelin_ms = 15.0
    t2_tissue_ms = 70.0
    f_myelin = 0.3

    tissue = symdwi.TissueParameters(
        f_csf_split=0.0, axon_radius=None,
        t2_intra_ms=t2_tissue_ms, t2_extra_ms=t2_tissue_ms,
        t2_myelin_ms=t2_myelin_ms, f_myelin=f_myelin,
    )
    bvals = np.array([0.0])
    bvecs = np.zeros((1, 3))

    tes = np.array([10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 150.0, 200.0])
    b = _bundle(tissue=tissue)
    dims = (30, 30, 36)

    b0_signal = []
    idx = None
    for te in tes:
        scan = symdwi.ScanParameters(te_ms=float(te))
        signal, _, gt = symdwi.simulate_dwi(
            [b], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
            voxel_size=1.0, sphere="repulsion100", return_groundtruth=True,
        )
        if idx is None:
            idx = np.unravel_index(np.argmax(gt["fiber_fraction"]), dims)
            assert gt["fiber_fraction"][idx] == 1.0
        b0_signal.append(signal[idx][0])
    b0_signal = np.array(b0_signal)

    from scipy.optimize import curve_fit

    def biexp(te, frac_fast, t2_fast, t2_slow):
        return frac_fast * np.exp(-te / t2_fast) + (1 - frac_fast) * np.exp(-te / t2_slow)

    popt, _ = curve_fit(
        biexp, tes, b0_signal,
        p0=[0.3, 20.0, 70.0], bounds=([0.0, 5.0, 50.0], [1.0, 45.0, 100.0]),
    )
    fitted_frac_fast, fitted_t2_fast, _ = popt

    assert abs(fitted_t2_fast - t2_myelin_ms) < 5.0
    assert abs(fitted_frac_fast - f_myelin) < 0.05


def test_per_bundle_myelin_fraction_propagation():
    # Two bundles differing only in f_myelin must show different TE-weighted b0 signal in their
    # own voxels, matched against a hand-computed two-term T2 mixture.
    tissue_low = symdwi.TissueParameters(
        f_myelin=0.0, axon_radius=None, t2_intra_ms=70.0, t2_extra_ms=70.0,
    )
    tissue_high = symdwi.TissueParameters(
        f_myelin=0.3, t2_myelin_ms=15.0, axon_radius=None,
        t2_intra_ms=70.0, t2_extra_ms=70.0,
    )
    geom_a = symdwi.BundleGeometry(
        np.array([[5, 5, 5], [5, 5, 15], [5, 5, 25]], float),
        n_streamlines=100, radius=2.0, seed=0,
    )
    geom_b = symdwi.BundleGeometry(
        np.array([[25, 25, 5], [25, 25, 15], [25, 25, 25]], float),
        n_streamlines=100, radius=2.0, seed=1,
    )
    a = symdwi.Bundle(geom_a, tissue=tissue_low)
    b = symdwi.Bundle(geom_b, tissue=tissue_high)
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 10)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=25.0)
    dims = (30, 30, 30)
    signal, _, gt = symdwi.simulate_dwi(
        [a, b], bvals, bvecs, scan, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, sphere="repulsion100", return_groundtruth=True,
    )
    ia = (5, 5, 15)
    ib = (25, 25, 15)
    assert gt["wm_mask"][ia] > 0
    assert gt["wm_mask"][ib] > 0

    w_intra_a = np.exp(-scan.te_ms / tissue_low.t2_intra_ms)
    expected_b0_a = gt["fiber_fraction"][ia] * w_intra_a + (1 - gt["fiber_fraction"][ia]) * w_intra_a
    w_tissue_b = np.exp(-scan.te_ms / tissue_high.t2_intra_ms)
    w_myelin_b = np.exp(-scan.te_ms / tissue_high.t2_myelin_ms)
    vf_intra_b = gt["fiber_fraction"][ib]
    expected_b0_b = (
        vf_intra_b * ((1 - tissue_high.f_myelin) * w_tissue_b + tissue_high.f_myelin * w_myelin_b)
        + (1 - vf_intra_b) * w_tissue_b
    )
    assert np.allclose(signal[ia][0], expected_b0_a, atol=1e-6)
    assert np.allclose(signal[ib][0], expected_b0_b, atol=1e-6)


def test_gm_isotropic_ball_replaced_matches_hand_computed_mixture():
    # A pure-GM voxel's signal must match f_is*soma + f_in*neurite + f_ec*ball, computed
    # directly from the gm_compartments kernels, proving the wiring is correct end-to-end.
    from symdwi.gm_compartments import gpd_sphere_attenuation, isotropic_stick_attenuation

    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 20), (2000, 20)], n_b0=2, seed=1)
    scan = symdwi.ScanParameters(te_ms=None)
    gm = symdwi.GMParameters(f_in=0.5, f_ec=0.35)
    dims = (10, 10, 10)

    signal, affine = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, gm=gm, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"gm": np.ones(dims)},
    )

    soma = gpd_sphere_attenuation(bvals, gm.big_delta_ms, gm.small_delta_ms, gm.d_is, gm.r_s)
    neurite = isotropic_stick_attenuation(bvals, gm.d_in)
    ball = np.exp(-bvals * gm.d_ec)
    expected = gm.f_is * soma + gm.f_in * neurite + gm.f_ec * ball

    assert np.allclose(signal[0, 0, 0], expected, atol=1e-10)


def test_gm_f_is_zero_reduces_to_neurite_plus_ball():
    # f_is=0 must reduce exactly to a neurite+ball mixture, with soma-only parameters
    # (r_s, d_is, big_delta_ms, small_delta_ms) having zero effect on output.
    from symdwi.gm_compartments import isotropic_stick_attenuation

    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1500, 20)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None)
    dims = (10, 10, 10)

    gm = symdwi.GMParameters(f_in=0.6, f_ec=0.4)
    assert gm.f_is == 0.0

    signal, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, gm=gm, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"gm": np.ones(dims)},
    )

    neurite = isotropic_stick_attenuation(bvals, gm.d_in)
    ball = np.exp(-bvals * gm.d_ec)
    expected = gm.f_in * neurite + gm.f_ec * ball
    assert np.allclose(signal[0, 0, 0], expected, atol=1e-10)

    gm_hot_soma = symdwi.GMParameters(
        f_in=0.6, f_ec=0.4, r_s=1.0, d_is=1.0e-3,
        big_delta_ms=100.0, small_delta_ms=5.0,
    )
    signal_hot, _ = symdwi.simulate_dwi(
        [], bvals, bvecs, scan, gm=gm_hot_soma, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, tissue_masks={"gm": np.ones(dims)},
    )
    assert np.allclose(signal, signal_hot, atol=1e-10)


def test_gm_soma_sensitive_to_delta_wm_compartments_are_not():
    # Two different (big_delta_ms, small_delta_ms) pairs must change the GM voxel's signal
    # but leave the WM voxel bit-identical (no leakage into the stick/zeppelin/ball path).
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(2000, 20)], n_b0=1, seed=1)
    scan = symdwi.ScanParameters(te_ms=None)
    dims = (30, 30, 30)

    geom = symdwi.BundleGeometry(
        np.array([[5, 5, 5], [5, 5, 15], [5, 5, 25]], float),
        n_streamlines=100, radius=2.0, seed=0,
    )
    b = symdwi.Bundle(geom, tissue=symdwi.TissueParameters(axon_radius=None))

    wm_mask = np.zeros(dims)
    wm_mask[3:8, 3:8, :] = 1.0
    gm_mask = np.zeros(dims)
    gm_mask[20:25, 20:25, :] = 1.0
    tissue_masks = {"wm": wm_mask, "gm": gm_mask}

    gm_a = symdwi.GMParameters(big_delta_ms=22.0, small_delta_ms=13.0)
    gm_b = symdwi.GMParameters(big_delta_ms=60.0, small_delta_ms=13.0)

    signal_a, _, gt = symdwi.simulate_dwi(
        [b], bvals, bvecs, scan, gm=gm_a, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, sphere="repulsion100", tissue_masks=tissue_masks,
        return_groundtruth=True,
    )
    signal_b, _ = symdwi.simulate_dwi(
        [b], bvals, bvecs, scan, gm=gm_b, origin=np.zeros(3), dims=dims,
        voxel_size=1.0, sphere="repulsion100", tissue_masks=tissue_masks,
    )

    wm_idx = tuple(np.argwhere(gt["wm_mask"] > 0)[0])
    gm_idx = (22, 22, 15)

    assert not np.allclose(signal_a[gm_idx], signal_b[gm_idx], atol=1e-10)
    assert np.array_equal(signal_a[wm_idx], signal_b[wm_idx])


def test_save_dwi_roundtrips_delta_when_provided(tmp_path):
    # acquisition_params.json is written only when both delta values are given.
    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 5)], n_b0=1, seed=1)
    signal = np.zeros((3, 3, 3, len(bvals)))
    affine = symdwi.make_affine(np.zeros(3), 1.0)

    out_with = tmp_path / "with_delta"
    out_with.mkdir()
    symdwi.save_dwi(signal, affine, bvals, bvecs, out_with,
                     big_delta_ms=22.0, small_delta_ms=13.0)
    sidecar = out_with / "acquisition_params.json"
    assert sidecar.exists()
    with open(sidecar) as f:
        params = json.load(f)
    assert params == {"big_delta_ms": 22.0, "small_delta_ms": 13.0}

    out_without = tmp_path / "without_delta"
    out_without.mkdir()
    symdwi.save_dwi(signal, affine, bvals, bvecs, out_without)
    assert not (out_without / "acquisition_params.json").exists()


def test_save_dwi_roundtrips_signal_bvals_affine(tmp_path):
    # dwi.nii.gz and bvals.txt must round-trip the array data and affine exactly.
    import nibabel as nib

    bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 5)], n_b0=1, seed=1)
    rng = np.random.default_rng(0)
    signal = rng.random((4, 4, 4, len(bvals)))
    affine = symdwi.make_affine(np.array([1.0, 2.0, 3.0]), 2.0)

    symdwi.save_dwi(signal, affine, bvals, bvecs, tmp_path)

    img = nib.load(tmp_path / "dwi.nii.gz")
    assert np.allclose(img.get_fdata(), signal, atol=1e-5)
    assert np.allclose(img.affine, affine)
    on_disk_bvals = np.loadtxt(tmp_path / "bvals.txt")
    assert np.array_equal(on_disk_bvals, bvals)


def test_save_groundtruth_roundtrips_masks_and_peaks(tmp_path):
    # wm_mask, fiber_fraction, and peaks (interleaved xyz) must round-trip through nibabel.
    import nibabel as nib

    signal, affine, gt = _sim(return_groundtruth=True)
    symdwi.save_groundtruth(gt, affine, tmp_path)

    wm_mask = nib.load(tmp_path / "wm_mask.nii.gz").get_fdata()
    assert np.array_equal(wm_mask, gt["wm_mask"])

    fiber_fraction = nib.load(tmp_path / "fiber_fraction.nii.gz").get_fdata()
    assert np.allclose(fiber_fraction, gt["fiber_fraction"], atol=1e-6)

    peaks_on_disk = nib.load(tmp_path / "peaks.nii.gz").get_fdata()
    expected_peaks = gt["peaks"].reshape(*gt["peaks"].shape[:3], -1)
    assert np.allclose(peaks_on_disk, expected_peaks, atol=1e-6)
