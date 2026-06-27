import symdwi
import numpy as np

dims = (60, 60, 60)
voxel_size = 1.0

b1_pts = np.array([
    [10, 50, 30], 
    [20, 45, 30],
    [28, 41, 30],
    [37, 38, 30],
    [43, 30, 30],
    [47, 20, 30],
    [50, 10, 30]
])

b1 = symdwi.Bundle(
    b1_pts,
    n_streamlines=500,
    radius=3.0,
    n_samples=100,
    degree=3,
    dispersion=0.25
)


b2_pts = np.array([
    [10, 29, 30], 
    [18, 28, 30],
    [24, 28, 30],
    [37, 28, 30],
    [42, 29, 30],
    [50, 33, 30],
    [53, 40, 30]
])

b2 = symdwi.Bundle(
    b2_pts,
    n_streamlines=500,
    radius=3.0,
    n_samples=100,
    degree=3,
    dispersion=0.25
)


bvals, bvecs = symdwi.generate_bvals_bvecs(shells=[(1000, 8), (2000, 8), (3000, 8)], n_b0=1)

params = symdwi.DWIParameters(
    f_intra=0.7,
    f_extra=0.3,
    f_csf=0.0,
    axon_radius_um=1.0,
    te_ms=80.0,
    background_csf=0.00,
)

dwi, affine, gt = symdwi.simulate_dwi(
    [b1, b2],
    bvals,
    bvecs,
    params,
    origin=np.array([0.0, 0.0, 0.0]),
    dims=dims,
    voxel_size=voxel_size,
    snr=None, # No rician noise
    return_groundtruth=True
)

symdwi.save_dwi(dwi, affine, bvals, bvecs, "results")
symdwi.save_groundtruth(gt, affine, "results")
symdwi.save_bundles([b1, b2], 'results/tractogram.tck')