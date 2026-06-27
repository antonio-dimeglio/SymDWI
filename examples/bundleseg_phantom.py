from pathlib import Path
import nibabel as nib
import numpy as np
from symdwi import (
    Bundle, 
    simulate_dwi, 
    generate_bvals_bvecs,
    DWIParameters,
    save_dwi
)

BUNDLESEG_DIR = Path('/Users/quantum/Desktop/PhD/Data/BundleSeg')
ATLAS_DIR = BUNDLESEG_DIR / 'atlas_tck'

print("Loading masks...")
wm = nib.load(BUNDLESEG_DIR / 'wm_2mm.nii.gz')
gm = nib.load(BUNDLESEG_DIR / 'gm_2mm.nii.gz')
csf = nib.load(BUNDLESEG_DIR / 'csf_2mm.nii.gz')

print('WM shape:', wm.shape, 'affine diag:', np.diag(wm.affine)[:3])
print('GM shape:', gm.shape)
print('CSF shape:', csf.shape)
print('WM range:', wm.get_fdata().min(), wm.get_fdata().max())


tcks = sorted(ATLAS_DIR.glob('*.tck'))
print(f'Found {len(tcks)} bundles')

bundles = []
print("Loading bundles...")
for tck in tcks:
    bundles.append(Bundle.from_tck(tck))
print("Done. Starting simulation...")

bvals, bvecs = generate_bvals_bvecs(shells=[(1000, 64), (2500, 64)], n_b0=1)
aff = wm.affine
origin = aff[:3, 3]
voxel_size = float(wm.header.get_zooms()[0])
dims = wm.shape

dwi, affine = simulate_dwi(
    bundles=bundles,
    bvals=bvals,
    bvecs=bvecs,
    params=DWIParameters(),
    origin=origin,
    dims=dims,
    voxel_size=voxel_size,
    snr=None,
    n_jobs=8,
    tissue_masks={"wm": wm.get_fdata(), "gm": gm.get_fdata(), "csf": csf.get_fdata()},
    verbose=1,
)

print("Saving dwi...")
save_dwi(
    dwi,
    affine,
    bvals,
    bvecs,
    '.'
)