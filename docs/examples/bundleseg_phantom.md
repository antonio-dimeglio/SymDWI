# BundleSeg full-brain phantom

This example simulates a full-brain DWI volume using the
[BundleSeg](https://github.com/scil-vital/BundleSeg) white matter atlas (51 bundles)
and FSL FAST probabilistic tissue maps on the MNI152 2mm grid.

The runnable script is at `examples/bundleseg_phantom.py`. Point `BUNDLESEG_DIR`
at your local BundleSeg directory before running.

## What you need

- BundleSeg atlas with `atlas_tck/` folder
- Probabilistic tissue maps at 2mm: `wm_2mm.nii.gz`, `gm_2mm.nii.gz`, `csf_2mm.nii.gz`
  (produced by FSL `fast` on the MNI brain, resampled with `flirt -applyisoxfm 2`)

## Key snippets

**Loading the grid from the mask header**

```python
wm = nib.load(BUNDLESEG_DIR / 'wm_2mm.nii.gz')
origin = wm.affine[:3, 3]
voxel_size = float(wm.header.get_zooms()[0])
dims = wm.shape
```

**Loading all bundles:**

```python
bundles = [Bundle.from_tck(p) for p in sorted(ATLAS_DIR.glob('*.tck'))]
```

**Custom gradient table**: two shells with an explicit b=0 count:

```python
bvals, bvecs = generate_bvals_bvecs(shells=[(1000, 64), (2500, 64)], n_b0=1)
```

**Running the simulation with tissue masks:**

```python
dwi, affine = simulate_dwi(
    bundles=bundles,
    bvals=bvals,
    bvecs=bvecs,
    scan=ScanParameters(),
    gm=GMParameters(),
    origin=origin,
    dims=dims,
    voxel_size=voxel_size,
    n_jobs=8,
    tissue_masks={
        "wm": wm.get_fdata(),
        "gm": gm.get_fdata(),
        "csf": csf.get_fdata(),
    },
    verbose=2,
)
```