# SymDWI

Synthetic diffusion MRI phantom library. Define white matter fiber bundles programmatically and simulate realistic DWI signal using the Standard Model (stick + zeppelin + ball compartments).

## Install

```bash
pip install -e .
```

For 3D bundle visualisation (requires [FURY](https://fury.gl)):

```bash
pip install -e ".[vis]"
```

## Quick start

```python
import numpy as np
import symdwi

# Define a bundle via 3D control points (mm)
pts = np.array([
    [30, 28,  5],
    [30, 29, 25],
    [30, 30, 45],
], dtype=float)
bundle = symdwi.Bundle(pts, n_streamlines=200, radius=4.0, seed=0)

# Generate gradient table (1 b=0 + 64 x b1000 + 64 x b2500 = 129 volumes)
bvals, bvecs = symdwi.generate_bvals_bvecs(seed=42)

# Simulate DWI
params = symdwi.DWIParameters(f_intra=0.7, f_extra=0.3, f_csf=0.0, te_ms=80.0)
dwi, affine = symdwi.simulate_dwi([bundle], bvals, bvecs, params, dims=(60, 60, 60))

# Save outputs
symdwi.save_dwi(dwi, affine, bvals, bvecs, "output/")
symdwi.save_bundles([bundle], "output/tractogram.tck")
```

## Tests

The test suite (and other developer tooling) lives in the `dev` extra:

```bash
pip install -e ".[dev]"
pytest
```
## Documentation

```bash
pip install -e ".[dev]"
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).
