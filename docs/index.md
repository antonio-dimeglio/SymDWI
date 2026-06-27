# SymDWI

Synthetic diffusion MRI phantom library.

SymDWI lets you define white matter fiber bundles as B-spline-based streamline families (or load them from existing tractograms) and simulate realistic DWI signal using the Standard Model of white matter diffusion (stick + zeppelin + ball compartments).

## Install

```bash
pip install -e .
```

## Minimal example
The following is a very minimal example for how to build a phantom DWI.

```python
import numpy as np
import symdwi

pts = np.array([[30, 28, 5], [30, 29, 25], [30, 30, 45]], dtype=float)
bundle = symdwi.Bundle(pts, n_streamlines=200, radius=4.0, seed=0)
bvals, bvecs = symdwi.generate_bvals_bvecs(seed=42)

params = symdwi.DWIParameters(f_intra=0.7, f_extra=0.3, f_csf=0.0, te_ms=80.0)
dwi, affine = symdwi.simulate_dwi([bundle], bvals, bvecs, params, dims=(60, 60, 60))

symdwi.save_dwi(dwi, affine, bvals, bvecs, "output/")
```

For a more in-depth example, alongside with an explanation for _how_ SymDWI generates a dwi, see the [getting started](examples/getting_started.md) section.
## API

- [Bundle](api/bundle.md): fiber bundle geometry
- [Simulate](api/simulate.md): DWI simulation and gradient tables

## Examples
- [Getting Started](examples/getting_started.md)
- [Building a full brain](examples/bundleseg_phantom.md)
- [GUI guide](examples/gui_guide.md)