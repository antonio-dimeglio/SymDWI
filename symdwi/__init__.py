"""SymDWI — synthetic diffusion MRI phantom library."""

from .bundle import Bundle, save_bundles
from .simulate import (
    DWIParameters,
    simulate_dwi,
    compute_signal,
    extract_groundtruth,
    make_affine,
    save_dwi,
    save_groundtruth,
    world_bvecs_to_fsl,
    generate_bvals_bvecs,
)

__all__ = [
    "Bundle",
    "save_bundles",
    "DWIParameters",
    "simulate_dwi",
    "compute_signal",
    "extract_groundtruth",
    "make_affine",
    "save_dwi",
    "save_groundtruth",
    "world_bvecs_to_fsl",
    "generate_bvals_bvecs",
]
