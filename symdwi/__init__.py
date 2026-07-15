"""SymDWI — synthetic diffusion MRI phantom library."""

from .bundle import Bundle, BundleGeometry, TissueParameters, save_bundles
from .simulate import (
    ScanParameters,
    GMParameters,
    simulate_dwi,
    compute_signal,
    extract_groundtruth,
    make_affine,
    save_dwi,
    save_groundtruth,
    world_bvecs_to_fsl,
    generate_bvals_bvecs,
)
from .gm_compartments import gpd_sphere_attenuation, isotropic_stick_attenuation

__all__ = [
    "Bundle",
    "BundleGeometry",
    "TissueParameters",
    "save_bundles",
    "ScanParameters",
    "GMParameters",
    "simulate_dwi",
    "compute_signal",
    "extract_groundtruth",
    "make_affine",
    "save_dwi",
    "save_groundtruth",
    "world_bvecs_to_fsl",
    "generate_bvals_bvecs",
    "gpd_sphere_attenuation",
    "isotropic_stick_attenuation",
]
