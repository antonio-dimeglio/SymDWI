from dataclasses import dataclass
from symdwi.simulate import DWIParameters

@dataclass
class VolumeConfig:
    x_mm: float = 60.0
    y_mm: float = 60.0
    z_mm: float = 60.0
    voxel_size: float = 1.0

@dataclass
class BundleConfig:
    control_points: list[tuple[float]]
    radius: float = 3.0
    n_streamlines: int = 100
    dispersion: float = 0.5
    taper: bool = False
    color: str = "blue"
    tck_path: str = ""
    visible: bool = True

@dataclass
class Scene:
    volume: VolumeConfig
    bundles: list[BundleConfig]
    active_points: list
    dwi_params: DWIParameters
    background_path: str = ""