"""Plain-data (dataclass) models describing the GUI's editable scene state.

These classes hold only serializable UI state — they are distinct from the
simulation-domain classes in :mod:`symdwi.bundle` and :mod:`symdwi.simulate`,
which they wrap or reference. See :mod:`symdwi.gui.io` for JSON persistence.
"""
from dataclasses import dataclass, field
from symdwi.simulate import ScanParameters, GMParameters
from symdwi.bundle import BundleGeometry, TissueParameters


@dataclass
class VolumeConfig:
    """Dimensions and resolution of the simulated imaging volume.

    Attributes:
        x_mm: Volume extent along x, in millimeters.
        y_mm: Volume extent along y, in millimeters.
        z_mm: Volume extent along z, in millimeters.
        voxel_size: Isotropic voxel edge length, in millimeters.
    """
    x_mm: float = 60.0
    y_mm: float = 60.0
    z_mm: float = 60.0
    voxel_size: float = 1.0


@dataclass
class BundleConfig:
    """GUI wrapper around a BundleGeometry: adds display/UI-only state
    (color, tck origin, visibility) and an optional per-bundle tissue
    override, without restating BundleGeometry's own fields.

    Attributes:
        geometry: The underlying fiber bundle geometry.
        color: Display color name/hex used when rendering the bundle.
        tck_path: Filesystem path of the ``.tck`` file the bundle was
            imported from, if any (empty string if not loaded from disk).
        visible: Whether the bundle is currently shown in the 2D/3D views.
        tissue: Per-bundle tissue parameter override. Only applied when
            ``tissue_override`` is True; otherwise the scene-level tissue
            parameters are used.
        tissue_override: Whether ``tissue`` should override the scene-level
            tissue parameters for this bundle.
        taper_linear: UI flag driving the linear-taper checkbox. Since
            ``geometry.taper`` holds a callable (or None) that can't
            round-trip through JSON scene files, this flag is translated to
            ``geometry.taper`` only when a :class:`~symdwi.bundle.Bundle` is
            built.
    """
    geometry: BundleGeometry
    color: str = "blue"
    tck_path: str = ""
    visible: bool = True
    tissue: TissueParameters | None = None
    tissue_override: bool = False
    taper_linear: bool = False


@dataclass
class Scene:
    """Top-level container for all state needed to reconstruct the GUI's
    editable session: the volume, fiber bundles, selected control points,
    and simulation parameters.

    Attributes:
        volume: Dimensions/resolution of the imaging volume.
        bundles: All fiber bundles currently in the scene.
        active_points: Currently selected/highlighted control points, as a
            list of coordinate tuples.
        scan: Scanner/acquisition parameters used for simulation.
        tissue: Scene-level (default) tissue parameters, used by bundles
            that do not set ``tissue_override``.
        gm: Gray-matter compartment parameters.
        gm_enabled: Whether the gray-matter compartment is included in
            simulation.
        background_path: Filesystem path of a background image, if loaded
            (empty string if none).
    """
    volume: VolumeConfig
    bundles: list[BundleConfig]
    active_points: list
    scan: ScanParameters = field(default_factory=ScanParameters)
    tissue: TissueParameters = field(default_factory=TissueParameters)
    gm: GMParameters = field(default_factory=GMParameters)
    gm_enabled: bool = False
    background_path: str = ""
