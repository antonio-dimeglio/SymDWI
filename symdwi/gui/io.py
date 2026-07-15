"""Serialization of GUI :class:`~symdwi.gui.scene.Scene` objects to and from
JSON scene files (used by the GUI's Save/Load scene actions)."""
from symdwi.gui.scene import Scene, VolumeConfig, BundleConfig
from symdwi.simulate import ScanParameters, GMParameters
from symdwi.bundle import BundleGeometry, TissueParameters
from pathlib import Path
from dataclasses import asdict
import numpy as np
import json


def save_scene(scene: Scene, path: str) -> None:
    """Serialize a :class:`Scene` to a JSON file.

    Args:
        scene: The scene to serialize.
        path: Destination file path. Its parent directory must already exist.

    Returns:
        None.

    Raises:
        NotADirectoryError: If the parent directory of ``path`` does not exist.
    """
    p = Path(path)

    if not p.parent.exists():
        raise NotADirectoryError(f"Directory {p.parent} does not exist")

    d = asdict(scene)
    for b in d["bundles"]:
        b["geometry"]["control_points"] = np.asarray(
            b["geometry"]["control_points"]
        ).tolist()
        b["geometry"]["taper"] = None  # not JSON-serializable; see taper_linear

    with open(p, "w") as f:
        json.dump(d, f, indent=4)


def load_scene(path: str) -> Scene:
    """Load a :class:`Scene` previously written by :func:`save_scene`.

    Args:
        path: Path to the JSON scene file to load.

    Returns:
        The reconstructed :class:`Scene`, including its volume, bundles
        (with geometry and optional per-bundle tissue overrides), active
        control points, scan/tissue/gray-matter parameters, and background
        image path. Fields absent from the file (e.g. from older scene
        files) fall back to their dataclass defaults.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File {p} does not exist.")

    with open(p, "r") as f:
        d = json.load(f)

    bundles = []
    for b in d["bundles"]:
        b = dict(b)
        geometry = BundleGeometry(**b.pop("geometry"))
        tissue = b.pop("tissue", None)
        bundles.append(BundleConfig(
            geometry=geometry,
            tissue=TissueParameters(**tissue) if tissue else None,
            **b,
        ))

    return Scene(
        volume=VolumeConfig(**d["volume"]),
        bundles=bundles,
        active_points=[tuple(p) for p in d["active_points"]],
        scan=ScanParameters(**d.get("scan", {})),
        tissue=TissueParameters(**d.get("tissue", {})),
        gm=GMParameters(**d.get("gm", {})),
        gm_enabled=d.get("gm_enabled", False),
        background_path=d.get("background_path", ""),
    )
