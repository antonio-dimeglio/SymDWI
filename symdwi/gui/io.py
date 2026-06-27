from symdwi.gui.scene import Scene, VolumeConfig, BundleConfig
from symdwi.simulate import DWIParameters
from pathlib import Path
from dataclasses import asdict
import json

def save_scene(scene: Scene, path: str) -> None:
    p = Path(path)

    if not p.parent.exists():
        raise NotADirectoryError(f"Directory {p.parent} does not exist")

    d = asdict(scene)
    with open(p, "w") as f:
        json.dump(d, f, indent=4)

def load_scene(path: str) -> Scene:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File {p} does not exist.")

    with open(p, "r") as f:
        d = json.load(f)

    return Scene(
        volume=VolumeConfig(**d["volume"]),
        bundles=[BundleConfig(**{**b, "control_points": [tuple(p) for p in b["control_points"]]}) for b in d["bundles"]],
        active_points=[tuple(p) for p in d["active_points"]],
        dwi_params=DWIParameters(**d["dwi_params"]),
        background_path=d.get("background_path", ""),
    )