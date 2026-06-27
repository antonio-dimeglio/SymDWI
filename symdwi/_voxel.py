import numpy as np
from .bundle import Bundle
from collections import defaultdict
from dipy.data import get_sphere


def directions_to_bins(tng: np.ndarray, verts: np.ndarray, chunk: int = 5000) -> np.ndarray:
    """
    Map tangent vectors to the nearest vertex on a sphere discretisation.

    Uses the absolute dot product because tangents are undirected.

    Parameters
    ----------
    tng : np.ndarray, shape (N, 3)
    verts : np.ndarray, shape (V, 3)
        Sphere vertices.
    chunk : int
        Batch size for memory-efficient processing.

    Returns
    -------
    np.ndarray, shape (N,), dtype int
        Index of the nearest sphere vertex for each tangent.
    """
    out = np.empty(len(tng), dtype=int)
    for s in range(0, len(tng), chunk):
        out[s:s+chunk] = np.argmax(np.abs(tng[s:s+chunk] @ verts.T), axis=1)
    return out


def voxelise_bundles(
    bundles: list[Bundle],
    origin: np.ndarray,
    dims: tuple[int] = (64, 64, 64),
    voxel_size: float = 1.0,
    sphere=get_sphere(name="repulsion724"),
) -> defaultdict:
    """
    Voxelise a list of bundles into a sparse ODF histogram grid.

    Parameters
    ----------
    bundles : list[Bundle]
    origin : np.ndarray, shape (3,)
        World-space corner of the volume (mm).
    dims : tuple[int]
        Volume shape in voxels.
    voxel_size : float
        Isotropic voxel size in mm.
    sphere : dipy Sphere
        Sphere used for direction discretisation (default: repulsion724).

    Returns
    -------
    defaultdict mapping (i, j, k) -> np.ndarray, shape (n_dirs,)
        Streamline-point counts per direction bin per voxel.
    """
    dims_arr = np.array(dims)
    n_dirs = len(sphere.vertices)
    verts = sphere.vertices

    origin = np.asarray(origin, dtype=float)
    flat_voxels = []
    dir_bins = []

    # Densify each streamline so consecutive samples are spaced below one voxel,
    # otherwise a thin/curved fiber can step over a voxel without any sample
    # landing inside it, leaving signal holes. Done per streamline so we never
    # interpolate across the gap between two separate streamlines.
    for b in bundles:
        for line, tang in zip(b.streamlines, b.tangents):
            pts, tng = _densify(np.asarray(line, dtype=float),
                                np.asarray(tang, dtype=float), voxel_size)
            idx = np.floor((pts - origin) / voxel_size + 0.5).astype(int)
            valid = ~((idx < 0) | (idx >= dims_arr)).any(axis=1)
            idx, tng = idx[valid], tng[valid]
            if len(idx) == 0:
                continue
            flat_voxels.append(np.ravel_multi_index(idx.T, dims))
            dir_bins.append(directions_to_bins(tng, verts))

    grid = defaultdict(lambda: np.zeros(n_dirs))
    if not flat_voxels:
        return grid

    flat_voxels = np.concatenate(flat_voxels)
    dir_bins = np.concatenate(dir_bins)

    combined = flat_voxels * n_dirs + dir_bins
    uniq, counts = np.unique(combined, return_counts=True)

    uniq_voxel = uniq // n_dirs
    uniq_dir = uniq % n_dirs

    voxel_ijk = np.array(np.unravel_index(uniq_voxel, dims)).T
    for (i, j, k), d, c in zip(map(tuple, voxel_ijk), uniq_dir, counts):
        grid[(i, j, k)][d] = c

    return grid


def _densify(pts: np.ndarray, tng: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert samples along a single streamline so no step exceeds ~half a voxel.

    Prevents signal holes where a sparsely-sampled fiber skips over a voxel.
    Each original segment is linearly subdivided; tangents are carried from the
    segment start (they vary slowly relative to the subdivision scale).

    Parameters
    ----------
    pts : np.ndarray, shape (M, 3)
        Ordered points of one streamline.
    tng : np.ndarray, shape (M, 3)
        Unit tangents at each point.
    voxel_size : float

    Returns
    -------
    (pts, tng) resampled to a sub-voxel step.
    """
    if len(pts) < 2:
        return pts, tng

    step = 0.5 * voxel_size
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    n_sub = np.maximum(1, np.ceil(seg_len / step).astype(int))

    out_pts = []
    out_tng = []
    for i in range(len(pts) - 1):
        ts = np.linspace(0.0, 1.0, n_sub[i], endpoint=False)
        out_pts.append(pts[i] + ts[:, None] * seg[i])
        out_tng.append(np.repeat(tng[i][None, :], len(ts), axis=0))
    out_pts.append(pts[-1][None, :])
    out_tng.append(tng[-1][None, :])

    return np.concatenate(out_pts), np.concatenate(out_tng)
