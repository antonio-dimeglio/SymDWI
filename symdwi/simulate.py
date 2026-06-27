import numpy as np
import concurrent.futures
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import tqdm as tqdm_mod

from dipy.data import get_sphere
from dipy.core.sphere import HemiSphere, disperse_charges
import nibabel as nib

from .bundle import Bundle
from ._voxel import voxelise_bundles


@dataclass
class DWIParameters:
    """
    Biophysical parameters for the Standard Model DWI simulation.

    Parameters
    ----------
    f_intra : float
        Intra-axonal signal fraction in pure-fiber voxels.
    f_extra : float
        Extra-axonal signal fraction in pure-fiber voxels.
    f_csf : float
        Free-water signal fraction in pure-fiber voxels.
        Must satisfy f_intra + f_extra + f_csf == 1.
    di_axial : float
        Intra-axonal axial diffusivity (mm^2/s).
    de_axial : float
        Extra-axonal axial diffusivity (mm^2/s).
    de_radial : float
        Extra-axonal radial diffusivity (mm^2/s).
    d_iso : float
        Free-water (CSF) isotropic diffusivity (mm^2/s).
    d_gm : float
        Gray matter isotropic diffusivity (mm^2/s). Used when a "gm" tissue
        mask is provided. T2 weighting uses t2_extra_ms as the closest analog.
    axon_radius_um : float or None
        Axon radius in micrometers. Converts streamline point density to a
        physical intra-axonal volume fraction. Set to None for relative density scaling.
    te_ms : float or None
        Echo time in ms. Set to None to disable T2 relaxation weighting.
    t2_intra_ms : float
        T2 relaxation time for the intra-axonal compartment (ms).
    t2_extra_ms : float
        T2 relaxation time for the extra-axonal compartment (ms).
        Also used as T2 weight for the GM compartment.
    t2_csf_ms : float
        T2 relaxation time for the CSF/free-water compartment (ms).
    background_csf : float
        Constant free-water signal added to every voxel (including background).
        Set to 0.0 for a silent background.
    """
    f_intra: float = 0.7
    f_extra: float = 0.3
    f_csf: float = 0.0
    di_axial: float = 1.7e-3
    de_axial: float = 1.7e-3
    de_radial: float = 0.5e-3
    d_iso: float = 3.0e-3
    d_gm: float = 0.8e-3
    axon_radius_um: float | None = 1.0
    te_ms: float | None = None
    t2_intra_ms: float = 70.0
    t2_extra_ms: float = 70.0
    t2_csf_ms: float = 2000.0
    background_csf: float = 0.0

    def __post_init__(self):
        total = self.f_intra + self.f_extra + self.f_csf
        assert abs(total - 1.0) < 1e-6, f"Compartment fractions must sum to 1.0 (got {total})"


def generate_bvals_bvecs(
    shells: list[tuple[int, int]] = [(1000, 64), (2500, 64)],
    n_b0: int = 1,
    iters: int = 500,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a multi-shell gradient table via electrostatic charge repulsion.

    Directions for each shell are independently optimised on the hemisphere.
    b=0 volumes are prepended.

    Parameters
    ----------
    shells : list of (b_value, n_directions) tuples
        Each entry defines one shell. Example: [(1000, 32), (3000, 64)].
    n_b0 : int
        Number of b=0 volumes prepended to the table.
    iters : int
        Number of charge-dispersion iterations per shell.
    seed : int or None
        RNG seed.

    Returns
    -------
    bvals : np.ndarray, shape (n_b0 + sum(n_dirs),)
        b-values in s/mm^2.
    bvecs : np.ndarray, shape (n_b0 + sum(n_dirs), 3)
        Unit gradient directions. b=0 rows are zero vectors.
    """
    rng = np.random.default_rng(seed)
    all_bvals = []
    all_bvecs = []

    for bval, n_pts in shells:
        theta = np.pi * rng.random(size=n_pts)
        phi = 2 * np.pi * rng.random(size=n_pts)
        hsph_initial = HemiSphere(theta=theta, phi=phi)
        hsph_updated, _ = disperse_charges(hsph_initial, iters=iters)
        all_bvals.append(bval * np.ones(n_pts))
        all_bvecs.append(hsph_updated.vertices)

    bvals = np.concatenate([[0] * n_b0, *all_bvals])
    bvecs = np.vstack([np.zeros((n_b0, 3)), *all_bvecs])
    return bvals, bvecs


def simulate_dwi(
    bundles: list[Bundle],
    bvals: np.ndarray,
    bvecs: np.ndarray,
    params: DWIParameters,
    origin: np.ndarray | None = None,
    dims: tuple[int, ...] = (64, 64, 64),
    voxel_size: float = 1.0,
    snr: float | None = None,
    n_jobs: int = 1,
    tissue_masks: dict[str, np.ndarray] | None = None,
    sphere: str = "repulsion724",
    return_groundtruth: bool = False,
    max_peaks: int = 3,
    seed: int | None = None,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate a DWI volume from a set of fiber bundles.

    Uses the Standard Model of white matter: stick (intra-axonal),
    zeppelin (extra-axonal), and ball (free water) compartments.

    Parameters
    ----------
    bundles : list[Bundle]
    bvals : np.ndarray, shape (N,)
        b-values in s/mm^2. Include a b=0 entry as a zero.
    bvecs : np.ndarray, shape (N, 3)
        Gradient directions. b=0 entries should be zero vectors.
    params : DWIParameters
    origin : np.ndarray or None
        World-space corner of the volume in mm.
        Auto-computed from bundle extents if None.
    dims : tuple[int, ...]
        Volume shape in voxels.
    voxel_size : float
        Isotropic voxel size in mm.
    snr : float or None
        If given, adds Rician noise with sigma = 1/snr (S0 = 1).
    n_jobs : int
        Number of threads for parallel voxel processing. Default 1 (no threading).
    tissue_masks : dict or None
        Optional tissue probability/binary maps. Accepted keys: "wm", "gm", "csf".
        Each value must be a 3D array matching `dims`, with values in [0, 1].
        Binary (0/1) and probabilistic masks are both accepted.
        Per-voxel tissue fractions are normalised to sum to 1 before use.
        The "gm" compartment is modelled as an isotropic ball with params.d_gm.
    sphere : str
        Name of the dipy sphere used to discretise fiber tangents into per-voxel
        direction bins (e.g. "repulsion100", "repulsion724"). Higher vertex counts
        reduce orientation quantisation error at a modest cost (~10% slower for
        724 vs 100 vertices, since only populated bins are evaluated per voxel).
    return_groundtruth : bool
        If True, also return a dict of per-voxel ground-truth labels derived
        directly from the fiber geometry (noise-independent). See Returns.
    max_peaks : int
        Maximum number of fiber-orientation peaks stored per voxel in the
        ground-truth output. Only used when return_groundtruth is True.
    seed : int or None
        Seed for the Rician noise RNG. Set for reproducible datasets (the
        signal itself is deterministic; only the noise draw uses this). None
        uses fresh, non-reproducible randomness. Ignored when snr is None.
    verbose : int
        Verbosity level. 0 = silent, 1 = step-level prints, 2 = 1 + tqdm progress bar
        during signal computation (per-voxel when n_jobs=1, per-chunk when threaded).

    Returns
    -------
    signal : np.ndarray, shape (*dims, N)
    affine : np.ndarray, shape (4, 4)
    groundtruth : dict, only if return_groundtruth is True
        Per-voxel ground truth in the same world frame as `affine`:
          - "wm_mask" : uint8 (*dims), 1 where a fiber is present.
          - "fiber_fraction" : float32 (*dims), intra-axonal volume fraction.
          - "peaks" : float32 (*dims, max_peaks, 3), unit peak directions
                               (world space), zero-padded, sorted by amplitude.
          - "peak_values" : float32 (*dims, max_peaks), peak amplitudes.
          - "sphere" : the dipy Sphere used for direction discretisation.
    """
    bvals = np.asarray(bvals, dtype=float)
    bvecs = np.asarray(bvecs, dtype=float)

    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-6, norms, 1.0)
    bvecs = np.where(norms > 1e-6, bvecs / safe_norms, 0.0)

    dims_arr = np.array(dims)

    if tissue_masks is not None:
        valid_keys = {"wm", "gm", "csf"}
        bad = set(tissue_masks) - valid_keys
        if bad:
            raise ValueError(f"tissue_masks has unknown keys: {bad}")
        tissue_masks = {
            key: np.clip(np.asarray(arr, dtype=float), 0.0, 1.0)
            for key, arr in tissue_masks.items()
        }
        for key, arr in tissue_masks.items():
            if arr.shape != tuple(dims):
                raise ValueError(
                    f"tissue_masks['{key}'] shape {arr.shape} != dims {dims}"
                )

    if origin is None:
        all_pts = np.concatenate(
            [b.points_with_orientation()[0] for b in bundles], axis=0
        )
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        origin = (lo + hi) / 2.0 - 0.5 * dims_arr * voxel_size

    if verbose >= 1:
        print(f"Voxelising {len(bundles)} bundle(s) into {dims} grid...")
    sphere = get_sphere(name=sphere)
    volume = voxelise_bundles(bundles, origin, dims, voxel_size, sphere=sphere)
    if verbose >= 1:
        print(f"  {len(volume)} fiber voxels found.")

    density = {k: v.sum() for k, v in volume.items()}

    if params.axon_radius_um is not None:
        r_mm = params.axon_radius_um * 1e-3
        voxel_vol = voxel_size ** 3
        fiber_vols = {k: np.pi * r_mm ** 2 * c for k, c in density.items()}
        max_fiber_vol = max(fiber_vols.values()) if fiber_vols else 1.0
        scale = params.f_intra * voxel_vol / max_fiber_vol
        density_frac = {k: min(params.f_intra, fv * scale / voxel_vol)
                        for k, fv in fiber_vols.items()}
    else:
        max_density = max(density.values()) if density else 1.0
        density_frac = {k: params.f_intra * c / max_density
                        for k, c in density.items()}

    for k, v in volume.items():
        volume[k] = v / v.sum()

    groundtruth = None
    if return_groundtruth:
        if verbose >= 1:
            print("Extracting ground-truth peaks...")
        groundtruth = extract_groundtruth(
            volume, density_frac, sphere, dims_arr, max_peaks=max_peaks
        )

    if verbose >= 1:
        tag = f"({n_jobs} threads)" if n_jobs > 1 else "(single-threaded)"
        print(f"Computing DWI signal {tag}...")
    signal = compute_signal(
        volume, density_frac, params, bvals, bvecs, dims_arr, sphere,
        n_jobs=n_jobs, tissue_masks=tissue_masks, verbose=verbose,
    )
    if verbose >= 1:
        print("  Signal computation done.")

    affine = make_affine(origin, voxel_size)

    if snr is not None:
        if verbose >= 1:
            print(f"Adding Rician noise (SNR={snr})...")
        rng = np.random.default_rng(seed)
        sigma = 1.0 / snr
        r = rng.normal(loc=0, scale=sigma, size=signal.shape)
        i = rng.normal(loc=0, scale=sigma, size=signal.shape)
        signal = np.sqrt((signal + r) ** 2 + i ** 2)

    if verbose >= 1:
        print("Done.")

    if return_groundtruth:
        return signal, affine, groundtruth
    return signal, affine


def extract_groundtruth(
    volume: defaultdict,
    density_frac: dict,
    sphere,
    dims: np.ndarray,
    max_peaks: int = 3,
    relative_peak_threshold: float = 0.5,
    min_separation_angle: float = 25.0,
) -> dict:
    """
    Extract per-voxel ground-truth fiber labels from a voxelised ODF grid.

    Derives a white-matter mask, intra-axonal fiber fraction, and discrete
    fiber-orientation peaks directly from the streamline geometry (independent
    of the diffusion signal or any added noise). Peaks are found as local
    maxima of each voxel's direction histogram on the discretisation sphere.

    Parameters
    ----------
    volume : defaultdict
        Sparse per-voxel direction distributions, normalised to sum to 1
        (as produced inside simulate_dwi after ODF normalisation).
    density_frac : dict
        Intra-axonal volume fraction per voxel key.
    sphere : dipy Sphere
        The sphere used to build `volume`. Peak directions are returned in the
        same (world) frame as its vertices.
    dims : np.ndarray, shape (3,)
        Volume dimensions in voxels.
    max_peaks : int
        Maximum number of peaks stored per voxel.
    relative_peak_threshold : float
        Discard peaks below this fraction of the largest peak in a voxel.
    min_separation_angle : float
        Merge peaks closer than this angle (degrees).

    Returns
    -------
    dict
        See simulate_dwi's `groundtruth` return for the keys.
    """
    from dipy.direction import peak_directions

    dims = tuple(int(d) for d in dims)
    wm_mask = np.zeros(dims, dtype=np.uint8)
    fiber_fraction = np.zeros(dims, dtype=np.float32)
    peaks = np.zeros((*dims, max_peaks, 3), dtype=np.float32)
    peak_values = np.zeros((*dims, max_peaks), dtype=np.float32)

    for k, odf in volume.items():
        wm_mask[k] = 1
        fiber_fraction[k] = density_frac.get(k, 0.0)
        dirs, vals, _ = peak_directions(
            odf, sphere,
            relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle,
        )
        n = min(len(dirs), max_peaks)
        if n:
            peaks[k][:n] = dirs[:n]
            peak_values[k][:n] = vals[:n]

    return {
        "wm_mask": wm_mask,
        "fiber_fraction": fiber_fraction,
        "peaks": peaks,
        "peak_values": peak_values,
        "sphere": sphere,
    }


def _compute_voxel_chunk(
    chunk,
    density_frac,
    params,
    bvals,
    bvecs,
    sphere,
    w_intra,
    w_extra,
    w_csf,
    water,
    water_gm,
    non_fiber_total,
    dwi,
    tissue_masks,
):
    for k, v in chunk:
        vf_intra = density_frac[k]
        remaining = 1.0 - vf_intra
        if non_fiber_total > 0:
            vf_extra = remaining * params.f_extra / non_fiber_total
            vf_csf_local = remaining * params.f_csf / non_fiber_total
        else:
            vf_extra = 0.0
            vf_csf_local = remaining

        mask = v > 0.0
        f_pop = v[mask]
        verts_active = sphere.vertices[mask]
        dot_sq = (bvecs @ verts_active.T) ** 2

        stick = np.exp(-bvals[:, None] * params.di_axial * dot_sq)
        zeppelin = (
            np.exp(-bvals[:, None] * params.de_radial)
            * np.exp(-bvals[:, None] * (params.de_axial - params.de_radial) * dot_sq)
        )
        wm_signal = (
            w_intra * vf_intra * (stick @ f_pop)
            + w_extra * vf_extra * (zeppelin @ f_pop)
            + w_csf * vf_csf_local * water
        )

        if tissue_masks is None:
            signal = wm_signal
        else:
            i, j, kk = k
            wm_frac = tissue_masks["wm"][i, j, kk] if "wm" in tissue_masks else 1.0
            gm_frac = tissue_masks["gm"][i, j, kk] if "gm" in tissue_masks else 0.0
            csf_frac = tissue_masks["csf"][i, j, kk] if "csf" in tissue_masks else 0.0
            total = wm_frac + gm_frac + csf_frac
            if total < 1e-9:
                dwi[k[0], k[1], k[2], :] = 0.0
                continue
            wm_frac /= total
            gm_frac /= total
            csf_frac /= total
            signal = (
                wm_frac  * wm_signal
                + gm_frac  * w_extra * water_gm
                + csf_frac * w_csf   * water
            )

        dwi[k[0], k[1], k[2], :] = signal


def compute_signal(
    volume: defaultdict,
    density_frac: dict,
    params: DWIParameters,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    dims: np.ndarray,
    sphere,
    n_jobs: int = 1,
    tissue_masks: dict[str, np.ndarray] | None = None,
    verbose: int = 0,
) -> np.ndarray:
    """
    Compute DWI signal using the Standard Model (stick + zeppelin + ball).

    Parameters
    ----------
    volume : defaultdict
        Sparse ODF grid from voxelise_bundles, with per-voxel direction
        probability distributions (must be normalised to sum to 1).
    density_frac : dict
        Intra-axonal volume fraction per voxel key.
    params : DWIParameters
    bvals : np.ndarray, shape (N,)
    bvecs : np.ndarray, shape (N, 3)
    dims : np.ndarray, shape (3,)
        Volume dimensions.
    sphere : dipy Sphere
        Must match the sphere used to build `volume`.
    n_jobs : int
        Number of threads. Default 1.
    tissue_masks : dict or None
        See simulate_dwi for details.
    verbose : int
        Verbosity level. 0 = silent, 1 = step prints, 2 = tqdm progress bar.

    Returns
    -------
    np.ndarray, shape (*dims, N)
    """
    dwi = np.zeros((*dims, len(bvals)))

    if params.te_ms is not None:
        w_intra = np.exp(-params.te_ms / params.t2_intra_ms)
        w_extra = np.exp(-params.te_ms / params.t2_extra_ms)
        w_csf = np.exp(-params.te_ms / params.t2_csf_ms)
    else:
        w_intra = w_extra = w_csf = 1.0

    water = np.exp(-bvals * params.d_iso)
    water_gm = np.exp(-bvals * params.d_gm)

    if params.background_csf > 0.0:
        dwi += params.background_csf * w_csf * water

    non_fiber_total = params.f_extra + params.f_csf
    items = list(volume.items())

    chunk_args = (
        density_frac, params, bvals, bvecs, sphere,
        w_intra, w_extra, w_csf, water, water_gm,
        non_fiber_total, dwi, tissue_masks,
    )

    if n_jobs == 1:
        if verbose >= 2:
            wrapped = tqdm_mod.tqdm(items, desc="voxels", unit="vox")
            _compute_voxel_chunk(wrapped, *chunk_args)
        else:
            _compute_voxel_chunk(items, *chunk_args)
    else:
        n_chunks = min(len(items), n_jobs * 8)
        chunk_size = max(1, (len(items) + n_chunks - 1) // n_chunks)
        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
            if verbose >= 2:
                bar = tqdm_mod.tqdm(
                    total=len(chunks), desc="chunks", unit="chunk",
                    postfix={"voxels": len(items)},
                )
                futs = {ex.submit(_compute_voxel_chunk, ch, *chunk_args): ch
                        for ch in chunks}
                for f in concurrent.futures.as_completed(futs):
                    f.result()
                    bar.update(1)
                bar.close()
            else:
                futs = [ex.submit(_compute_voxel_chunk, ch, *chunk_args) for ch in chunks]
                for f in futs:
                    f.result()

    return dwi


def make_affine(origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Build a 4x4 voxel-to-world affine matrix.

    Parameters
    ----------
    origin : np.ndarray, shape (3,)
        World-space coordinates of voxel (0, 0, 0) in mm.
    voxel_size : float
        Isotropic voxel size in mm.

    Returns
    -------
    np.ndarray, shape (4, 4)
    """
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = voxel_size
    aff[:3, 3] = np.asarray(origin, dtype=float)
    return aff


def world_bvecs_to_fsl(bvecs: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert world-space gradient directions to FSL `.bvec` convention.

    SymDWI generates the signal using `bvecs` as directions in world (scanner)
    space. The on-disk `dwi.nii.gz` + `bvals`/`bvecs` triple is the FSL format,
    and tools that read it via the FSL convention (FSL itself, and MRtrix's
    `-fslgrad`) express gradients in the *image voxel* frame and additionally
    negate the first component when the voxel-to-world affine has a positive
    determinant (the "radiological" flip).

    Parameters
    ----------
    bvecs : np.ndarray, shape (N, 3)
        Unit gradient directions in world space (b=0 rows may be zero).
    affine : np.ndarray, shape (4, 4)
        The voxel-to-world affine the NIfTI is saved with.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Gradient directions in FSL `.bvec` convention.
    """
    R = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(R, axis=0)
    norms[norms == 0] = 1.0
    Rn = R / norms
    bvecs_vox = np.asarray(bvecs, dtype=float) @ Rn
    if np.linalg.det(Rn) > 0:
        bvecs_vox = bvecs_vox.copy()
        bvecs_vox[:, 0] *= -1.0
    return bvecs_vox


def save_dwi(
    signal: np.ndarray,
    affine: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    path: str | Path,
    to_fsl: bool = True,
) -> None:
    """
    Save DWI data to a directory.

    Writes three files into `path`: `dwi.nii.gz`, `bvals.txt`, `bvecs.txt`.
    `bvecs.txt` is written in FSL convention (see world_bvecs_to_fsl) so it
    reconstructs correctly with FSL and with MRtrix's `-fslgrad` option.

    Parameters
    ----------
    signal : np.ndarray, shape (*dims, N)
    affine : np.ndarray, shape (4, 4)
    bvals : np.ndarray, shape (N,)
    bvecs : np.ndarray, shape (N, 3)
        Unit gradient directions in world space.
    path : str or Path
        Output directory.
    """
    p = Path(path)
    img = nib.Nifti1Image(signal, affine)
    nib.save(img, p / "dwi.nii.gz")

    if to_fsl:
        bvecs = world_bvecs_to_fsl(bvecs, affine)
    
    np.savetxt(p / "bvals.txt", bvals[None, :], fmt="%g")
    np.savetxt(p / "bvecs.txt", bvecs.T, fmt="%g")


def save_groundtruth(
    groundtruth: dict,
    affine: np.ndarray,
    path: str | Path,
) -> None:
    """
    Save per-voxel ground-truth labels to a directory.

    Writes into `path`:
      - `wm_mask.nii.gz` : uint8 fiber-presence mask.
      - `fiber_fraction.nii.gz` : float32 intra-axonal volume fraction.
      - `peaks.nii.gz` : float32 (*dims, max_peaks*3), peak directions
                                  interleaved [x0,y0,z0,x1,y1,z1,...] — the same
                                  layout as MRtrix `sh2peaks` output, so it can be
                                  compared/overlaid directly.

    Parameters
    ----------
    groundtruth : dict
        The dict returned by simulate_dwi(..., return_groundtruth=True).
    affine : np.ndarray, shape (4, 4)
    path : str or Path
        Output directory.
    """
    p = Path(path)
    gt = groundtruth
    nib.save(nib.Nifti1Image(gt["wm_mask"], affine), p / "wm_mask.nii.gz")
    nib.save(nib.Nifti1Image(gt["fiber_fraction"], affine), p / "fiber_fraction.nii.gz")
    peaks = gt["peaks"]
    peaks_flat = peaks.reshape(*peaks.shape[:3], -1).astype(np.float32)
    nib.save(nib.Nifti1Image(peaks_flat, affine), p / "peaks.nii.gz")
