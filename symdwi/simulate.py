import numpy as np
import concurrent.futures
import json
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import tqdm as tqdm_mod

from dipy.data import get_sphere
from dipy.core.sphere import HemiSphere, disperse_charges
import nibabel as nib

from .bundle import Bundle, TissueParameters, DEFAULT_TISSUE_PARAMETERS
from ._voxel import voxelise_bundles
from .gm_compartments import gpd_sphere_attenuation, isotropic_stick_attenuation


@dataclass
class ScanParameters:
    """
    Biophysical/sequence parameters.

    Parameters
    ----------
    te_ms : float or None
        Echo time in ms. Set to None to disable T2 relaxation weighting.
    d_iso : float
        Free-water (CSF) isotropic diffusivity (mm^2/s).
    t2_csf_ms : float
        T2 relaxation time for the CSF/free-water compartment (ms).
    background_csf : float
        Constant free-water signal added to every voxel (including background).
        Set to 0.0 for a silent background.
    """
    te_ms: float | None = None
    d_iso: float = 3.0e-3
    t2_csf_ms: float = 2000.0
    background_csf: float = 0.0


@dataclass
class GMParameters:
    """
    SANDI 3-compartment gray matter model: soma (restricted sphere), dispersed
    neurite (isotropically-averaged stick), and extracellular water (ball).

    Parameters
    ----------
    f_in : float
        Intra-neurite (stick) volume fraction. Must be in [0, 1].
    f_ec : float
        Extracellular (ball) volume fraction. Must be in [0, 1].
    d_in : float
        Intra-neurite diffusivity (mm^2/s).
    d_ec : float
        Extracellular diffusivity (mm^2/s).
    r_s : float
        Soma (sphere) radius (um).
    d_is : float
        Intra-soma intrinsic diffusivity (mm^2/s).
    big_delta_ms : float
        Diffusion time Delta (ms). Fixed, single value for the whole
        acquisition.
    small_delta_ms : float
        Gradient pulse duration delta (ms). Fixed, single value for the
        whole acquisition.
    """
    f_in: float = 0.5
    f_ec: float = 0.35
    d_in: float = 1.7e-3
    d_ec: float = 1.5e-3
    r_s: float = 6.0
    d_is: float = 3.0e-3
    big_delta_ms: float = 22.0
    small_delta_ms: float = 13.0

    def __post_init__(self):
        assert 0.0 <= self.f_in <= 1.0, f"f_in must be in [0, 1], got {self.f_in}"
        assert 0.0 <= self.f_ec <= 1.0, f"f_ec must be in [0, 1], got {self.f_ec}"
        assert self.f_in + self.f_ec <= 1.0 + 1e-9, (
            f"f_in + f_ec must be <= 1 (remainder is soma fraction), "
            f"got f_in={self.f_in}, f_ec={self.f_ec}, sum={self.f_in + self.f_ec}"
        )
        assert self.d_in > 0.0, f"d_in must be > 0, got {self.d_in}"
        assert self.d_ec > 0.0, f"d_ec must be > 0, got {self.d_ec}"
        assert self.d_is > 0.0, f"d_is must be > 0, got {self.d_is}"
        assert self.r_s > 0.0, f"r_s must be > 0, got {self.r_s}"
        assert self.big_delta_ms > 0.0, f"big_delta_ms must be > 0, got {self.big_delta_ms}"
        assert self.small_delta_ms > 0.0, f"small_delta_ms must be > 0, got {self.small_delta_ms}"
        assert self.small_delta_ms <= self.big_delta_ms, (
            f"small_delta_ms (pulse duration) must be <= big_delta_ms "
            f"(diffusion time), got delta={self.small_delta_ms}, Delta={self.big_delta_ms}"
        )

    @property
    def f_is(self) -> float:
        """Soma volume fraction, derived: 1 - f_in - f_ec."""
        return 1.0 - self.f_in - self.f_ec


DEFAULT_GM_PARAMETERS = GMParameters()


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


def _resolve_tissue(bundle: Bundle, default_tissue: TissueParameters) -> TissueParameters:
    return bundle.tissue if bundle.tissue is not None else default_tissue


def simulate_dwi(
    bundles: list[Bundle],
    bvals: np.ndarray,
    bvecs: np.ndarray,
    scan: ScanParameters,
    tissue: TissueParameters | None = None,
    gm: GMParameters | None = None,
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
    progress_cb=None,
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
    scan : ScanParameters
        Volume-wide sequence/tissue constants.
    tissue : TissueParameters or None
        Bundle specific tissue properties. If not specified, the default
        parameters are used.
    gm : GMParameters or None
        SANDI-style gray matter compartment model, used when a "gm" tissue
        mask is provided. If not specified, the default parameters are used.
    origin : np.ndarray or None
        World-space corner of the volume in mm.
        Auto-computed from bundle extents if None.
    dims : tuple[int, ...]
        Volume shape in voxels.
    voxel_size : float
        Isotropic voxel size in mm.
    snr : float or None
        If given, adds Rician noise with sigma = (per-voxel S0) / snr, where
        S0 is that voxel's own noise-free mean b=0 signal.
    n_jobs : int
        Number of threads for parallel voxel processing. Default 1 (no threading).
    tissue_masks : dict or None
        Optional tissue probability/binary maps. Accepted keys: "wm", "gm", "csf".
        Each value must be a 3D array matching `dims`, with values in [0, 1].
        Binary (0/1) and probabilistic masks are both accepted.
        Per-voxel tissue fractions are normalised to sum to 1 before use.
        The "gm" compartment is modelled as a 3-compartment SANDI-style
        soma/neurite/extracellular mixture (see `gm`/`GMParameters`). 
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
    progress_cb : callable or None
        Optional ``(done, total, stage) -> None``, called at the start of each
        major stage (voxelising, ground truth, signal computation) and as
        chunks complete during signal computation. See compute_signal for the
        thread-safety contract.

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
    default_tissue = tissue if tissue is not None else DEFAULT_TISSUE_PARAMETERS

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
    if progress_cb is not None:
        progress_cb(0, 1, "Voxelising bundles")
    sphere = get_sphere(name=sphere)
    volume = voxelise_bundles(bundles, origin, dims, voxel_size, sphere=sphere)
    if verbose >= 1:
        print(f"  {len(volume)} fiber voxels found.")

    bundle_tissues = [_resolve_tissue(b, default_tissue) for b in bundles]

    fiber_vols = {}
    for key, per_bundle in volume.items():
        for bundle_idx, hist in per_bundle.items():
            count = hist.sum()
            r = bundle_tissues[bundle_idx].axon_radius
            if r is not None:
                r_mm = r * 1e-3
                fiber_vols[(key, bundle_idx)] = np.pi * r_mm ** 2 * count
            else:
                fiber_vols[(key, bundle_idx)] = count

    max_fiber_vol = max(fiber_vols.values()) if fiber_vols else 1.0
    max_fiber_vol = max_fiber_vol if max_fiber_vol > 0 else 1.0
    density_frac = {k: min(1.0, fv / max_fiber_vol) for k, fv in fiber_vols.items()}

    for key, per_bundle in volume.items():
        for bundle_idx, hist in per_bundle.items():
            total = hist.sum()
            if total > 0:
                per_bundle[bundle_idx] = hist / total

    groundtruth = None
    if return_groundtruth:
        if verbose >= 1:
            print("Extracting ground-truth peaks...")
        if progress_cb is not None:
            progress_cb(0, 1, "Extracting ground-truth peaks")
        groundtruth = extract_groundtruth(
            volume, density_frac, sphere, dims_arr, max_peaks=max_peaks
        )

    if verbose >= 1:
        tag = f"({n_jobs} threads)" if n_jobs > 1 else "(single-threaded)"
        print(f"Computing DWI signal {tag}...")
    signal = compute_signal(
        volume, density_frac, bundle_tissues, scan, bvals, bvecs, dims_arr, sphere,
        n_jobs=n_jobs, tissue_masks=tissue_masks, gm=gm, verbose=verbose,
        progress_cb=progress_cb,
    )
    if verbose >= 1:
        print("  Signal computation done.")

    affine = make_affine(origin, voxel_size)

    if snr is not None:
        if verbose >= 1:
            print(f"Adding Rician noise (SNR={snr}, relative to per-voxel S0)...")
        if progress_cb is not None:
            progress_cb(0, 1, "Adding noise")
        rng = np.random.default_rng(seed)
        b0_mask = bvals <= 1e-6
        if not np.any(b0_mask):
            raise ValueError("snr requires at least one b=0 volume to define S0.")
        s0 = signal[..., b0_mask].mean(axis=-1, keepdims=True)
        sigma = s0 / snr
        r = rng.normal(loc=0, scale=sigma, size=signal.shape)
        i = rng.normal(loc=0, scale=sigma, size=signal.shape)
        signal = np.sqrt((signal + r) ** 2 + i ** 2).astype(np.float32)

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
    maxima of each voxel's pooled (across-bundle) direction histogram on the
    discretisation sphere.

    Parameters
    ----------
    volume : defaultdict
        Sparse per-voxel, per-bundle direction distributions, each normalised
        to sum to 1 (as produced inside simulate_dwi after ODF normalisation).
    density_frac : dict
        Intra-axonal volume fraction per (voxel, bundle_idx) key.
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

    for k, per_bundle in volume.items():
        wm_mask[k] = 1
        pooled = sum(per_bundle.values())
        pooled_sum = pooled.sum()
        if pooled_sum > 0:
            pooled = pooled / pooled_sum
        fiber_fraction[k] = sum(
            density_frac.get((k, bundle_idx), 0.0) for bundle_idx in per_bundle
        )
        dirs, vals, _ = peak_directions(
            pooled, sphere,
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
    bundle_tissues,
    scan,
    bvals,
    bvecs,
    sphere,
    water,
    gm_signal,
    dwi,
    tissue_masks,
):
    for k, per_bundle in chunk:
        wm_signal = np.zeros(len(bvals))

        for bundle_idx, v in per_bundle.items():
            tissue = bundle_tissues[bundle_idx]

            vf_intra = density_frac[(k, bundle_idx)]
            remaining = 1.0 - vf_intra

            if scan.te_ms is not None:
                w_intra = np.exp(-scan.te_ms / tissue.t2_intra_ms)
                w_extra = np.exp(-scan.te_ms / tissue.t2_extra_ms)
                w_csf = np.exp(-scan.te_ms / scan.t2_csf_ms)
                w_myelin = np.exp(-scan.te_ms / tissue.t2_myelin_ms)
            else:
                w_intra = w_extra = w_csf = w_myelin = 1.0

            csf_mask_present = tissue_masks is not None and "csf" in tissue_masks
            if csf_mask_present:
                i, j, kk = k
                csf_frac_mask = tissue_masks["csf"][i, j, kk]
                vf_csf_local = remaining * csf_frac_mask
                vf_extra = remaining * (1.0 - csf_frac_mask)
            else:
                vf_csf_local = remaining * tissue.f_csf_split
                vf_extra = remaining * (1.0 - tissue.f_csf_split)

            mask = v > 0.0
            f_pop = v[mask]
            verts_active = sphere.vertices[mask]
            dot_sq = (bvecs @ verts_active.T) ** 2

            stick = np.exp(-bvals[:, None] * tissue.di_axial * dot_sq)
            zeppelin = (
                np.exp(-bvals[:, None] * tissue.de_radial)
                * np.exp(-bvals[:, None] * (tissue.de_axial - tissue.de_radial) * dot_sq)
            )
            vf_myelin = vf_intra * tissue.f_myelin
            vf_intra_tissue = vf_intra * (1.0 - tissue.f_myelin)
            wm_signal = wm_signal + (
                w_intra * vf_intra_tissue * (stick @ f_pop)
                + w_myelin * vf_myelin * (stick @ f_pop)
                + w_extra * vf_extra * (zeppelin @ f_pop)
                + w_csf * vf_csf_local * water
            )

        wm_signal = wm_signal / max(1, len(per_bundle))

        if tissue_masks is None:
            signal = wm_signal
        else:
            i, j, kk = k
            if "wm" in tissue_masks:
                wm_frac = tissue_masks["wm"][i, j, kk]
            else:
                wm_frac = 1.0 if per_bundle else 0.0
            gm_frac = tissue_masks["gm"][i, j, kk] if "gm" in tissue_masks else 0.0
            csf_frac = tissue_masks["csf"][i, j, kk] if "csf" in tissue_masks else 0.0
            total = wm_frac + gm_frac + csf_frac
            if total < 1e-9:
                dwi[k[0], k[1], k[2], :] = 0.0
                continue
            wm_frac /= total
            gm_frac /= total
            csf_frac /= total
            w_csf_bg = np.exp(-scan.te_ms / scan.t2_csf_ms) if scan.te_ms is not None else 1.0
            signal = (
                wm_frac * wm_signal
                + gm_frac * w_csf_bg * gm_signal
                + csf_frac * w_csf_bg * water
            )

        dwi[k[0], k[1], k[2], :] = signal


def compute_signal(
    volume: defaultdict,
    density_frac: dict,
    bundle_tissues: list[TissueParameters],
    scan: ScanParameters,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    dims: np.ndarray,
    sphere,
    n_jobs: int = 1,
    tissue_masks: dict[str, np.ndarray] | None = None,
    gm: GMParameters | None = None,
    verbose: int = 0,
    progress_cb=None,
) -> np.ndarray:
    """
    Compute DWI signal using the Standard Model (stick + zeppelin + ball)
    for white matter, and a SANDI-like 3-compartment model (soma + neurite
    + ball) for gray matter.

    Parameters
    ----------
    volume : defaultdict
        Sparse ODF grid from voxelise_bundles: (i,j,k) -> {bundle_idx: hist},
        each hist normalised to sum to 1.
    density_frac : dict
        Intra-axonal volume fraction per (voxel, bundle_idx) key.
    bundle_tissues : list[TissueParameters]
        Resolved tissue parameters, one per bundle (by index into the
        original `bundles` list passed to simulate_dwi).
    scan : ScanParameters
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
    gm : GMParameters or None
        SANDI-style gray matter compartment model.
    verbose : int
        Verbosity level. 0 = silent, 1 = step prints, 2 = tqdm progress bar.
    progress_cb : callable or None
        Optional ``(done, total, stage) -> None`` called as voxel chunks
        complete (per-voxel when n_jobs=1, per-chunk when threaded). Intended
        for UI progress reporting; independent of `verbose`, and always
        called from the same thread `compute_signal` itself runs in — safe
        to connect to a Qt signal's `.emit`, never to a widget method
        directly, since this may run off the main thread.

    Returns
    -------
    np.ndarray, shape (*dims, N), dtype float32
    """
    gm = gm if gm is not None else DEFAULT_GM_PARAMETERS

    dwi = np.zeros((*dims, len(bvals)), dtype=np.float32)

    water = np.exp(-bvals * scan.d_iso)

    soma_signal = gpd_sphere_attenuation(
        bvals, gm.big_delta_ms, gm.small_delta_ms, gm.d_is, gm.r_s,
    )
    neurite_signal = isotropic_stick_attenuation(bvals, gm.d_in)
    ec_signal = np.exp(-bvals * gm.d_ec)
    gm_signal = (
        gm.f_is * soma_signal
        + gm.f_in * neurite_signal
        + gm.f_ec * ec_signal
    )

    if scan.background_csf > 0.0:
        w_csf_bg = np.exp(-scan.te_ms / scan.t2_csf_ms) if scan.te_ms is not None else 1.0
        dwi += scan.background_csf * w_csf_bg * water

    items = list(volume.items())

    if tissue_masks is not None:
        any_frac = np.zeros(tuple(dims), dtype=float)
        for arr in tissue_masks.values():
            any_frac += arr
        missing = np.argwhere(any_frac > 0.0)
        for idx in missing:
            key = tuple(int(x) for x in idx)
            if key not in volume:
                items.append((key, {}))

    chunk_args = (
        density_frac, bundle_tissues, scan, bvals, bvecs, sphere,
        water, gm_signal, dwi, tissue_masks,
    )

    if n_jobs == 1:
        n_chunks = min(len(items), 200) if progress_cb is not None else 1
        chunk_size = max(1, (len(items) + n_chunks - 1) // max(1, n_chunks))
        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)] or [[]]
        wrapped = tqdm_mod.tqdm(chunks, desc="voxels", unit="chunk") if verbose >= 2 else chunks
        for i, ch in enumerate(wrapped):
            _compute_voxel_chunk(ch, *chunk_args)
            if progress_cb is not None:
                progress_cb(i + 1, len(chunks), "Computing DWI signal")
    else:
        n_chunks = min(len(items), n_jobs * 8)
        chunk_size = max(1, (len(items) + n_chunks - 1) // n_chunks)
        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
            bar = tqdm_mod.tqdm(
                total=len(chunks), desc="chunks", unit="chunk",
                postfix={"voxels": len(items)},
            ) if verbose >= 2 else None
            futs = {ex.submit(_compute_voxel_chunk, ch, *chunk_args): ch
                    for ch in chunks}
            n_done = 0
            for f in concurrent.futures.as_completed(futs):
                f.result()
                n_done += 1
                if bar is not None:
                    bar.update(1)
                if progress_cb is not None:
                    progress_cb(n_done, len(chunks), "Computing DWI signal")
            if bar is not None:
                bar.close()

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
    big_delta_ms: float | None = None,
    small_delta_ms: float | None = None,
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
    big_delta_ms : float or None
        Diffusion time Delta (ms), if the acquisition used one fixed value
        (e.g. for the GMParameters soma compartment). If given (together
        with small_delta_ms), also writes acquisition_params.json.
    small_delta_ms : float or None
        Gradient pulse duration delta (ms). See big_delta_ms.
    """
    p = Path(path)
    img = nib.Nifti1Image(np.asarray(signal, dtype=np.float32), affine)
    nib.save(img, p / "dwi.nii.gz")

    if to_fsl:
        bvecs = world_bvecs_to_fsl(bvecs, affine)

    np.savetxt(p / "bvals.txt", bvals[None, :], fmt="%g")
    np.savetxt(p / "bvecs.txt", bvecs.T, fmt="%g")

    if big_delta_ms is not None and small_delta_ms is not None:
        with open(p / "acquisition_params.json", "w") as f:
            json.dump(
                {"big_delta_ms": big_delta_ms, "small_delta_ms": small_delta_ms}, f,
            )


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