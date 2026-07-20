import numpy as np
from scipy.interpolate import splprep, splev
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from nibabel.streamlines import Tractogram
import nibabel as nib
from dataclasses import dataclass


@dataclass
class BundleGeometry:
    """
    Geometrical properties of a bundle.

    Parameters
    ----------
    control_points : np.ndarray, shape (N, 3)
        3D points the centerline passes through.
    n_streamlines : int
        Number of streamlines in the bundle.
    radius : float
        Thickness of the bundle (max offset of streamlines from center).
    n_samples : int
        Number of points sampled along each streamline.
    degree : int
        Spline degree (3 = cubic).
    smoothing : float
        How strictly the centerline hits the control points (0 = exact).
    taper : callable or None
        Optional function controlling how radius varies along the bundle.
    dispersion : float
        Std-dev of per-control-point wander (same units as coordinates).
        Controls how much fibers drift from a perfectly parallel tube.
    seed : int or None
        RNG seed.
    """
    control_points: np.ndarray
    n_streamlines: int = 10000
    radius: float = 2.0
    n_samples: int = 128
    degree: int = 3
    smoothing: float = 0.0
    taper: object = None
    dispersion: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        self.control_points = np.asarray(self.control_points, dtype=float)
        assert self.control_points.shape[1] == 3, "Cannot use non 3D set of points."


@dataclass
class TissueParameters:
    """
    Standard Model compartment description for one or multiple bundles.
    Parameters
    ----------
    f_csf_split : float
        Fraction of non-axonal voxel space attributed to free
        water (CSF) rather than extra-axonal water, in pure-fiber voxels with
        no CSF tissue mask contribution. Must satisfy 0 <= f_csf_split <= 1.
    di_axial : float
        Intra-axonal axial diffusivity (mm^2/s).
    de_axial : float
        Extra-axonal axial diffusivity (mm^2/s).
    de_radial : float
        Extra-axonal radial diffusivity (mm^2/s).
    axon_radius : float or None
        Effective packing-density calibration radius, in micrometers. Converts
        streamline point density to a *relative* intra-axonal volume fraction.
        Set to None for uncalibrated (raw count) relative density scaling.
    t2_intra_ms : float
        T2 relaxation time for the intra-axonal compartment (ms).
    t2_extra_ms : float
        T2 relaxation time for the extra-axonal compartment (ms).
    t2_myelin_ms : float
        T2 relaxation time for the short-T2 myelin-water pool (ms). Only
        contributes to the signal when f_myelin > 0.
    f_myelin : float
        Fraction of the intra-axonal-adjacent signal reassigned to the
        myelin-water pool.
    """

    f_csf_split: float = 0.0

    di_axial: float = 1.7e-3
    de_axial: float = 1.7e-3
    de_radial: float = 0.5e-3

    axon_radius: float | None = 1.0

    t2_intra_ms: float = 80.0
    t2_extra_ms: float = 45.0
    t2_myelin_ms: float = 15.0
    f_myelin: float = 0.0

    def __post_init__(self):
        assert 0.0 <= self.f_csf_split <= 1.0, (
            f"f_csf_split must be in [0, 1], got {self.f_csf_split}"
        )
        assert 0.0 <= self.f_myelin <= 1.0, (
            f"f_myelin must be in [0, 1], got {self.f_myelin}"
        )


DEFAULT_TISSUE_PARAMETERS = TissueParameters()


class Bundle:
    def __init__(
        self,
        geometry: BundleGeometry,
        tissue: TissueParameters | None = None,
    ):
        """
        Fiber bundle modeled via B-splines.

        A B-spline is a smooth curve defined via a set of control points it passes
        through/near. A bundle is built here using a large set of such curves.

        The main path of a bundle is built by using a singular centerline B-spline, that
        is explicitly fit through the specified control points.
        Then, sideways from the centerline and within the radius specified, streamlines
        are placed by discretizing it as a streamline of n_samples.

        Parameters
        ----------
        geometry : BundleGeometry
            The bundle's shape in space.
        tissue : TissueParameters or None
            Per-bundle override of the Standard Model tissue parameters. If
            None, the bundle falls back to whatever default `TissueParameters`
            the simulation is run with.
        """
        self.geometry = geometry
        self.tissue = tissue

        self.control_points = geometry.control_points
        self.n_streamlines = geometry.n_streamlines
        self.radius = geometry.radius
        self.n_samples = geometry.n_samples
        self.degree = geometry.degree
        self.smoothing = geometry.smoothing
        self.taper = geometry.taper
        self.dispersion = geometry.dispersion
        self.rng = np.random.default_rng(geometry.seed)

        self._build_bundle()

    @classmethod
    def from_tck(cls, path: str, n_samples: int = 100,
                 tissue: TissueParameters | None = None) -> "Bundle":
        """
        Load a bundle from a .tck tractogram file.

        Each streamline is resampled to `n_samples` points via arc-length
        interpolation. Tangents are computed from central differences.

        Parameters
        ----------
        path : str
            Path to a .tck file.
        n_samples : int
            Number of points to resample each streamline to.
        tissue : TissueParameters or None
            Per-bundle override of the Standard Model tissue parameters.

        Returns
        -------
        Bundle
        """
        raw = nib.streamlines.load(path).streamlines
        streamlines = []
        tangents = []
        u_new = np.linspace(0.0, 1.0, n_samples)
        for pts in raw:
            pts = np.asarray(pts, dtype=float)
            if len(pts) < 2:
                continue
            diffs = np.diff(pts, axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            if seg_lens.sum() < 1e-9:
                continue
            arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
            arc_norm = arc / arc[-1]
            resampled = np.stack(
                [np.interp(u_new, arc_norm, pts[:, d]) for d in range(3)], axis=1
            )
            t = np.empty_like(resampled)
            t[1:-1] = resampled[2:] - resampled[:-2]
            t[0] = resampled[1] - resampled[0]
            t[-1] = resampled[-1] - resampled[-2]
            norms = np.linalg.norm(t, axis=1, keepdims=True)
            t = t / np.where(norms < 1e-12, 1.0, norms)
            streamlines.append(resampled)
            tangents.append(t)
        if not streamlines:
            raise ValueError(f"No valid streamlines found in {path}")
        b = cls.__new__(cls)
        b.geometry = None
        b.tissue = tissue
        b.streamlines = streamlines
        b.tangents = tangents
        b.n_streamlines = len(streamlines)
        b.n_samples = n_samples
        return b

    def as_array(self) -> np.ndarray:
        """Stack all streamlines into a single array of shape (n_streamlines, n_samples, 3)."""
        return np.stack(self.streamlines)

    def points_with_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        pts : np.ndarray, shape (n_streamlines * n_samples, 3)
        tng : np.ndarray, shape (n_streamlines * n_samples, 3)
        """
        pts = np.concatenate(self.streamlines, axis=0)
        tng = np.concatenate(self.tangents, axis=0)
        return pts, tng

    def visualize(self, interactive=True, save_path=None):
        """
        Render the bundle using FURY (optional dependency).

        Parameters
        ----------
        interactive : bool
            Open an interactive window.
        save_path : str or None
            If given, save a screenshot to this path.
        """
        from fury import actor, window

        scene = window.Scene()
        scene.background((1, 1, 1))

        colors = [np.abs(t) for t in self.tangents]
        stream_actor = actor.line(self.streamlines, colors=colors, linewidth=2)
        scene.add(stream_actor)

        center_actor = actor.line([self.centerline], colors=(0, 0, 0), linewidth=4)
        scene.add(center_actor)

        if save_path:
            window.record(scene, out_path=save_path, size=(900, 900))
        if interactive:
            window.show(scene, size=(900, 900), title="Bundle")
        return scene

    def _build_bundle(self) -> None:
        self._fit_centerline()
        self.cp_frames = self._control_point_frames()
        self._generate_streamlines()

    def _fit_centerline(self) -> None:
        cp = self.control_points
        k = min(self.degree, len(cp) - 1)
        self.tck, self.u_cp = splprep(
            [cp[:, 0], cp[:, 1], cp[:, 2]], k=k, s=self.smoothing
        )
        self.u_cp = np.asarray(self.u_cp)

        u = np.linspace(0, 1, self.n_samples)
        self.u = u
        self.centerline = np.stack(splev(u, self.tck), axis=-1)
        deriv = np.stack(splev(u, self.tck, der=1), axis=-1)
        self.center_tangents = _normalize(deriv)

    def _control_point_frames(self) -> np.ndarray:
        """Orthonormal frame at each control point (where offsets are applied)."""
        deriv = np.stack(splev(self.u_cp, self.tck, der=1), axis=-1)
        tangents = _normalize(deriv)
        return self._parallel_transport_frame(tangents)

    def _parallel_transport_frame(self, tangents):
        """Rotation-minimizing frame: avoids the twist you'd get from Frenet."""
        M = len(tangents)
        frames = np.zeros((M, 3, 3))
        t0 = tangents[0]
        a = np.array([1.0, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1.0, 0])
        n0 = _normalize(np.cross(t0, a))
        b0 = np.cross(t0, n0)
        frames[0] = np.stack([t0, n0, b0])
        for i in range(1, M):
            t_prev, t_cur = tangents[i - 1], tangents[i]
            n_prev = frames[i - 1, 1]
            v = np.cross(t_prev, t_cur)
            s = np.linalg.norm(v)
            if s < 1e-8:
                n_cur = n_prev
            else:
                axis = v / s
                c = np.clip(np.dot(t_prev, t_cur), -1, 1)
                ang = np.arccos(c)
                n_cur = (
                    n_prev * np.cos(ang)
                    + np.cross(axis, n_prev) * np.sin(ang)
                    + axis * np.dot(axis, n_prev) * (1 - np.cos(ang))
                )
            n_cur = _normalize(n_cur - np.dot(n_cur, t_cur) * t_cur)
            b_cur = np.cross(t_cur, n_cur)
            frames[i] = np.stack([t_cur, n_cur, b_cur])
        return frames

    def _generate_streamlines(self) -> None:
        self.streamlines = []
        self.tangents = []

        cp = self.control_points
        K = len(cp)
        k = min(self.degree, K - 1)

        taper = self.taper or (lambda uu: np.ones_like(uu))
        cp_scale = np.asarray(taper(self.u_cp))

        normals = self.cp_frames[:, 1, :]
        binormals = self.cp_frames[:, 2, :]

        for _ in range(self.n_streamlines):
            r = self.radius * np.sqrt(self.rng.random())
            a = 2 * np.pi * self.rng.random()
            off_n, off_b = r * np.cos(a), r * np.sin(a)
            home = cp_scale[:, None] * (off_n * normals + off_b * binormals)

            if self.dispersion:
                w = self.dispersion * self.rng.standard_normal((K, 2))
                w[0] = 0.0
                w[-1] = 0.0
                wander = w[:, 0:1] * normals + w[:, 1:2] * binormals
            else:
                wander = 0.0

            perturbed_cp = cp + home + wander

            tck, _ = splprep(
                [perturbed_cp[:, 0], perturbed_cp[:, 1], perturbed_cp[:, 2]],
                k=k,
                s=self.smoothing,
            )
            line = np.stack(splev(self.u, tck), axis=-1)
            t = np.stack(splev(self.u, tck, der=1), axis=-1)

            self.streamlines.append(line)
            self.tangents.append(_normalize(t))


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def save_bundles(
    bundles: list[Bundle],
    path: str,
    reference: np.ndarray | None = None,
) -> None:
    """
    Save a list of bundles to a single .tck file.

    Parameters
    ----------
    bundles : list[Bundle]
    path : str
        Output path (e.g. "tractogram.tck").
    reference : np.ndarray or None
        If given, use as reference image affine for RASMM space output.
    """
    streamlines = []
    for b in bundles:
        streamlines.extend([np.asarray(s, dtype=np.float32) for s in b.streamlines])

    if not streamlines:
        raise ValueError("No streamlines to save.")

    tractogram = Tractogram(streamlines=streamlines, affine_to_rasmm=np.eye(4))

    if reference is None:
        nib.streamlines.save(tractogram, path)
    else:
        sft = StatefulTractogram(streamlines, reference, Space.RASMM)
        save_tractogram(sft, path)