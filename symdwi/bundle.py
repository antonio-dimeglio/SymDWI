import numpy as np
from scipy.interpolate import splprep, splev
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from nibabel.streamlines import Tractogram
import nibabel as nib


class Bundle:
    def __init__(
        self,
        control_points: np.ndarray,
        n_streamlines: int = 100,
        radius: float = 2.0,
        n_samples: int = 100,
        degree: int = 3,
        smoothing: float = 0.0,
        taper=None,
        dispersion: float = 0.0,
        seed: int | None = None,
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
        self.control_points = np.asarray(control_points, dtype=float)
        assert self.control_points.shape[1] == 3, "Cannot use non 3D set of points."
        self.n_streamlines = n_streamlines
        self.radius = radius
        self.n_samples = n_samples
        self.degree = degree
        self.smoothing = smoothing
        self.taper = taper
        self.dispersion = dispersion
        self.rng = np.random.default_rng(seed)

        self._build_bundle()

    @classmethod
    def from_tck(cls, path: str, n_samples: int = 100) -> "Bundle":
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
