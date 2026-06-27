import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSlider, QApplication
from PySide6.QtCore import Signal, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle


class OrthoView(QWidget):
    point_picked = Signal(str, float, float, float)

    _PLANE_LABELS = {
        "axial":    ("X (mm)", "Y (mm)"),
        "coronal":  ("X (mm)", "Z (mm)"),
        "sagittal": ("Y (mm)", "Z (mm)"),
    }

    def __init__(self, plane: str, parent=None):
        super().__init__(parent)
        assert plane in self._PLANE_LABELS, f"Unknown plane: {plane}"
        self.plane = plane

        self._vol = (60.0, 60.0, 60.0)
        self._voxel = 1.0
        self._origin = (0.0, 0.0, 0.0)
        self._dots = []
        self._active_points = []
        self._bundles = []

        self._bg_data = None
        self._bg_clim = (0.0, 1.0)
        self._slice_idx = 0

        self._view_limits = None
        self._nav = None

        fig = Figure(figsize=(3, 3), tight_layout=True)
        fig.patch.set_facecolor("#1e1e1e")
        self.canvas = FigureCanvas(fig)
        self.ax = fig.add_subplot(111)
        self._style_axes()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.valueChanged.connect(self._on_slice_change)
        layout.addWidget(self._slider)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()

    def _style_axes(self):
        self.ax.set_facecolor("#2d2d2d")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#555555")
        self.ax.tick_params(colors="#aaaaaa", labelsize=7)
        xlabel, ylabel = self._PLANE_LABELS[self.plane]
        self.ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        self.ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        self.ax.set_title(self.plane.capitalize(), color="#cccccc", fontsize=9)

    def set_volume(self, x_mm: float, y_mm: float, z_mm: float, voxel_size: float = None,
                   origin=None):
        self._vol = (x_mm, y_mm, z_mm)
        if voxel_size:
            self._voxel = float(voxel_size)
        if origin is not None:
            self._origin = tuple(float(o) for o in origin)
        self._view_limits = None
        self._update_slider_range()
        self._draw()

    def _plane_origin(self):
        """World mm origin (voxel-0 centre) of this plane's (u, v) axes."""
        ox, oy, oz = self._origin
        if self.plane == "axial":
            return ox, oy
        if self.plane == "coronal":
            return ox, oz
        return oy, oz

    def _outplane_origin(self) -> float:
        """World mm origin (voxel-0 centre) of the axis this plane slices."""
        ox, oy, oz = self._origin
        return {"axial": oz, "coronal": oy, "sagittal": ox}[self.plane]

    def _outplane_mm(self) -> float:
        """Extent (mm) of the axis this plane slices through."""
        x, y, z = self._vol
        return {"axial": z, "coronal": y, "sagittal": x}[self.plane]

    def _slice_count_volume(self) -> int:
        """Number of slices implied by the volume grid (no baseline loaded)."""
        if self._voxel <= 0:
            return 1
        return max(1, int(self._outplane_mm() / self._voxel))

    def _update_slider_range(self):
        """Configure the slider from the volume grid when no baseline is loaded.
        With a baseline, set_background owns the slider range."""
        if self._bg_data is not None:
            return
        n = self._slice_count_volume()
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, n - 1))
        if self._slice_idx <= 0 or self._slice_idx > n - 1:
            self._slice_idx = n // 2
        self._slider.setValue(self._slice_idx)
        self._slider.blockSignals(False)
        self._slider.show()

    def current_slice_mm(self) -> float:
        """Out-of-plane world position (mm) of the current slice."""
        return self._outplane_origin() + self._slice_idx * self._voxel

    def set_slice_mm(self, mm: float):
        """Move the slider to the slice containing the given world mm position."""
        if self._voxel <= 0:
            return
        n = (self._slice_axis_len() if self._bg_data is not None
             else self._slice_count_volume())
        idx = int(np.clip(round((mm - self._outplane_origin()) / self._voxel),
                          0, max(0, n - 1)))
        if idx == self._slice_idx:
            return
        self._slice_idx = idx
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._draw()

    def set_active_points(self, points: list):
        """Update the projected overlay of resolved 3D control points."""
        self._active_points = list(points)
        self._draw()

    def set_bundles(self, bundles: list):
        """Set the semi-transparent overlays of confirmed bundles.

        ``bundles`` is a list of (control_points, color) for every bundle that
        should be shown, projected onto this plane.
        """
        self._bundles = list(bundles)
        self._draw()

    def set_background(self, data):
        """
        Set (or clear) the baseline T1/T2 volume shown behind the drawing.

        ``data`` is a 3D array in canonical RAS voxel order (nx, ny, nz),
        or None to remove the background.
        """
        self._bg_data = data
        self._view_limits = None
        if data is None:
            self._update_slider_range()
            self._draw()
            return

        n = self._slice_count()
        self._slice_idx = n // 2
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, n - 1))
        self._slider.setValue(self._slice_idx)
        self._slider.blockSignals(False)

        finite = data[np.isfinite(data)]
        if finite.size:
            self._bg_clim = (float(np.percentile(finite, 1.0)),
                             float(np.percentile(finite, 99.0)))
        else:
            self._bg_clim = (0.0, 1.0)

        self._slider.show()
        self._draw()

    def _slice_axis_len(self):
        """Length of the volume axis this plane slices through."""
        if self._bg_data is None:
            return 0
        nx, ny, nz = self._bg_data.shape[:3]
        return {"axial": nz, "coronal": ny, "sagittal": nx}[self.plane]

    def _slice_count(self):
        return self._slice_axis_len()

    def _current_slice(self):
        """Return the 2D slice (rows=v, cols=u) for the current index."""
        data = self._bg_data
        n = self._slice_axis_len()
        k = int(np.clip(self._slice_idx, 0, n - 1))
        if self.plane == "axial":
            return data[:, :, k].T
        if self.plane == "coronal":
            return data[:, k, :].T
        return data[k, :, :].T

    def _on_slice_change(self, value):
        self._slice_idx = value
        self._draw()

    def _on_scroll(self, event):
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl and event.xdata is not None and event.ydata is not None:
            factor = 0.8 if event.button == "up" else 1.25
            self._zoom_about(event.xdata, event.ydata, factor)
            return
        if self._bg_data is None:
            return
        step = 1 if event.button == "up" else -1
        n = self._slice_count()
        new_idx = int(np.clip(self._slice_idx + step, 0, n - 1))
        if new_idx != self._slice_idx:
            self._slider.setValue(new_idx)


    def _apply_limits(self, xlim, ylim):
        """Set view limits without a full redraw (smooth during a drag)."""
        self._view_limits = (xlim, ylim)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.canvas.draw_idle()

    def _zoom_about(self, cx, cy, factor):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self._apply_limits(
            (cx - (cx - x0) * factor, cx + (x1 - cx) * factor),
            (cy - (cy - y0) * factor, cy + (y1 - cy) * factor),
        )

    def _on_motion(self, event):
        nav = self._nav
        if nav is None:
            return
        dx_px = event.x - nav["px"][0]
        dy_px = event.y - nav["px"][1]
        (x0, x1), (y0, y1) = nav["xlim"], nav["ylim"]
        bbox = self.ax.get_window_extent()
        if nav["mode"] == "pan":
            dx = dx_px * (x1 - x0) / max(bbox.width, 1)
            dy = dy_px * (y1 - y0) / max(bbox.height, 1)
            self._apply_limits((x0 - dx, x1 - dx), (y0 - dy, y1 - dy))
        else:
            factor = float(np.exp(-dy_px * 0.01))
            cx, cy = nav["anchor"]
            self._apply_limits(
                (cx - (cx - x0) * factor, cx + (x1 - cx) * factor),
                (cy - (cy - y0) * factor, cy + (y1 - cy) * factor),
            )

    def _on_release(self, event):
        self._nav = None

    def _on_key(self, event):
        if event.key == "r":
            self._view_limits = None
            self._draw()

    def clear_dots(self):
        self._dots.clear()
        self._draw()

    def _uv_from_point(self, pt):
        """Project a 3D point [x,y,z] onto this view's (u, v) axes."""
        x, y, z = pt
        if self.plane == "axial":
            return x, y
        if self.plane == "coronal":
            return x, z
        return y, z

    def _draw(self):
        x_mm, y_mm, z_mm = self._vol
        if self.plane == "axial":
            w, h = x_mm, y_mm
        elif self.plane == "coronal":
            w, h = x_mm, z_mm
        else:
            w, h = y_mm, z_mm

        ou, ov = self._plane_origin()
        half = 0.5 * self._voxel
        u0, v0 = ou - half, ov - half
        extent = (u0, u0 + w, v0, v0 + h)

        self.ax.cla()
        self._style_axes()

        if self._bg_data is not None:
            self.ax.imshow(
                self._current_slice(),
                cmap="gray", origin="lower",
                extent=extent, aspect="equal",
                vmin=self._bg_clim[0], vmax=self._bg_clim[1],
                interpolation="nearest", zorder=0,
            )

        rect = Rectangle((u0, v0), w, h,
                          linewidth=1.5, edgecolor="#5599ff",
                          facecolor="none", linestyle="--")
        self.ax.add_patch(rect)

        for u, v in self._dots:
            self.ax.plot(u, v, "o", color="#ff9900", markersize=5, zorder=5)

        for points, color in self._bundles:
            if len(points) < 1:
                continue
            us = [self._uv_from_point(p)[0] for p in points]
            vs = [self._uv_from_point(p)[1] for p in points]
            self.ax.plot(us, vs, "-o", color=color, alpha=0.45,
                         linewidth=1.2, markersize=4, zorder=3)

        if self._active_points:
            us = [self._uv_from_point(p)[0] for p in self._active_points]
            vs = [self._uv_from_point(p)[1] for p in self._active_points]
            self.ax.plot(us, vs, "-", color="#aaffaa", linewidth=1.2, zorder=4)
            self.ax.plot(us, vs, "o", color="#44ff44", markersize=6, zorder=5)

        self.ax.set_aspect("equal")
        if self._view_limits is not None:
            (x0, x1), (y0, y1) = self._view_limits
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y0, y1)
        else:
            margin = max(w, h) * 0.02
            self.ax.set_xlim(u0 - margin, u0 + w + margin)
            self.ax.set_ylim(v0 - margin, v0 + h + margin)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax:
            return
        
        if event.button in (2, 3):
            self._nav = {
                "mode": "pan" if event.button == 3 else "zoom",
                "px": (event.x, event.y),
                "xlim": self.ax.get_xlim(),
                "ylim": self.ax.get_ylim(),
                "anchor": (event.xdata, event.ydata),
            }
            return
        if event.button != 1:
            return
        u, v = event.xdata, event.ydata
        if u is None or v is None:
            return
        
        self.point_picked.emit(self.plane, float(u), float(v),
                               self.current_slice_mm())