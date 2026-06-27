import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout


class Preview3D(QWidget):
    """PyVista 3D render widget embedded in Qt."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        layout.addWidget(self.plotter)

        self._box_actor = None
        self._preview_actor = None
        self._bundle_actors = []


    def set_volume(self, x_mm: float, y_mm: float, z_mm: float, origin=(0.0, 0.0, 0.0)):
        """Draw (or redraw) a wireframe bounding box in world mm."""
        if self._box_actor is not None:
            self.plotter.remove_actor(self._box_actor)
            self._box_actor = None

        ox, oy, oz = origin
        box = pv.Box(bounds=(ox, ox + x_mm, oy, oy + y_mm, oz, oz + z_mm))
        self._box_actor = self.plotter.add_mesh(
            box, style="wireframe", color="#5599ff", line_width=2
        )
        self.plotter.reset_camera()

    
    def _streamlines_to_polydata(self, streamlines: np.ndarray) -> pv.PolyData:
        """
        Convert (n_streamlines, n_samples, 3) array to a single PolyData
        with one cell per streamline.
        """
        n_sl, n_pts, _ = streamlines.shape
        points = streamlines.reshape(-1, 3)

        lines = np.empty(n_sl * (n_pts + 1), dtype=int)
        for i in range(n_sl):
            offset = i * (n_pts + 1)
            lines[offset] = n_pts
            lines[offset + 1 : offset + 1 + n_pts] = np.arange(
                i * n_pts, i * n_pts + n_pts
            )

        pd = pv.PolyData()
        pd.points = points.astype(np.float32)
        pd.lines = lines
        return pd

    def set_preview_bundle(self, streamlines: np.ndarray, color: str = "cyan"):
        """Show a temporary preview bundle (overwritten on each call)."""
        if self._preview_actor is not None:
            self.plotter.remove_actor(self._preview_actor)
            self._preview_actor = None

        pd = self._streamlines_to_polydata(streamlines)
        self._preview_actor = self.plotter.add_mesh(
            pd, color=color, line_width=1.5, render_lines_as_tubes=False
        )

    def clear_preview(self):
        if self._preview_actor is not None:
            self.plotter.remove_actor(self._preview_actor)
            self._preview_actor = None

    def add_bundle(self, streamlines: np.ndarray, color: str = "white"):
        """Add a confirmed bundle (persists until clear_bundles is called)."""
        pd = self._streamlines_to_polydata(streamlines)
        actor = self.plotter.add_mesh(
            pd, color=color, line_width=1.5, render_lines_as_tubes=False
        )
        self._bundle_actors.append(actor)

    def clear_bundles(self):
        for actor in self._bundle_actors:
            self.plotter.remove_actor(actor)
        self._bundle_actors.clear()

    def remove_last_bundle(self):
        if self._bundle_actors:
            self.plotter.remove_actor(self._bundle_actors.pop())