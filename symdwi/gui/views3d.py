"""3D scene preview widget for the SymDWI GUI.

Embeds a PyVista/VTK render window in a Qt layout (via pyvistaqt's
``QtInteractor``) to visualize the simulation volume bounding box together
with fiber-bundle streamlines. Bundles are split into a transient "preview"
actor (used while a bundle is being configured) and a list of "confirmed"
bundle actors that persist in the scene until explicitly cleared.
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout

from symdwi.gui.theme import BG_0


class Preview3D(QWidget):
    """PyVista 3D render widget embedded in Qt.

    Hosts a single ``QtInteractor`` render view and manages three
    categories of actors: an optional bounding-box wireframe, a single
    transient preview-bundle actor, and a list of confirmed bundle actors.
    Defines no custom Qt signals.
    """

    # Cap on streamlines actually drawn per bundle: large tractograms (tens
    # of thousands of streamlines) make interactive rotation/pan laggy, and
    # the 3D view is a geometry preview, not the simulation itself, so a
    # random subsample is a faithful-enough stand-in.
    MAX_PREVIEW_STREAMLINES = 300

    def __init__(self, parent=None):
        """Build the render view and initialize empty actor state.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background(BG_0)
        layout.addWidget(self.plotter)

        self._box_actor = None
        self._preview_actor = None
        self._bundle_actors = []


    def set_volume(self, x_mm: float, y_mm: float, z_mm: float, origin=(0.0, 0.0, 0.0)):
        """Draw (or redraw) a wireframe bounding box in world mm.

        Removes any previously drawn box before adding the new one and
        resets the camera to frame it.

        Args:
            x_mm: Box extent along X, in millimeters.
            y_mm: Box extent along Y, in millimeters.
            z_mm: Box extent along Z, in millimeters.
            origin: (x, y, z) world-space coordinate of the box's
                minimum corner, in millimeters.
        """
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
        with one cell per streamline. Randomly subsamples down to
        MAX_PREVIEW_STREAMLINES first, since this is a geometry preview, not
        the simulation itself, and full tractograms make the view laggy.

        Args:
            streamlines: Array of shape (n_streamlines, n_samples, 3) with
                per-point (x, y, z) coordinates for each streamline.

        Returns:
            A ``pv.PolyData`` containing all (possibly subsampled) points
            with one polyline cell per streamline.
        """
        if len(streamlines) > self.MAX_PREVIEW_STREAMLINES:
            idx = np.random.default_rng(0).choice(
                len(streamlines), self.MAX_PREVIEW_STREAMLINES, replace=False
            )
            streamlines = streamlines[idx]

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
        """Show a temporary preview bundle (overwritten on each call).

        Replaces any existing preview actor, so only one preview bundle is
        ever shown at a time.

        Args:
            streamlines: Array of shape (n_streamlines, n_samples, 3) with
                per-point (x, y, z) coordinates for each streamline.
            color: PyVista color spec for the preview lines.
        """
        if self._preview_actor is not None:
            self.plotter.remove_actor(self._preview_actor)
            self._preview_actor = None

        pd = self._streamlines_to_polydata(streamlines)
        self._preview_actor = self.plotter.add_mesh(
            pd, color=color, line_width=1.5, render_lines_as_tubes=False
        )

    def clear_preview(self):
        """Remove the transient preview bundle actor, if one is shown."""
        if self._preview_actor is not None:
            self.plotter.remove_actor(self._preview_actor)
            self._preview_actor = None

    def add_bundle(self, streamlines: np.ndarray, color: str = "white"):
        """Add a confirmed bundle (persists until clear_bundles is called).

        Args:
            streamlines: Array of shape (n_streamlines, n_samples, 3) with
                per-point (x, y, z) coordinates for each streamline.
            color: PyVista color spec for the bundle's lines.
        """
        pd = self._streamlines_to_polydata(streamlines)
        actor = self.plotter.add_mesh(
            pd, color=color, line_width=1.5, render_lines_as_tubes=False
        )
        self._bundle_actors.append(actor)

    def clear_bundles(self):
        """Remove all confirmed bundle actors from the scene."""
        for actor in self._bundle_actors:
            self.plotter.remove_actor(actor)
        self._bundle_actors.clear()

    def remove_last_bundle(self):
        """Remove the most recently added confirmed bundle, if any."""
        if self._bundle_actors:
            self.plotter.remove_actor(self._bundle_actors.pop())