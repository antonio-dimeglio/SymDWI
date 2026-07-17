"""Main application window for the SymDWI GUI.

Wires together the sidebar controls, the three linked orthogonal 2D views,
and the 3D preview into a single scene-editing workflow for building fiber
bundles, configuring tissue/scan/gray-matter parameters, and launching a
diffusion-weighted MRI simulation. The simulation itself (and the subsequent
file saves) runs on a background QThread via `_SimWorker` so the UI stays
responsive during long BundleSeg-scale runs.
"""

import os
import numpy as np
import nibabel as nib
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QToolBar, QPushButton, QSplitter, QGridLayout,
    QFileDialog, QMessageBox, QProgressDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QKeySequence, QShortcut

from symdwi.gui.sidebar import Sidebar
from symdwi.gui.scene import Scene, VolumeConfig, BundleConfig
from symdwi.gui.views2d import OrthoView
from symdwi.gui.views3d import Preview3D
from symdwi.gui.theme import BUNDLE_COLORS as _BUNDLE_COLORS
from symdwi.bundle import BundleGeometry
import symdwi
import symdwi.gui.io as scene_io


class _SimWorker(QObject):
    """Runs simulate_dwi() and the subsequent file saves entirely off the
    Qt main thread, so a slow BundleSeg-sized save doesn't freeze the UI the
    way it used to when saving happened in the `finished` slot.

    Thread-safety: this object lives on `_sim_thread`, not the GUI thread.
    `progress` is a Qt signal, so `.emit()` is the only thing this class (or
    anything it calls, e.g. compute_signal's progress_cb) is allowed to touch
    on the UI's behalf — Qt marshals the connected slot onto the main thread
    automatically. Never store or call a widget method directly from here.
    """

    finished = Signal(str)  # out_dir, once simulation + all saves are done
    error = Signal(str)
    progress = Signal(int, int, str)  # done, total, stage label

    def __init__(self, bundles, shells, n_b0, scan, tissue, gm, dims, voxel_size, snr,
                 out_dir, origin=None, tissue_masks=None, n_jobs=1, return_groundtruth=True,
                 loaded_gradients=None, big_delta_ms=None, small_delta_ms=None):
        """Store all parameters needed to run and save a simulation.

        Args:
            bundles: List of `symdwi.Bundle` objects to simulate.
            shells: B-value shell configuration used to generate gradients
                (ignored if `loaded_gradients` is given).
            n_b0: Number of b=0 volumes to generate (ignored if
                `loaded_gradients` is given).
            scan: Scan parameters passed to `symdwi.simulate_dwi`.
            tissue: Default `TissueParameters` for bundles without an override.
            gm: Gray-matter compartment parameters, or None to disable.
            dims: Volume dimensions in voxels, as a 3-tuple.
            voxel_size: Isotropic voxel size in mm.
            snr: Target SNR, or None to disable noise.
            out_dir: Output directory to save the DWI volume, tractogram,
                and ground truth into.
            origin: World-space origin of the volume, or None for (0, 0, 0).
            tissue_masks: Optional dict of tissue-name to mask array.
            n_jobs: Number of parallel worker processes to use.
            return_groundtruth: Whether to compute and save ground truth.
            loaded_gradients: Optional (bvals, bvecs) tuple to use instead of
                generating gradients from `shells`/`n_b0`.
            big_delta_ms: Gray-matter diffusion time (big delta), in ms.
            small_delta_ms: Gray-matter gradient pulse duration (small delta),
                in ms.
        """
        super().__init__()
        self._bundles = bundles
        self._shells = shells
        self._n_b0 = n_b0
        self._scan = scan
        self._tissue = tissue
        self._gm = gm
        self._dims = dims
        self._voxel_size = voxel_size
        self._snr = snr
        self._out_dir = out_dir
        self._origin = origin
        self._tissue_masks = tissue_masks
        self._n_jobs = n_jobs
        self._return_groundtruth = return_groundtruth
        self._loaded_gradients = loaded_gradients
        self._big_delta_ms = big_delta_ms
        self._small_delta_ms = small_delta_ms

    def run(self):
        """Execute the simulation and save all outputs.

        Intended to run on `_sim_thread` (connected to `QThread.started`).
        Generates or reuses gradients, calls `symdwi.simulate_dwi`, then
        saves the DWI volume, tractogram, and (optionally) ground truth to
        `self._out_dir`, emitting `progress` updates along the way. Emits
        `finished` with the output directory on success, or `error` with the
        exception message on failure (exceptions are caught, not raised).
        """
        try:
            if self._loaded_gradients is not None:
                bvals, bvecs = self._loaded_gradients
            else:
                bvals, bvecs = symdwi.generate_bvals_bvecs(
                    shells=self._shells, n_b0=self._n_b0
                )
            result = symdwi.simulate_dwi(
                bundles=self._bundles,
                bvals=bvals,
                bvecs=bvecs,
                scan=self._scan,
                tissue=self._tissue,
                gm=self._gm,
                origin=self._origin,
                dims=self._dims,
                voxel_size=self._voxel_size,
                snr=self._snr,
                tissue_masks=self._tissue_masks,
                n_jobs=self._n_jobs,
                return_groundtruth=self._return_groundtruth,
                verbose=2,
                progress_cb=self.progress.emit,
            )
            if self._return_groundtruth:
                signal, affine, groundtruth = result
            else:
                signal, affine = result
                groundtruth = None

            self.progress.emit(0, 1, "Saving DWI volume")
            symdwi.save_dwi(
                signal, affine, bvals, bvecs, self._out_dir,
                big_delta_ms=self._big_delta_ms, small_delta_ms=self._small_delta_ms,
            )
            self.progress.emit(0, 1, "Saving tractogram")
            symdwi.save_bundles(self._bundles, os.path.join(self._out_dir, "tractogram.tck"))
            if groundtruth is not None:
                self.progress.emit(0, 1, "Saving ground truth")
                symdwi.save_groundtruth(groundtruth, affine, self._out_dir)

            self.finished.emit(self._out_dir)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    """Top-level SymDWI application window.

    Owns the scene state (`self.scene`) and coordinates the sidebar, the
    three linked `OrthoView` widgets (axial/coronal/sagittal), and the
    `Preview3D` widget. Handles bundle drawing/editing, loading baseline and
    tissue-mask NIfTI images, tissue/scan/gray-matter parameter edits, scene
    save/load, and dispatching simulation runs to a background `_SimWorker`
    thread so the UI remains responsive.
    """

    def __init__(self, parent=None):
        """Build the window chrome, scene state, and all child widgets.

        Args:
            parent: Optional parent widget, forwarded to `QMainWindow`.
        """
        super().__init__(parent)
        self.setWindowTitle("SymDWI")
        self.setMinimumSize(1280, 840)
        self.resize(1600, 1000)

        self.scene = Scene(
            volume=VolumeConfig(),
            bundles=[],
            active_points=[],
        )

        self._editing_index = None
        self._editing_color = None
        self._editing_tissue = None
        self._last_added_index = None

        self._ref_origin = None
        self._ref_dims = None
        self._ref_voxel_size = None
        self._tissue_masks = None
        self._baseline_data = None

        self._build_toolbar()
        self._build_layout()
        self._connect_sidebar()
        self._add_shortcuts()

        self._sync_volume_to_views()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_toolbar(self):
        """Create the top toolbar (New/Undo/Save/Load/Run buttons) and wire
        each button's `clicked` signal to its handler."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        self._new_btn = QPushButton("New Bundle")
        self._undo_btn = QPushButton("Undo Last Bundle")
        self._save_btn = QPushButton("Save Scene")
        self._load_btn = QPushButton("Load Scene")
        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.setProperty("variant", "primary")
        self._undo_btn.setEnabled(False)

        for btn in (self._new_btn, self._undo_btn):
            toolbar.addWidget(btn)
        toolbar.addSeparator()
        for btn in (self._save_btn, self._load_btn):
            toolbar.addWidget(btn)
        toolbar.addSeparator()
        toolbar.addWidget(self._run_btn)

        self._new_btn.clicked.connect(self.on_new_bundle)
        self._undo_btn.clicked.connect(self.on_undo_last_bundle)
        self._save_btn.clicked.connect(self.on_save_scene)
        self._load_btn.clicked.connect(self.on_load_scene)
        self._run_btn.clicked.connect(self.on_run)

    def _build_layout(self):
        """Build the central widget: a splitter holding the sidebar on the
        left and a 2x2 grid of the three ortho views plus the 3D preview on
        the right."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        root.addWidget(splitter)

        self.sidebar = Sidebar(scene=self.scene)
        splitter.addWidget(self.sidebar)

        viewport_container = QWidget()
        grid = QGridLayout(viewport_container)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)

        self.view_axial = OrthoView("axial")
        self.view_coronal = OrthoView("coronal")
        self.view_sagittal = OrthoView("sagittal")
        self.view3d = Preview3D()

        grid.addWidget(self.view_axial, 0, 0)
        grid.addWidget(self.view_coronal, 0, 1)
        grid.addWidget(self.view_sagittal, 1, 0)
        grid.addWidget(self.view3d, 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        splitter.addWidget(viewport_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1260])

    def _connect_sidebar(self):
        """Connect every sidebar panel signal (volume, points, bundle list,
        tissue, scan) and each ortho view's `point_picked` signal to their
        respective handlers on this window."""
        sp = self.sidebar
        sp.volume_panel.x_mm.valueChanged.connect(self._sync_volume_to_views)
        sp.volume_panel.y_mm.valueChanged.connect(self._sync_volume_to_views)
        sp.volume_panel.z_mm.valueChanged.connect(self._sync_volume_to_views)
        sp.volume_panel.voxel_size.valueChanged.connect(self._sync_volume_to_views)
        sp.volume_panel.load_requested.connect(self._on_volume_load_requested)

        self.view_axial.point_picked.connect(self._on_point_picked)
        self.view_coronal.point_picked.connect(self._on_point_picked)
        self.view_sagittal.point_picked.connect(self._on_point_picked)

        sp.bundle_geometry.preview_btn.clicked.connect(self.on_preview_bundle)
        sp.bundle_geometry.add_bundle_btn.clicked.connect(self.on_add_bundle)
        sp.bundle_geometry.points_edited.connect(self._on_points_edited)

        sp.bundle_list.load_tck_btn.clicked.connect(self.on_load_tck)
        sp.bundle_list.edit_requested.connect(self._on_edit_bundle)
        sp.bundle_list.visibility_changed.connect(self._on_bundle_visibility_changed)
        sp.bundle_list.bundle_deleted.connect(self._on_bundle_deleted)
        sp.bundle_list.selection_changed.connect(self._on_bundle_selection_changed)

        sp.tissue_panel.target_changed.connect(self._on_tissue_target_changed)
        sp.tissue_panel.override_toggled.connect(self._on_tissue_override_toggled)
        sp.tissue_panel.params_changed.connect(self._on_tissue_params_changed)

        sp.scan_panel.load_gradients_requested.connect(self.on_load_gradients)

    def _add_shortcuts(self):
        """Bind Backspace to delete the last placed control point."""
        sc = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        sc.activated.connect(self._delete_last_point)

    # ------------------------------------------------------------------
    # Volume / views
    # ------------------------------------------------------------------

    def _sync_volume_to_views(self):
        """Push the current volume geometry (extents, voxel size, origin) to
        the three ortho views and the 3D preview so they all render the same
        box. Connected to the volume spinboxes' `valueChanged` signals."""
        v = self.scene.volume
        origin = (tuple(float(o) for o in self._ref_origin)
                  if self._ref_origin is not None else (0.0, 0.0, 0.0))
        for view in (self.view_axial, self.view_coronal, self.view_sagittal):
            view.set_volume(v.x_mm, v.y_mm, v.z_mm, v.voxel_size, origin=origin)
        half = 0.5 * v.voxel_size
        corner = tuple(o - half for o in origin)
        self.view3d.set_volume(v.x_mm, v.y_mm, v.z_mm, origin=corner)

    def _on_point_picked(self, plane: str, u: float, v: float, slice_mm: float):
        """Place a full 3D control point from a single click in one view.

        Two coordinates come from the in-plane click; the third comes from the
        view's current slice. Drawing an entire bundle in one plane just means
        keeping that view on the same slice and clicking each point.

        Args:
            plane: Which ortho view the click came from ("axial", "coronal",
                or "sagittal").
            u: First in-plane click coordinate, in mm.
            v: Second in-plane click coordinate, in mm.
            slice_mm: The view's current out-of-plane slice position, in mm.
        """
        if plane == "axial":
            pt = [u, v, slice_mm]
        elif plane == "coronal":
            pt = [u, slice_mm, v]
        else:
            pt = [slice_mm, u, v]

        self.scene.active_points.append(pt)
        self._link_slices_to_point(pt)
        self._redraw_2d_active()
        self._sync_points_to_table()

    def _link_slices_to_point(self, pt):
        """Recenter every view's slice on the given point so it shows
        consistently across all three planes (linked crosshairs).

        Args:
            pt: 3-element sequence of (x, y, z) world coordinates in mm.
        """
        x, y, z = pt
        self.view_axial.set_slice_mm(z)
        self.view_coronal.set_slice_mm(y)
        self.view_sagittal.set_slice_mm(x)

    def _on_points_edited(self):
        """User edited the control-point table directly: it is the source of
        truth, so mirror it back into the scene and 2D overlays."""
        pts = self.sidebar.bundle_geometry._read_control_points(
            on_invalid_row=self._warn_invalid_row
        )
        self.scene.active_points = [list(p) for p in pts]
        self._redraw_2d_active()

    def _warn_invalid_row(self, row: int):
        """Show a status-bar message that a control-point table row was
        skipped for having non-numeric coordinates.

        Args:
            row: Zero-based index of the invalid row.
        """
        self.statusBar().showMessage(
            f"Row {row + 1} has invalid (non-numeric) coordinates and was skipped.",
            5000,
        )

    def _redraw_2d_active(self):
        """Push the in-progress `scene.active_points` to all three ortho
        views so the not-yet-confirmed bundle outline stays in sync."""
        pts = self.scene.active_points
        for view in (self.view_axial, self.view_coronal, self.view_sagittal):
            view.set_active_points(pts)

    def _redraw_2d_bundles(self):
        """Project every visible confirmed bundle's control-point line into the
        three ortho views, so crossing/combined configurations stay visible."""
        overlays = [
            (cfg.geometry.control_points, cfg.color)
            for cfg in self.scene.bundles
            if cfg.visible and cfg.geometry is not None and len(cfg.geometry.control_points)
        ]
        for view in (self.view_axial, self.view_coronal, self.view_sagittal):
            view.set_bundles(overlays)

    def _on_bundle_visibility_changed(self):
        """Refresh both the 2D overlays and the 3D scene after a bundle's
        visibility checkbox was toggled in the sidebar list."""
        self._redraw_2d_bundles()
        self._rebuild_3d_bundles()

    def _sync_points_to_table(self):
        """Mirror scene.active_points into the sidebar control-point table."""
        self.sidebar.bundle_geometry.sync_table(self.scene.active_points)

    def _delete_last_point(self):
        """Remove the most recently placed active control point, if any.
        Bound to the Backspace shortcut."""
        if self.scene.active_points:
            self.scene.active_points.pop()
            self._redraw_2d_active()
            self._sync_points_to_table()

    # ------------------------------------------------------------------
    # Bundle construction
    # ------------------------------------------------------------------

    def _current_geometry(self):
        """Build a BundleGeometry from the sidebar + active points, or None
        with a warning dialog if there aren't enough points yet.

        Returns:
            The constructed geometry, or None if fewer than 2 active points
            are placed (a warning dialog is shown in that case).
        """
        bp = self.sidebar.bundle_geometry
        pts = self.scene.active_points or []
        if len(pts) < 2:
            QMessageBox.warning(
                self, "Not enough points",
                "Place at least 2 control points first."
            )
            return None
        return bp.current_geometry(pts, on_invalid_row=self._warn_invalid_row)

    def _build_symdwi_bundle(self, cfg: BundleConfig) -> "symdwi.Bundle":
        """Build a symdwi.Bundle from a BundleConfig (tck-sourced or drawn).

        Args:
            cfg: Bundle configuration; either has a `tck_path` (loaded from a
                .tck file) or a `geometry` (drawn in the GUI).

        Returns:
            The constructed `symdwi.Bundle`, with the config's tissue
            override applied if `cfg.tissue_override` is set.
        """
        tissue = cfg.tissue if cfg.tissue_override else None
        if cfg.tck_path:
            return symdwi.Bundle.from_tck(
                cfg.tck_path, n_samples=cfg.geometry.n_samples if cfg.geometry else 100,
                tissue=tissue,
            )
        geometry = BundleGeometry(
            control_points=np.array(cfg.geometry.control_points, dtype=float),
            n_streamlines=cfg.geometry.n_streamlines,
            radius=cfg.geometry.radius,
            n_samples=cfg.geometry.n_samples,
            degree=cfg.geometry.degree,
            smoothing=cfg.geometry.smoothing,
            taper=(lambda u: 1 - u) if cfg.taper_linear else None,
            dispersion=cfg.geometry.dispersion,
            seed=cfg.geometry.seed,
        )
        return symdwi.Bundle(geometry, tissue=tissue)

    def on_preview_bundle(self):
        """Render a temporary cyan preview of the bundle currently being
        drawn, without adding it to the scene. Slot for the Preview button."""
        geometry = self._current_geometry()
        if geometry is None:
            return
        taper_linear = self.sidebar.bundle_geometry.taper.isChecked()
        geometry.taper = (lambda u: 1 - u) if taper_linear else None
        bundle = symdwi.Bundle(geometry)
        self.view3d.set_preview_bundle(bundle.as_array(), color="cyan")

    def on_add_bundle(self):
        """Confirm the bundle currently being drawn: build it, add or
        reinsert it into `scene.bundles`, and refresh the 3D/2D views and
        sidebar list. Slot for the Add Bundle button.

        If `_editing_index` is set, this reinserts the edited bundle at its
        original position (preserving its color and tissue override)
        instead of appending a new one.
        """
        geometry = self._current_geometry()
        if geometry is None:
            return

        bp = self.sidebar.bundle_geometry
        taper_linear = bp.taper.isChecked()

        editing = self._editing_index is not None
        color = self._editing_color if editing else (
            _BUNDLE_COLORS[len(self.scene.bundles) % len(_BUNDLE_COLORS)]
        )
        # Carry forward whatever override the bundle already had (set via the
        # Tissue tab while editing); a brand-new bundle starts with none and
        # gets one later by selecting it in the Tissue tab.
        tissue_override = self._editing_tissue if editing else None

        cfg = BundleConfig(
            geometry=geometry,
            color=color,
            tissue=tissue_override,
            tissue_override=tissue_override is not None,
            taper_linear=taper_linear,
        )

        if editing:
            self.scene.bundles.insert(self._editing_index, cfg)
            self._editing_index = None
            self._editing_color = None
        else:
            self.scene.bundles.append(cfg)

        self.view3d.clear_preview()
        try:
            bundle = self._build_symdwi_bundle(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Bundle error", str(exc))
            return

        if editing:
            self._rebuild_3d_bundles()
        else:
            self.view3d.add_bundle(bundle.as_array(), color=cfg.color)
            self._last_added_index = len(self.scene.bundles) - 1
            self._undo_btn.setEnabled(True)

        self.sidebar.bundle_list.refresh()
        self._refresh_tissue_targets()

        self.scene.active_points.clear()
        self._editing_tissue = None
        self._redraw_2d_active()
        self._redraw_2d_bundles()
        self._sync_points_to_table()

    def on_undo_last_bundle(self):
        """Remove the most recently added bundle without needing to select
        it in the list and click Delete. Slot for the Undo Last Bundle
        button; no-ops (and disables itself) if nothing is available to
        undo."""
        if self._last_added_index is None or self._last_added_index >= len(self.scene.bundles):
            self._undo_btn.setEnabled(False)
            return
        self.scene.bundles.pop(self._last_added_index)
        self.view3d.remove_last_bundle()
        self._last_added_index = None
        self._undo_btn.setEnabled(False)
        self.sidebar.bundle_list.refresh()
        self._refresh_tissue_targets()
        self._redraw_2d_bundles()

    def on_new_bundle(self):
        """Clear active points to start placing a new bundle."""
        self._editing_index = None
        self._editing_color = None
        self._editing_tissue = None
        self.scene.active_points.clear()
        self.view3d.clear_preview()
        self._redraw_2d_active()
        self._sync_points_to_table()
        self.sidebar.tabs.setCurrentWidget(self.sidebar.bundles_tab)
        self.sidebar.draw_section.set_expanded(True)

    def _refresh_tissue_targets(self):
        names = [f"Bundle {i + 1}" for i in range(len(self.scene.bundles))]
        self.sidebar.tissue_panel.set_bundle_names(names)

    # ------------------------------------------------------------------
    # Baseline / reference image / tissue mask loading (all from the Volume
    # panel: each of baseline/wm/gm/csf is loaded and cleared independently)
    # ------------------------------------------------------------------

    def _on_volume_load_requested(self, kind: str):
        if kind == "baseline":
            path, _ = QFileDialog.getOpenFileName(
                self, "Load baseline T1/T2 image", "",
                "NIfTI files (*.nii *.nii.gz)"
            )
            if path:
                self._load_baseline_from_path(path)
        elif kind == "clear_baseline":
            self._baseline_data = None
            self.scene.background_path = ""
            for view in (self.view_axial, self.view_coronal, self.view_sagittal):
                view.set_background(None)
            self.sidebar.volume_panel.set_status("baseline", "not loaded")
        elif kind in ("wm", "gm", "csf"):
            self._load_mask(kind)
        elif kind.startswith("clear_"):
            mask_kind = kind[len("clear_"):]
            if self._tissue_masks is not None:
                self._tissue_masks.pop(mask_kind, None)
                if not self._tissue_masks:
                    self._tissue_masks = None
            self.sidebar.volume_panel.set_status(mask_kind, "not loaded")

    def on_load_baseline(self):
        self._on_volume_load_requested("baseline")

    def _load_baseline_from_path(self, path: str):
        """Load a T1/T2 NIfTI as the drawing background and adopt its geometry."""
        try:
            img = nib.as_closest_canonical(nib.load(path))
            data = np.asarray(img.get_fdata(), dtype=float)
        except Exception as exc:
            QMessageBox.critical(self, "Baseline error", str(exc))
            return

        self._baseline_data = data
        self.scene.background_path = path

        self._set_geometry(
            img.affine[:3, 3].copy(),
            float(img.header.get_zooms()[0]),
            img.shape[:3],
        )
        self._sync_volume_to_views()

        for view in (self.view_axial, self.view_coronal, self.view_sagittal):
            view.set_background(data)

        self.sidebar.volume_panel.set_status(
            "baseline", f"{os.path.basename(path)} ({data.shape[0]}x{data.shape[1]}x{data.shape[2]})"
        )
        self.statusBar().showMessage(f"Baseline loaded: {os.path.basename(path)}", 5000)

    def _load_mask(self, kind: str):
        """Load a single WM/GM/CSF mask independently of the others. If no
        volume geometry has been set yet (no baseline/other mask loaded),
        this mask's own image geometry is adopted as the reference box."""
        label = {"wm": "WM", "gm": "GM", "csf": "CSF"}[kind]
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {label} mask", "", "NIfTI files (*.nii *.nii.gz)"
        )
        if not path:
            return
        try:
            img = nib.as_closest_canonical(nib.load(path))
            data = np.asarray(img.get_fdata(), dtype=float)
        except Exception as exc:
            QMessageBox.critical(self, f"{label} mask error", str(exc))
            return

        if self._ref_dims is None:
            self._set_geometry(
                img.affine[:3, 3].copy(), float(img.header.get_zooms()[0]), img.shape[:3]
            )
        elif data.shape != self._ref_dims:
            QMessageBox.warning(
                self, "Shape mismatch",
                f"{label} mask shape {data.shape} does not match the current "
                f"volume geometry {self._ref_dims}. Not loaded."
            )
            return

        self._tissue_masks = dict(self._tissue_masks or {})
        self._tissue_masks[kind] = data
        self.sidebar.volume_panel.set_status(
            kind, f"{os.path.basename(path)} ({data.shape[0]}x{data.shape[1]}x{data.shape[2]})"
        )
        if kind == "gm":
            self.sidebar.gm_panel.enabled.setChecked(True)
        self.statusBar().showMessage(f"{label} mask loaded: {os.path.basename(path)}", 5000)

    def on_load_tck(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load .tck bundles", "", "Tractogram files (*.tck)"
        )
        if not paths:
            return

        # .tck streamlines are stored in world (RASMM) coordinates. Without a
        # reference image telling the GUI what world-space box the volume
        # occupies, the streamlines will very likely fall outside the default
        # synthetic grid and silently voxelise to nothing.
        if self._ref_dims is None:
            load_ref = QMessageBox.question(
                self, "Reference image",
                "No reference geometry is set yet. Load a reference NIfTI to "
                "set the volume's world-space box?\n\n"
                "Without one, the volume keeps its current box (default: "
                "60mm cube at the origin). If these .tck streamlines were "
                "tracked in scanner space, they will very likely fall "
                "outside that box and be silently dropped when you run the "
                "simulation.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            ) == QMessageBox.Yes

            if load_ref:
                ref_path, _ = QFileDialog.getOpenFileName(
                    self, "Load reference NIfTI", "",
                    "NIfTI files (*.nii *.nii.gz)"
                )
                if ref_path:
                    self._load_baseline_from_path(ref_path)

        errors = []
        empty = []
        for path in paths:
            try:
                bundle = symdwi.Bundle.from_tck(path)
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")
                continue

            if not self._bundle_within_volume(bundle):
                empty.append(os.path.basename(path))

            cfg = BundleConfig(
                geometry=None,
                color=_BUNDLE_COLORS[len(self.scene.bundles) % len(_BUNDLE_COLORS)],
                tck_path=path,
            )
            self.scene.bundles.append(cfg)
            self.view3d.add_bundle(bundle.as_array(), color=cfg.color)

        self.sidebar.bundle_list.refresh()
        self._redraw_2d_bundles()

        if errors:
            QMessageBox.warning(
                self, "Some files failed",
                "Could not load:\n" + "\n".join(errors)
            )
        if empty:
            QMessageBox.warning(
                self, "Streamlines outside volume",
                "These bundles fall entirely outside the current volume box "
                "and will voxelise to nothing in the simulation:\n\n"
                + "\n".join(empty)
                + "\n\nLoad a matching reference image (or adjust the volume "
                "geometry) so the box covers these streamlines."
            )

    def _bundle_within_volume(self, bundle) -> bool:
        """Quick world-space bounds check: does any point of this bundle fall
        inside the current volume box? Mirrors the bounds test voxelise_bundles
        applies internally, so we can warn before running instead of after."""
        v = self.scene.volume
        origin = np.array(self._ref_origin if self._ref_origin is not None else (0.0, 0.0, 0.0))
        voxel_size = self._ref_voxel_size or v.voxel_size
        dims = self._ref_dims or (
            int(v.x_mm / v.voxel_size), int(v.y_mm / v.voxel_size), int(v.z_mm / v.voxel_size)
        )
        extent_hi = origin + np.array(dims) * voxel_size
        pts, _ = bundle.points_with_orientation()
        inside = np.all((pts >= origin - 0.5 * voxel_size) & (pts < extent_hi + 0.5 * voxel_size), axis=1)
        return bool(np.any(inside))

    def _set_geometry(self, origin, voxel_size: float, dims):
        """Set the volume geometry (origin/voxel/dims) from a reference image
        and reflect it in the volume spinboxes."""
        self._ref_origin = origin
        self._ref_voxel_size = float(voxel_size)
        self._ref_dims = tuple(int(d) for d in dims)

        v = self.scene.volume
        v.voxel_size = self._ref_voxel_size
        v.x_mm = self._ref_dims[0] * self._ref_voxel_size
        v.y_mm = self._ref_dims[1] * self._ref_voxel_size
        v.z_mm = self._ref_dims[2] * self._ref_voxel_size

        vp = self.sidebar.volume_panel
        for box, val in ((vp.voxel_size, v.voxel_size), (vp.x_mm, v.x_mm),
                         (vp.y_mm, v.y_mm), (vp.z_mm, v.z_mm)):
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        vp._update_grid()
        self._sync_volume_to_views()

    # ------------------------------------------------------------------
    # Bundle list interactions
    # ------------------------------------------------------------------

    def _on_bundle_selection_changed(self):
        """When exactly one bundle is selected in the list, point the Tissue
        tab's target selector at it so its override can be reviewed/edited."""
        idx = self.sidebar.bundle_list.selected_index()
        tp = self.sidebar.tissue_panel
        if idx is None or idx >= len(self.scene.bundles):
            tp.target.setCurrentIndex(0)
            return
        tp.target.setCurrentIndex(idx + 1)

    def _get_default_tissue(self):
        """The scene-wide default TissueParameters, picking up any live edit
        currently shown in the form if 'All bundles' is the active target."""
        tp = self.sidebar.tissue_panel
        if tp.current_target() is None:
            self.scene.tissue = tp.get_params()
        return self.scene.tissue

    def _on_tissue_params_changed(self):
        """Live-sync every keystroke/spin edit back to whichever target the
        form currently shows, so switching targets never loses an edit."""
        tp = self.sidebar.tissue_panel
        idx = tp.current_target()
        if idx is None:
            self.scene.tissue = tp.get_params()
        elif idx < len(self.scene.bundles):
            cfg = self.scene.bundles[idx]
            if cfg.tissue_override:
                cfg.tissue = tp.get_params()
                if self._editing_index == idx:
                    self._editing_tissue = cfg.tissue

    def _on_tissue_target_changed(self, bundle_idx: int):
        """bundle_idx is -1 for 'All bundles', else a bundle index. Load
        that target's current parameters into the form."""
        tp = self.sidebar.tissue_panel
        if bundle_idx < 0:
            tp.set_params(self.scene.tissue)
            return
        if bundle_idx >= len(self.scene.bundles):
            return
        cfg = self.scene.bundles[bundle_idx]
        tp.set_override_enabled(cfg.tissue_override)
        tp.set_params(cfg.tissue if cfg.tissue_override else self.scene.tissue)

    def _on_tissue_override_toggled(self, checked: bool):
        idx = self.sidebar.tissue_panel.current_target()
        if idx is None or idx >= len(self.scene.bundles):
            return
        cfg = self.scene.bundles[idx]
        cfg.tissue_override = checked
        cfg.tissue = self.sidebar.tissue_panel.get_params() if checked else None
        if self._editing_index == idx:
            self._editing_tissue = cfg.tissue
        self.sidebar.bundle_list.refresh()

    def _on_edit_bundle(self, idx: int):
        cfg = self.scene.bundles[idx]
        if cfg.tck_path:
            QMessageBox.information(
                self, "TCK bundle",
                "This bundle was loaded from a .tck file and cannot be edited."
            )
            return

        bp = self.sidebar.bundle_geometry
        bp.load_geometry(cfg.geometry, cfg.taper_linear)

        self.scene.active_points = [list(p) for p in cfg.geometry.control_points]
        self._redraw_2d_active()
        self._sync_points_to_table()

        self._editing_index = idx
        self._editing_color = cfg.color
        self._editing_tissue = cfg.tissue if cfg.tissue_override else None
        self.scene.bundles.pop(idx)
        self._rebuild_3d_bundles()
        self._redraw_2d_bundles()
        self.sidebar.bundle_list.refresh()
        self._refresh_tissue_targets()
        self.sidebar.draw_section.set_expanded(True)

    def _on_bundle_deleted(self, deleted_rows: list[int]):
        """One or more bundles were removed from the list: refresh the 3D/2D
        overlays and keep any pending edit's reinsertion index correct."""
        if self._editing_index is not None:
            self._editing_index -= sum(1 for r in deleted_rows if r < self._editing_index)
        if self._last_added_index is not None:
            if self._last_added_index in deleted_rows:
                self._last_added_index = None
                self._undo_btn.setEnabled(False)
            else:
                self._last_added_index -= sum(1 for r in deleted_rows if r < self._last_added_index)
        self._rebuild_3d_bundles()
        self._redraw_2d_bundles()
        self._refresh_tissue_targets()

    def _rebuild_3d_bundles(self):
        """Redraw all visible confirmed bundles from scratch."""
        self.view3d.clear_bundles()
        for cfg in self.scene.bundles:
            if not cfg.visible:
                continue
            try:
                bundle = self._build_symdwi_bundle(cfg)
                self.view3d.add_bundle(bundle.as_array(), color=cfg.color)
            except Exception as exc:
                self.statusBar().showMessage(f"Could not display bundle: {exc}", 4000)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def on_load_gradients(self):
        """Load bvals/bvecs from FSL-style text files, as an alternative to
        generating them from the shells table. bvecs are expected as unit
        direction vectors in world (scanner) space, matching the convention
        symdwi.generate_bvals_bvecs() itself produces — the same convention
        simulate_dwi() requires. FSL .bvec files exported from a real scan
        are in the image voxel frame instead and are NOT directly usable
        here without converting them to world space first."""
        bval_path, _ = QFileDialog.getOpenFileName(
            self, "Load bvals file", "", "Text files (*.txt *.bval);;All files (*)"
        )
        if not bval_path:
            return
        bvec_path, _ = QFileDialog.getOpenFileName(
            self, "Load bvecs file", os.path.dirname(bval_path),
            "Text files (*.txt *.bvec);;All files (*)"
        )
        if not bvec_path:
            return

        try:
            bvals = np.loadtxt(bval_path).reshape(-1)
            bvecs = np.loadtxt(bvec_path)
            if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
                bvecs = bvecs.T
            if bvecs.shape != (len(bvals), 3):
                raise ValueError(
                    f"bvecs shape {bvecs.shape} does not match {len(bvals)} bvals"
                )
        except Exception as exc:
            QMessageBox.critical(self, "Gradient file error", str(exc))
            return

        norms = np.linalg.norm(bvecs, axis=1)
        if np.any((norms > 1e-6) & (np.abs(norms - 1.0) > 1e-2)):
            QMessageBox.warning(
                self, "Unnormalized bvecs",
                "Some non-zero bvecs are not unit vectors. Continuing, but "
                "double-check these are world-space directions, not FSL "
                "voxel-frame vectors from a real scan."
            )

        self.sidebar.scan_panel.set_loaded_gradients(bval_path, bvec_path, bvals, bvecs)

    def on_run(self):
        if not self.scene.bundles:
            QMessageBox.warning(
                self, "No bundles",
                "Add at least one bundle before running the simulation."
            )
            return

        try:
            bundles = [self._build_symdwi_bundle(cfg) for cfg in self.scene.bundles]
        except Exception as exc:
            QMessageBox.critical(self, "Bundle error", str(exc))
            return

        v = self.scene.volume
        dims = self._ref_dims or (
            int(v.x_mm / v.voxel_size),
            int(v.y_mm / v.voxel_size),
            int(v.z_mm / v.voxel_size),
        )
        vs = self._ref_voxel_size or v.voxel_size
        origin = self._ref_origin

        empty = [
            f"Bundle {i + 1}" for i, b in enumerate(bundles)
            if not self._bundle_within_volume(b)
        ]
        if empty:
            proceed = QMessageBox.warning(
                self, "Streamlines outside volume",
                "These bundles fall entirely outside the current volume box "
                "and will contribute no signal:\n\n" + "\n".join(empty)
                + "\n\nRun anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if proceed != QMessageBox.Yes:
                return

        sp = self.sidebar
        run_cfg = sp.scan_panel.get_run_config()
        loaded_gradients = sp.scan_panel.loaded_gradients()
        if loaded_gradients is None:
            if not run_cfg["shells"]:
                QMessageBox.warning(self, "No shells", "Add at least one b-value shell.")
                return
            if run_cfg["snr"] is not None and run_cfg["n_b0"] < 1:
                QMessageBox.warning(
                    self, "SNR requires b=0",
                    "SNR is enabled but n_b0 is 0. Set n_b0 >= 1 or disable SNR."
                )
                return
        elif run_cfg["snr"] is not None and not np.any(loaded_gradients[0] <= 1e-6):
            QMessageBox.warning(
                self, "SNR requires b=0",
                "SNR is enabled but the loaded bvals contain no b=0 volume."
            )
            return

        scan = sp.scan_panel.get_scan()
        tissue = self._get_default_tissue()
        gm_panel = sp.gm_panel
        gm = gm_panel.get_gm() if gm_panel.is_enabled() else None

        if gm is not None and (self._tissue_masks is None or "gm" not in self._tissue_masks):
            proceed = QMessageBox.warning(
                self, "No GM mask loaded",
                "Gray matter is enabled but no 'gm' tissue mask is loaded, so "
                "it will have no effect. Continue anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if proceed != QMessageBox.Yes:
                return

        # Ask for the output directory up front rather than after the run:
        # saving now happens in the background worker together with the
        # simulation, so there's no natural pause afterward to prompt in,
        # and failing fast here avoids wasting a long run on a cancelled dialog.
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if not out_dir:
            return

        self._run_btn.setEnabled(False)
        self._progress = QProgressDialog("Running simulation...", None, 0, 100, self)
        self._progress.setWindowTitle("SymDWI")
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.setAutoReset(False)
        self._progress.setCancelButton(None)
        self._progress.setValue(0)
        self._progress.show()

        big_delta = gm_panel.big_delta_ms.value() if gm_panel.is_enabled() else None
        small_delta = gm_panel.small_delta_ms.value() if gm_panel.is_enabled() else None

        self._sim_thread = QThread()
        self._sim_worker = _SimWorker(
            bundles=bundles,
            shells=run_cfg["shells"],
            n_b0=run_cfg["n_b0"],
            scan=scan,
            tissue=tissue,
            gm=gm,
            dims=dims,
            voxel_size=vs,
            snr=run_cfg["snr"],
            out_dir=out_dir,
            origin=origin,
            tissue_masks=self._tissue_masks,
            n_jobs=run_cfg.get("n_jobs", os.cpu_count() or 1),
            return_groundtruth=run_cfg.get("export_gt", True),
            loaded_gradients=loaded_gradients,
            big_delta_ms=big_delta,
            small_delta_ms=small_delta,
        )
        self._sim_worker.moveToThread(self._sim_thread)
        self._sim_thread.started.connect(self._sim_worker.run)
        self._sim_worker.progress.connect(self._on_sim_progress)
        self._sim_worker.finished.connect(self._on_sim_done)
        self._sim_worker.error.connect(self._on_sim_error)
        self._sim_worker.finished.connect(self._sim_thread.quit)
        self._sim_worker.error.connect(self._sim_thread.quit)
        self._sim_thread.start()

    def _on_sim_progress(self, done: int, total: int, stage: str):
        # Runs on the main thread: this slot is the only place progress
        # from the worker thread is allowed to touch the dialog widget.
        pct = int(100 * done / total) if total > 0 else 0
        self._progress.setLabelText(
            f"{stage}... ({done}/{total})" if total > 1 else f"{stage}..."
        )
        self._progress.setValue(pct)

    def _on_sim_done(self, out_dir: str):
        self._run_btn.setEnabled(True)
        self._progress.close()
        QMessageBox.information(self, "Done", f"Outputs saved to:\n{out_dir}")

    def _on_sim_error(self, msg):
        self._run_btn.setEnabled(True)
        self._progress.close()
        QMessageBox.critical(self, "Simulation error", msg)

    # ------------------------------------------------------------------
    # Scene persistence
    # ------------------------------------------------------------------

    def on_save_scene(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Scene", "", "JSON files (*.json)"
        )
        if not path:
            return
        self.scene.scan = self.sidebar.scan_panel.get_scan()
        self.scene.tissue = self._get_default_tissue()
        self.scene.gm = self.sidebar.gm_panel.get_gm()
        self.scene.gm_enabled = self.sidebar.gm_panel.is_enabled()
        try:
            scene_io.save_scene(self.scene, path)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def on_load_scene(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Scene", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            loaded = scene_io.load_scene(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self.scene.volume = loaded.volume
        self.scene.bundles = loaded.bundles
        self.scene.active_points = loaded.active_points
        self.scene.background_path = loaded.background_path
        self.scene.scan = loaded.scan
        self.scene.tissue = loaded.tissue
        self.scene.gm = loaded.gm
        self.scene.gm_enabled = loaded.gm_enabled

        self._editing_index = None
        self._editing_color = None
        self._editing_tissue = None
        self._last_added_index = None
        self._undo_btn.setEnabled(False)
        self._ref_origin = None
        self._ref_dims = None
        self._ref_voxel_size = None
        self._tissue_masks = None

        sp = self.sidebar
        sp.volume_panel.x_mm.setValue(loaded.volume.x_mm)
        sp.volume_panel.y_mm.setValue(loaded.volume.y_mm)
        sp.volume_panel.z_mm.setValue(loaded.volume.z_mm)
        sp.volume_panel.voxel_size.setValue(loaded.volume.voxel_size)
        for kind in ("baseline", "wm", "gm", "csf"):
            sp.volume_panel.set_status(kind, "not loaded")
        sp.tissue_panel.target.setCurrentIndex(0)
        sp.tissue_panel.set_params(loaded.tissue)
        sp.gm_panel.enabled.setChecked(loaded.gm_enabled)
        self._apply_gm_params(loaded.gm)
        self._apply_scan_params(loaded.scan)
        self._refresh_tissue_targets()
        sp.bundle_list.refresh()

        self._baseline_data = None
        for view in (self.view_axial, self.view_coronal, self.view_sagittal):
            view.set_background(None)
        if loaded.background_path and os.path.exists(loaded.background_path):
            self._load_baseline_from_path(loaded.background_path)
        elif loaded.background_path:
            self.statusBar().showMessage(
                f"Baseline image not found: {loaded.background_path}", 5000
            )

        self._sync_volume_to_views()
        self._rebuild_3d_bundles()
        self._redraw_2d_active()
        self._redraw_2d_bundles()
        self._sync_points_to_table()

    def _apply_gm_params(self, gm):
        p = self.sidebar.gm_panel
        p.f_in.setValue(gm.f_in)
        p.f_ec.setValue(gm.f_ec)
        p.d_in.setValue(gm.d_in * 1e3)
        p.d_ec.setValue(gm.d_ec * 1e3)
        p.r_s.setValue(gm.r_s)
        p.d_is.setValue(gm.d_is * 1e3)
        p.big_delta_ms.setValue(gm.big_delta_ms)
        p.small_delta_ms.setValue(gm.small_delta_ms)

    def _apply_scan_params(self, scan):
        p = self.sidebar.scan_panel
        if scan.te_ms is None:
            p._te_none.setChecked(True)
        else:
            p._te_none.setChecked(False)
            p._te_spin.setValue(scan.te_ms)
        p.d_iso.setValue(scan.d_iso * 1e3)
        p.t2_csf_ms.setValue(scan.t2_csf_ms)
        p.background_csf.setValue(scan.background_csf)
