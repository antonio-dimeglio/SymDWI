import os
import numpy as np
import nibabel as nib
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QToolBar, QPushButton, QSplitter, QGridLayout,
 QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QKeySequence, QShortcut

from symdwi.gui.sidebar import Sidebar
from symdwi.gui.scene import Scene, VolumeConfig, BundleConfig
from symdwi.gui.views2d import OrthoView
from symdwi.gui.views3d import Preview3D
from symdwi.simulate import DWIParameters
import symdwi
import symdwi.gui.io as scene_io


_BUNDLE_COLORS = [
    "#e06c75", "#61afef", "#98c379", "#e5c07b",
    "#c678dd", "#56b6c2", "#d19a66", "#abb2bf",
]


class _SimWorker(QObject):
    finished = Signal(object, object, object, object, object)
    error = Signal(str)

    def __init__(self, bundles, shells, n_b0, params, dims, voxel_size, snr,
                 origin=None, tissue_masks=None, n_jobs=1, return_groundtruth=True):
        super().__init__()
        self._bundles = bundles
        self._shells = shells
        self._n_b0 = n_b0
        self._params = params
        self._dims = dims
        self._voxel_size = voxel_size
        self._snr = snr
        self._origin = origin
        self._tissue_masks = tissue_masks
        self._n_jobs = n_jobs
        self._return_groundtruth = return_groundtruth

    def run(self):
        try:
            bvals, bvecs = symdwi.generate_bvals_bvecs(
                shells=self._shells, n_b0=self._n_b0
            )
            result = symdwi.simulate_dwi(
                bundles=self._bundles,
                bvals=bvals,
                bvecs=bvecs,
                params=self._params,
                origin=self._origin,
                dims=self._dims,
                voxel_size=self._voxel_size,
                snr=self._snr,
                tissue_masks=self._tissue_masks,
                n_jobs=self._n_jobs,
                return_groundtruth=self._return_groundtruth,
                verbose=2,
            )
            if self._return_groundtruth:
                signal, affine, groundtruth = result
            else:
                signal, affine = result
                groundtruth = None
            self.finished.emit(signal, affine, bvals, bvecs, groundtruth)
        except Exception as exc:
            self.error.emit(str(exc))

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SymDWI")
        self.setMinimumSize(1200, 800)
        self.resize(1500, 950)

        self.scene = Scene(
            volume=VolumeConfig(),
            bundles=[],
            active_points=[],
            dwi_params=DWIParameters(),
        )

        self._sim_result = None
        self._sim_groundtruth = None
        self._sim_bundles = None
        
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

    def _build_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        self._new_btn = QPushButton("New Bundle")
        self._baseline_btn = QPushButton("Load Baseline")
        self._save_btn = QPushButton("Save Scene")
        self._load_btn = QPushButton("Load Scene")
        self._run_btn = QPushButton("Run")

        for btn in (self._new_btn, self._baseline_btn, self._save_btn, self._load_btn, self._run_btn):
            toolbar.addWidget(btn)

        self._new_btn.clicked.connect(self.on_new_bundle)
        self._baseline_btn.clicked.connect(self.on_load_baseline)
        self._save_btn.clicked.connect(self.on_save_scene)
        self._load_btn.clicked.connect(self.on_load_scene)
        self._run_btn.clicked.connect(self.on_run)

    def _build_layout(self):
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
        splitter.setSizes([300, 1200])

    def _connect_sidebar(self):
        sp = self.sidebar
        sp.volume_panel.x_mm.valueChanged.connect(self._sync_volume_to_views)
        sp.volume_panel.y_mm.valueChanged.connect(self._sync_volume_to_views)
        sp.volume_panel.z_mm.valueChanged.connect(self._sync_volume_to_views)
        sp.volume_panel.voxel_size.valueChanged.connect(self._sync_volume_to_views)

        self.view_axial.point_picked.connect(self._on_point_picked)
        self.view_coronal.point_picked.connect(self._on_point_picked)
        self.view_sagittal.point_picked.connect(self._on_point_picked)

        sp.bundle_params.preview_btn.clicked.connect(self.on_preview_bundle)
        sp.bundle_params.add_bundle_btn.clicked.connect(self.on_add_bundle)
        sp.bundle_params.points_edited.connect(self._on_points_edited)


        sp.bundle_list.load_tck_btn.clicked.connect(self.on_load_tck)
        sp.bundle_list.edit_btn.clicked.connect(self._on_edit_bundle)
        sp.bundle_list.delete_btn.clicked.connect(self._on_delete_bundle)
        sp.bundle_list.visibility_changed.connect(self._on_bundle_visibility_changed)


    def _add_shortcuts(self):

        sc = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        sc.activated.connect(self._delete_last_point)


    def _sync_volume_to_views(self):
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
        consistently across all three planes (linked crosshairs)."""
        x, y, z = pt
        self.view_axial.set_slice_mm(z)
        self.view_coronal.set_slice_mm(y)
        self.view_sagittal.set_slice_mm(x)

    def _on_points_edited(self):
        """User edited the control-point table directly: it is the source of
        truth, so mirror it back into the scene and 2D overlays."""
        pts = self.sidebar.bundle_params._read_control_points()
        self.scene.active_points = [list(p) for p in pts]
        self._redraw_2d_active()

    def _redraw_2d_active(self):
        pts = self.scene.active_points
        for view in (self.view_axial, self.view_coronal, self.view_sagittal):
            view.set_active_points(pts)

    def _redraw_2d_bundles(self):
        """Project every visible confirmed bundle's control-point line into the
        three ortho views, so crossing/combined configurations stay visible."""
        overlays = [
            (cfg.control_points, cfg.color)
            for cfg in self.scene.bundles
            if cfg.visible and cfg.control_points
        ]
        for view in (self.view_axial, self.view_coronal, self.view_sagittal):
            view.set_bundles(overlays)

    def _on_bundle_visibility_changed(self):
        self._redraw_2d_bundles()
        self._rebuild_3d_bundles()

    def _sync_points_to_table(self):
        """Mirror scene.active_points into the sidebar control-point table."""
        from PySide6.QtWidgets import QTableWidgetItem
        table = self.sidebar.bundle_params.table
        table.blockSignals(True)
        table.setRowCount(0)
        for pt in self.scene.active_points:
            row = table.rowCount()
            table.insertRow(row)
            for col, val in enumerate(pt):
                table.setItem(row, col, QTableWidgetItem(f"{val:.2f}"))
        table.blockSignals(False)

    def _delete_last_point(self):
        if self.scene.active_points:
            self.scene.active_points.pop()
            self._redraw_2d_active()
            self._sync_points_to_table()

    def _build_bundle_from_scene(self) -> "symdwi.Bundle | None":
        cfg = self.sidebar.bundle_params.current_config()
        pts = self.scene.active_points or cfg.control_points
        if len(pts) < 2:
            QMessageBox.warning(
                self, "Not enough points",
                "Place at least 2 control points before previewing."
            )
            return None
        return symdwi.Bundle(
            control_points=np.array(pts, dtype=float),
            n_streamlines=cfg.n_streamlines,
            radius=cfg.radius,
            dispersion=cfg.dispersion,
            taper=(lambda u: 1 - u) if cfg.taper else None,
        )

    def on_preview_bundle(self):
        bundle = self._build_bundle_from_scene()
        if bundle is None:
            return
        sl = bundle.as_array()
        self.view3d.set_preview_bundle(sl, color="cyan")

    def on_add_bundle(self):
        cfg = self.sidebar.bundle_params.current_config()
        pts = self.scene.active_points or cfg.control_points
        if len(pts) < 2:
            QMessageBox.warning(
                self, "Not enough points",
                "Place at least 2 control points before adding a bundle."
            )
            return

        cfg = BundleConfig(
            control_points=[list(p) for p in pts],
            radius=cfg.radius,
            n_streamlines=cfg.n_streamlines,
            dispersion=cfg.dispersion,
            taper=cfg.taper,
            color=_BUNDLE_COLORS[len(self.scene.bundles) % len(_BUNDLE_COLORS)],
        )
        self.scene.bundles.append(cfg)

        bundle = symdwi.Bundle(
            control_points=np.array(cfg.control_points, dtype=float),
            n_streamlines=cfg.n_streamlines,
            radius=cfg.radius,
            dispersion=cfg.dispersion,
            taper=(lambda u: 1 - u) if cfg.taper else None,
        )
        self.view3d.clear_preview()
        self.view3d.add_bundle(bundle.as_array(), color=cfg.color)

        self.sidebar.bundle_list.refresh()

        self.scene.active_points.clear()
        self._redraw_2d_active()
        self._redraw_2d_bundles()
        self._sync_points_to_table()

    def on_new_bundle(self):
        """Clear active points to start placing a new bundle."""
        self.scene.active_points.clear()
        self.view3d.clear_preview()
        self._redraw_2d_active()
        self._sync_points_to_table()

    def on_load_baseline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load baseline T1/T2 image", "",
            "NIfTI files (*.nii *.nii.gz)"
        )
        if not path:
            return
        self._load_baseline_from_path(path)

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

        self.statusBar().showMessage(
            f"Baseline loaded: {os.path.basename(path)}  "
            f"({data.shape[0]}×{data.shape[1]}×{data.shape[2]})",
            5000,
        )


    def _build_symdwi_bundle(self, cfg: BundleConfig) -> "symdwi.Bundle":
        """Build a symdwi.Bundle from a BundleConfig (tck or drawn)."""
        if cfg.tck_path:
            return symdwi.Bundle.from_tck(cfg.tck_path)
        return symdwi.Bundle(
            control_points=np.array(cfg.control_points, dtype=float),
            n_streamlines=cfg.n_streamlines,
            radius=cfg.radius,
            dispersion=cfg.dispersion,
            taper=(lambda u: 1 - u) if cfg.taper else None,
        )

    def on_load_tck(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load .tck bundles", "", "Tractogram files (*.tck)"
        )
        if not paths:
            return

        load_ref = QMessageBox.question(
            self, "Reference image",
            "Load a reference NIfTI to set volume geometry and tissue masks?",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes

        if load_ref:
            ref_path, _ = QFileDialog.getOpenFileName(
                self, "Load reference NIfTI", "",
                "NIfTI files (*.nii *.nii.gz)"
            )
            if ref_path:
                self._load_reference_nifti(ref_path)

        errors = []
        for path in paths:
            try:
                cfg = BundleConfig(
                    control_points=[],
                    color=_BUNDLE_COLORS[len(self.scene.bundles) % len(_BUNDLE_COLORS)],
                    tck_path=path,
                )
                bundle = symdwi.Bundle.from_tck(path)
                self.scene.bundles.append(cfg)
                self.view3d.add_bundle(bundle.as_array(), color=cfg.color)
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        self.sidebar.bundle_list.refresh()
        self._redraw_2d_bundles()

        if errors:
            QMessageBox.warning(
                self, "Some files failed",
                "Could not load:\n" + "\n".join(errors)
            )

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

    def _load_reference_nifti(self, path: str):
        """Load affine/dims/voxel_size from a NIfTI and optionally tissue masks."""
        img = nib.load(path)
        self._set_geometry(
            img.affine[:3, 3].copy(),
            float(img.header.get_zooms()[0]),
            img.shape[:3],
        )

        load_masks = QMessageBox.question(
            self, "Tissue masks",
            "Load WM, GM, and CSF masks?",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes

        if not load_masks:
            self._tissue_masks = None
            return

        masks = {}
        for key, label in [("wm", "WM"), ("gm", "GM"), ("csf", "CSF")]:
            p, _ = QFileDialog.getOpenFileName(
                self, f"Load {label} mask", os.path.dirname(path),
                "NIfTI files (*.nii *.nii.gz)"
            )
            if p:
                masks[key] = nib.load(p).get_fdata()

        self._tissue_masks = masks if masks else None
        self.statusBar().showMessage(
            f"Reference loaded: {os.path.basename(path)}"
            + (f"  |  masks: {', '.join(masks)}" if masks else ""),
            5000
        )

    def _on_edit_bundle(self):
        idx = self.sidebar.bundle_list.selected_index()
        if idx is None:
            return
        cfg = self.scene.bundles[idx]
        if cfg.tck_path:
            QMessageBox.information(
                self, "TCK bundle",
                "This bundle was loaded from a .tck file and cannot be edited."
            )
            return
        bp = self.sidebar.bundle_params
        bp.radius.setValue(cfg.radius)
        bp.n_streamlines.setValue(cfg.n_streamlines)
        bp.dispersion.setValue(cfg.dispersion)
        bp.taper.setChecked(cfg.taper)

        self.scene.active_points = [list(p) for p in cfg.control_points]
        self._redraw_2d_active()
        self._sync_points_to_table()

        self.scene.bundles.pop(idx)
        self._rebuild_3d_bundles()
        self._redraw_2d_bundles()
        self.sidebar.bundle_list.refresh()

    def _on_delete_bundle(self):
        self._rebuild_3d_bundles()
        self._redraw_2d_bundles()

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

    def on_run(self):
        if not self.scene.bundles:
            QMessageBox.warning(
                self, "No bundles",
                "Add at least one bundle before running the simulation."
            )
            return

        dwi_cfg = self.sidebar.dwi_panel.get_params()

        try:
            bundles = [self._build_symdwi_bundle(cfg) for cfg in self.scene.bundles]
        except Exception as exc:
            QMessageBox.critical(self, "Bundle error", str(exc))
            return

        self._sim_bundles = bundles

        shells = [(s["b_value"], s["n_directions"]) for s in dwi_cfg["shells"]]

        params = DWIParameters(
            f_intra=dwi_cfg["f_intra"],
            f_extra=1.0 - dwi_cfg["f_intra"],
            axon_radius_um=dwi_cfg["axon_radius_um"],
            te_ms=dwi_cfg["te"],
        )

        v = self.scene.volume
        dims = self._ref_dims or  \
            (int(v.x_mm / v.voxel_size),
            int(v.y_mm / v.voxel_size),
            int(v.z_mm / v.voxel_size))
        vs = self._ref_voxel_size or v.voxel_size
        origin = self._ref_origin

        self._run_btn.setEnabled(False)
        self.statusBar().showMessage("Running simulation…")

        self._sim_thread = QThread()
        self._sim_worker = _SimWorker(
            bundles=bundles,
            shells=shells,
            n_b0=dwi_cfg["n_b0"],
            params=params,
            dims=dims,
            voxel_size=vs,
            snr=dwi_cfg["snr"],
            origin=origin,
            tissue_masks=self._tissue_masks,
            n_jobs=dwi_cfg.get("n_jobs", os.cpu_count() or 1),
            return_groundtruth=dwi_cfg.get("export_gt", True),
        )
        self._sim_worker.moveToThread(self._sim_thread)
        self._sim_thread.started.connect(self._sim_worker.run)
        self._sim_worker.finished.connect(self._on_sim_done)
        self._sim_worker.error.connect(self._on_sim_error)
        self._sim_worker.finished.connect(self._sim_thread.quit)
        self._sim_worker.error.connect(self._sim_thread.quit)
        self._sim_thread.start()

    def _on_sim_done(self, signal, affine, bvals, bvecs, groundtruth):
        self._run_btn.setEnabled(True)
        self.statusBar().clearMessage()
        self._sim_result = (signal, affine, bvals, bvecs)
        self._sim_groundtruth = groundtruth

        out_dir = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if not out_dir:
            return

        try:
            symdwi.save_dwi(signal, affine, bvals, bvecs, out_dir)
            bundles = self._sim_bundles or [
                self._build_symdwi_bundle(cfg) for cfg in self.scene.bundles
            ]
            symdwi.save_bundles(bundles, os.path.join(out_dir, "tractogram.tck"))
            if groundtruth is not None:
                symdwi.save_groundtruth(groundtruth, affine, out_dir)
            QMessageBox.information(
                self, "Done", f"Outputs saved to:\n{out_dir}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _on_sim_error(self, msg):
        self._run_btn.setEnabled(True)
        self.statusBar().clearMessage()
        QMessageBox.critical(self, "Simulation error", msg)

    def on_save_scene(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Scene", "", "JSON files (*.json)"
        )
        if not path:
            return
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

        self.sidebar.volume_panel.x_mm.setValue(loaded.volume.x_mm)
        self.sidebar.volume_panel.y_mm.setValue(loaded.volume.y_mm)
        self.sidebar.volume_panel.z_mm.setValue(loaded.volume.z_mm)
        self.sidebar.volume_panel.voxel_size.setValue(loaded.volume.voxel_size)
        self.sidebar.bundle_list.refresh()

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