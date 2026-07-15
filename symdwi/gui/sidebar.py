"""Sidebar panels for the SymDWI scene-building GUI.

This module implements the tabbed sidebar used to configure a diffusion-MRI
simulation scene: volume geometry and reference/mask images (``VolumePanel``),
the bundle list and per-bundle authoring controls (``BundleListPanel``,
``BundleGeometryPanel``), tissue microstructure parameters with per-bundle
overrides (``TissueParamsPanel``), acquisition/scan parameters
(``ScanParamsPanel``), gray-matter (SANDI) compartment parameters
(``GMParamsPanel``), and the top-level container that assembles all of the
above into tabs (``Sidebar``). Small module-private helpers build commonly
styled widgets (headings, hints, separators, spinboxes) shared across panels.
"""

import os
from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QFormLayout,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QAbstractItemView,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QWidget,
    QTabWidget,
    QFrame,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from symdwi.bundle import BundleGeometry, TissueParameters
from symdwi.simulate import ScanParameters, GMParameters
from symdwi.gui.scene import VolumeConfig, Scene, BundleConfig
from symdwi.gui.widgets import Collapsible, SegmentedToggle


def _spinbox(min_, max_, default, step=1.0, decimals=None):
    """Build a ``QDoubleSpinBox`` with the given range, initial value and step.

    Args:
        min_: Minimum allowed value.
        max_: Maximum allowed value.
        default: Initial value.
        step: Single-step increment used by the spinbox arrows.
        decimals: Number of decimal places to display, or None to keep the
            widget default.

    Returns:
        The configured ``QDoubleSpinBox``.
    """
    box = QDoubleSpinBox()
    box.setRange(min_, max_)
    box.setValue(default)
    box.setSingleStep(step)
    if decimals is not None:
        box.setDecimals(decimals)
    return box


def _heading(text: str) -> QLabel:
    """Build a ``QLabel`` styled as a section heading via the "heading" role property.

    Args:
        text: Heading text.

    Returns:
        The styled ``QLabel``.
    """
    lbl = QLabel(text)
    lbl.setProperty("role", "heading")
    return lbl


def _hint(text: str) -> QLabel:
    """Build a word-wrapped ``QLabel`` styled as secondary hint text.

    Args:
        text: Hint text.

    Returns:
        The styled ``QLabel``.
    """
    lbl = QLabel(text)
    lbl.setProperty("role", "hint")
    lbl.setWordWrap(True)
    return lbl


def _hline() -> QFrame:
    """Build a styled horizontal divider line.

    Returns:
        A ``QFrame`` configured as a horizontal rule.
    """
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #3a3f4b;")
    return line


class VolumePanel(QGroupBox):
    """Panel for configuring volume/voxel geometry and reference/mask images.

    Lets the user set the physical volume size and voxel size (with a live
    computed-grid readout), and trigger loading of the baseline reference
    image plus optional WM/GM/CSF tissue masks.

    Signals:
        load_requested(str): Emitted with a "kind" string identifying which
            file to load: "baseline", "wm", "gm", "csf", or, for the
            clearable rows, "clear_<kind>" to clear a previously loaded file.
            The file dialog itself and the resulting state update are handled
            by the owning window; this panel only requests the action.
    """

    # Emitted with (kind, path) where kind is "baseline", "wm", "gm", or "csf".
    load_requested = Signal(str)

    def __init__(self, config: VolumeConfig, parent=None):
        """Build the volume-size, voxel-size and reference/mask controls.

        Args:
            config: The scene's ``VolumeConfig``, edited in place as the user
                changes size/voxel-size fields.
            parent: Optional parent widget.
        """
        super().__init__("Volume", parent)
        self.config = config

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Volume size (mm)"))
        size_row = QHBoxLayout()
        self.x_mm = _spinbox(10, 500, config.x_mm)
        self.y_mm = _spinbox(10, 500, config.y_mm)
        self.z_mm = _spinbox(10, 500, config.z_mm)
        for label, box in [("X", self.x_mm), ("Y", self.y_mm), ("Z", self.z_mm)]:
            size_row.addWidget(QLabel(label))
            size_row.addWidget(box)
        layout.addLayout(size_row)

        layout.addWidget(QLabel("Voxel size (mm)"))
        self.voxel_size = _spinbox(0.1, 5.0, config.voxel_size, step=0.1, decimals=2)
        layout.addWidget(self.voxel_size)

        self.grid_label = QLabel()
        self.grid_label.setProperty("role", "hint")
        layout.addWidget(self.grid_label)
        self._update_grid()

        layout.addWidget(_hline())
        layout.addWidget(QLabel("Reference / baseline"))

        self.baseline_row = self._file_row(
            "Baseline (T1/T2)", "baseline", clearable=True
        )
        layout.addLayout(self.baseline_row)

        layout.addWidget(_hint(
            "Tissue masks (optional, any subset). Loading a mask adopts its "
            "image geometry as the volume's world-space box if none is set yet."
        ))
        self.wm_row = self._file_row("WM mask", "wm", clearable=True)
        self.gm_row = self._file_row("GM mask", "gm", clearable=True)
        self.csf_row = self._file_row("CSF mask", "csf", clearable=True)
        layout.addLayout(self.wm_row)
        layout.addLayout(self.gm_row)
        layout.addLayout(self.csf_row)

        self.x_mm.valueChanged.connect(self._on_change)
        self.y_mm.valueChanged.connect(self._on_change)
        self.z_mm.valueChanged.connect(self._on_change)
        self.voxel_size.valueChanged.connect(self._on_change)

    def _file_row(self, label: str, kind: str, clearable: bool = False) -> QHBoxLayout:
        """Build a labeled row with a status text, a "Load..." button, and
        optionally a clear ("x") button.

        Stores the status ``QLabel`` on ``self`` as ``_{kind}_status`` so
        :meth:`set_status` can update it later. Both buttons emit
        ``load_requested`` with ``kind`` (or ``clear_{kind}`` for the clear
        button) rather than performing any I/O themselves.

        Args:
            label: Row label text.
            kind: Identifier used in the emitted ``load_requested`` signal
                and in the stored status-label attribute name.
            clearable: If True, also add a small "x" button to clear the file.

        Returns:
            The assembled row layout.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        status = QLabel("not loaded")
        status.setProperty("role", "hint")
        row.addWidget(status, stretch=1)
        load_btn = QPushButton("Load...")
        load_btn.setProperty("variant", "flat")
        load_btn.clicked.connect(lambda: self.load_requested.emit(kind))
        row.addWidget(load_btn)
        setattr(self, f"_{kind}_status", status)
        if clearable:
            clear_btn = QPushButton("x")
            clear_btn.setProperty("variant", "flat")
            clear_btn.setFixedWidth(22)
            clear_btn.clicked.connect(lambda: self.load_requested.emit(f"clear_{kind}"))
            row.addWidget(clear_btn)
        return row

    def set_status(self, kind: str, text: str):
        """Update the status text for a given file row, if it exists.

        Args:
            kind: Row identifier ("baseline", "wm", "gm", or "csf").
            text: New status text to display.
        """
        lbl = getattr(self, f"_{kind}_status", None)
        if lbl is not None:
            lbl.setText(text)

    def _on_change(self):
        """Sync spinbox values back into ``self.config`` and refresh the grid readout."""
        self.config.x_mm = self.x_mm.value()
        self.config.y_mm = self.y_mm.value()
        self.config.z_mm = self.z_mm.value()
        self.config.voxel_size = self.voxel_size.value()
        self._update_grid()

    def _update_grid(self):
        """Recompute and display the voxel-grid dimensions from the current size/voxel-size values."""
        v = self.voxel_size.value()
        gx = int(self.x_mm.value() / v)
        gy = int(self.y_mm.value() / v)
        gz = int(self.z_mm.value() / v)
        self.grid_label.setText(f"Grid: {gx} x {gy} x {gz} voxels")


class BundleListPanel(QGroupBox):
    """Bundle list with a single selection model: the checkbox on each row is
    purely visibility (show/hide in the 3D and 2D views); which row(s) are
    selected (via normal list click/ctrl/shift-click) drives Edit and Delete.
    Edit requires exactly one selected bundle; Delete accepts one or many.

    Signals:
        visibility_changed(): Emitted when a row's checkbox is toggled.
        bundle_deleted(list): Emitted after one or more bundles are removed
            from the scene, with the sorted list of deleted indices.
        selection_changed(): Emitted whenever the set of selected rows changes.
        edit_requested(int): Emitted with the bundle index when "Edit" is
            clicked with exactly one row selected.
    """

    visibility_changed = Signal()
    bundle_deleted = Signal(list)  # sorted list of deleted indices
    selection_changed = Signal()
    edit_requested = Signal(int)

    def __init__(self, scene: Scene, parent=None):
        """Build the bundle list widget and its Load/Edit/Delete controls.

        Args:
            scene: The ``Scene`` whose ``bundles`` list this panel displays
                and mutates (on delete).
            parent: Optional parent widget.
        """
        super().__init__("Bundles", parent)
        self.scene = scene

        layout = QVBoxLayout(self)
        layout.addWidget(_hint(
            "Check to show/hide a bundle. Select one row to edit, or "
            "multiple to delete together."
        ))

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.load_tck_btn = QPushButton("Load .tck...")
        self.edit_btn = QPushButton("Edit")
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setProperty("variant", "danger")
        btn_row.addWidget(self.load_tck_btn)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.delete_btn)
        layout.addLayout(btn_row)

        self.edit_btn.clicked.connect(self._on_edit_clicked)
        self.delete_btn.clicked.connect(self.on_delete)

        self.refresh()
        self._on_selection_changed()

    def refresh(self):
        """Rebuild the list from scene.bundles."""
        self.list_widget.blockSignals(True)
        selected = set(self.selected_indices())
        self.list_widget.clear()
        for i, cfg in enumerate(self.scene.bundles):
            n_pts = len(cfg.geometry.control_points) if cfg.geometry is not None else 0
            src = os.path.basename(cfg.tck_path) if cfg.tck_path else f"{n_pts} pts"
            override = "  [tissue override]" if cfg.tissue_override else ""
            label = f"Bundle {i + 1}  ({src}){override}"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if cfg.visible else Qt.Unchecked)
            item.setSelected(i in selected)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        self._on_selection_changed()

    def _on_item_changed(self, item):
        """Mirror a checkbox toggle back into the bundle's visibility flag.

        Args:
            item: The ``QListWidgetItem`` whose check state changed.
        """
        row = self.list_widget.row(item)
        if 0 <= row < len(self.scene.bundles):
            self.scene.bundles[row].visible = (item.checkState() == Qt.Checked)
            self.visibility_changed.emit()

    def _on_selection_changed(self):
        """Update Edit/Delete button enabled state to match the current selection and emit ``selection_changed``."""
        n = len(self.selected_indices())
        self.edit_btn.setEnabled(n == 1)
        self.delete_btn.setEnabled(n >= 1)
        self.selection_changed.emit()

    def selected_indices(self) -> list[int]:
        """Return the sorted row indices currently selected in the list.

        Returns:
            Sorted list of selected row indices (possibly empty).
        """
        return sorted(self.list_widget.row(item) for item in self.list_widget.selectedItems())

    def selected_index(self):
        """Single selected index, or None if zero/multiple rows are selected."""
        idx = self.selected_indices()
        return idx[0] if len(idx) == 1 else None

    def _on_edit_clicked(self):
        """Handle the "Edit" button: warn if selection isn't exactly one bundle, else emit ``edit_requested``."""
        idx = self.selected_indices()
        if len(idx) != 1:
            QMessageBox.warning(
                self, "Select one bundle",
                "Select exactly one bundle to edit."
            )
            return
        self.edit_requested.emit(idx[0])

    def on_delete(self):
        """Remove all currently selected bundles from the scene and emit ``bundle_deleted``.

        Removes indices in reverse order so earlier indices remain valid
        while later ones are popped.
        """
        idx = self.selected_indices()
        if not idx:
            return
        for i in reversed(idx):
            self.scene.bundles.pop(i)
        self.refresh()
        self.bundle_deleted.emit(idx)


class BundleGeometryPanel(QWidget):
    """The bundle-authoring controls (shape parameters + control points).
    Undecorated: it's nested inside a Collapsible titled "Draw Bundle", so
    its own container adds no title/border of its own.

    Signals:
        points_edited(): Emitted whenever the control-points table is
            modified (row add/remove or cell edit).
    """

    points_edited = Signal()

    def __init__(self, parent=None):
        """Build the shape-parameter form and the control-points table/editor.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)

        layout.addWidget(_hint(
            "Click in any 2D view to place control points; the streamline "
            "path follows them. Backspace removes the last point."
        ))

        form = QFormLayout()

        self.radius = _spinbox(0.1, 20.0, 3.0, step=0.5)
        form.addRow("Radius (mm)", self.radius)

        self.n_streamlines = QSpinBox()
        self.n_streamlines.setRange(1, 100000)
        self.n_streamlines.setValue(200)
        form.addRow("Streamlines", self.n_streamlines)

        self.n_samples = QSpinBox()
        self.n_samples.setRange(4, 2000)
        self.n_samples.setValue(128)
        form.addRow("Samples/streamline", self.n_samples)

        self.dispersion = _spinbox(0.0, 10.0, 0.5, step=0.1)
        form.addRow("Dispersion (mm)", self.dispersion)

        self.smoothing = _spinbox(0.0, 10.0, 0.0, step=0.1)
        form.addRow("Smoothing", self.smoothing)

        self.taper = QCheckBox()
        self.taper.setToolTip("Linearly taper streamline offsets to zero at the bundle end.")
        form.addRow("Taper", self.taper)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setSpecialValueText("random")
        form.addRow("Seed", self.seed)

        layout.addLayout(form)

        # Kept right under the geometry form (not after the points table
        # below) so it's reachable without scrolling once points are placed.
        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Bundle")
        self.add_bundle_btn = QPushButton("Add Bundle")
        self.add_bundle_btn.setProperty("variant", "primary")
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.add_bundle_btn)
        layout.addLayout(btn_row)

        layout.addWidget(_hline())

        points_body = QWidget()
        points_layout = QVBoxLayout(points_body)
        points_layout.setContentsMargins(0, 4, 0, 0)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setFixedHeight(150)
        self.table.setAlternatingRowColors(True)
        points_layout.addWidget(self.table)

        table_btns = QHBoxLayout()
        add_row_btn = QPushButton("+")
        del_row_btn = QPushButton("-")
        add_row_btn.setFixedWidth(30)
        del_row_btn.setFixedWidth(30)
        table_btns.addWidget(add_row_btn)
        table_btns.addWidget(del_row_btn)
        table_btns.addStretch()
        points_layout.addLayout(table_btns)

        self.points_section = Collapsible("Edit points as numbers", points_body, expanded=False)
        layout.addWidget(self.points_section)

        add_row_btn.clicked.connect(self._add_row)
        del_row_btn.clicked.connect(self._del_row)
        self.table.cellChanged.connect(lambda *_: self.points_edited.emit())

        self._add_row()

    def _add_row(self):
        """Append a new control-point row to the table, initialized to (0.0, 0.0, 0.0)."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col in range(3):
            self.table.setItem(row, col, QTableWidgetItem("0.0"))

    def _del_row(self):
        """Remove the currently selected row from the control-points table, if any, and emit ``points_edited``."""
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.points_edited.emit()

    def sync_table(self, points: list):
        """Rebuild the table from an authoritative points list (e.g. after a
        2D-view click). Auto-expands the disclosure once there's something to
        show, so the table is never silently out of sync with the 2D views."""
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for pt in points:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, val in enumerate(pt):
                self.table.setItem(row, col, QTableWidgetItem(f"{val:.2f}"))
        self.table.blockSignals(False)
        if points and not self.points_section.is_expanded():
            self.points_section.set_expanded(True)

    def _read_control_points(self, on_invalid_row=None):
        """Parse the control-points table into a list of (x, y, z) float tuples.

        Args:
            on_invalid_row: Optional callback invoked with the row index for
                any row that fails to parse (missing/non-numeric cells);
                that row is skipped in the result.

        Returns:
            List of valid (x, y, z) tuples, in table order.
        """
        points = []
        for row in range(self.table.rowCount()):
            try:
                x = float(self.table.item(row, 0).text())
                y = float(self.table.item(row, 1).text())
                z = float(self.table.item(row, 2).text())
                points.append((x, y, z))
            except (ValueError, AttributeError):
                if on_invalid_row is not None:
                    on_invalid_row(row)
        return points

    def load_geometry(self, geometry: BundleGeometry, taper_linear: bool):
        """Populate the shape-parameter form from an existing ``BundleGeometry``.

        Note: control points themselves are not loaded here; callers use
        :meth:`sync_table` separately for that.

        Args:
            geometry: Bundle geometry to load values from.
            taper_linear: Whether the "Taper" checkbox should be checked.
        """
        self.radius.setValue(geometry.radius)
        self.n_streamlines.setValue(geometry.n_streamlines)
        self.n_samples.setValue(geometry.n_samples)
        self.dispersion.setValue(geometry.dispersion)
        self.smoothing.setValue(geometry.smoothing)
        self.taper.setChecked(taper_linear)
        self.seed.setValue(geometry.seed if geometry.seed is not None else 0)

    def current_geometry(self, control_points, on_invalid_row=None) -> BundleGeometry:
        """Build a ``BundleGeometry`` from the current form values and given control points.

        Args:
            control_points: Sequence of (x, y, z) points to use as the
                bundle's control points (typically from
                :meth:`_read_control_points`).
            on_invalid_row: Unused here; accepted for call-site symmetry with
                :meth:`_read_control_points` but not referenced in this method.

        Returns:
            A new ``BundleGeometry`` populated from the form fields.
        """
        import numpy as np
        return BundleGeometry(
            control_points=np.array(control_points, dtype=float),
            n_streamlines=self.n_streamlines.value(),
            radius=self.radius.value(),
            n_samples=self.n_samples.value(),
            smoothing=self.smoothing.value(),
            taper=None,
            dispersion=self.dispersion.value(),
            seed=self.seed.value() or None,
        )


class TissueParamsPanel(QGroupBox):
    """Editor for a TissueParameters instance. A target selector at the top
    switches between the scene-wide default and a per-bundle override, so
    there is a single set of controls rather than duplicated ones."""

    target_changed = Signal(int)  # -1 = default for all bundles, else bundle index
    override_toggled = Signal(bool)  # only meaningful when a bundle is targeted
    params_changed = Signal()    # any field edited by the user, current target

    _DEFAULT_LABEL = "All bundles (default)"

    def __init__(self, parent=None):
        super().__init__("Tissue", parent)
        layout = QVBoxLayout(self)
        layout.addWidget(_hint(
            "Standard Model compartments: intra-axonal (stick), extra-axonal "
            "(zeppelin), and CSF (ball)."
        ))

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Applies to"))
        self.target = QComboBox()
        self.target.addItem(self._DEFAULT_LABEL)
        target_row.addWidget(self.target, stretch=1)
        layout.addLayout(target_row)

        # Default/Custom is always visible (not conditionally shown) so its
        # state is legible at a glance instead of requiring a tooltip; it's
        # simply disabled while "All bundles" is the target.
        override_row = QHBoxLayout()
        override_row.addWidget(QLabel("Uses"))
        self.override_toggle = SegmentedToggle(["Default", "Custom"])
        override_row.addWidget(self.override_toggle)
        override_row.addStretch()
        layout.addLayout(override_row)

        form = QFormLayout()

        self.axon_radius = _spinbox(0.05, 20.0, 1.0, step=0.1, decimals=2)
        self.axon_radius.setToolTip(
            "Packing-density calibration radius (um). Determines intra-axonal "
            "volume fraction from streamline density."
        )
        form.addRow("Axon radius (um)", self.axon_radius)

        self.f_csf_split = _spinbox(0.0, 1.0, 0.0, step=0.05)
        self.f_csf_split.setToolTip(
            "Fraction of non-axonal space that is free water (CSF) rather "
            "than extra-axonal, in pure-fiber voxels."
        )
        form.addRow("CSF split", self.f_csf_split)

        self.di_axial = _spinbox(0.0, 5.0, 1.7, step=0.1, decimals=3)
        form.addRow("Intra-axonal Da (um2/ms)", self.di_axial)

        self.de_axial = _spinbox(0.0, 5.0, 1.7, step=0.1, decimals=3)
        form.addRow("Extra-axonal Da (um2/ms)", self.de_axial)

        self.de_radial = _spinbox(0.0, 5.0, 0.5, step=0.1, decimals=3)
        form.addRow("Extra-axonal Dr (um2/ms)", self.de_radial)

        self.t2_intra_ms = _spinbox(1.0, 500.0, 70.0, step=1.0)
        form.addRow("T2 intra-axonal (ms)", self.t2_intra_ms)

        self.t2_extra_ms = _spinbox(1.0, 500.0, 70.0, step=1.0)
        form.addRow("T2 extra-axonal (ms)", self.t2_extra_ms)

        layout.addLayout(form)

        adv_body = QWidget()
        adv_form = QFormLayout(adv_body)
        adv_form.setContentsMargins(0, 4, 0, 0)

        self.f_myelin = _spinbox(0.0, 1.0, 0.0, step=0.05)
        adv_form.addRow("Myelin fraction", self.f_myelin)

        self.t2_myelin_ms = _spinbox(1.0, 500.0, 15.0, step=1.0)
        adv_form.addRow("T2 myelin (ms)", self.t2_myelin_ms)

        layout.addWidget(Collapsible("Advanced (myelin)", adv_body, expanded=False))

        self.target.currentIndexChanged.connect(self._on_target_changed)
        self.override_toggle.currentChanged.connect(self._on_override_changed)
        for w in (self.axon_radius, self.f_csf_split, self.di_axial, self.de_axial,
                  self.de_radial, self.t2_intra_ms, self.t2_extra_ms,
                  self.f_myelin, self.t2_myelin_ms):
            w.valueChanged.connect(self.params_changed)

        self._bundle_names: list[str] = []
        self._update_form_enabled()

    # -- target selector: index 0 is the scene default, N+1 is bundle N ----

    def set_bundle_names(self, names: list[str]):
        """Repopulate the target dropdown, preserving the current selection
        (by bundle identity, not index) where possible."""
        current_target = self.current_target()
        self._bundle_names = list(names)

        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem(self._DEFAULT_LABEL)
        for name in names:
            self.target.addItem(name)
        if current_target is not None and current_target < len(names):
            self.target.setCurrentIndex(current_target + 1)
        else:
            self.target.setCurrentIndex(0)
        self.target.blockSignals(False)
        self._on_target_changed()

    def current_target(self) -> int | None:
        """Currently selected bundle index, or None if 'All bundles' is selected."""
        idx = self.target.currentIndex()
        return None if idx <= 0 else idx - 1

    def _on_target_changed(self):
        is_bundle = self.current_target() is not None
        self.override_toggle.setEnabled(is_bundle)
        self.target_changed.emit(-1 if self.current_target() is None else self.current_target())
        self._update_form_enabled()

    def _on_override_changed(self, index: int):
        if self.current_target() is not None:
            self.override_toggled.emit(index == 1)
        self._update_form_enabled()

    def _update_form_enabled(self):
        disabled = self.current_target() is not None and self.override_toggle.current() == 0
        for w in (self.axon_radius, self.f_csf_split, self.di_axial, self.de_axial,
                  self.de_radial, self.t2_intra_ms, self.t2_extra_ms,
                  self.f_myelin, self.t2_myelin_ms):
            w.setEnabled(not disabled)

    def get_params(self) -> TissueParameters:
        return TissueParameters(
            f_csf_split=self.f_csf_split.value(),
            di_axial=self.di_axial.value() * 1e-3,
            de_axial=self.de_axial.value() * 1e-3,
            de_radial=self.de_radial.value() * 1e-3,
            axon_radius=self.axon_radius.value(),
            t2_intra_ms=self.t2_intra_ms.value(),
            t2_extra_ms=self.t2_extra_ms.value(),
            t2_myelin_ms=self.t2_myelin_ms.value(),
            f_myelin=self.f_myelin.value(),
        )

    def set_params(self, t: TissueParameters):
        self.f_csf_split.setValue(t.f_csf_split)
        self.di_axial.setValue(t.di_axial * 1e3)
        self.de_axial.setValue(t.de_axial * 1e3)
        self.de_radial.setValue(t.de_radial * 1e3)
        self.axon_radius.setValue(t.axon_radius if t.axon_radius is not None else 1.0)
        self.t2_intra_ms.setValue(t.t2_intra_ms)
        self.t2_extra_ms.setValue(t.t2_extra_ms)
        self.t2_myelin_ms.setValue(t.t2_myelin_ms)
        self.f_myelin.setValue(t.f_myelin)

    def is_override_enabled(self) -> bool:
        return self.current_target() is not None and self.override_toggle.current() == 1

    def set_override_enabled(self, enabled: bool):
        self.override_toggle.set_current(1 if enabled else 0)
        self._update_form_enabled()


class ScanParamsPanel(QGroupBox):
    # Emitted when the user picks "Load bvals/bvecs..."; app.py handles the
    # file dialogs and calls set_loaded_gradients()/clear_loaded_gradients().
    load_gradients_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("Scan", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Gradient scheme"))

        src_row = QHBoxLayout()
        self.gradients_status = _hint("Using generated shells below.")
        src_row.addWidget(self.gradients_status, stretch=1)
        self.load_gradients_btn = QPushButton("Load bvals/bvecs...")
        self.load_gradients_btn.setProperty("variant", "flat")
        self.load_gradients_btn.clicked.connect(self.load_gradients_requested)
        src_row.addWidget(self.load_gradients_btn)
        self.clear_gradients_btn = QPushButton("x")
        self.clear_gradients_btn.setProperty("variant", "flat")
        self.clear_gradients_btn.setFixedWidth(22)
        self.clear_gradients_btn.setVisible(False)
        self.clear_gradients_btn.clicked.connect(self.clear_loaded_gradients)
        src_row.addWidget(self.clear_gradients_btn)
        layout.addLayout(src_row)

        self.shells_table = QTableWidget(0, 2)
        self.shells_table.setHorizontalHeaderLabels(["b-value", "n-directions"])
        self.shells_table.horizontalHeader().setStretchLastSection(True)
        self.shells_table.setFixedHeight(110)
        self.shells_table.setAlternatingRowColors(True)
        layout.addWidget(self.shells_table)

        shell_btns = QHBoxLayout()
        add_shell_btn = QPushButton("+")
        del_shell_btn = QPushButton("-")
        add_shell_btn.setFixedWidth(30)
        del_shell_btn.setFixedWidth(30)
        shell_btns.addWidget(add_shell_btn)
        shell_btns.addWidget(del_shell_btn)
        shell_btns.addStretch()
        layout.addLayout(shell_btns)

        add_shell_btn.clicked.connect(self._add_shell)
        del_shell_btn.clicked.connect(self._del_shell)
        self._add_shell(1000, 64)
        self._add_shell(2500, 64)

        self._loaded_bvals = None
        self._loaded_bvecs = None

        layout.addWidget(_hline())

        form = QFormLayout()

        self.n_b0 = QSpinBox()
        self.n_b0.setRange(1, 20)
        self.n_b0.setValue(1)
        form.addRow("n_b0", self.n_b0)

        self.d_iso = _spinbox(0.1, 5.0, 3.0, step=0.1, decimals=3)
        self.d_iso.setToolTip("Free-water (CSF) isotropic diffusivity.")
        form.addRow("CSF D (um2/ms)", self.d_iso)

        self.t2_csf_ms = _spinbox(1.0, 5000.0, 2000.0, step=10.0)
        form.addRow("T2 CSF (ms)", self.t2_csf_ms)

        layout.addLayout(form)

        layout.addWidget(self._make_nullable_row("TE (ms)", 1.0, 500.0, 80.0, 1.0, "_te", start_none=False))
        layout.addWidget(self._make_nullable_row("SNR (noise-free if disabled)", 1.0, 200.0, 20.0, 1.0, "_snr", start_none=True))

        adv_body = QWidget()
        adv_layout = QVBoxLayout(adv_body)
        adv_layout.setContentsMargins(0, 4, 0, 0)

        adv_form = QFormLayout()
        self.background_csf = _spinbox(0.0, 1.0, 0.0, step=0.01, decimals=3)
        self.background_csf.setToolTip(
            "Constant free-water signal added to every voxel, including background."
        )
        adv_form.addRow("Background CSF", self.background_csf)

        n_cores = os.cpu_count() or 1
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(1, n_cores)
        self.n_jobs.setValue(n_cores)
        self.n_jobs.setToolTip(f"Threads for the simulation (1-{n_cores}).")
        adv_form.addRow("CPU cores", self.n_jobs)

        self.export_gt = QCheckBox()
        self.export_gt.setChecked(True)
        self.export_gt.setToolTip(
            "Also save per-voxel ground truth (WM mask, fiber fraction, peaks) on export."
        )
        adv_form.addRow("Export ground truth", self.export_gt)
        adv_layout.addLayout(adv_form)

        layout.addWidget(Collapsible("Advanced", adv_body, expanded=False))

    def _add_shell(self, b=1000, n=64):
        row = self.shells_table.rowCount()
        self.shells_table.insertRow(row)
        self.shells_table.setItem(row, 0, QTableWidgetItem(str(b)))
        self.shells_table.setItem(row, 1, QTableWidgetItem(str(n)))

    def _del_shell(self):
        row = self.shells_table.currentRow()
        if row >= 0:
            self.shells_table.removeRow(row)

    def _make_nullable_row(self, label, min_, max_, default, step, attr, start_none=False) -> QGroupBox:
        """Creates a row with a spinbox and a 'disabled' checkbox that, when
        checked, makes the value read as None."""
        row = QHBoxLayout()

        spinbox = _spinbox(min_, max_, default, step)
        spinbox.setDisabled(start_none)
        none_cb = QCheckBox("disabled")
        none_cb.setChecked(start_none)

        def on_toggle(is_none):
            spinbox.setDisabled(is_none)

        none_cb.toggled.connect(on_toggle)

        row.addWidget(QLabel(label))
        row.addWidget(spinbox)
        row.addWidget(none_cb)
        row.addStretch()

        setattr(self, f"{attr}_spin", spinbox)
        setattr(self, f"{attr}_none", none_cb)

        container = QGroupBox()
        container.setProperty("flat", "true")
        container.setFlat(True)
        container.setLayout(row)
        return container

    def read_shells(self, on_invalid_row=None):
        shells = []
        for row in range(self.shells_table.rowCount()):
            try:
                b = float(self.shells_table.item(row, 0).text())
                n = int(self.shells_table.item(row, 1).text())
                shells.append((b, n))
            except (ValueError, AttributeError):
                if on_invalid_row is not None:
                    on_invalid_row(row)
        return shells

    def set_loaded_gradients(self, path_bvals: str, path_bvecs: str, bvals, bvecs):
        self._loaded_bvals = bvals
        self._loaded_bvecs = bvecs
        self.shells_table.setEnabled(False)
        self.n_b0.setEnabled(False)
        self.clear_gradients_btn.setVisible(True)
        self.gradients_status.setText(
            f"Loaded {len(bvals)} directions from "
            f"{os.path.basename(path_bvals)} / {os.path.basename(path_bvecs)}."
        )

    def clear_loaded_gradients(self):
        self._loaded_bvals = None
        self._loaded_bvecs = None
        self.shells_table.setEnabled(True)
        self.n_b0.setEnabled(True)
        self.clear_gradients_btn.setVisible(False)
        self.gradients_status.setText("Using generated shells below.")

    def loaded_gradients(self):
        """(bvals, bvecs) if a gradient file pair was loaded, else None."""
        if self._loaded_bvals is None:
            return None
        return self._loaded_bvals, self._loaded_bvecs

    def get_scan(self) -> ScanParameters:
        return ScanParameters(
            te_ms=None if self._te_none.isChecked() else self._te_spin.value(),
            d_iso=self.d_iso.value() * 1e-3,
            t2_csf_ms=self.t2_csf_ms.value(),
            background_csf=self.background_csf.value(),
        )

    def get_snr(self):
        return None if self._snr_none.isChecked() else self._snr_spin.value()

    def get_run_config(self) -> dict:
        return {
            "shells": self.read_shells(),
            "n_b0": self.n_b0.value(),
            "n_jobs": self.n_jobs.value(),
            "export_gt": self.export_gt.isChecked(),
            "snr": self.get_snr(),
        }


class GMParamsPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Gray Matter", parent)
        layout = QVBoxLayout(self)

        self.enabled = QCheckBox("Enable SANDI gray-matter compartment")
        self.enabled.setToolTip(
            "Applies wherever a 'gm' tissue mask fraction is present. "
            "Requires a reference image with a GM mask loaded."
        )
        layout.addWidget(self.enabled)
        layout.addWidget(_hint(
            "3-compartment SANDI model: intra-neurite stick, extracellular "
            "ball, and restricted soma sphere (derived: f_is = 1 - f_in - f_ec)."
        ))

        self.form_container = QWidget()
        form = QFormLayout(self.form_container)
        form.setContentsMargins(0, 6, 0, 0)

        self.f_in = _spinbox(0.0, 1.0, 0.5, step=0.05)
        form.addRow("Neurite fraction (f_in)", self.f_in)

        self.f_ec = _spinbox(0.0, 1.0, 0.35, step=0.05)
        form.addRow("Extracellular fraction (f_ec)", self.f_ec)

        self.f_is_label = QLabel()
        self.f_is_label.setProperty("role", "value")
        form.addRow("Soma fraction (f_is, derived)", self.f_is_label)

        self.d_in = _spinbox(0.0, 5.0, 1.7, step=0.1, decimals=3)
        form.addRow("Neurite D (um2/ms)", self.d_in)

        self.d_ec = _spinbox(0.0, 5.0, 1.5, step=0.1, decimals=3)
        form.addRow("Extracellular D (um2/ms)", self.d_ec)

        self.r_s = _spinbox(0.5, 20.0, 6.0, step=0.5)
        form.addRow("Soma radius (um)", self.r_s)

        self.d_is = _spinbox(0.1, 5.0, 3.0, step=0.1, decimals=3)
        form.addRow("Soma intrinsic D (um2/ms)", self.d_is)

        self.big_delta_ms = _spinbox(0.1, 200.0, 22.0, step=0.5)
        form.addRow("Diffusion time Delta (ms)", self.big_delta_ms)

        self.small_delta_ms = _spinbox(0.1, 200.0, 13.0, step=0.5)
        form.addRow("Pulse duration delta (ms)", self.small_delta_ms)

        layout.addWidget(self.form_container)

        self.f_in.valueChanged.connect(self._update_f_is)
        self.f_ec.valueChanged.connect(self._update_f_is)
        self.enabled.toggled.connect(self.form_container.setEnabled)
        self.form_container.setEnabled(False)
        self._update_f_is()

    def _update_f_is(self):
        f_is = 1.0 - self.f_in.value() - self.f_ec.value()
        color = "#e5677a" if f_is < 0 else "#e6e8ec"
        self.f_is_label.setText(f"{f_is:.3f}")
        self.f_is_label.setStyleSheet(f"color: {color};")

    def is_enabled(self) -> bool:
        return self.enabled.isChecked()

    def get_gm(self) -> GMParameters:
        return GMParameters(
            f_in=self.f_in.value(),
            f_ec=self.f_ec.value(),
            d_in=self.d_in.value() * 1e-3,
            d_ec=self.d_ec.value() * 1e-3,
            r_s=self.r_s.value(),
            d_is=self.d_is.value() * 1e-3,
            big_delta_ms=self.big_delta_ms.value(),
            small_delta_ms=self.small_delta_ms.value(),
        )


class Sidebar(QWidget):
    def __init__(self, scene: Scene, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(420)

        tabs = QTabWidget()
        self.tabs = tabs

        # --- Setup tab: volume size/voxel + reference image and masks;
        # configured once per scene, so it's kept separate from the
        # per-bundle workflow below ---
        setup_scroll, setup_layout = self._scrollable()
        self.volume_panel = VolumePanel(config=scene.volume)
        setup_layout.addWidget(self.volume_panel)
        tabs.addTab(setup_scroll, "Setup")

        # --- Bundles tab: the list (management) plus a collapsible "Draw
        # Bundle" section (authoring) that's only relevant while placing or
        # editing control points. Starts expanded when the scene has no
        # bundles yet, since there's nothing else to do on this tab then. ---
        bundles_scroll, bundles_layout = self._scrollable()
        self.bundle_list = BundleListPanel(scene=scene)
        self.bundle_geometry = BundleGeometryPanel()
        self.draw_section = Collapsible(
            "Draw Bundle", self.bundle_geometry, expanded=not scene.bundles
        )
        bundles_layout.addWidget(self.draw_section)
        bundles_layout.addWidget(self.bundle_list)
        self.bundles_tab = bundles_scroll
        tabs.addTab(bundles_scroll, "Bundles")

        # --- Tissue tab: one panel, switchable between the scene default
        # and a per-bundle override via its own target selector ---
        tissue_scroll, tissue_layout = self._scrollable()
        self.tissue_panel = TissueParamsPanel()
        tissue_layout.addWidget(self.tissue_panel)
        tabs.addTab(tissue_scroll, "Tissue")

        # --- Scan tab: shells, sequence constants, run settings ---
        scan_scroll, scan_layout = self._scrollable()
        self.scan_panel = ScanParamsPanel()
        scan_layout.addWidget(self.scan_panel)
        tabs.addTab(scan_scroll, "Scan")

        # --- Gray matter tab ---
        gm_scroll, gm_layout = self._scrollable()
        self.gm_panel = GMParamsPanel()
        gm_layout.addWidget(self.gm_panel)
        tabs.addTab(gm_scroll, "Gray Matter")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tabs)

    @staticmethod
    def _scrollable() -> tuple[QScrollArea, QVBoxLayout]:
        """Build a scroll area with a top-aligned inner layout, returning
        (scroll_area, inner_layout)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        scroll.setWidget(container)
        return scroll, layout
