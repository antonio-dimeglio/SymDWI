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
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QWidget,
)
from PySide6.QtCore import Qt, Signal
from symdwi.gui.scene import VolumeConfig, Scene, BundleConfig

class VolumePanel(QGroupBox):
    def __init__(self, config: VolumeConfig, parent=None):
        super().__init__("Volume", parent)
        self.config = config

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Volume size (mm)"))
        size_row = QHBoxLayout()
        self.x_mm = self._spinbox(10, 300, config.x_mm)
        self.y_mm = self._spinbox(10, 300, config.y_mm)
        self.z_mm = self._spinbox(10, 300, config.z_mm)
        for label, box in [("X", self.x_mm), ("Y", self.y_mm), ("Z", self.z_mm)]:
            size_row.addWidget(QLabel(label))
            size_row.addWidget(box)
        layout.addLayout(size_row)

        layout.addWidget(QLabel("Voxel size (mm)"))
        self.voxel_size = self._spinbox(0.5, 5.0, config.voxel_size, step=0.5)
        layout.addWidget(self.voxel_size)

        self.grid_label = QLabel()
        layout.addWidget(self.grid_label)
        self._update_grid()

        self.x_mm.valueChanged.connect(self._on_change)
        self.y_mm.valueChanged.connect(self._on_change)
        self.z_mm.valueChanged.connect(self._on_change)
        self.voxel_size.valueChanged.connect(self._on_change)

    def _spinbox(self, min_, max_, default, step=1.0):
        box = QDoubleSpinBox()
        box.setRange(min_, max_)
        box.setValue(default)
        box.setSingleStep(step)
        return box

    def _on_change(self):
        self.config.x_mm = self.x_mm.value()
        self.config.y_mm = self.y_mm.value()
        self.config.z_mm = self.z_mm.value()
        self.config.voxel_size = self.voxel_size.value()
        self._update_grid()

    def _update_grid(self):
        v = self.voxel_size.value()
        gx = int(self.x_mm.value() / v)
        gy = int(self.y_mm.value() / v)
        gz = int(self.z_mm.value() / v)
        self.grid_label.setText(f"Grid: {gx}×{gy}×{gz} voxels")

class BundleListPanel(QGroupBox):
    visibility_changed = Signal()

    def __init__(self, scene: Scene, parent=None):
        super().__init__("Bundles", parent)
        self.scene = scene

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.load_tck_btn = QPushButton("Load .tck…")
        self.edit_btn = QPushButton("Edit")
        self.delete_btn = QPushButton("Delete")
        btn_row.addWidget(self.load_tck_btn)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.delete_btn)
        layout.addLayout(btn_row)

        self.delete_btn.clicked.connect(self.on_delete)

        self.refresh()

    def refresh(self):
        """Rebuild the list from scene.bundles."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, cfg in enumerate(self.scene.bundles):
            label = (f"Bundle {i + 1}  [{os.path.basename(cfg.tck_path)}]"
                     if cfg.tck_path else f"Bundle {i + 1}")
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if cfg.visible else Qt.Unchecked)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _on_item_changed(self, item):
        """Mirror a checkbox toggle back into the bundle's visibility flag."""
        row = self.list_widget.row(item)
        if 0 <= row < len(self.scene.bundles):
            self.scene.bundles[row].visible = (item.checkState() == Qt.Checked)
            self.visibility_changed.emit()

    def selected_index(self):
        row = self.list_widget.currentRow()
        return row if row >= 0 else None

    def on_delete(self):
        idx = self.selected_index()
        if idx is None:
            return
        self.scene.bundles.pop(idx)
        self.refresh()


class BundleParamsPanel(QGroupBox):
    points_edited = Signal()

    def __init__(self, parent=None):
        super().__init__("Bundle Parameters", parent)
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.0, 20.0)
        self.radius.setValue(3.0)
        self.radius.setSingleStep(0.5)
        form.addRow("Radius (mm)", self.radius)

        self.n_streamlines = QSpinBox()
        self.n_streamlines.setRange(1, 1000)
        self.n_streamlines.setValue(100)
        form.addRow("Streamlines", self.n_streamlines)

        self.dispersion = QDoubleSpinBox()
        self.dispersion.setRange(0.0, 5.0)
        self.dispersion.setValue(0.5)
        self.dispersion.setSingleStep(0.1)
        form.addRow("Dispersion (mm)", self.dispersion)

        self.taper = QCheckBox()
        form.addRow("Taper", self.taper)

        layout.addLayout(form)

        layout.addWidget(QLabel("Control points"))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setFixedHeight(150)
        layout.addWidget(self.table)

        table_btns = QHBoxLayout()
        add_row_btn = QPushButton("+")
        del_row_btn = QPushButton("-")
        add_row_btn.setFixedWidth(30)
        del_row_btn.setFixedWidth(30)
        table_btns.addWidget(add_row_btn)
        table_btns.addWidget(del_row_btn)
        table_btns.addStretch()
        layout.addLayout(table_btns)

        add_row_btn.clicked.connect(self._add_row)
        del_row_btn.clicked.connect(self._del_row)
        self.table.cellChanged.connect(lambda *_: self.points_edited.emit())

        self._add_row()

        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Bundle")
        self.add_bundle_btn = QPushButton("Add Bundle")
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.add_bundle_btn)
        layout.addLayout(btn_row)


    def _add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col in range(3):
            self.table.setItem(row, col, QTableWidgetItem("0.0"))

    def _del_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.points_edited.emit()

    def _read_control_points(self):
        points = []
        for row in range(self.table.rowCount()):
            try:
                x = float(self.table.item(row, 0).text())
                y = float(self.table.item(row, 1).text())
                z = float(self.table.item(row, 2).text())
                points.append((x, y, z))
            except (ValueError, AttributeError):
                print(f"WARNING: row {row} has invalid values, skipping")
        return points

    def current_config(self) -> BundleConfig:
        return BundleConfig(
            control_points=self._read_control_points(),
            radius=self.radius.value(),
            n_streamlines=self.n_streamlines.value(),
            dispersion=self.dispersion.value(),
            taper=self.taper.isChecked(),
        )

class DWIParamsPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("DWI Parameters", parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Shells"))

        self.shells_table = QTableWidget(0, 2)
        self.shells_table.setHorizontalHeaderLabels(["b-value", "n-directions"])
        self.shells_table.horizontalHeader().setStretchLastSection(True)
        self.shells_table.setFixedHeight(120)
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

        self._add_shell()

        form = QFormLayout()

        self.n_b0 = QSpinBox()
        self.n_b0.setRange(1, 20)
        self.n_b0.setValue(1)
        form.addRow("n_b0", self.n_b0)

        self.f_intra = QDoubleSpinBox()
        self.f_intra.setRange(0.0, 1.0)
        self.f_intra.setValue(0.7)
        self.f_intra.setSingleStep(0.05)
        form.addRow("f_intra", self.f_intra)

        self.f_extra = QDoubleSpinBox()
        self.f_extra.setRange(0.0, 1.0)
        self.f_extra.setValue(0.3)
        self.f_extra.setSingleStep(0.05)
        form.addRow("f_extra", self.f_extra)

        self.axon_radius_um = QDoubleSpinBox()
        self.axon_radius_um.setRange(0.1, 20.0)
        self.axon_radius_um.setValue(1.0)
        self.axon_radius_um.setSingleStep(0.1)
        form.addRow("Axon radius (μm)", self.axon_radius_um)

        n_cores = os.cpu_count() or 1
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(1, n_cores)
        self.n_jobs.setValue(n_cores)
        self.n_jobs.setToolTip(f"Threads for the simulation (1–{n_cores}).")
        form.addRow("CPU cores", self.n_jobs)

        self.export_gt = QCheckBox()
        self.export_gt.setChecked(True)
        self.export_gt.setToolTip(
            "Also save per-voxel ground truth (WM mask, fiber fraction, peaks) on export."
        )
        form.addRow("Export ground truth", self.export_gt)

        layout.addLayout(form)

        layout.addWidget(self._make_nullable_row("SNR", 1.0, 200.0, 20.0, 1.0, "_snr"))
        layout.addWidget(self._make_nullable_row("TE (ms)", 1.0, 500.0, 80.0, 1.0, "_te"))

    def _add_shell(self):
        row = self.shells_table.rowCount()
        self.shells_table.insertRow(row)
        self.shells_table.setItem(row, 0, QTableWidgetItem("1000"))
        self.shells_table.setItem(row, 1, QTableWidgetItem("64"))

    def _del_shell(self):
        row = self.shells_table.currentRow()
        if row >= 0:
            self.shells_table.removeRow(row)


    def _make_nullable_row(self, label, min_, max_, default, step, attr) -> QGroupBox:
        """Creates a row with a spinbox and a 'None' checkbox to disable it."""
        row = QHBoxLayout()

        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_, max_)
        spinbox.setValue(default)
        spinbox.setSingleStep(step)

        none_cb = QCheckBox("None")

        def on_toggle(checked):
            spinbox.setDisabled(checked)

        none_cb.toggled.connect(on_toggle)

        row.addWidget(QLabel(label))
        row.addWidget(spinbox)
        row.addWidget(none_cb)

        setattr(self, f"{attr}_spin", spinbox)
        setattr(self, f"{attr}_none", none_cb)

        container = QGroupBox()
        container.setFlat(True)
        container.setLayout(row)
        return container

    def get_params(self) -> dict:
        shells = []
        for row in range(self.shells_table.rowCount()):
            try:
                b = float(self.shells_table.item(row, 0).text())
                n = int(self.shells_table.item(row, 1).text())
                shells.append({"b_value": b, "n_directions": n})
            except (ValueError, AttributeError):
                print(f"WARNING: shell row {row} has invalid values, skipping")

        return {
            "shells": shells,
            "n_b0": self.n_b0.value(),
            "f_intra": self.f_intra.value(),
            "f_extra": self.f_extra.value(),
            "axon_radius_um":  self.axon_radius_um.value(),
            "snr": None if self._snr_none.isChecked() else self._snr_spin.value(),
            "te":  None if self._te_none.isChecked() else self._te_spin.value(),
            "n_jobs": self.n_jobs.value(),
            "export_gt": self.export_gt.isChecked(),
        }
    
class Sidebar(QWidget):
    def __init__(self, scene: Scene, parent=None):
        super().__init__(parent)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        inner_layout = QVBoxLayout(container)
        inner_layout.setAlignment(Qt.AlignTop)
        inner_layout.setSpacing(4)
        inner_layout.setContentsMargins(4, 4, 4, 4)

        self.volume_panel = VolumePanel(config=scene.volume)
        self.bundle_list = BundleListPanel(scene=scene)
        self.bundle_params = BundleParamsPanel()
        self.dwi_panel = DWIParamsPanel()

        inner_layout.addWidget(self.volume_panel)
        inner_layout.addWidget(self.bundle_list)
        inner_layout.addWidget(self.bundle_params)
        inner_layout.addWidget(self.dwi_panel)

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)