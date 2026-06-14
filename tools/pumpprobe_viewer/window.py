from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.figure import Figure

try:
    from .controller import PumpProbeController
except ImportError:
    from controller import PumpProbeController


class ChirpFitDialog(QDialog):
    def __init__(
        self,
        wavelengths: np.ndarray,
        delays: np.ndarray,
        values: np.ndarray,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fit chirp by eye")
        self.resize(920, 660)
        self.setMinimumSize(760, 540)

        order = np.argsort(wavelengths)
        self.wavelengths = np.asarray(wavelengths, dtype=float)[order]
        delay_order = np.argsort(delays)
        self.delays = np.asarray(delays, dtype=float)[delay_order]
        self.values = np.asarray(values, dtype=float)[delay_order, :][:, order]
        self.points: list[tuple[float, float]] = []
        self.coeffs: list[float] | None = None
        self.plot_ax = None

        self.figure = Figure(figsize=(8.8, 5.6), constrained_layout=False)
        self.figure.patch.set_facecolor("#f4f6f8")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(0, 8)
        self.degree_spin.setValue(3)
        self.coeffs_edit = QLineEdit()
        self.coeffs_edit.setReadOnly(True)
        self.coeffs_edit.setPlaceholderText("Click at least degree + 1 points")

        self.btn_start_over = QPushButton("Start over")
        self.btn_undo = QPushButton("Undo last")
        self.btn_done = QPushButton("Done")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_done.setProperty("role", "primary")

        controls = QGridLayout()
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(QLabel("Degree"), 0, 0)
        controls.addWidget(self.degree_spin, 0, 1)
        controls.addWidget(QLabel("Coeffs"), 0, 2)
        controls.addWidget(self.coeffs_edit, 0, 3)
        controls.addWidget(self.btn_start_over, 1, 0)
        controls.addWidget(self.btn_undo, 1, 1)
        controls.addWidget(self.btn_cancel, 1, 2)
        controls.addWidget(self.btn_done, 1, 3)
        controls.setColumnStretch(3, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)

        self.degree_spin.valueChanged.connect(self.update_fit)
        self.btn_start_over.clicked.connect(self.start_over)
        self.btn_undo.clicked.connect(self.undo_last)
        self.btn_done.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self.update_fit()

    def on_plot_click(self, event) -> None:
        if event.inaxes is not self.plot_ax or event.xdata is None or event.ydata is None:
            return
        if not (-2.0 <= event.ydata <= 2.0):
            return
        wl = float(np.clip(event.xdata, self.wavelengths[0], self.wavelengths[-1]))
        delay = float(np.clip(event.ydata, -2.0, 2.0))
        self.points.append((wl, delay))
        self.update_fit()

    def start_over(self) -> None:
        self.points.clear()
        self.update_fit()

    def undo_last(self) -> None:
        if self.points:
            self.points.pop()
            self.update_fit()

    def update_fit(self) -> None:
        degree = int(self.degree_spin.value())
        self.coeffs = None
        if len(self.points) >= degree + 1 and len({point[0] for point in self.points}) >= degree + 1:
            point_array = np.array(self.points, dtype=float)
            coeffs = np.polyfit(point_array[:, 0], point_array[:, 1], degree)
            self.coeffs = [float(value) for value in coeffs]
            self.coeffs_edit.setText(self.format_coeffs(self.coeffs))
        else:
            needed = degree + 1
            self.coeffs_edit.setText(f"Need {needed} points with distinct wavelengths")
        self.btn_done.setEnabled(self.coeffs is not None)
        self.draw()

    def draw(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.plot_ax = ax
        ax.set_facecolor("#ffffff")
        ax.set_title("Click the chirp trace from -2 to 2 ps", fontsize=12, fontweight="semibold")
        ax.set_xlabel("Probe wavelength")
        ax.set_ylabel("Delay (ps)")

        mask = (self.delays >= -2.0) & (self.delays <= 2.0)
        plot_delays = self.delays[mask]
        plot_values = self.values[mask, :]
        if plot_delays.size == 0:
            nearest = int(np.argmin(np.abs(self.delays)))
            plot_delays = self.delays[max(0, nearest - 1) : nearest + 2]
            plot_values = self.values[max(0, nearest - 1) : nearest + 2, :]

        cmap, norm = MainWindow._discrete_color_scale(plot_values)

        image = ax.pcolormesh(
            MainWindow._axis_edges(self.wavelengths),
            MainWindow._axis_edges(plot_delays),
            plot_values,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        colorbar = self.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        colorbar.ax.tick_params(labelsize=8)

        if self.points:
            point_array = np.array(self.points, dtype=float)
            ax.scatter(
                point_array[:, 0],
                point_array[:, 1],
                s=38,
                color="#1f6feb",
                edgecolors="#ffffff",
                linewidths=0.9,
                zorder=4,
                label="selected points",
            )

        if self.coeffs is not None:
            poly = np.poly1d(self.coeffs)
            fitted_delay = poly(self.wavelengths)
            ax.plot(
                self.wavelengths,
                fitted_delay,
                color="#d12828",
                linewidth=2.0,
                zorder=5,
                label=f"degree {self.degree_spin.value()} fit",
            )

        ax.set_xlim(float(self.wavelengths[0]), float(self.wavelengths[-1]))
        ax.set_ylim(-2.0, 2.0)
        ax.grid(color="#dbe2ec", linewidth=0.6, alpha=0.8)
        if self.points or self.coeffs is not None:
            ax.legend(loc="upper right", fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def format_coeffs(coeffs: list[float]) -> str:
        return ", ".join(f"{value:.8g}" for value in coeffs)


class MainWindow(QMainWindow):
    APP_VERSION = "1.0.1"
    SLICE_COLORS = ("red", "#00d900", "blue", "#00d5d5", "magenta", "#ff8c00", "#7f3fbf")
    CARPET_CMAP = LinearSegmentedColormap.from_list(
        "carpetview_like",
        [
            (0.00, "#cc66ff"),
            (0.16, "#0000ff"),
            (0.36, "#ffffff"),
            (0.50, "#f7fbf2"),
            (0.60, "#7fbd70"),
            (0.74, "#42a321"),
            (0.88, "#ffff00"),
            (1.00, "#ff0000"),
        ],
    )

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(f"Pump-Probe Loop Processor {self.APP_VERSION}")
        self.resize(1450, 900)
        self.setMinimumSize(1280, 780)

        self.settings = QSettings("A304Data", "PumpProbeViewerRelease")
        self.controller = PumpProbeController()
        self.current_folder = self.settings.value("last_data_folder", "", type=str) or None
        self.colorbar = None
        self.section_layouts: list[QVBoxLayout] = []
        self.compact_sidebar = False

        self._build_controls()
        self._build_plot_area()
        self._build_layout()
        self._connect_signals()
        self._apply_style()

        self.statusBar().showMessage("Ready")

    def _build_controls(self) -> None:
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setObjectName("PathLabel")
        self.folder_label.setWordWrap(True)
        if self.current_folder:
            self.folder_label.setText(self.current_folder)

        self.btn_open = QPushButton("Choose folder")
        self.btn_choose_data = QPushButton("Choose Data")
        self.btn_clean = QPushButton("Clean jumps")
        self.btn_average = QPushButton("Calculate average")
        self.btn_chirp = QPushButton("Correct chirp")
        self.btn_visual_chirp = QPushButton("Fit chirp by eye")
        self.btn_background = QPushButton("Subtract background")
        self.btn_save = QPushButton("Save current average")
        self.btn_clear_delay = QPushButton("Clear")
        self.btn_clear_probe = QPushButton("Clear")
        for button in (self.btn_average, self.btn_chirp, self.btn_background, self.btn_save):
            button.setProperty("role", "primary")
        for button in (self.btn_clear_delay, self.btn_clear_probe):
            button.setProperty("role", "subtle")

        self.cb_avg_only = QCheckBox("Read averaged file only")
        self.combo_data_source = QComboBox()

        self.spin_wl_min = QDoubleSpinBox()
        self.spin_wl_max = QDoubleSpinBox()
        for spin in (self.spin_wl_min, self.spin_wl_max):
            spin.setRange(-100000, 100000)
            spin.setDecimals(2)
            spin.setSingleStep(10)
        self.spin_wl_min.setValue(450)
        self.spin_wl_max.setValue(750)

        self.spin_ref_wl = QDoubleSpinBox()
        self.spin_ref_wl.setRange(-100000, 100000)
        self.spin_ref_wl.setDecimals(2)
        self.spin_ref_wl.setSingleStep(10)
        self.spin_ref_wl.setValue(650)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0, 100)
        self.spin_threshold.setDecimals(5)
        self.spin_threshold.setSingleStep(0.001)
        self.spin_threshold.setValue(0.02)

        self.spin_clean_delay_min = QDoubleSpinBox()
        self.spin_clean_delay_max = QDoubleSpinBox()
        for spin in (self.spin_clean_delay_min, self.spin_clean_delay_max):
            spin.setRange(-100000, 100000)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
        self.spin_clean_delay_min.setValue(-100000)
        self.spin_clean_delay_max.setValue(100000)

        self.loops_edit = QLineEdit("all")
        self.coeffs_edit = QLineEdit()
        self.coeffs_edit.setPlaceholderText("highest order first, e.g. 1e-8, -2e-5, 0.03, -1.2")

        self.spin_bg_before = QDoubleSpinBox()
        self.spin_bg_before.setRange(-100000, 100000)
        self.spin_bg_before.setDecimals(4)
        self.spin_bg_before.setSingleStep(0.1)
        self.spin_bg_before.setValue(0.0)

        self.delay_slices_edit = QLineEdit("0, 1, 10")
        self.wavelength_slices_edit = QLineEdit("500, 550, 600")
        self.cb_symlog = QCheckBox("Signed-log delay axis")
        self.cb_symlog.setChecked(True)

        self.info_label = QLabel("Load a folder to see data information.")
        self.info_label.setObjectName("InfoLabel")
        self.info_label.setWordWrap(True)

        self.metric_loops = self._metric_value("--")
        self.metric_average = self._metric_value("--")
        self.metric_probe = self._metric_value("--")
        self.metric_delay = self._metric_value("--")

    def _build_plot_area(self) -> None:
        self.figure = Figure(figsize=(10.6, 8.2), constrained_layout=False)
        self.figure.patch.set_facecolor("#f4f6f8")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("PlotCanvas")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

    def _build_layout(self) -> None:
        left = QWidget()
        left.setObjectName("Sidebar")
        self.sidebar_content = left
        self.sidebar_layout = QGridLayout(left)
        left_layout = self.sidebar_layout
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setHorizontalSpacing(8)
        left_layout.setVerticalSpacing(8)

        brand = QWidget()
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(2, 0, 2, 2)
        brand_layout.setSpacing(10)
        brand_title = QLabel("Pump-Probe Processor")
        brand_title.setObjectName("BrandTitle")
        self.status_pill = QLabel("Ready")
        self.status_pill.setObjectName("StatusPill")
        brand_layout.addWidget(brand_title)
        brand_layout.addStretch(1)
        brand_layout.addWidget(self.status_pill)

        load_box, load_layout = self._make_section("DATA FOLDER")
        load_layout.addWidget(self.folder_label)
        source_buttons = QHBoxLayout()
        source_buttons.setSpacing(8)
        source_buttons.addWidget(self.btn_choose_data)
        source_buttons.addWidget(self.btn_open)
        load_layout.addLayout(source_buttons)
        wl_row = QGridLayout()
        wl_row.setHorizontalSpacing(8)
        wl_row.setVerticalSpacing(6)
        wl_row.addWidget(QLabel("wl min"), 0, 0)
        wl_row.addWidget(self.spin_wl_min, 0, 1)
        wl_row.addWidget(QLabel("wl max"), 0, 2)
        wl_row.addWidget(self.spin_wl_max, 0, 3)
        wl_row.setColumnStretch(1, 1)
        wl_row.setColumnStretch(3, 1)
        load_layout.addLayout(wl_row)
        load_layout.addWidget(self.cb_avg_only)

        dataset_box, dataset_layout = self._make_section("DATASET")
        dataset_form = QFormLayout()
        dataset_form.setHorizontalSpacing(8)
        dataset_form.setVerticalSpacing(7)
        dataset_form.addRow("View", self.combo_data_source)
        dataset_form.addRow("Loops", self.loops_edit)
        dataset_layout.addLayout(dataset_form)
        dataset_layout.addWidget(self.btn_average)

        clean_box, clean_layout = self._make_section("JUMP POINT")
        clean_form = QFormLayout()
        clean_form.setHorizontalSpacing(8)
        clean_form.setVerticalSpacing(7)
        clean_form.addRow("Jump wl", self.spin_ref_wl)
        clean_form.addRow("Threshold", self.spin_threshold)
        clean_form.addRow("Delay min", self.spin_clean_delay_min)
        clean_form.addRow("Delay max", self.spin_clean_delay_max)
        clean_layout.addLayout(clean_form)
        clean_layout.addWidget(self.btn_clean)

        chirp_box, chirp_layout = self._make_section("CHIRP")
        chirp_form = QFormLayout()
        chirp_form.setHorizontalSpacing(8)
        chirp_form.setVerticalSpacing(7)
        chirp_form.addRow("Coeffs", self.coeffs_edit)
        chirp_layout.addLayout(chirp_form)
        chirp_layout.addWidget(self.btn_visual_chirp)
        chirp_layout.addWidget(self.btn_chirp)

        background_box, background_layout = self._make_section("BACKGROUND")
        background_form = QFormLayout()
        background_form.setHorizontalSpacing(8)
        background_form.setVerticalSpacing(7)
        background_form.addRow("Before ps", self.spin_bg_before)
        background_layout.addLayout(background_form)
        background_layout.addWidget(self.btn_background)

        save_box, save_layout = self._make_section("SAVE")
        save_layout.addWidget(self.btn_save)

        view_box, view_layout = self._make_section("VIEW")
        delay_row = QGridLayout()
        delay_row.setHorizontalSpacing(8)
        delay_row.addWidget(QLabel("delay cuts"), 0, 0)
        delay_row.addWidget(self.delay_slices_edit, 0, 1)
        delay_row.addWidget(self.btn_clear_delay, 0, 2)
        delay_row.setColumnStretch(1, 1)
        probe_row = QGridLayout()
        probe_row.setHorizontalSpacing(8)
        probe_row.addWidget(QLabel("probe cuts"), 0, 0)
        probe_row.addWidget(self.wavelength_slices_edit, 0, 1)
        probe_row.addWidget(self.btn_clear_probe, 0, 2)
        probe_row.setColumnStretch(1, 1)
        view_layout.addLayout(delay_row)
        view_layout.addLayout(probe_row)
        view_layout.addWidget(self.cb_symlog)
        hint = QLabel("Ctrl+click adds a probe slice. Shift+click adds a delay slice.")
        hint.setObjectName("HintLabel")
        view_layout.addWidget(hint)

        info_box, info_layout = self._make_section("INFO")
        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(6)
        info_grid.setVerticalSpacing(6)
        info_grid.addWidget(self._metric("Loops", self.metric_loops), 0, 0)
        info_grid.addWidget(self._metric("Average", self.metric_average), 0, 1)
        info_grid.addWidget(self._metric("Probe", self.metric_probe), 1, 0)
        info_grid.addWidget(self._metric("Delay", self.metric_delay), 1, 1)
        info_layout.addLayout(info_grid)
        info_layout.addWidget(self.info_label)

        left_layout.addWidget(brand, 0, 0, 1, 2)
        left_layout.addWidget(load_box, 1, 0, 1, 2)
        left_layout.addWidget(dataset_box, 2, 0)
        left_layout.addWidget(clean_box, 2, 1)
        left_layout.addWidget(chirp_box, 3, 0)
        left_layout.addWidget(background_box, 3, 1)
        left_layout.addWidget(view_box, 4, 0, 1, 2)
        left_layout.addWidget(save_box, 5, 0)
        left_layout.addWidget(info_box, 5, 1)
        left_layout.setRowStretch(6, 1)

        right = QWidget()
        right.setObjectName("PlotPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(12)
        toolbar = QWidget()
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(2)
        self.plot_title = QLabel("Averaged data")
        self.plot_title.setObjectName("PlotTitle")
        self.plot_subtitle = QLabel("")
        self.plot_subtitle.setObjectName("PlotSubtitle")
        self.plot_subtitle.hide()
        toolbar_layout.addWidget(self.plot_title)
        toolbar_layout.addWidget(self.plot_subtitle)
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas, 1)

        main = QWidget()
        main.setObjectName("MainSurface")
        main_layout = QHBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        left_scroll = QScrollArea()
        left_scroll.setObjectName("SidebarScroll")
        left_scroll.setFixedWidth(500)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)
        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right, 1)
        self.setCentralWidget(main)
        self._apply_sidebar_density(self.height())

    def _make_section(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        section = QFrame()
        section.setObjectName("Section")
        section.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(7)
        self.section_layouts.append(layout)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        return section, layout

    def _metric(self, title: str, value: QLabel) -> QFrame:
        box = QFrame()
        box.setObjectName("Metric")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(1)
        label = QLabel(title)
        label.setObjectName("MetricLabel")
        layout.addWidget(label)
        layout.addWidget(value)
        return box

    @staticmethod
    def _metric_value(text: str) -> QLabel:
        value = QLabel(text)
        value.setObjectName("MetricValue")
        return value

    def _connect_signals(self) -> None:
        self.btn_open.clicked.connect(self.choose_folder)
        self.btn_choose_data.clicked.connect(self.choose_data_file)
        self.btn_clean.clicked.connect(self.clean_jump_points)
        self.btn_average.clicked.connect(self.calculate_average)
        self.btn_chirp.clicked.connect(self.apply_chirp)
        self.btn_visual_chirp.clicked.connect(self.fit_chirp_visually)
        self.btn_background.clicked.connect(self.subtract_background)
        self.btn_save.clicked.connect(self.save_processed)
        self.btn_clear_delay.clicked.connect(self.clear_delay_slices)
        self.btn_clear_probe.clicked.connect(self.clear_probe_slices)
        self.combo_data_source.currentIndexChanged.connect(self.on_data_source_changed)

        for widget in (
            self.delay_slices_edit,
            self.wavelength_slices_edit,
            self.cb_symlog,
        ):
            if isinstance(widget, QLineEdit):
                widget.editingFinished.connect(self.update_plot)
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(self.update_plot)
            else:
                widget.stateChanged.connect(self.update_plot)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget#MainSurface {
                background: #f4f6f8;
                color: #18212f;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 13px;
            }
            QWidget#Sidebar {
                background: #fbfcfe;
                border-right: 1px solid #d4dae2;
            }
            QScrollArea#SidebarScroll {
                background: #fbfcfe;
                border-right: 1px solid #d4dae2;
            }
            QWidget#PlotPanel {
                background: #f4f6f8;
            }
            QLabel#BrandTitle {
                color: #07111f;
                font-size: 18px;
                font-weight: 650;
            }
            QLabel#StatusPill {
                color: #1c8a55;
                background: #e8f7ef;
                border: 1px solid #bfe7d0;
                border-radius: 13px;
                padding: 3px 9px;
                font-size: 12px;
            }
            QLabel#PlotTitle {
                color: #07111f;
                font-size: 18px;
                font-weight: 650;
            }
            QLabel#PlotSubtitle {
                color: #657386;
                font-size: 12px;
            }
            QFrame#Section {
                background: #ffffff;
                border: 1px solid #d4dae2;
                border-radius: 8px;
            }
            QLabel#SectionTitle {
                color: #344155;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel {
                color: #3f4b5c;
            }
            QLabel#PathLabel {
                background: #f8fafc;
                border: 1px solid #c9d2df;
                border-radius: 6px;
                color: #223047;
                padding: 7px;
                font-size: 12px;
                line-height: 130%;
            }
            QLabel#InfoLabel {
                color: #657386;
                font-size: 12px;
            }
            QLabel#HintLabel {
                background: #e8f1ff;
                border: 1px solid #c7ddff;
                border-radius: 6px;
                color: #24405f;
                padding: 6px;
                font-size: 12px;
            }
            QFrame#Metric {
                background: #f7f9fc;
                border: 1px solid #e0e6ef;
                border-radius: 6px;
            }
            QLabel#MetricLabel {
                color: #657386;
                font-size: 12px;
            }
            QLabel#MetricValue {
                color: #07111f;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton {
                min-height: 28px;
                border: 1px solid #bac4d1;
                border-radius: 6px;
                background: #ffffff;
                color: #1c2b3f;
                padding: 3px 10px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #f7f9fc;
                border-color: #8fa2ba;
            }
            QPushButton:pressed {
                background: #e8eef7;
            }
            QPushButton[role="primary"] {
                background: #1f6feb;
                border-color: #0f5bd7;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton[role="primary"]:hover {
                background: #2d7cf0;
                border-color: #1768e5;
            }
            QPushButton[role="primary"]:pressed {
                background: #1758c9;
            }
            QPushButton[role="subtle"] {
                min-width: 48px;
                background: #f7f9fc;
                color: #46566d;
                padding: 3px 8px;
            }
            QLineEdit, QComboBox, QDoubleSpinBox {
                min-height: 26px;
                border: 1px solid #c5cedb;
                border-radius: 5px;
                background: #ffffff;
                color: #1b2430;
                padding: 1px 8px;
                selection-background-color: #1f6feb;
            }
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
                border-color: #1f6feb;
                background: #fbfdff;
            }
            QComboBox::drop-down, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                border: 0;
                width: 18px;
            }
            QCheckBox {
                color: #27364a;
                spacing: 7px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #aeb9c8;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #1f6feb;
                border-color: #1f6feb;
            }
            QStatusBar {
                background: #fbfcfe;
                border-top: 1px solid #d4dae2;
                color: #657386;
            }
            QWidget#PlotCanvas {
                background: #f4f6f8;
            }
            """
        )
        self.statusBar().setSizeGripEnabled(False)

    def choose_folder(self) -> None:
        start_dir = self.current_folder if self.current_folder and os.path.isdir(self.current_folder) else os.getcwd()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose pump-probe data folder",
            start_dir,
        )
        if folder:
            self.current_folder = folder
            self.settings.setValue("last_data_folder", folder)
            self.folder_label.setText(folder)
            self.coeffs_edit.clear()
            self.load_data()

    def choose_data_file(self) -> None:
        last_file = self.settings.value("last_data_file", "", type=str) or ""
        if last_file and os.path.isfile(last_file):
            start_dir = os.path.dirname(last_file)
        elif self.current_folder and os.path.isdir(self.current_folder):
            start_dir = self.current_folder
        else:
            start_dir = os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose pump-probe data file",
            start_dir,
            "Data files (*.dat *.txt);;All files (*)",
        )
        if path:
            self.current_folder = os.path.dirname(path)
            self.settings.setValue("last_data_file", path)
            self.settings.setValue("last_data_folder", self.current_folder)
            self.folder_label.setText(path)
            self.coeffs_edit.clear()
            self.load_single_data(path)

    def load_data(self) -> None:
        if not self.current_folder:
            self.choose_folder()
            return
        try:
            summary = self.controller.load_dataset(
                self.current_folder,
                self.spin_wl_min.value(),
                self.spin_wl_max.value(),
                self.cb_avg_only.isChecked(),
            )
            self._apply_summary(summary)
            self.refresh_data_sources()
            self.update_plot()
            self.status_pill.setText("Loaded")
            self.statusBar().showMessage("Data loaded")
        except Exception as exc:
            self.show_error(exc)

    def load_single_data(self, path: str) -> None:
        try:
            summary = self.controller.load_single_file(
                path,
                self.spin_wl_min.value(),
                self.spin_wl_max.value(),
            )
            self._apply_summary(summary)
            self.refresh_data_sources()
            self.update_plot()
            self.status_pill.setText("Loaded")
            self.statusBar().showMessage("Data file loaded")
        except Exception as exc:
            self.show_error(exc)

    def clean_jump_points(self) -> None:
        try:
            self.controller.clean_jump_points(
                ref_wavelength=self.spin_ref_wl.value(),
                threshold=self.spin_threshold.value(),
                delay_min=self.spin_clean_delay_min.value(),
                delay_max=self.spin_clean_delay_max.value(),
            )
            self.statusBar().showMessage("Jump points cleaned")
        except Exception as exc:
            self.show_error(exc)

    def calculate_average(self) -> None:
        try:
            self.controller.calculate_average(self.loops_edit.text())
            self.refresh_data_sources(preferred="avg")
            self.update_plot()
            self.statusBar().showMessage("Average calculated")
        except Exception as exc:
            self.show_error(exc)

    def apply_chirp(self) -> None:
        try:
            self.controller.apply_chirp(self.coeffs_edit.text())
            self.update_plot()
            self.statusBar().showMessage("Chirp corrected")
        except Exception as exc:
            self.show_error(exc)

    def fit_chirp_visually(self) -> None:
        try:
            data = self.controller.get_matrix()
            wavelengths = data.columns.to_numpy(dtype=float)
            delays = data.index.to_numpy(dtype=float)
            values = data.to_numpy(dtype=float)
            dialog = ChirpFitDialog(wavelengths, delays, values, self)
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.coeffs is None:
                self.statusBar().showMessage("Fit chirp by eye canceled")
                return
            self.coeffs_edit.setText(ChirpFitDialog.format_coeffs(dialog.coeffs))
            self.controller.apply_chirp(self.coeffs_edit.text())
            self.update_plot()
            self.statusBar().showMessage("Fit chirp by eye applied")
        except Exception as exc:
            self.show_error(exc)

    def subtract_background(self) -> None:
        try:
            self.controller.subtract_background(self.spin_bg_before.value())
            self.refresh_data_sources(preferred="avg")
            self.update_plot()
            self.statusBar().showMessage("Background subtracted")
        except Exception as exc:
            self.show_error(exc)

    def save_processed(self) -> None:
        try:
            start_dir = self.settings.value("last_save_folder", "", type=str) or self.current_folder or os.getcwd()
            if not os.path.isdir(start_dir):
                start_dir = self.current_folder if self.current_folder and os.path.isdir(self.current_folder) else os.getcwd()
            default_path = os.path.join(start_dir, "processed_averaged.dat")
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save current average",
                default_path,
                "DAT files (*.dat);;All files (*)",
            )
            if not path:
                self.statusBar().showMessage("Save canceled")
                return
            if not path.lower().endswith(".dat"):
                path = f"{path}.dat"
            self.settings.setValue("last_save_folder", os.path.dirname(path))
            path = self.controller.save_current_average(path)
            self.statusBar().showMessage(f"Saved {path}")
            QMessageBox.information(self, "Saved", path)
        except Exception as exc:
            self.show_error(exc)

    def clear_delay_slices(self) -> None:
        self.delay_slices_edit.clear()
        self.update_plot()

    def clear_probe_slices(self) -> None:
        self.wavelength_slices_edit.clear()
        self.update_plot()

    def update_plot(self) -> None:
        try:
            data = self.controller.get_matrix()
        except Exception:
            return

        wavelengths = data.columns.to_numpy(dtype=float)
        delays = data.index.to_numpy(dtype=float)
        values = data.to_numpy(dtype=float)
        order = np.argsort(wavelengths)
        wavelengths = wavelengths[order]
        values = values[:, order]
        x_label, y_label = self.controller.get_axis_labels()
        use_signed_log = self.cb_symlog.isChecked()
        delay_plot = self._signed_log(delays) if use_signed_log else delays

        self.figure.clear()
        self.plot_title.setText(self.controller.get_view_source_label())
        self.figure.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.08, wspace=0.52, hspace=0.42)
        grid = self.figure.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1])
        ax_img = self.figure.add_subplot(grid[0, 0])
        ax_wl = self.figure.add_subplot(grid[0, 1])
        ax_delay = self.figure.add_subplot(grid[1, 0])
        ax_empty = self.figure.add_subplot(grid[1, 1])
        ax_empty.axis("off")
        for ax in (ax_img, ax_wl, ax_delay):
            ax.set_facecolor("#ffffff")
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#6f7d8f")
                spine.set_linewidth(0.8)
            ax.tick_params(colors="#172033", labelsize=9)
            ax.title.set_fontsize(12)
            ax.title.set_fontweight("semibold")

        cmap, norm = self._discrete_color_scale(values)

        image = ax_img.pcolormesh(
            self._axis_edges(wavelengths),
            self._axis_edges(delay_plot),
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        ax_img.set_xlabel(x_label)
        ax_img.set_ylabel(y_label)
        ax_img.set_title(self.controller.get_view_source_label())
        self._format_delay_axis(ax_img, "y", delays, delay_plot, use_signed_log)
        self._draw_delay_reference_lines(ax_img, "y", delays, use_signed_log)
        ax_img.grid(axis="x", color="0.86", linewidth=0.6)
        colorbar = self.figure.colorbar(image, ax=ax_img, fraction=0.046, pad=0.035)
        colorbar.ax.tick_params(labelsize=8)

        delay_slices = self._parse_number_list(self.delay_slices_edit.text())
        for index, delay in enumerate(delay_slices):
            nearest_delay = self.controller.get_nearest_delay(delay)
            nearest_delay_plot = self._signed_log_scalar(nearest_delay) if use_signed_log else nearest_delay
            color = self._slice_color(index)
            ax_img.axhline(nearest_delay_plot, color=color, linewidth=1.0, alpha=0.9)
            ax_delay.plot(
                wavelengths,
                data.loc[nearest_delay, :].to_numpy(dtype=float)[order],
                color=color,
                linewidth=1.6,
                label=f"{nearest_delay:.1f} ps",
            )
        ax_delay.axhline(0, color="k", linewidth=0.7, alpha=0.5)
        ax_delay.grid(color="0.86", linewidth=0.6)
        ax_delay.set_xlabel(x_label)
        ax_delay.set_ylabel("Intensity")
        ax_delay.set_title(f"Slices at {y_label}")
        if delay_slices:
            ax_delay.legend(fontsize=8)

        wl_slices = self._parse_number_list(self.wavelength_slices_edit.text())
        for index, wavelength in enumerate(wl_slices):
            nearest_wl = self.controller.get_nearest_wavelength(wavelength)
            color = self._slice_color(index)
            ax_img.axvline(nearest_wl, color=color, linewidth=1.0, alpha=0.9)
            ax_wl.plot(
                delay_plot,
                data.loc[:, nearest_wl].to_numpy(dtype=float),
                color=color,
                linewidth=1.6,
                label=f"{nearest_wl:.0f} nm",
            )
        ax_wl.axhline(0, color="k", linewidth=0.7, alpha=0.5)
        ax_wl.set_xlabel(y_label)
        ax_wl.set_ylabel("Intensity")
        ax_wl.set_title("Slices at probe")
        self._format_delay_axis(ax_wl, "x", delays, delay_plot, use_signed_log)
        self._draw_delay_reference_lines(ax_wl, "x", delays, use_signed_log)
        ax_wl.grid(axis="y", color="0.86", linewidth=0.6)
        if wl_slices:
            ax_wl.legend(fontsize=8)

        self.canvas.draw_idle()

    def on_plot_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        try:
            if event.inaxes.get_title() == self.controller.get_view_source_label():
                wl = self.controller.get_nearest_wavelength(event.xdata)
                delay_value = self._delay_from_plot_value(event.ydata)
                delay = self.controller.get_nearest_delay(delay_value)
                modifiers = QApplication.keyboardModifiers()
                event_key = (getattr(event, "key", "") or "").lower()
                ctrl_pressed = bool(modifiers & Qt.KeyboardModifier.ControlModifier) or "control" in event_key or "ctrl" in event_key
                shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier) or "shift" in event_key
                if ctrl_pressed:
                    self.wavelength_slices_edit.setText(self._append_unique(self.wavelength_slices_edit.text(), wl))
                    self.statusBar().showMessage(f"Added probe slice {wl:g}")
                    self.update_plot()
                elif shift_pressed:
                    self.delay_slices_edit.setText(self._append_unique(self.delay_slices_edit.text(), delay))
                    self.statusBar().showMessage(f"Added delay slice {delay:g} ps")
                    self.update_plot()
                else:
                    self.statusBar().showMessage("Ctrl+click adds probe; Shift+click adds delay")
        except Exception as exc:
            self.show_error(exc)

    def on_data_source_changed(self) -> None:
        source = self.combo_data_source.currentData()
        if not source:
            return
        self.controller.set_view_source(source)
        self.update_plot()

    def refresh_data_sources(self, preferred: str | None = None) -> None:
        try:
            sources = self.controller.list_view_sources()
        except Exception:
            return

        current = preferred or self.combo_data_source.currentData() or self.controller.view_source
        self.combo_data_source.blockSignals(True)
        self.combo_data_source.clear()
        selected_index = 0
        for index, (source, label) in enumerate(sources):
            self.combo_data_source.addItem(label, source)
            if source == current:
                selected_index = index
        self.combo_data_source.setCurrentIndex(selected_index)
        self.combo_data_source.blockSignals(False)
        self.controller.set_view_source(self.combo_data_source.currentData() or "avg")

    def _apply_summary(self, summary) -> None:
        wl_low = min(summary.wavelength_min, summary.wavelength_max)
        wl_high = max(summary.wavelength_min, summary.wavelength_max)
        delay_low = min(summary.delay_min, summary.delay_max)
        delay_high = max(summary.delay_min, summary.delay_max)

        self.spin_ref_wl.setRange(wl_low, wl_high)
        self.spin_ref_wl.setValue(wl_high)
        for spin in (self.spin_clean_delay_min, self.spin_clean_delay_max):
            spin.setRange(delay_low, delay_high)
        self.spin_clean_delay_min.setValue(delay_low)
        self.spin_clean_delay_max.setValue(delay_high)
        self.spin_bg_before.setRange(delay_low, delay_high)
        if delay_low <= -1.0 <= delay_high:
            self.spin_bg_before.setValue(-1.0)
        else:
            self.spin_bg_before.setValue(delay_low if abs(delay_low + 1.0) < abs(delay_high + 1.0) else delay_high)

        self.wavelength_slices_edit.setText(self._default_values(wl_low, wl_high))
        self.delay_slices_edit.setText(self._default_values(delay_low, delay_high))
        self.metric_loops.setText(f"{summary.loop_count:g}")
        self.metric_average.setText("Yes" if summary.has_average else "No")
        self.metric_probe.setText(f"{wl_low:.0f}-{wl_high:.0f}")
        self.metric_delay.setText(f"{delay_low:g}~{delay_high:g}")
        self.info_label.setText(
            f"Data loaded. Current view uses "
            f"{'the averaged file' if summary.has_average else 'calculated/loop data'}."
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_sidebar_density(event.size().height())

    def _apply_sidebar_density(self, window_height: int) -> None:
        compact = window_height < 850
        if compact == self.compact_sidebar:
            return
        self.compact_sidebar = compact
        outer_margin = 8 if compact else 12
        outer_spacing = 5 if compact else 8
        section_margin = (8, 6, 8, 7) if compact else (10, 8, 10, 10)
        section_spacing = 5 if compact else 7
        if hasattr(self, "sidebar_layout"):
            self.sidebar_layout.setContentsMargins(outer_margin, outer_margin, outer_margin, outer_margin)
            self.sidebar_layout.setHorizontalSpacing(outer_spacing)
            self.sidebar_layout.setVerticalSpacing(outer_spacing)
        for layout in getattr(self, "section_layouts", []):
            layout.setContentsMargins(*section_margin)
            layout.setSpacing(section_spacing)

    @staticmethod
    def _parse_number_list(text: str) -> list[float]:
        numbers = []
        for token in text.replace(";", ",").split(","):
            token = token.strip()
            if token:
                numbers.append(float(token))
        return numbers

    @staticmethod
    def _append_unique(text: str, value: float) -> str:
        values = MainWindow._parse_number_list(text)
        if not any(abs(value - existing) <= max(1e-9, abs(value) * 1e-9) for existing in values):
            values.append(value)
        return ", ".join(f"{item:g}" for item in values)

    @staticmethod
    def _default_values(low: float, high: float) -> str:
        if not np.isfinite(low) or not np.isfinite(high):
            return ""
        values = np.linspace(low, high, 3)
        return ", ".join(f"{value:g}" for value in values)

    @staticmethod
    def _signed_log(values, linthresh: float = 1.0):
        values = np.asarray(values, dtype=float)
        return np.sign(values) * np.log10(1.0 + np.abs(values) / linthresh)

    @staticmethod
    def _signed_log_inverse(values, linthresh: float = 1.0):
        values = np.asarray(values, dtype=float)
        return np.sign(values) * linthresh * (np.power(10.0, np.abs(values)) - 1.0)

    @staticmethod
    def _signed_log_scalar(value: float, linthresh: float = 1.0) -> float:
        return float(np.sign(value) * np.log10(1.0 + abs(value) / linthresh))

    @staticmethod
    def _axis_edges(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
        mids = (values[:-1] + values[1:]) / 2
        first = values[0] - (mids[0] - values[0])
        last = values[-1] + (values[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]])

    @classmethod
    def _discrete_color_scale(cls, values: np.ndarray, level_count: int = 17) -> tuple[LinearSegmentedColormap, BoundaryNorm]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        vmax = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        levels = np.linspace(-vmax, vmax, level_count + 1)
        cmap = cls.CARPET_CMAP.resampled(level_count)
        norm = BoundaryNorm(levels, cmap.N, clip=True)
        return cmap, norm

    def _format_delay_axis(self, ax, axis: str, delays: np.ndarray, delay_plot: np.ndarray, use_signed_log: bool) -> None:
        if not use_signed_log:
            return
        ticks = self._delay_ticks(delays)
        tick_positions = self._signed_log(ticks)
        tick_labels = [f"{tick:g}" for tick in ticks]
        if axis == "x":
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
        else:
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)

    def _delay_ticks(self, delays: np.ndarray) -> np.ndarray:
        delay_min = float(np.nanmin(delays))
        delay_max = float(np.nanmax(delays))
        ticks = [0.0]
        if delay_max > 0:
            min_power = -3 if delay_max < 1 else 0
            max_power = int(np.floor(np.log10(delay_max)))
            for power in range(min_power, max_power + 1):
                ticks.append(10.0 ** power)
        if delay_min < 0:
            min_abs = abs(delay_min)
            min_power = -3 if min_abs < 1 else 0
            max_power = int(np.floor(np.log10(min_abs)))
            for power in range(min_power, max_power + 1):
                ticks.append(-(10.0 ** power))
        valid = sorted({tick for tick in ticks if delay_min <= tick <= delay_max})
        return np.array(valid, dtype=float)

    def _draw_delay_reference_lines(self, ax, axis: str, delays: np.ndarray, use_signed_log: bool) -> None:
        ticks = self._delay_ticks(delays)
        positions = self._signed_log(ticks) if use_signed_log else ticks
        for value, position in zip(ticks, positions):
            if abs(value) < 1e-12:
                continue
            if axis == "x":
                ax.axvline(position, color="0.84", linewidth=0.6, zorder=2)
            else:
                ax.axhline(position, color="0.84", linewidth=0.6, zorder=2)

    def _slice_color(self, index: int) -> str:
        return self.SLICE_COLORS[index % len(self.SLICE_COLORS)]

    def _delay_from_plot_value(self, plot_value: float) -> float:
        if not self.cb_symlog.isChecked():
            return plot_value
        return float(self._signed_log_inverse(plot_value))

    def show_error(self, exc: Exception) -> None:
        QMessageBox.critical(self, "Error", str(exc))
        self.statusBar().showMessage(str(exc))
