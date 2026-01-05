from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox,
    QLabel, QComboBox, QCheckBox,
    QDoubleSpinBox, QSpinBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from controller import UVVisController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UV-Vis Viewer")
        self.resize(1000, 700)

        self.controller = UVVisController()
        self.ds = None
        self.ax = None

        # ===== Controls =====
        self.btn_open = QPushButton("选择数据目录")
        self.btn_open.clicked.connect(self.open_folder)

        self.combo_index = QComboBox()
        self.combo_index.currentIndexChanged.connect(self.update_plot)

        self.cb_peak = QCheckBox("查找峰值")
        self.cb_peak.stateChanged.connect(self.update_plot)

        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0, 10)
        self.spin_height.setDecimals(3)
        self.spin_height.setPrefix("height=")
        self.spin_height.valueChanged.connect(self.update_plot)

        self.spin_prom = QDoubleSpinBox()
        self.spin_prom.setRange(0, 10)
        self.spin_prom.setDecimals(3)
        self.spin_prom.setPrefix("prom=")
        self.spin_prom.valueChanged.connect(self.update_plot)

        self.spin_dist = QSpinBox()
        self.spin_dist.setRange(0, 10000)
        self.spin_dist.setPrefix("dist=")
        self.spin_dist.valueChanged.connect(self.update_plot)

        self.spin_wl_min = QDoubleSpinBox()
        self.spin_wl_max = QDoubleSpinBox()
        for spin in (self.spin_wl_min, self.spin_wl_max):
            spin.setRange(0, 2000)
            spin.setDecimals(1)
            spin.valueChanged.connect(self.update_plot)

        self.btn_save = QPushButton("保存 PNG")
        self.btn_save.clicked.connect(self.save_figure)

        # ===== Layout =====
        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_open)
        ctrl.addWidget(QLabel("光谱"))
        ctrl.addWidget(self.combo_index)
        ctrl.addWidget(self.cb_peak)
        ctrl.addWidget(self.spin_height)
        ctrl.addWidget(self.spin_prom)
        ctrl.addWidget(self.spin_dist)
        ctrl.addWidget(QLabel("wl_min"))
        ctrl.addWidget(self.spin_wl_min)
        ctrl.addWidget(QLabel("wl_max"))
        ctrl.addWidget(self.spin_wl_max)
        ctrl.addWidget(self.btn_save)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # ===== 十字光标相关 =====
        self.vline = None
        self.hline = None

        self.canvas.mpl_connect(
            "motion_notify_event",
            self.on_mouse_move
        )

        self.statusBar().showMessage("Ready")

        main_layout = QVBoxLayout()
        main_layout.addLayout(ctrl)
        main_layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ===== Logic =====
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 UV-Vis 数据目录")
        if not folder:
            return

        try:
            self.ds = self.controller.load_dataset(folder)

            self.combo_index.clear()
            for i in range(len(self.ds.uvvis_data)):
                self.combo_index.addItem(f"#{i+1}")

            wl = self.ds.bg_data['Wavelength']
            self.spin_wl_min.setValue(wl.min())
            self.spin_wl_max.setValue(wl.max())

            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def update_plot(self):
        if self.ds is None:
            return

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        # 峰值
        if self.cb_peak.isChecked():
            self.controller.apply_find_peak(
                self.ds,
                self.spin_height.value(),
                self.spin_prom.value(),
                self.spin_dist.value()
            )
        else:
            if hasattr(self.ds, "peaks"):
                delattr(self.ds, "peaks")

        self.controller.plot(
            self.ds,
            self.combo_index.currentIndex(),
            self.ax,
            self.spin_wl_min.value(),
            self.spin_wl_max.value()
        )

        # 重置十字光标
        self.vline = self.ax.axvline(
            color='gray',
            linestyle='--',
            linewidth=0.8,
            visible=False
        )
        self.hline = self.ax.axhline(
            color='gray',
            linestyle='--',
            linewidth=0.8,
            visible=False
        )

        self.canvas.draw()

    def on_mouse_move(self, event):
        if self.ax is None:
            return

        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            x = event.xdata
            y = event.ydata

            self.vline.set_xdata([x,x])
            self.hline.set_ydata([y,y])
            self.vline.set_visible(True)
            self.hline.set_visible(True)

            self.statusBar().showMessage(
                f"Wavelength = {x:.2f} nm | Absorbance = {y:.3f}"
            )

            self.canvas.draw_idle()
        else:
            if self.vline:
                self.vline.set_visible(False)
                self.hline.set_visible(False)
                self.canvas.draw_idle()
            self.statusBar().clearMessage()

    def save_figure(self):
        if self.ds is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            "uvvis.png",
            "PNG (*.png)"
        )
        if path:
            self.figure.savefig(path, dpi=300)
