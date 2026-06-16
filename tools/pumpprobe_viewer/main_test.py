import os
import sys

from PyQt6.QtWidgets import QApplication


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from .window import MainWindow
except ImportError:
    from window import MainWindow


def main() -> None:
    MainWindow.APP_VERSION = "1.0.3-test"
    MainWindow.SETTINGS_NAME = "PumpProbeViewerTest"
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
