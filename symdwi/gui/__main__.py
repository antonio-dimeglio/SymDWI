import os
import sys

# VTK/pyvista's OpenGL rendering does not work on Qt's native Wayland backend,
# so force the X11 (xcb) platform plugin, which runs through XWayland. Must be
# set before QApplication is created. Respect an explicit user override.
if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ["QT_QPA_PLATFORM"] = "xcb"

from PySide6.QtWidgets import QApplication
from symdwi.gui.app import MainWindow

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()