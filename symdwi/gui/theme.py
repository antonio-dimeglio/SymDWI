"""Neutral, tool-like dark theme shared across the SymDWI GUI.

Defines the color palette, font stack, and the Qt style sheet (QSS) used to
skin the PySide6 widgets throughout the application. Modeled after the
restrained, hairline-bordered look of scientific desktop tools (3D Slicer,
MRIcroGL) rather than a generic SaaS dark mode: muted warm-neutral grays,
a single desaturated accent reserved for primary actions/selection, and
borders instead of glow/shadow for focus states. Other GUI modules import
the color constants for custom painting (e.g. canvas/scene drawing) and
apply ``STYLESHEET`` to the ``QApplication`` for widget styling.
"""

# Core surface/background colors, from darkest (window) to lightest (hover).
# Warm neutral grays (a touch of brown/yellow undertone) rather than the
# cool blue-grays typical of default dark-mode templates.
BG_0 = "#1c1b1a"       # window background
BG_1 = "#242322"       # panel background
BG_2 = "#2a2928"       # input background
BG_3 = "#333130"       # hover
BORDER = "#43413e"             # default border color for panels/controls
BORDER_FOCUS = "#a8a29a"       # border color when a control has focus (neutral, not accent-colored)
TEXT = "#e8e6e2"               # primary text color
TEXT_DIM = "#a8a5a0"           # secondary/muted text color
TEXT_FAINT = "#726f6a"         # tertiary/disabled text color
ACCENT = "#c17a4f"             # primary accent color (buttons, selection) -- muted terracotta
ACCENT_HOVER = "#d18f66"       # accent color on hover
ACCENT_TEXT = "#1c1b1a"        # text color used on top of accent backgrounds
DANGER = "#b8564f"             # destructive/error action color
DANGER_HOVER = "#c96a63"       # destructive action color on hover
OK = "#7a9b6e"                 # success/positive status color

# Cycling palette used to assign distinct colors to fiber bundles/overlays
# in the 2D/3D views; indexed modulo len(BUNDLE_COLORS) as bundles are added.
# Desaturated, print-plot-like hues instead of neon/saturated ones.
BUNDLE_COLORS = [
    "#c17a4f", "#5f8f7a", "#7b8fc4", "#c4a24f",
    "#9a6fa0", "#5fa3ac", "#b0714f", "#8a8f6f",
]

# Preferred system font stack, falling back across platforms (macOS, Windows,
# generic sans-serif).
FONT_FAMILY = (
    '-apple-system, "SF Pro Text", "Segoe UI", "Helvetica Neue", '
    "Arial, sans-serif"
)

# Application-wide QSS style sheet, built from the color/font constants
# above. Selectors are grouped by widget type (QPushButton, QLineEdit,
# QTableWidget, etc.) and use Qt dynamic properties (e.g. `variant`,
# `role`) to style custom widget states/variants. Applied wholesale to the
# QApplication instance. Content is intentionally left unmodified below.
STYLESHEET = f"""
* {{
    font-family: {FONT_FAMILY};
    font-size: 12.5px;
    color: {TEXT};
}}

QMainWindow, QWidget {{
    background-color: {BG_0};
}}

QToolBar {{
    background-color: {BG_1};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    spacing: 6px;
}}

QStatusBar {{
    background-color: {BG_1};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
}}

QSplitter::handle {{
    background-color: {BG_0};
}}

QLabel {{
    color: {TEXT_DIM};
    background: transparent;
}}
QLabel[role="heading"] {{
    color: {TEXT};
    font-weight: 600;
    font-size: 12.5px;
}}
QLabel[role="hint"] {{
    color: {TEXT_FAINT};
    font-size: 11px;
}}
QLabel[role="value"] {{
    color: {TEXT};
}}

QGroupBox {{
    background-color: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 14px;
    padding: 10px 10px 12px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT};
}}
QGroupBox[flat="true"] {{
    border: none;
    background: transparent;
    margin-top: 0;
    padding: 0;
}}

QPushButton {{
    background-color: {BG_2};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 6px 12px;
    color: {TEXT};
}}
QPushButton:hover {{
    background-color: {BG_3};
    border-color: {TEXT_FAINT};
}}
QPushButton:pressed {{
    background-color: {BG_1};
}}
QPushButton:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER};
    background-color: {BG_1};
}}
QPushButton[variant="primary"] {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: {ACCENT_TEXT};
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[variant="danger"]:hover {{
    background-color: {DANGER};
    border-color: {DANGER};
    color: {ACCENT_TEXT};
}}
QPushButton[variant="flat"] {{
    background: transparent;
    border: none;
    padding: 2px 6px;
}}
QPushButton[variant="flat"]:hover {{
    background-color: {BG_3};
}}

QPushButton[variant="disclosure"] {{
    background: transparent;
    border: none;
    text-align: left;
    padding: 4px 2px;
    color: {TEXT};
    font-weight: 600;
}}
QPushButton[variant="disclosure"]:hover {{
    color: {TEXT_DIM};
}}

QPushButton[variant="segment"] {{
    background-color: {BG_2};
    border: 1px solid {BORDER};
    border-radius: 0;
    padding: 5px 10px;
    color: {TEXT_DIM};
}}
QPushButton[variant="segment"][segment-pos="first"] {{
    border-top-left-radius: 3px;
    border-bottom-left-radius: 3px;
}}
QPushButton[variant="segment"][segment-pos="last"] {{
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    border-left: none;
}}
QPushButton[variant="segment"][segment-pos="mid"] {{
    border-left: none;
}}
QPushButton[variant="segment"]:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: {ACCENT_TEXT};
    font-weight: 600;
}}
QPushButton[variant="segment"]:hover:!checked {{
    background-color: {BG_3};
    color: {TEXT};
}}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background-color: {BG_2};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 6px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {BORDER_FOCUS};
}}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{
    color: {TEXT_FAINT};
    background-color: {BG_1};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_2};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}

QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 3px;
    border: 1px solid {BORDER};
    background-color: {BG_2};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QTableWidget {{
    background-color: {BG_2};
    alternate-background-color: {BG_1};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
QHeaderView::section {{
    background-color: {BG_1};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 4px;
    font-weight: 600;
}}
QTableWidget::item {{
    padding: 2px;
}}
QTableWidget::item:selected {{
    background-color: {BORDER_FOCUS};
    color: {ACCENT_TEXT};
}}

QListWidget {{
    background-color: {BG_2};
    border: 1px solid {BORDER};
    border-radius: 3px;
    outline: none;
}}
QListWidget::item {{
    padding: 6px 4px;
    border-radius: 2px;
}}
QListWidget::item:selected {{
    background-color: {ACCENT};
    color: {ACCENT_TEXT};
}}
QListWidget::item:hover:!selected {{
    background-color: {BG_3};
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    top: -1px;
    background-color: {BG_1};
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_DIM};
    padding: 7px 12px;
    margin-right: 2px;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}}
QTabBar::tab:selected {{
    background-color: {BG_1};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-bottom: none;
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT};
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_FAINT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_2};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QToolTip {{
    background-color: {BG_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 6px;
    border-radius: 3px;
}}

QSplitter::handle:hover {{
    background-color: {TEXT_FAINT};
}}
"""
