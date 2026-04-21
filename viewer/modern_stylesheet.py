"""
Modern Theme Stylesheets for Navier-Stokes Flow Simulator
Provides cohesive, professional light and dark themes with excellent readability
"""

MODERN_DARK_THEME = """
/* ===== MAIN WINDOW ===== */
QMainWindow {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11px;
}

/* ===== GROUP BOXES ===== */
QGroupBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
    color: #cba6f7;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px 0 8px;
    color: #cba6f7;
    font-size: 12px;
    font-weight: 600;
}

/* ===== BUTTONS ===== */
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 500;
    font-size: 11px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #585b70;
    border-color: #7f849c;
}

QPushButton:pressed {
    background-color: #313244;
}

QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
    border-color: #45475a;
}

/* ===== LINE EDIT & SPIN BOX ===== */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #585b70;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #89b4fa;
    background-color: #313244;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #45475a;
    border-top-right-radius: 4px;
    background-color: #45475a;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #45475a;
    border-bottom-right-radius: 4px;
    background-color: #45475a;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #585b70;
}

/* ===== COMBO BOX ===== */
QComboBox {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px 4px 32px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #585b70;
}

QComboBox:focus {
    border-color: #89b4fa;
    background-color: #313244;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #cdd6f4;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    outline: none;
}

/* ===== CHECK BOX ===== */
QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #45475a;
    border-radius: 4px;
    background-color: #1e1e2e;
}

QCheckBox::indicator:hover {
    border-color: #585b70;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
    image: none;
}

QCheckBox::indicator:checked:hover {
    background-color: #b4befe;
    border-color: #b4befe;
}

/* ===== RADIO BUTTON ===== */
QRadioButton {
    spacing: 8px;
    color: #cdd6f4;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #45475a;
    border-radius: 9px;
    background-color: #1e1e2e;
}

QRadioButton::indicator:hover {
    border-color: #585b70;
}

QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
    width: 10px;
    height: 10px;
    margin: 4px;
}

QRadioButton::indicator:checked:hover {
    background-color: #b4befe;
    border-color: #b4befe;
}

/* ===== SLIDERS ===== */
QSlider::groove:horizontal {
    height: 6px;
    background: #45475a;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #89b4fa;
    border: none;
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -5px 0;
}

QSlider::handle:horizontal:hover {
    background: #b4befe;
}

QSlider::groove:horizontal:disabled {
    background: #313244;
}

QSlider::handle:horizontal:disabled {
    background: #6c7086;
}

/* ===== SCROLL BAR ===== */
QScrollBar:vertical {
    background: #313244;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #585b70;
    min-height: 30px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #6c7086;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #313244;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background: #585b70;
    min-width: 30px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal:hover {
    background: #6c7086;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ===== LABELS ===== */
QLabel {
    color: #cdd6f4;
    background: transparent;
}

QLabel[class="heading"] {
    font-size: 14px;
    font-weight: 700;
    color: #cba6f7;
}

QLabel[class="subheading"] {
    font-size: 12px;
    font-weight: 600;
    color: #89b4fa;
}

QLabel[class="value"] {
    font-size: 11px;
    font-weight: 500;
    color: #a6e3a1;
    font-family: 'Consolas', 'Monaco', monospace;
}

QLabel[class="info"] {
    color: #94e2d5;
}

QLabel[class="warning"] {
    color: #f9e2af;
}

QLabel[class="error"] {
    color: #f38ba8;
}

/* ===== TAB WIDGET ===== */
QTabWidget::pane {
    background-color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 8px;
}

QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}

QTabBar::tab:hover:!selected {
    background-color: #45475a;
}

/* ===== SPLITTER ===== */
QSplitter::handle {
    background-color: #45475a;
}

QSplitter::handle:hover {
    background-color: #89b4fa;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* ===== PROGRESS BAR ===== */
QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}

/* ===== TOOL TIP ===== */
QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}

/* ===== MENU BAR ===== */
QMenuBar {
    background-color: #313244;
    color: #cdd6f4;
    border-bottom: 1px solid #45475a;
    padding: 4px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #45475a;
}

QMenuBar::item:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}

/* ===== MENU ===== */
QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #45475a;
}

QMenu::item:disabled {
    color: #6c7086;
}

QMenu::separator {
    height: 1px;
    background-color: #45475a;
    margin: 4px 8px;
}

/* ===== DOCK WIDGET ===== */
QDockWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background-color: #313244;
    padding: 8px;
    border-bottom: 1px solid #45475a;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    font-weight: 600;
}

/* ===== STATUS BAR ===== */
QStatusBar {
    background-color: #313244;
    color: #cdd6f4;
    border-top: 1px solid #45475a;
}

/* ===== SCROLL AREA ===== */
QScrollArea {
    background-color: #1e1e2e;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: #1e1e2e;
}

/* ===== FRAME ===== */
QFrame {
    background-color: #1e1e2e;
    border: none;
}

QFrame[class="card"] {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 12px;
}

QFrame[class="separator"] {
    background-color: #45475a;
    max-height: 1px;
    min-height: 1px;
}
"""


MODERN_LIGHT_THEME = """
/* ===== MAIN WINDOW ===== */
QMainWindow {
    background-color: #ffffff;
    color: #2c3e50;
}

QWidget {
    background-color: #ffffff;
    color: #2c3e50;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11px;
}

/* ===== GROUP BOXES ===== */
QGroupBox {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
    color: #495057;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px 0 8px;
    color: #495057;
    font-size: 12px;
    font-weight: 600;
}

/* ===== BUTTONS ===== */
QPushButton {
    background-color: #e9ecef;
    color: #212529;
    border: 1px solid #ced4da;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 500;
    font-size: 11px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #dee2e6;
    border-color: #adb5bd;
}

QPushButton:pressed {
    background-color: #ced4da;
}

QPushButton:disabled {
    background-color: #f8f9fa;
    color: #6c757d;
    border-color: #dee2e6;
}

/* ===== LINE EDIT & SPIN BOX ===== */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #ced4da;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #007bff;
    selection-color: #ffffff;
}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #adb5bd;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #007bff;
    background-color: #f8f9fa;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #ced4da;
    border-top-right-radius: 4px;
    background-color: #e9ecef;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #ced4da;
    border-bottom-right-radius: 4px;
    background-color: #e9ecef;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #dee2e6;
}

/* ===== COMBO BOX ===== */
QComboBox {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #ced4da;
    border-radius: 4px;
    padding: 4px 8px 4px 32px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #adb5bd;
}

QComboBox:focus {
    border-color: #007bff;
    background-color: #f8f9fa;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #2c3e50;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #ced4da;
    selection-background-color: #007bff;
    selection-color: #ffffff;
    outline: none;
}

/* ===== CHECK BOX ===== */
QCheckBox {
    spacing: 8px;
    color: #2c3e50;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #ced4da;
    border-radius: 4px;
    background-color: #ffffff;
}

QCheckBox::indicator:hover {
    border-color: #adb5bd;
}

QCheckBox::indicator:checked {
    background-color: #007bff;
    border-color: #007bff;
    image: none;
}

QCheckBox::indicator:checked:hover {
    background-color: #0056b3;
    border-color: #0056b3;
}

/* ===== RADIO BUTTON ===== */
QRadioButton {
    spacing: 8px;
    color: #2c3e50;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #ced4da;
    border-radius: 9px;
    background-color: #ffffff;
}

QRadioButton::indicator:hover {
    border-color: #adb5bd;
}

QRadioButton::indicator:checked {
    background-color: #007bff;
    border-color: #007bff;
    width: 10px;
    height: 10px;
    margin: 4px;
}

QRadioButton::indicator:checked:hover {
    background-color: #0056b3;
    border-color: #0056b3;
}

/* ===== SLIDERS ===== */
QSlider::groove:horizontal {
    height: 6px;
    background: #dee2e6;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #007bff;
    border: none;
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -5px 0;
}

QSlider::handle:horizontal:hover {
    background: #0056b3;
}

QSlider::groove:horizontal:disabled {
    background: #e9ecef;
}

QSlider::handle:horizontal:disabled {
    background: #adb5bd;
}

/* ===== SCROLL BAR ===== */
QScrollBar:vertical {
    background: #f8f9fa;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #dee2e6;
    min-height: 30px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #ced4da;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #f8f9fa;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background: #dee2e6;
    min-width: 30px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal:hover {
    background: #ced4da;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ===== LABELS ===== */
QLabel {
    color: #2c3e50;
    background: transparent;
}

QLabel[class="heading"] {
    font-size: 14px;
    font-weight: 700;
    color: #495057;
}

QLabel[class="subheading"] {
    font-size: 12px;
    font-weight: 600;
    color: #007bff;
}

QLabel[class="value"] {
    font-size: 11px;
    font-weight: 500;
    color: #28a745;
    font-family: 'Consolas', 'Monaco', monospace;
}

QLabel[class="info"] {
    color: #17a2b8;
}

QLabel[class="warning"] {
    color: #ffc107;
}

QLabel[class="error"] {
    color: #dc3545;
}

/* ===== TAB WIDGET ===== */
QTabWidget::pane {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
}

QTabBar::tab {
    background-color: #f8f9fa;
    color: #2c3e50;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #007bff;
    border-bottom: 2px solid #007bff;
}

QTabBar::tab:hover:!selected {
    background-color: #e9ecef;
}

/* ===== SPLITTER ===== */
QSplitter::handle {
    background-color: #dee2e6;
}

QSplitter::handle:hover {
    background-color: #007bff;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* ===== PROGRESS BAR ===== */
QProgressBar {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    text-align: center;
    color: #2c3e50;
}

QProgressBar::chunk {
    background-color: #007bff;
    border-radius: 3px;
}

/* ===== TOOL TIP ===== */
QToolTip {
    background-color: #343a40;
    color: #ffffff;
    border: 1px solid #495057;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}

/* ===== MENU BAR ===== */
QMenuBar {
    background-color: #f8f9fa;
    color: #2c3e50;
    border-bottom: 1px solid #dee2e6;
    padding: 4px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #e9ecef;
}

QMenuBar::item:pressed {
    background-color: #007bff;
    color: #ffffff;
}

/* ===== MENU ===== */
QMenu {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #e9ecef;
}

QMenu::item:disabled {
    color: #adb5bd;
}

QMenu::separator {
    height: 1px;
    background-color: #dee2e6;
    margin: 4px 8px;
}

/* ===== DOCK WIDGET ===== */
QDockWidget {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background-color: #f8f9fa;
    padding: 8px;
    border-bottom: 1px solid #dee2e6;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    font-weight: 600;
}

/* ===== STATUS BAR ===== */
QStatusBar {
    background-color: #f8f9fa;
    color: #2c3e50;
    border-top: 1px solid #dee2e6;
}

/* ===== SCROLL AREA ===== */
QScrollArea {
    background-color: #ffffff;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: #ffffff;
}

/* ===== FRAME ===== */
QFrame {
    background-color: #ffffff;
    border: none;
}

QFrame[class="card"] {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px;
}

QFrame[class="separator"] {
    background-color: #dee2e6;
    max-height: 1px;
    min-height: 1px;
}
"""
