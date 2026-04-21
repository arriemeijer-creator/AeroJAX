"""
Top console toolbar for core simulation controls.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QDoubleSpinBox, QCheckBox, QSlider
)
from .collapsible_groupbox import CollapsibleGroupBox


class TopConsole(QFrame):
    """Top horizontal toolbar with core simulation controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Create the top horizontal toolbar with core simulation controls (buttons only)"""
        self.setFrameShape(QFrame.Shape.StyledPanel)

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 5, 10, 5)

        # Simulation Controls - always visible (not collapsible)
        self.start_btn = QPushButton("▶ Start")
        self.pause_btn = QPushButton("⏸ Pause")
        self.reset_btn = QPushButton("↺ Reset")
        self.theme_toggle_btn = QPushButton("☀️")
        self.theme_toggle_btn.setToolTip("Toggle Light/Dark Mode")
        self.theme_toggle_btn.setMaximumWidth(40)
        for btn in (self.start_btn, self.pause_btn, self.reset_btn):
            btn.setMinimumWidth(70)
        main_layout.addWidget(self.start_btn)
        main_layout.addWidget(self.pause_btn)
        main_layout.addWidget(self.reset_btn)
        main_layout.addWidget(self.theme_toggle_btn)
        main_layout.addStretch()
