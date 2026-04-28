"""
Top console toolbar for core simulation controls.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QDoubleSpinBox, QCheckBox, QSlider, QGroupBox
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

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(10, 5, 10, 5)

        # Simulation Controls Group
        sim_group = QGroupBox("Simulation Controls")
        sim_layout = QHBoxLayout()
        sim_layout.setSpacing(10)

        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        for btn in (self.start_btn, self.pause_btn, self.reset_btn):
            btn.setMinimumWidth(70)
        sim_layout.addWidget(self.start_btn)
        sim_layout.addWidget(self.pause_btn)
        sim_layout.addWidget(self.reset_btn)
        sim_layout.addStretch()
        sim_group.setLayout(sim_layout)

        # Tools Group
        tools_group = QGroupBox("Tools")
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(10)

        self.inverse_design_btn = QPushButton("Inverse Design")
        self.thermal_btn = QPushButton("Thermal")
        self.theme_toggle_btn = QPushButton("☀️")
        self.theme_toggle_btn.setToolTip("Toggle Light/Dark Mode")
        self.theme_toggle_btn.setMaximumWidth(40)
        self.inverse_design_btn.setMinimumWidth(120)
        self.thermal_btn.setMinimumWidth(100)
        tools_layout.addWidget(self.inverse_design_btn)
        tools_layout.addWidget(self.thermal_btn)
        tools_layout.addWidget(self.theme_toggle_btn)
        tools_layout.addStretch()
        tools_group.setLayout(tools_layout)

        main_layout.addWidget(sim_group)
        main_layout.addWidget(tools_group)
