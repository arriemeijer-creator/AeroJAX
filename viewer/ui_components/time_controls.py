"""
Time stepping and LES controls.
"""

from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QCheckBox, QComboBox
)
from .collapsible_groupbox import CollapsibleGroupBox


class TimeControls(CollapsibleGroupBox):
    """Group for time step controls, CFL, and LES"""

    def __init__(self, parent=None):
        super().__init__("Time Stepping")
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup time stepping controls"""
        layout = QHBoxLayout()
        layout.setSpacing(8)

        layout.addWidget(QLabel("dt:"))
        self.dt_spinbox = QDoubleSpinBox()
        self.dt_spinbox.setRange(0.0001, 0.01)
        self.dt_spinbox.setDecimals(4)
        self.dt_spinbox.setSingleStep(0.0001)
        self.dt_spinbox.setValue(0.01)  # Match default dt from config
        self.dt_spinbox.setMaximumWidth(150)
        layout.addWidget(self.dt_spinbox)

        self.apply_dt_btn = QPushButton("Apply")
        self.apply_dt_btn.setMaximumWidth(60)
        layout.addWidget(self.apply_dt_btn)

        self.adaptive_dt_checkbox = QCheckBox("Adaptive")
        self.adaptive_dt_checkbox.setChecked(False)  # Disabled by default to prevent instability
        layout.addWidget(self.adaptive_dt_checkbox)

        self.cfl_label = QLabel("CFL: 0.00")
        self.cfl_label.setMinimumWidth(80)
        layout.addWidget(self.cfl_label)

        layout.addStretch()
        self.setLayout(layout)
