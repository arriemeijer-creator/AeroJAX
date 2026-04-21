"""
Dye injection controls.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSlider
)
from .collapsible_groupbox import CollapsibleGroupBox


class DyeControls(CollapsibleGroupBox):
    """Group for dye injection controls"""

    def __init__(self, parent=None):
        super().__init__("Dye Injection")
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup dye injection controls"""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # X position
        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("X position:"))
        self.dye_x_input = QDoubleSpinBox()
        self.dye_x_input.setRange(0.0, 20.0)
        self.dye_x_input.setValue(2.0)
        self.dye_x_input.setSingleStep(0.5)
        self.dye_x_input.setMaximumWidth(100)
        row_x.addWidget(self.dye_x_input)
        row_x.addStretch()
        layout.addLayout(row_x)

        # Y position
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Y position:"))
        self.dye_y_input = QDoubleSpinBox()
        self.dye_y_input.setRange(0.0, 3.8)
        self.dye_y_input.setValue(1.9)
        self.dye_y_input.setSingleStep(0.5)
        self.dye_y_input.setMaximumWidth(100)
        row_y.addWidget(self.dye_y_input)
        row_y.addStretch()
        layout.addLayout(row_y)

        # Dye amount slider
        row_amount = QHBoxLayout()
        row_amount.addWidget(QLabel("Amount:"))
        self.dye_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self.dye_amount_slider.setRange(0, 100)
        self.dye_amount_slider.setValue(50)
        self.dye_amount_slider.setMaximumWidth(150)
        row_amount.addWidget(self.dye_amount_slider)
        self.dye_amount_label = QLabel("0.50")
        self.dye_amount_label.setMaximumWidth(40)
        row_amount.addWidget(self.dye_amount_label)
        row_amount.addStretch()
        layout.addLayout(row_amount)

        # Connect slider to label update
        self.dye_amount_slider.valueChanged.connect(
            lambda v: self.dye_amount_label.setText(f"{v/100:.2f}")
        )

        # Dye position sliders (for marker position updates)
        row_sliders = QHBoxLayout()
        row_sliders.addWidget(QLabel("Marker X:"))
        self.dye_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.dye_x_slider.setRange(0, 100)
        self.dye_x_slider.setValue(50)
        self.dye_x_slider.setMaximumWidth(150)
        row_sliders.addWidget(self.dye_x_slider)

        row_sliders.addWidget(QLabel("Y:"))
        self.dye_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.dye_y_slider.setRange(0, 100)
        self.dye_y_slider.setValue(50)
        self.dye_y_slider.setMaximumWidth(150)
        row_sliders.addWidget(self.dye_y_slider)
        row_sliders.addStretch()
        layout.addLayout(row_sliders)

        # Inject button
        row_btn = QHBoxLayout()
        self.inject_dye_btn = QPushButton("💧 Inject Dye")
        self.inject_dye_btn.setMaximumWidth(120)
        row_btn.addWidget(self.inject_dye_btn)
        row_btn.addStretch()
        layout.addLayout(row_btn)

        self.setLayout(layout)
