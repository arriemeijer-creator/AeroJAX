"""
Floating control bar for quick access to frequently used controls.
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox,
                             QFrame, QSizePolicy, QCheckBox, QSlider)
from PyQt6.QtCore import Qt
from ..config import ConfigManager


class FloatingControlBar(QFrame):
    """Floating control bar with Start/Pause/Reset and colormap controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup floating control bar UI"""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QPushButton {
                font-size: 10px;
                padding: 2px 6px;
            }
            QComboBox {
                font-size: 10px;
                padding: 2px;
            }
            QLabel {
                font-size: 10px;
            }
        """)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(3)
        main_layout.setContentsMargins(5, 3, 5, 3)

        # Row 1: Start / Pause / Reset controls
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(5)

        # Start button
        self.start_btn = QPushButton("▶")
        self.start_btn.setMaximumWidth(30)
        self.start_btn.setToolTip("Start")
        row1_layout.addWidget(self.start_btn)

        # Pause button
        self.pause_btn = QPushButton("⏸")
        self.pause_btn.setMaximumWidth(30)
        self.pause_btn.setToolTip("Pause")
        row1_layout.addWidget(self.pause_btn)

        # Reset button
        self.reset_btn = QPushButton("↺")
        self.reset_btn.setMaximumWidth(30)
        self.reset_btn.setToolTip("Reset")
        row1_layout.addWidget(self.reset_btn)

        row1_layout.addStretch()
        main_layout.addLayout(row1_layout)

        # Row 2: Error metrics and airfoil metrics checkboxes
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(5)

        # Error metrics toggle
        self.error_metrics_cb = QCheckBox("Err")
        self.error_metrics_cb.setChecked(False)
        self.error_metrics_cb.setToolTip("Error Metrics")
        row2_layout.addWidget(self.error_metrics_cb)

        # Airfoil metrics toggle
        self.airfoil_metrics_cb = QCheckBox("Airfoil")
        self.airfoil_metrics_cb.setChecked(False)
        self.airfoil_metrics_cb.setToolTip("Airfoil Metrics")
        row2_layout.addWidget(self.airfoil_metrics_cb)

        row2_layout.addStretch()
        main_layout.addLayout(row2_layout)

        # Row 3: Colour scheme dropdowns
        row3_layout = QHBoxLayout()
        row3_layout.setSpacing(5)

        # Velocity colormap
        row3_layout.addWidget(QLabel("V:"))
        self.velocity_colormap_combo = QComboBox()
        self.velocity_colormap_combo.setMaximumWidth(80)
        self._populate_velocity_colormaps()
        row3_layout.addWidget(self.velocity_colormap_combo)

        # Vorticity colormap
        row3_layout.addWidget(QLabel("ω:"))
        self.vorticity_colormap_combo = QComboBox()
        self.vorticity_colormap_combo.setMaximumWidth(80)
        self._populate_vorticity_colormaps()
        row3_layout.addWidget(self.vorticity_colormap_combo)

        # Pressure colormap
        row3_layout.addWidget(QLabel("P:"))
        self.pressure_colormap_combo = QComboBox()
        self.pressure_colormap_combo.setMaximumWidth(80)
        self._populate_pressure_colormaps()
        row3_layout.addWidget(self.pressure_colormap_combo)

        row3_layout.addStretch()
        main_layout.addLayout(row3_layout)

        # Row 4: Dye injection controls
        row4_layout = QHBoxLayout()
        row4_layout.setSpacing(5)

        # Dye X slider
        row4_layout.addWidget(QLabel("Dye X:"))
        self.dye_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.dye_x_slider.setRange(0, 100)
        self.dye_x_slider.setValue(50)
        self.dye_x_slider.setMaximumWidth(80)
        row4_layout.addWidget(self.dye_x_slider)

        # Dye Y slider
        row4_layout.addWidget(QLabel("Y:"))
        self.dye_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.dye_y_slider.setRange(0, 100)
        self.dye_y_slider.setValue(50)
        self.dye_y_slider.setMaximumWidth(80)
        row4_layout.addWidget(self.dye_y_slider)

        # Dye injection button
        self.inject_dye_btn = QPushButton("💧")
        self.inject_dye_btn.setMaximumWidth(30)
        self.inject_dye_btn.setToolTip("Inject Dye")
        row4_layout.addWidget(self.inject_dye_btn)

        row4_layout.addStretch()
        main_layout.addLayout(row4_layout)

        self.setLayout(main_layout)

    def _populate_velocity_colormaps(self):
        """Populate velocity colormap dropdown"""
        velocity_colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
            'CET-C1', 'CET-C2', 'CET-C3', 'CET-C4', 'CET-C5', 'CET-C6', 'CET-C7',
            'CET-D1', 'CET-D2', 'CET-D3', 'CET-D4', 'CET-D6', 'CET-D7', 'CET-D8',
            'CET-D9', 'CET-D10', 'CET-D11', 'CET-D12', 'CET-D13',
            'CET-L1', 'CET-L2', 'CET-L3', 'CET-L4', 'CET-L5', 'CET-L6', 'CET-L7',
            'CET-L8', 'CET-L9', 'CET-L10', 'CET-L11', 'CET-L12', 'CET-L13', 'CET-L14',
            'CET-L15', 'CET-L16', 'CET-L17', 'CET-L18', 'CET-L19',
            'PAL-relaxed', 'PAL-relaxed_bright'
        ]
        self.velocity_colormap_combo.addItems(velocity_colormaps)
        config = ConfigManager()
        self.velocity_colormap_combo.setCurrentText(config.viz_config.default_velocity_colormap)

    def _populate_vorticity_colormaps(self):
        """Populate vorticity colormap dropdown"""
        vorticity_colormaps = [
            'CET-CBC1', 'coolwarm', 'RdBu', 'seismic', 'bwr', 'PiYG', 'PRGn', 'BrBG',
            'CET-CBC2', 'CET-CBD1', 'CET-CBL1', 'CET-CBL2',
            'CET-CBTC1', 'CET-CBTC2', 'CET-CBTD1', 'CET-CBTL1', 'CET-CBTL2',
            'CET-I1', 'CET-I2', 'CET-I3',
            'CET-R1', 'CET-R2', 'CET-R3', 'CET-R4',
            '--- Sequential (Magnitude) ---',
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'PAL-relaxed', 'PAL-relaxed_bright'
        ]
        self.vorticity_colormap_combo.addItems(vorticity_colormaps)
        config = ConfigManager()
        self.vorticity_colormap_combo.setCurrentText(config.viz_config.default_vorticity_colormap)

    def _populate_pressure_colormaps(self):
        """Populate pressure colormap dropdown"""
        pressure_colormaps = [
            'CET-CBC1', 'coolwarm', 'RdBu', 'seismic', 'bwr', 'PiYG', 'PRGn', 'BrBG',
            'CET-CBC2', 'CET-CBD1', 'CET-CBL1', 'CET-CBL2',
            'CET-CBTC1', 'CET-CBTC2', 'CET-CBTD1', 'CET-CBTL1', 'CET-CBTL2',
            'CET-I1', 'CET-I2', 'CET-I3',
            'CET-R1', 'CET-R2', 'CET-R3', 'CET-R4',
            '--- Sequential (Magnitude) ---',
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'PAL-relaxed', 'PAL-relaxed_bright'
        ]
        self.pressure_colormap_combo.addItems(pressure_colormaps)
        # Default to RdBu for pressure (diverging colormap suitable for pressure)
        self.pressure_colormap_combo.setCurrentText('RdBu')
