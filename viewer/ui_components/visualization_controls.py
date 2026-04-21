"""
Visualization settings controls.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QCheckBox, QSlider, QComboBox, QGridLayout
)
from ..config import ConfigManager
from .collapsible_groupbox import CollapsibleGroupBox


class VisualizationControls(CollapsibleGroupBox):
    """Group for all visualization settings (performance, toggles, colormaps, export)"""

    def __init__(self, parent=None):
        super().__init__("Visualization")
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup visualization controls"""
        layout = QGridLayout()
        layout.setSpacing(5)
        layout.setColumnStretch(2, 1)  # Stretch last column

        # Row 0: Frame skip
        layout.addWidget(QLabel("Frame skip:"), 0, 0)
        self.frame_skip_input = QSpinBox()
        self.frame_skip_input.setRange(1, 100)
        self.frame_skip_input.setValue(1)
        self.frame_skip_input.setSingleStep(1)
        self.frame_skip_input.setSuffix("x")
        self.frame_skip_input.setMaximumWidth(80)
        layout.addWidget(self.frame_skip_input, 0, 1)
        self.apply_frame_skip_btn = QPushButton("Apply")
        self.apply_frame_skip_btn.setMaximumWidth(60)
        layout.addWidget(self.apply_frame_skip_btn, 0, 2)

        # Row 1: Target FPS
        layout.addWidget(QLabel("Target FPS:"), 1, 0)
        self.vis_fps_input = QSpinBox()
        self.vis_fps_input.setRange(10, 120)
        self.vis_fps_input.setValue(60)
        self.vis_fps_input.setSingleStep(5)
        self.vis_fps_input.setSuffix(" Hz")
        self.vis_fps_input.setMaximumWidth(80)
        layout.addWidget(self.vis_fps_input, 1, 1)
        self.apply_vis_fps_btn = QPushButton("Apply")
        self.apply_vis_fps_btn.setMaximumWidth(60)
        layout.addWidget(self.apply_vis_fps_btn, 1, 2)

        # Row 2: Display toggles - horizontal layout within grid cell
        display_toggle_row = QHBoxLayout()
        self.show_velocity_checkbox = QCheckBox("Velocity")
        self.show_velocity_checkbox.setChecked(True)
        self.show_vorticity_checkbox = QCheckBox("Vorticity")
        self.show_vorticity_checkbox.setChecked(True)
        self.show_sdf_checkbox = QCheckBox("SDF Mask")
        self.show_sdf_checkbox.setChecked(False)
        self.show_streamlines_checkbox = QCheckBox("Streamlines")
        self.show_streamlines_checkbox.setChecked(False)
        display_toggle_row.addWidget(self.show_velocity_checkbox)
        display_toggle_row.addWidget(self.show_vorticity_checkbox)
        display_toggle_row.addWidget(self.show_sdf_checkbox)
        display_toggle_row.addWidget(self.show_streamlines_checkbox)
        display_toggle_row.addStretch()
        layout.addLayout(display_toggle_row, 2, 0, 1, 3)  # Span all columns

        # Row 3: Color scale options - horizontal layout within grid cell
        colorscale_row = QHBoxLayout()
        self.log_colorscale_checkbox = QCheckBox("Log Color Scale")
        self.log_colorscale_checkbox.setChecked(True)
        self.spatial_colorscale_checkbox = QCheckBox("Spatial Weighting")
        self.spatial_colorscale_checkbox.setChecked(False)
        self.adaptive_colorscale_checkbox = QCheckBox("Adaptive Scale")
        self.adaptive_colorscale_checkbox.setChecked(True)
        self.adaptive_colorscale_checkbox.setToolTip("When enabled, color scales adjust automatically to data range. Disable to allow manual adjustment.")
        colorscale_row.addWidget(self.log_colorscale_checkbox)
        colorscale_row.addWidget(self.spatial_colorscale_checkbox)
        colorscale_row.addWidget(self.adaptive_colorscale_checkbox)
        colorscale_row.addStretch()
        layout.addLayout(colorscale_row, 3, 0, 1, 3)  # Span all columns

        # Row 4: Visualization smoothing
        layout.addWidget(QLabel("Smooth:"), 4, 0)
        self.upscale_slider = QSlider(Qt.Orientation.Horizontal)
        self.upscale_slider.setRange(1, 10)
        self.upscale_slider.setValue(1)
        self.upscale_slider.setMaximumWidth(120)
        layout.addWidget(self.upscale_slider, 4, 1)
        self.upscale_label = QLabel("1x")
        layout.addWidget(self.upscale_label, 4, 2)

        # Row 5: Velocity colormap
        layout.addWidget(QLabel("Velocity colormap:"), 5, 0)
        self.velocity_colormap_combo = QComboBox()
        self.velocity_colormap_combo.setMaximumWidth(150)
        self._populate_velocity_colormaps()
        layout.addWidget(self.velocity_colormap_combo, 5, 1, 1, 2)  # Span 2 columns

        # Row 6: Vorticity colormap
        layout.addWidget(QLabel("Vorticity colormap:"), 6, 0)
        self.vorticity_colormap_combo = QComboBox()
        self.vorticity_colormap_combo.setMaximumWidth(150)
        self._populate_vorticity_colormaps()
        layout.addWidget(self.vorticity_colormap_combo, 6, 1, 1, 2)  # Span 2 columns

        # Row 7: Pressure colormap
        layout.addWidget(QLabel("Pressure colormap:"), 7, 0)
        self.pressure_colormap_combo = QComboBox()
        self.pressure_colormap_combo.setMaximumWidth(150)
        self._populate_pressure_colormaps()
        layout.addWidget(self.pressure_colormap_combo, 7, 1, 1, 2)  # Span 2 columns

        # Row 8: Export buttons
        self.export_btn = QPushButton("Export Frame")
        self.export_btn.setMaximumWidth(100)
        layout.addWidget(self.export_btn, 8, 0)
        self.record_btn = QPushButton("Record")
        self.record_btn.setMaximumWidth(80)
        layout.addWidget(self.record_btn, 8, 1)
        self.save_btn = QPushButton("Save State")
        self.save_btn.setEnabled(False)
        self.save_btn.setMaximumWidth(90)
        layout.addWidget(self.save_btn, 8, 2)

        # Row 9: Auto-scale buttons
        layout.addWidget(QLabel("Auto-scale:"), 9, 0)
        self.autofit_velocity_btn = QPushButton("Velocity")
        self.autofit_velocity_btn.setMaximumWidth(70)
        layout.addWidget(self.autofit_velocity_btn, 9, 1)
        self.autofit_vorticity_btn = QPushButton("Vorticity")
        self.autofit_vorticity_btn.setMaximumWidth(70)
        layout.addWidget(self.autofit_vorticity_btn, 9, 2)
        self.autofit_both_btn = QPushButton("Both")
        self.autofit_both_btn.setMaximumWidth(50)
        layout.addWidget(self.autofit_both_btn, 9, 3)

        self.setLayout(layout)

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
        config = ConfigManager()
        # Default to RdBu for pressure (diverging colormap suitable for pressure)
        self.pressure_colormap_combo.setCurrentText('RdBu')
