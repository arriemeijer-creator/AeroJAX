"""
Main control panel orchestrating all UI components.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QPushButton, QSpinBox, QComboBox, QDoubleSpinBox,
    QCheckBox, QSlider
)
from PyQt6.QtCore import Qt
from .top_console import TopConsole
from .obstacle_controls import ObstacleControls
from .time_controls import TimeControls
from .dye_controls import DyeControls
from .visualization_controls import VisualizationControls
from .collapsible_groupbox import CollapsibleGroupBox


class ControlPanel(QWidget):
    """Main control panel with top console and sidebar for advanced controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup the complete control panel UI: top console + scrollable sidebar"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # ========== TOP CONSOLE (horizontal bar with buttons only) ==========
        self.top_console = TopConsole(self)
        main_layout.addWidget(self.top_console)

        # ========== SCROLLABLE SIDEBAR (all groupboxes at same level) ==========
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Sidebar content widget
        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setSpacing(15)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        # ========== GRID SIZE GROUP ==========
        grid_group = CollapsibleGroupBox("Grid Size", start_collapsed=True)
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid:"))
        self.grid_x_spinbox = QSpinBox()
        self.grid_x_spinbox.setRange(64, 4096)
        self.grid_x_spinbox.setValue(512)
        self.grid_x_spinbox.setSingleStep(64)
        self.grid_x_spinbox.setMaximumWidth(90)
        grid_layout.addWidget(self.grid_x_spinbox)
        grid_layout.addWidget(QLabel("×"))
        self.grid_y_spinbox = QSpinBox()
        self.grid_y_spinbox.setRange(32, 2048)
        self.grid_y_spinbox.setValue(128)
        self.grid_y_spinbox.setSingleStep(32)
        self.grid_y_spinbox.setMaximumWidth(90)
        grid_layout.addWidget(self.grid_y_spinbox)
        self.apply_grid_btn = QPushButton("Apply")
        self.apply_grid_btn.setMaximumWidth(60)
        grid_layout.addWidget(self.apply_grid_btn)
        grid_layout.addStretch()
        grid_group.setLayout(grid_layout)
        sidebar_layout.addWidget(grid_group)

        # ========== REYNOLDS NUMBER GROUP ==========
        re_group = CollapsibleGroupBox("Reynolds Number", start_collapsed=True)
        from PyQt6.QtWidgets import QGridLayout
        re_layout = QGridLayout()
        re_layout.setSpacing(5)
        re_layout.setColumnStretch(3, 1)  # Stretch last column

        # Row 0: U input
        re_layout.addWidget(QLabel("U (m/s):"), 0, 0)
        self.u_input = QDoubleSpinBox()
        self.u_input.setRange(0.01, 100.0)
        self.u_input.setSingleStep(0.1)
        self.u_input.setValue(0.5)
        self.u_input.setMaximumWidth(110)
        re_layout.addWidget(self.u_input, 0, 1)
        self.lock_u_cb = QCheckBox("Lock")
        self.lock_u_cb.setChecked(False)
        re_layout.addWidget(self.lock_u_cb, 0, 2)

        # Row 1: ν input
        re_layout.addWidget(QLabel("ν (m²/s):"), 1, 0)
        self.nu_input = QDoubleSpinBox()
        self.nu_input.setRange(1e-6, 1.0)
        self.nu_input.setSingleStep(1e-4)
        self.nu_input.setDecimals(6)
        self.nu_input.setValue(0.001667)
        self.nu_input.setMaximumWidth(110)
        re_layout.addWidget(self.nu_input, 1, 1)
        self.lock_nu_cb = QCheckBox("Lock")
        self.lock_nu_cb.setChecked(True)
        re_layout.addWidget(self.lock_nu_cb, 1, 2)

        # Row 2: Re input
        re_layout.addWidget(QLabel("Re:"), 2, 0)
        self.re_input = QDoubleSpinBox()
        self.re_input.setRange(1.0, 100000.0)
        self.re_input.setSingleStep(1.0)
        self.re_input.setValue(2000.0)
        self.re_input.setMaximumWidth(110)
        re_layout.addWidget(self.re_input, 2, 1)
        self.lock_re_cb = QCheckBox("Lock")
        self.lock_re_cb.setChecked(True)
        re_layout.addWidget(self.lock_re_cb, 2, 2)

        # Row 3: Apply button
        self.apply_re_btn = QPushButton("Apply")
        self.apply_re_btn.setMaximumWidth(60)
        re_layout.addWidget(self.apply_re_btn, 3, 0, 1, 2)  # Span 2 columns

        re_group.setLayout(re_layout)
        sidebar_layout.addWidget(re_group)

        # ========== FLOW TYPE GROUP ==========
        flow_group = CollapsibleGroupBox("Flow Type", start_collapsed=True)
        flow_layout = QHBoxLayout()
        flow_layout.addWidget(QLabel("Flow:"))
        self.flow_combo = QComboBox()
        self.flow_combo.addItems(["von_karman", "taylor_green"])
        self.flow_combo.setMaximumWidth(150)
        flow_layout.addWidget(self.flow_combo)
        flow_layout.addStretch()
        flow_group.setLayout(flow_layout)
        sidebar_layout.addWidget(flow_group)

        # ========== PRECISION GROUP ==========
        precision_group = CollapsibleGroupBox("Precision", start_collapsed=True)
        precision_layout = QHBoxLayout()
        precision_layout.addWidget(QLabel("Precision:"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["float32", "float64"])
        self.precision_combo.setMaximumWidth(100)
        precision_layout.addWidget(self.precision_combo)
        self.apply_precision_btn = QPushButton("Apply")
        self.apply_precision_btn.setMaximumWidth(60)
        precision_layout.addWidget(self.apply_precision_btn)
        precision_layout.addStretch()
        precision_group.setLayout(precision_layout)
        sidebar_layout.addWidget(precision_group)

        # ========== SOLVER PARAMETERS GROUP ==========
        solver_group = CollapsibleGroupBox("Solver Parameters", start_collapsed=True)
        solver_layout = QGridLayout()
        solver_layout.setSpacing(5)
        solver_layout.setColumnStretch(4, 1)  # Stretch last column

        # Row 0: MG V-cycles
        solver_layout.addWidget(QLabel("MG V-cycles:"), 0, 0)
        self.vcycles_slider = QSlider(Qt.Orientation.Horizontal)
        self.vcycles_slider.setRange(1, 10)
        self.vcycles_slider.setValue(7)
        self.vcycles_slider.setMaximumWidth(100)
        solver_layout.addWidget(self.vcycles_slider, 0, 1)
        self.vcycles_label = QLabel("7")
        self.vcycles_label.setMinimumWidth(20)
        solver_layout.addWidget(self.vcycles_label, 0, 2)
        self.apply_vcycles_btn = QPushButton("Apply")
        self.apply_vcycles_btn.setMaximumWidth(50)
        solver_layout.addWidget(self.apply_vcycles_btn, 0, 3)
        self.vcycles_slider.valueChanged.connect(self._update_vcycles_label)

        # Row 1: Hyper ν
        solver_layout.addWidget(QLabel("Hyper ν:"), 1, 0)
        self.hyper_viscosity_slider = QSlider(Qt.Orientation.Horizontal)
        self.hyper_viscosity_slider.setRange(0, 100)
        self.hyper_viscosity_slider.setValue(0)
        self.hyper_viscosity_slider.setMaximumWidth(100)
        solver_layout.addWidget(self.hyper_viscosity_slider, 1, 1)
        self.hyper_viscosity_spinbox = QDoubleSpinBox()
        self.hyper_viscosity_spinbox.setRange(0.0, 0.05)
        self.hyper_viscosity_spinbox.setSingleStep(0.0001)
        self.hyper_viscosity_spinbox.setDecimals(4)
        self.hyper_viscosity_spinbox.setValue(0.0)
        self.hyper_viscosity_spinbox.setMaximumWidth(80)
        solver_layout.addWidget(self.hyper_viscosity_spinbox, 1, 2)
        self.apply_hyper_viscosity_btn = QPushButton("Apply")
        self.apply_hyper_viscosity_btn.setMaximumWidth(50)
        solver_layout.addWidget(self.apply_hyper_viscosity_btn, 1, 3)
        self.hyper_viscosity_slider.valueChanged.connect(self._update_hyper_viscosity_label)
        self.hyper_viscosity_spinbox.valueChanged.connect(self._update_hyper_viscosity_spinbox)

        # Row 2: Fast mode (RK2)
        solver_layout.addWidget(QLabel("Fast Mode (RK2):"), 2, 0)
        self.fast_mode_checkbox = QCheckBox("Enable")
        self.fast_mode_checkbox.setChecked(False)
        self.fast_mode_checkbox.setToolTip("RK2 for speed (Real-Time Interaction)\nRK3 for accuracy")
        solver_layout.addWidget(self.fast_mode_checkbox, 2, 1, 1, 2)

        # Row 3: LES controls
        solver_layout.addWidget(QLabel("LES:"), 3, 0)
        self.les_checkbox = QCheckBox("Enable")
        self.les_checkbox.setChecked(False)
        self.les_checkbox.stateChanged.connect(self._on_les_checkbox_changed)
        solver_layout.addWidget(self.les_checkbox, 3, 1)
        self.les_model_combo = QComboBox()
        self.les_model_combo.addItems(["dynamic_smagorinsky", "smagorinsky"])
        self.les_model_combo.setMaximumWidth(150)
        self.les_model_combo.setEnabled(False)
        solver_layout.addWidget(self.les_model_combo, 3, 2)
        self.apply_les_btn = QPushButton("Apply")
        self.apply_les_btn.setMaximumWidth(50)
        self.apply_les_btn.setEnabled(False)
        solver_layout.addWidget(self.apply_les_btn, 3, 3)

        solver_group.setLayout(solver_layout)
        sidebar_layout.addWidget(solver_group)

        # ========== BOUNDARY CONDITIONS GROUP ==========
        boundary_group = CollapsibleGroupBox("Boundary Conditions", start_collapsed=True)
        boundary_layout = QVBoxLayout()
        slip_row = QHBoxLayout()
        self.slip_walls_checkbox = QCheckBox("Slip Walls")
        self.slip_walls_checkbox.setChecked(True)
        slip_row.addWidget(self.slip_walls_checkbox)
        slip_row.addStretch()
        boundary_layout.addLayout(slip_row)
        epsilon_row = QHBoxLayout()
        epsilon_row.addWidget(QLabel("Mask ε:"))
        self.epsilon_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_slider.setRange(1, 100)
        self.epsilon_slider.setValue(1)
        self.epsilon_slider.setMaximumWidth(100)
        epsilon_row.addWidget(self.epsilon_slider)
        self.epsilon_label = QLabel("0.01")
        self.epsilon_label.setMinimumWidth(35)
        epsilon_row.addWidget(self.epsilon_label)
        self.apply_epsilon_btn = QPushButton("Apply")
        self.apply_epsilon_btn.setMaximumWidth(50)
        epsilon_row.addWidget(self.apply_epsilon_btn)
        epsilon_row.addStretch()
        boundary_layout.addLayout(epsilon_row)
        self.epsilon_slider.valueChanged.connect(self._update_epsilon_label)
        boundary_group.setLayout(boundary_layout)
        sidebar_layout.addWidget(boundary_group)

        # ========== SIMULATION INFO GROUP ==========
        info_group = CollapsibleGroupBox("Simulation Info", start_collapsed=True)
        info_layout = QVBoxLayout()
        solver_row = QHBoxLayout()
        self.solver_status_label = QLabel("Solver: Not initialized")
        self.solver_status_label.setMinimumWidth(150)
        solver_row.addWidget(self.solver_status_label)
        solver_row.addStretch()
        info_layout.addLayout(solver_row)
        fps_row = QHBoxLayout()
        self.sim_fps_label = QLabel("Sim FPS: 0")
        self.sim_fps_label.setMinimumWidth(80)
        fps_row.addWidget(self.sim_fps_label)
        self.viz_fps_label = QLabel("Vis FPS: 0")
        self.viz_fps_label.setMinimumWidth(80)
        fps_row.addWidget(self.viz_fps_label)
        fps_row.addStretch()
        info_layout.addLayout(fps_row)
        time_row = QHBoxLayout()
        self.sim_time_label = QLabel("Time: 0.000")
        self.sim_time_label.setMinimumWidth(100)
        time_row.addWidget(self.sim_time_label)
        self.dt_label = QLabel("dt: 0.0000")
        self.dt_label.setMinimumWidth(80)
        time_row.addWidget(self.dt_label)
        time_row.addStretch()
        info_layout.addLayout(time_row)
        div_row = QHBoxLayout()
        self.max_div_label = QLabel("RMS Divergence: 0.000")
        self.max_div_label.setMinimumWidth(150)
        div_row.addWidget(self.max_div_label)
        div_row.addStretch()
        info_layout.addLayout(div_row)
        info_group.setLayout(info_layout)
        sidebar_layout.addWidget(info_group)

        # ========== OBSTACLE CONTROLS ==========
        self.obstacle_controls = ObstacleControls(self)
        sidebar_layout.addWidget(self.obstacle_controls)

        # ========== VISUALIZATION CONTROLS ==========
        self.visualization_controls = VisualizationControls(self)
        sidebar_layout.addWidget(self.visualization_controls)

        # ========== DYE INJECTION CONTROLS ==========
        self.dye_controls = DyeControls(self)
        sidebar_layout.addWidget(self.dye_controls)

        # ========== TIME STEPPING CONTROLS ==========
        self.time_controls = TimeControls(self)
        sidebar_layout.addWidget(self.time_controls)

        sidebar_layout.addStretch()

        # Set sidebar content as scroll area widget
        scroll_area.setWidget(sidebar_content)
        main_layout.addWidget(scroll_area, 1)

        self.setLayout(main_layout)

    def _update_epsilon_label(self):
        """Update epsilon label when slider changes."""
        value = self.epsilon_slider.value() / 100.0
        self.epsilon_label.setText(f"{value:.2f}")

    def _update_hyper_viscosity_label(self):
        """Update hyperviscosity label when slider changes."""
        value = self.hyper_viscosity_slider.value()
        percentage = value  # 0-100%
        ratio = percentage / 100.0 * 0.05  # Convert to 0.0-0.05 range
        self.hyper_viscosity_label.setText(f"{percentage}%")
        self.hyper_viscosity_spinbox.blockSignals(True)
        self.hyper_viscosity_spinbox.setValue(ratio)
        self.hyper_viscosity_spinbox.blockSignals(False)

    def _update_hyper_viscosity_spinbox(self):
        """Update hyperviscosity slider when spinbox changes."""
        value = self.hyper_viscosity_spinbox.value()
        percentage = int((value / 0.05) * 100)  # Convert to 0-100% range
        self.hyper_viscosity_slider.blockSignals(True)
        self.hyper_viscosity_slider.setValue(percentage)
        self.hyper_viscosity_slider.blockSignals(False)
        self.hyper_viscosity_label.setText(f"{percentage}%")

    def _update_vcycles_label(self):
        """Update V-cycles label when slider changes."""
        value = self.vcycles_slider.value()
        self.vcycles_label.setText(str(value))

    def _on_les_checkbox_changed(self, state):
        """Enable/disable LES controls when checkbox is toggled"""
        is_checked = state == 2  # Qt.CheckState.Checked
        self.les_model_combo.setEnabled(is_checked)
        self.apply_les_btn.setEnabled(is_checked)

    def _on_slip_walls_changed(self, state):
        """Handle slip walls checkbox state change."""
        is_slip = (state == 2)  # Qt.CheckState.Checked
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_wall_boundary_condition'):
                self.parent_viewer.apply_wall_boundary_condition(is_slip)

    # Expose child component attributes for backward compatibility
    @property
    def start_btn(self):
        return self.top_console.start_btn

    @property
    def pause_btn(self):
        return self.top_console.pause_btn

    @property
    def reset_btn(self):
        return self.top_console.reset_btn

    @property
    def theme_toggle_btn(self):
        return self.top_console.theme_toggle_btn

    # Obstacle controls
    @property
    def naca_combo(self):
        return self.obstacle_controls.naca_combo

    @property
    def chord_spinbox(self):
        return self.obstacle_controls.chord_spinbox

    @property
    def angle_spinbox(self):
        return self.obstacle_controls.angle_spinbox

    @property
    def angle_slider(self):
        return self.obstacle_controls.angle_slider

    @property
    def apply_naca_btn(self):
        return self.obstacle_controls.apply_naca_btn

    @property
    def cylinder_radius_spinbox(self):
        return self.obstacle_controls.cylinder_radius_spinbox

    @property
    def apply_cylinder_btn(self):
        return self.obstacle_controls.apply_cylinder_btn

    @property
    def cylinder_diameter_spinbox(self):
        return self.obstacle_controls.cylinder_diameter_spinbox

    @property
    def cylinder_spacing_spinbox(self):
        return self.obstacle_controls.cylinder_spacing_spinbox

    @property
    def apply_cylinder_array_btn(self):
        return self.obstacle_controls.apply_cylinder_array_btn

    @property
    def x_position_slider(self):
        return self.obstacle_controls.x_position_slider

    @property
    def x_position_label(self):
        return self.obstacle_controls.x_position_label

    @property
    def y_position_slider(self):
        return self.obstacle_controls.y_position_slider

    @property
    def y_position_label(self):
        return self.obstacle_controls.y_position_label

    @property
    def cylinder_radio(self):
        return self.obstacle_controls.cylinder_radio

    @property
    def naca_radio(self):
        return self.obstacle_controls.naca_radio

    @property
    def cow_radio(self):
        return self.obstacle_controls.cow_radio

    @property
    def cylinder_array_radio(self):
        return self.obstacle_controls.cylinder_array_radio

    @property
    def obstacle_button_group(self):
        return self.obstacle_controls.obstacle_button_group

    @property
    def draw_custom_btn(self):
        return self.obstacle_controls.draw_custom_btn

    # Time controls
    @property
    def dt_spinbox(self):
        return self.time_controls.dt_spinbox

    @property
    def apply_dt_btn(self):
        return self.time_controls.apply_dt_btn

    @property
    def adaptive_dt_checkbox(self):
        return self.time_controls.adaptive_dt_checkbox

    @property
    def cfl_label(self):
        return self.time_controls.cfl_label

    # Visualization controls
    @property
    def frame_skip_input(self):
        return self.visualization_controls.frame_skip_input

    @property
    def apply_frame_skip_btn(self):
        return self.visualization_controls.apply_frame_skip_btn

    @property
    def vis_fps_input(self):
        return self.visualization_controls.vis_fps_input

    @property
    def apply_vis_fps_btn(self):
        return self.visualization_controls.apply_vis_fps_btn

    @property
    def show_velocity_checkbox(self):
        return self.visualization_controls.show_velocity_checkbox

    @property
    def show_vorticity_checkbox(self):
        return self.visualization_controls.show_vorticity_checkbox

    @property
    def show_sdf_checkbox(self):
        return self.visualization_controls.show_sdf_checkbox

    @property
    def show_streamlines_checkbox(self):
        return self.visualization_controls.show_streamlines_checkbox

    @property
    def log_colorscale_checkbox(self):
        return self.visualization_controls.log_colorscale_checkbox

    @property
    def spatial_colorscale_checkbox(self):
        return self.visualization_controls.spatial_colorscale_checkbox

    @property
    def adaptive_colorscale_checkbox(self):
        return self.visualization_controls.adaptive_colorscale_checkbox

    @property
    def upscale_slider(self):
        return self.visualization_controls.upscale_slider

    @property
    def upscale_label(self):
        return self.visualization_controls.upscale_label

    @property
    def velocity_colormap_combo(self):
        return self.visualization_controls.velocity_colormap_combo

    @property
    def vorticity_colormap_combo(self):
        return self.visualization_controls.vorticity_colormap_combo

    @property
    def pressure_colormap_combo(self):
        return self.visualization_controls.pressure_colormap_combo

    @property
    def export_btn(self):
        return self.visualization_controls.export_btn

    @property
    def record_btn(self):
        return self.visualization_controls.record_btn

    @property
    def save_btn(self):
        return self.visualization_controls.save_btn

    @property
    def autofit_velocity_btn(self):
        return self.visualization_controls.autofit_velocity_btn

    @property
    def autofit_vorticity_btn(self):
        return self.visualization_controls.autofit_vorticity_btn

    @property
    def autofit_both_btn(self):
        return self.visualization_controls.autofit_both_btn

    # Dye controls
    @property
    def dye_x_input(self):
        return self.dye_controls.dye_x_input

    @property
    def dye_y_input(self):
        return self.dye_controls.dye_y_input

    @property
    def dye_amount_slider(self):
        return self.dye_controls.dye_amount_slider

    @property
    def dye_amount_label(self):
        return self.dye_controls.dye_amount_label

    @property
    def inject_dye_btn(self):
        return self.dye_controls.inject_dye_btn

    @property
    def dye_x_slider(self):
        return self.dye_controls.dye_x_slider

    @property
    def dye_y_slider(self):
        return self.dye_controls.dye_y_slider

    # Methods for backward compatibility
    def set_chord_range_for_domain(self, max_chord: float):
        """Update chord spinbox range based on domain size"""
        self.obstacle_controls.set_chord_range_for_domain(max_chord)

    def show_naca_controls(self, show: bool) -> None:
        """Show/hide NACA controls based on obstacle selection"""
        self.obstacle_controls.show_naca_controls(show)

    def _on_les_checkbox_changed(self, state):
        """Enable/disable LES controls when checkbox is toggled"""
        is_checked = state == 2  # Qt.CheckState.Checked
        self.les_model_combo.setEnabled(is_checked)
        self.apply_les_btn.setEnabled(is_checked)

    def _on_obstacle_radio_changed(self, button):
        """Handle obstacle type radio button selection changes"""
        self.obstacle_controls._on_obstacle_radio_changed(button)

    def _on_x_position_changed(self, value):
        """Handle x-position slider changes"""
        self.obstacle_controls._on_x_position_changed(value)

    def _on_y_position_changed(self, value):
        """Handle y-position slider changes"""
        self.obstacle_controls._on_y_position_changed(value)

    def _on_naca_hover(self, index):
        """Show airfoil preview when selection changes"""
        self.obstacle_controls._on_naca_hover(index)


    def _check_naca_availability(self):
        """Check if NACA airfoils are available"""
        return self.obstacle_controls._check_naca_availability()

    def _populate_velocity_colormaps(self):
        """Populate velocity colormap dropdown"""
        self.visualization_controls._populate_velocity_colormaps()

    def _populate_vorticity_colormaps(self):
        """Populate vorticity colormap dropdown"""
        self.visualization_controls._populate_vorticity_colormaps()

    def on_obstacle_type_selected(self, obstacle_type: str) -> None:
        """Delegate obstacle type selection to parent viewer (FlowManager)"""
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_obstacle_type_selected'):
                self.parent_viewer.on_obstacle_type_selected(obstacle_type)
