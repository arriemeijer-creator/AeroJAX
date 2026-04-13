import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox, QCheckBox, QSlider,
    QGroupBox, QScrollArea, QFrame, QApplication
)
from .config import ConfigManager


class ControlPanel(QWidget):
    """Main control panel with top console and sidebar for advanced controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup the complete control panel UI: top console + sidebar"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # ========== TOP CONSOLE (horizontal bar) ==========
        top_console = self._create_top_console()
        main_layout.addWidget(top_console)

        # ========== SIDEBAR (non-scrollable area for all other controls) ==========
        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setSpacing(15)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        # Add all control groups
        sidebar_layout.addWidget(self._create_numerical_group())
        sidebar_layout.addWidget(self._create_obstacle_group())      # includes NACA

        # Visualization and Dye Injection side by side
        viz_dye_row = QHBoxLayout()
        viz_dye_row.addWidget(self._create_visualization_group())
        viz_dye_row.addWidget(self._create_dye_group())
        sidebar_layout.addLayout(viz_dye_row)

        sidebar_layout.addWidget(self._create_time_group())
        sidebar_layout.addStretch()

        main_layout.addWidget(sidebar_content, 1)  # stretch factor 1

        self.setLayout(main_layout)

    def _create_top_console(self):
        """Create the top horizontal toolbar with core controls"""
        console = QFrame()
        console.setFrameShape(QFrame.Shape.StyledPanel)
        console.setStyleSheet("QFrame { background-color: #2d2d2d; border-radius: 4px; } QLabel { color: white; } QPushButton { color: white; background-color: #404040; border: 1px solid #606060; border-radius: 3px; padding: 2px 6px; } QPushButton:hover { background-color: #505050; } QComboBox { color: white; background-color: #404040; border: 1px solid #606060; border-radius: 3px; padding: 2px 6px; } QSpinBox { color: white; background-color: #404040; border: 1px solid #606060; border-radius: 3px; padding: 2px 6px; } QCheckBox { color: white; } QDoubleSpinBox { color: white; background-color: #404040; border: 1px solid #606060; border-radius: 3px; padding: 2px 6px; }")
        
        # Main vertical layout
        main_layout = QVBoxLayout(console)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(10, 5, 10, 5)
        
        # Top row: start/stop/reset buttons | grid size
        top_row = QHBoxLayout()
        
        # --- Simulation buttons (left) ---
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        for btn in (self.start_btn, self.pause_btn, self.reset_btn):
            btn.setMinimumWidth(60)
        top_row.addWidget(self.start_btn)
        top_row.addWidget(self.pause_btn)
        top_row.addWidget(self.reset_btn)
        top_row.addStretch()
        
        # --- Grid size (right) ---
        top_row.addWidget(QLabel("Grid:"))
        self.grid_x_spinbox = QSpinBox()
        self.grid_x_spinbox.setRange(64, 4096)
        self.grid_x_spinbox.setValue(512)
        self.grid_x_spinbox.setSingleStep(64)
        self.grid_x_spinbox.setMaximumWidth(90)
        top_row.addWidget(self.grid_x_spinbox)

        top_row.addWidget(QLabel("×"))
        self.grid_y_spinbox = QSpinBox()
        self.grid_y_spinbox.setRange(32, 2048)
        self.grid_y_spinbox.setValue(96)
        self.grid_y_spinbox.setSingleStep(32)
        self.grid_y_spinbox.setMaximumWidth(90)
        top_row.addWidget(self.grid_y_spinbox)

        self.apply_grid_btn = QPushButton("Apply")
        self.apply_grid_btn.setMaximumWidth(60)
        top_row.addWidget(self.apply_grid_btn)
        
        # Bottom row: Reynolds number | flow type
        bottom_row = QHBoxLayout()
        
        # --- Flow parameters with constraint locks ---
        # Velocity (U)
        bottom_row.addWidget(QLabel("U (m/s):"))
        self.u_input = QDoubleSpinBox()
        self.u_input.setRange(0.01, 100.0)
        self.u_input.setSingleStep(0.1)
        self.u_input.setValue(0.5)
        self.u_input.setMaximumWidth(110)
        bottom_row.addWidget(self.u_input)
        self.lock_u_cb = QCheckBox("Lock")
        self.lock_u_cb.setChecked(False)
        self.lock_u_cb.setToolTip("Lock velocity (U) - will not be derived")
        bottom_row.addWidget(self.lock_u_cb)
        
        bottom_row.addWidget(QLabel("|"))
        
        # Viscosity (nu)
        bottom_row.addWidget(QLabel("ν (m²/s):"))
        self.nu_input = QDoubleSpinBox()
        self.nu_input.setRange(1e-6, 1.0)
        self.nu_input.setSingleStep(1e-4)
        self.nu_input.setDecimals(6)
        self.nu_input.setValue(0.001667)
        self.nu_input.setMaximumWidth(110)
        bottom_row.addWidget(self.nu_input)
        self.lock_nu_cb = QCheckBox("Lock")
        self.lock_nu_cb.setChecked(True)
        self.lock_nu_cb.setToolTip("Lock viscosity (ν) - will not be derived")
        bottom_row.addWidget(self.lock_nu_cb)
        
        bottom_row.addWidget(QLabel("|"))

        # Reynolds number (derived)
        bottom_row.addWidget(QLabel("Re:"))
        self.re_input = QDoubleSpinBox()
        self.re_input.setRange(1.0, 100000.0)
        self.re_input.setSingleStep(1.0)
        self.re_input.setValue(3000.0)
        self.re_input.setMaximumWidth(110)
        bottom_row.addWidget(self.re_input)
        self.lock_re_cb = QCheckBox("Lock")
        self.lock_re_cb.setChecked(True)
        self.lock_re_cb.setToolTip("Lock Reynolds number (Re) - will not be derived")
        bottom_row.addWidget(self.lock_re_cb)

        self.apply_re_btn = QPushButton("Apply")
        self.apply_re_btn.setMaximumWidth(60)
        bottom_row.addWidget(self.apply_re_btn)

        bottom_row.addStretch()
        main_layout.addLayout(top_row)
        main_layout.addLayout(bottom_row)

        # Third row: Flow type | Precision
        third_row = QHBoxLayout()

        # Flow type
        third_row.addWidget(QLabel("Flow:"))
        self.flow_combo = QComboBox()
        self.flow_combo.addItem("von_karman")
        self.flow_combo.addItem("lid_driven_cavity")
        self.flow_combo.setMaximumWidth(150)
        third_row.addWidget(self.flow_combo)

        third_row.addWidget(QLabel("|"))

        # Precision
        third_row.addWidget(QLabel("Precision:"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["float32", "float64"])
        self.precision_combo.setMaximumWidth(100)
        third_row.addWidget(self.precision_combo)

        self.apply_precision_btn = QPushButton("Apply")
        self.apply_precision_btn.setMaximumWidth(60)
        third_row.addWidget(self.apply_precision_btn)

        third_row.addStretch()
        main_layout.addLayout(third_row)

        # Fourth row: Simulation information | Cylinder mask
        fourth_row = QHBoxLayout()

        # --- Simulation information (left) ---
        self.solver_status_label = QLabel("Solver: Not initialized")
        self.solver_status_label.setMinimumWidth(150)
        fourth_row.addWidget(self.solver_status_label)

        fourth_row.addWidget(QLabel("|"))

        self.sim_fps_label = QLabel("Sim FPS: 0")
        self.sim_fps_label.setMinimumWidth(80)
        fourth_row.addWidget(self.sim_fps_label)

        self.viz_fps_label = QLabel("Vis FPS: 0")
        self.viz_fps_label.setMinimumWidth(80)
        fourth_row.addWidget(self.viz_fps_label)

        fourth_row.addSpacing(20)

        # --- Cylinder mask (right) ---
        fourth_row.addWidget(QLabel("Radius:"))
        self.cylinder_radius_spinbox = QDoubleSpinBox()
        self.cylinder_radius_spinbox.setRange(0.05, 2.0)
        self.cylinder_radius_spinbox.setValue(0.18)
        self.cylinder_radius_spinbox.setDecimals(3)
        self.cylinder_radius_spinbox.setSingleStep(0.01)
        self.cylinder_radius_spinbox.setMaximumWidth(80)
        fourth_row.addWidget(self.cylinder_radius_spinbox)

        self.apply_cylinder_btn = QPushButton("Apply")
        self.apply_cylinder_btn.setMaximumWidth(50)
        fourth_row.addWidget(self.apply_cylinder_btn)

        fourth_row.addWidget(QLabel("|"))

        fourth_row.addWidget(QLabel("Mask ε:"))
        self.epsilon_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_slider.setRange(1, 50)
        self.epsilon_slider.setValue(1)
        self.epsilon_slider.setMaximumWidth(100)
        fourth_row.addWidget(self.epsilon_slider)

        self.epsilon_label = QLabel("1")
        self.epsilon_label.setMinimumWidth(25)
        fourth_row.addWidget(self.epsilon_label)

        self.apply_epsilon_btn = QPushButton("Apply")
        self.apply_epsilon_btn.setMaximumWidth(50)
        fourth_row.addWidget(self.apply_epsilon_btn)

        self.epsilon_slider.valueChanged.connect(self._update_epsilon_label)

        fourth_row.addStretch()
        main_layout.addLayout(fourth_row)

        # Fifth row: Simulation time | Max divergence
        fifth_row = QHBoxLayout()

        # --- Simulation time (left) ---
        self.sim_time_label = QLabel("Time: 0.000")
        self.sim_time_label.setMinimumWidth(100)
        fifth_row.addWidget(self.sim_time_label)

        fifth_row.addWidget(QLabel("|"))

        self.dt_label = QLabel("dt: 0.0000")
        self.dt_label.setMinimumWidth(80)
        fifth_row.addWidget(self.dt_label)

        fifth_row.addSpacing(20)

        # --- Max divergence (right) ---
        fifth_row.addWidget(QLabel("Max Div:"))
        self.max_div_label = QLabel("0.000000")
        self.max_div_label.setMinimumWidth(80)
        fifth_row.addWidget(self.max_div_label)

        fifth_row.addStretch()
        main_layout.addLayout(fifth_row)

        return console

    def _update_epsilon_label(self):
        """Update epsilon label when slider changes."""
        value = self.epsilon_slider.value()
        self.epsilon_label.setText(str(value))

    def _update_solver_info(self):
        """Update solver information display"""
        # This method is called by the viewer to update solver info
        pass  # InfoPanel doesn't need to display solver info

    def _create_numerical_group(self):
        """Group for advection scheme, pressure solver, Jacobi iterations, LES"""
        group = QGroupBox("Numerical Schemes")
        layout = QHBoxLayout()
        layout.setSpacing(8)

        layout.addWidget(QLabel("Advection:"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems(["rk3"])
        self.scheme_combo.setMaximumWidth(150)
        layout.addWidget(self.scheme_combo)
        self.apply_scheme_btn = QPushButton("Apply")
        self.apply_scheme_btn.setMaximumWidth(60)
        layout.addWidget(self.apply_scheme_btn)

        layout.addWidget(QLabel("|"))

        layout.addWidget(QLabel("Pressure:"))
        self.pressure_combo = QComboBox()
        self.pressure_combo.addItems(["fft", "cg", "multigrid"])
        self.pressure_combo.setMaximumWidth(150)
        layout.addWidget(self.pressure_combo)

        # Jacobi iterations (hidden initially)
        self.jacobi_iter_label = QLabel("Jacobi Iter:")
        self.jacobi_iter_input = QSpinBox()
        self.jacobi_iter_input.setRange(1, 1000)
        self.jacobi_iter_input.setValue(50)
        self.jacobi_iter_input.setSingleStep(10)
        self.jacobi_iter_input.setMaximumWidth(100)
        self.jacobi_iter_label.setVisible(False)
        self.jacobi_iter_input.setVisible(False)
        layout.addWidget(self.jacobi_iter_label)
        layout.addWidget(self.jacobi_iter_input)

        self.apply_pressure_btn = QPushButton("Apply")
        self.apply_pressure_btn.setMaximumWidth(60)
        layout.addWidget(self.apply_pressure_btn)

        layout.addWidget(QLabel("|"))

        # LES controls
        self.les_checkbox = QCheckBox("LES")
        self.les_checkbox.setChecked(False)
        self.les_checkbox.stateChanged.connect(self._on_les_checkbox_changed)
        layout.addWidget(self.les_checkbox)

        self.les_model_combo = QComboBox()
        self.les_model_combo.addItems(["dynamic_smagorinsky", "smagorinsky"])
        self.les_model_combo.setMaximumWidth(150)
        self.les_model_combo.setEnabled(False)  # Disabled by default

        layout.addWidget(QLabel("|"))

        self.apply_les_btn = QPushButton("Apply")
        self.apply_les_btn.setMaximumWidth(60)
        self.apply_les_btn.setEnabled(False)
        layout.addWidget(self.apply_les_btn)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def _on_les_checkbox_changed(self, state):
        """Enable/disable LES controls when checkbox is toggled"""
        is_checked = state == 2  # Qt.CheckState.Checked
        self.les_model_combo.setEnabled(is_checked)
        self.apply_les_btn.setEnabled(is_checked)

    def _on_naca_hover(self, index):
        """Show airfoil preview when selection changes"""
        if not self._check_naca_availability():
            return
        
        designation = self.naca_combo.currentText()
        if not designation or designation == "NACA 0012":
            return
        
        try:
            from solver.naca_airfoils import NACA_AIRFOILS, parse_naca_4digit, parse_naca_5digit
            if designation not in NACA_AIRFOILS:
                return
            
            # Parse designation to get parameters
            digits = ''.join(filter(str.isdigit, designation))
            tooltip_text = f"<b>{designation}</b><br><br>"
            
            if len(digits) == 4:
                m, p, t = parse_naca_4digit(designation)
                tooltip_text += f"Type: 4-digit series<br>"
                tooltip_text += f"Max camber: {m*100:.1f}%<br>"
                tooltip_text += f"Camber position: {p*100:.0f}% chord<br>"
                tooltip_text += f"Max thickness: {t*100:.1f}% chord"
            elif len(digits) == 5:
                cl, p, m, t = parse_naca_5digit(designation)
                tooltip_text += f"Type: 5-digit series<br>"
                tooltip_text += f"Design lift coeff: {cl:.2f}<br>"
                tooltip_text += f"Camber position: {p*100:.0f}% chord<br>"
                tooltip_text += f"Max camber: {m*10:.1f}%<br>"
                tooltip_text += f"Max thickness: {t*100:.1f}% chord"
            
            self.naca_combo.setToolTip(tooltip_text)
        except Exception as e:
            pass

    def _update_solver_info(self):
        """Update solver information display"""
        # This method is called by the viewer to update solver info
        pass  # InfoPanel doesn't need to display solver info

    def _create_obstacle_group(self):
        """Group for obstacle selection (cylinder / NACA airfoil) and NACA parameters"""
        group = QGroupBox("Obstacle Configuration")
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Row 1: Obstacle type
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self.obstacle_combo = QComboBox()
        self.obstacle_combo.addItems(["cylinder", "naca_airfoil"])
        self.obstacle_combo.setMaximumWidth(180)
        row1.addWidget(self.obstacle_combo)
        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: NACA controls (initially hidden)
        self.naca_widget = QWidget()
        naca_layout = QHBoxLayout(self.naca_widget)
        naca_layout.setContentsMargins(0, 0, 0, 0)
        naca_layout.setSpacing(8)

        naca_layout.addWidget(QLabel("NACA:"))
        self.naca_combo = QComboBox()
        # Check availability
        if self._check_naca_availability():
            from solver.naca_airfoils import NACA_AIRFOILS
            self.naca_combo.addItems(list(NACA_AIRFOILS.keys()))
            self.naca_combo.setCurrentText("NACA 2412")
        else:
            self.naca_combo.addItems(["NACA 2412"])
        self.naca_combo.setMaximumWidth(150)
        # Enable hover tracking for airfoil preview
        self.naca_combo.setMouseTracking(True)
        self.naca_combo.currentIndexChanged.connect(self._on_naca_hover)
        naca_layout.addWidget(self.naca_combo)

        naca_layout.addWidget(QLabel("Chord:"))
        self.chord_spinbox = QDoubleSpinBox()
        self.chord_spinbox.setRange(0.1, 3.0)
        self.chord_spinbox.setValue(3.0)
        self.chord_spinbox.setDecimals(2)
        self.chord_spinbox.setSingleStep(0.1)
        self.chord_spinbox.setMaximumWidth(120)
        naca_layout.addWidget(self.chord_spinbox)

        naca_layout.addWidget(QLabel("AoA:"))
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(-20.0, 20.0)
        self.angle_spinbox.setValue(-10.0)
        self.angle_spinbox.setDecimals(1)
        self.angle_spinbox.setSingleStep(1.0)
        self.angle_spinbox.setMaximumWidth(120)
        naca_layout.addWidget(self.angle_spinbox)

        naca_layout.addWidget(QLabel("Real-time:"))
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(-200, 200)
        self.angle_slider.setValue(0)
        self.angle_slider.setMaximumWidth(150)
        naca_layout.addWidget(self.angle_slider)

        self.apply_naca_btn = QPushButton("Apply")
        self.apply_naca_btn.setMaximumWidth(60)
        naca_layout.addWidget(self.apply_naca_btn)

        naca_layout.addStretch()
        self.naca_widget.setVisible(False)
        layout.addWidget(self.naca_widget)

        group.setLayout(layout)

        # If NACA module missing, still keep attributes but disable
        if not self._check_naca_availability():
            self.naca_combo = None
            self.chord_spinbox = None
            self.angle_spinbox = None
            self.angle_slider = None
            self.apply_naca_btn = None

        return group

    def _create_time_group(self):
        """Group for time step controls and CFL"""
        group = QGroupBox("Time Stepping")
        layout = QHBoxLayout()
        layout.setSpacing(8)

        layout.addWidget(QLabel("dt:"))
        self.dt_spinbox = QDoubleSpinBox()
        self.dt_spinbox.setRange(0.0001, 0.01)
        self.dt_spinbox.setDecimals(4)
        self.dt_spinbox.setSingleStep(0.0001)
        self.dt_spinbox.setMaximumWidth(150)
        layout.addWidget(self.dt_spinbox)

        self.apply_dt_btn = QPushButton("Apply")
        self.apply_dt_btn.setMaximumWidth(60)
        layout.addWidget(self.apply_dt_btn)

        self.adaptive_dt_checkbox = QCheckBox("Adaptive")
        self.adaptive_dt_checkbox.setChecked(True)  # Enable adaptive dt by default
        layout.addWidget(self.adaptive_dt_checkbox)

        self.cfl_label = QLabel("CFL: 0.00")
        self.cfl_label.setMinimumWidth(80)
        layout.addWidget(self.cfl_label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def _create_dye_group(self):
        """Group for dye injection controls"""
        group = QGroupBox("Dye Injection")
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

        # Inject button
        row_btn = QHBoxLayout()
        self.inject_dye_btn = QPushButton("Inject Dye")
        self.inject_dye_btn.setMaximumWidth(120)
        row_btn.addWidget(self.inject_dye_btn)
        row_btn.addStretch()
        layout.addLayout(row_btn)

        group.setLayout(layout)
        return group

    def _update_solver_info(self):
        """Update solver information display"""
        # This method is called by the viewer to update solver info
        pass  # InfoPanel doesn't need to display solver info

    def _create_visualization_group(self):
        """Group for all visualization settings (performance, toggles, colormaps, export)"""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Row 1: Performance (frame skip, FPS)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Frame skip:"))
        self.frame_skip_input = QSpinBox()
        self.frame_skip_input.setRange(1, 100)
        self.frame_skip_input.setValue(1)
        self.frame_skip_input.setSingleStep(1)
        self.frame_skip_input.setSuffix("x")
        self.frame_skip_input.setMaximumWidth(120)
        row1.addWidget(self.frame_skip_input)
        self.apply_frame_skip_btn = QPushButton("Apply")
        self.apply_frame_skip_btn.setMaximumWidth(60)
        row1.addWidget(self.apply_frame_skip_btn)

        row1.addSpacing(20)
        row1.addWidget(QLabel("Target FPS:"))
        self.vis_fps_input = QSpinBox()
        self.vis_fps_input.setRange(10, 120)
        self.vis_fps_input.setValue(60)
        self.vis_fps_input.setSingleStep(5)
        self.vis_fps_input.setSuffix(" Hz")
        self.vis_fps_input.setMaximumWidth(120)
        row1.addWidget(self.vis_fps_input)
        self.apply_vis_fps_btn = QPushButton("Apply")
        self.apply_vis_fps_btn.setMaximumWidth(60)
        row1.addWidget(self.apply_vis_fps_btn)
        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Display toggles
        row2 = QHBoxLayout()
        self.show_velocity_checkbox = QCheckBox("Velocity")
        self.show_velocity_checkbox.setChecked(True)
        self.show_vorticity_checkbox = QCheckBox("Vorticity")
        self.show_vorticity_checkbox.setChecked(True)
        self.show_sdf_checkbox = QCheckBox("SDF Mask")
        self.show_sdf_checkbox.setChecked(False)
        self.show_scalar_checkbox = QCheckBox("Dye")
        self.show_scalar_checkbox.setChecked(False)
        row2.addWidget(self.show_velocity_checkbox)
        row2.addWidget(self.show_vorticity_checkbox)
        row2.addWidget(self.show_sdf_checkbox)
        row2.addWidget(self.show_scalar_checkbox)
        row2.addStretch()
        layout.addLayout(row2)
        
        # Row 3: Dye injection controls
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Dye X:"))
        self.dye_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.dye_x_slider.setRange(0, 100)
        self.dye_x_slider.setValue(50)
        self.dye_x_slider.setMaximumWidth(150)
        row3.addWidget(self.dye_x_slider)
        
        row3.addWidget(QLabel("Y:"))
        self.dye_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.dye_y_slider.setRange(0, 100)
        self.dye_y_slider.setValue(50)
        self.dye_y_slider.setMaximumWidth(150)
        row3.addWidget(self.dye_y_slider)
        
        self.inject_dye_btn = QPushButton("Inject Dye")
        self.inject_dye_btn.setMaximumWidth(100)
        row3.addWidget(self.inject_dye_btn)
        row3.addStretch()
        layout.addLayout(row3)

        # Row 4: Colormaps
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Velocity colormap:"))
        self.velocity_colormap_combo = QComboBox()
        self.velocity_colormap_combo.setMaximumWidth(200)
        self._populate_velocity_colormaps()
        row4.addWidget(self.velocity_colormap_combo)

        row4.addSpacing(15)
        row4.addWidget(QLabel("Vorticity colormap:"))
        self.vorticity_colormap_combo = QComboBox()
        self.vorticity_colormap_combo.setMaximumWidth(200)
        self._populate_vorticity_colormaps()
        row4.addWidget(self.vorticity_colormap_combo)
        row4.addStretch()
        layout.addLayout(row4)

        # Row 5: Export and auto-fit buttons
        row5 = QHBoxLayout()
        self.export_btn = QPushButton("Export Frame")
        self.export_btn.setMaximumWidth(100)
        self.record_btn = QPushButton("Record")
        self.record_btn.setMaximumWidth(80)
        self.save_btn = QPushButton("Save State")
        self.save_btn.setEnabled(False)
        self.save_btn.setMaximumWidth(90)
        row5.addWidget(self.export_btn)
        row5.addWidget(self.record_btn)
        row5.addWidget(self.save_btn)

        row5.addWidget(QLabel("|"))
        self.autofit_velocity_btn = QPushButton("Auto-Vel")
        self.autofit_velocity_btn.setMaximumWidth(80)
        self.autofit_vorticity_btn = QPushButton("Auto-Vor")
        self.autofit_vorticity_btn.setMaximumWidth(80)
        self.autofit_both_btn = QPushButton("Auto-Both")
        self.autofit_both_btn.setMaximumWidth(90)
        row5.addWidget(self.autofit_velocity_btn)
        row5.addWidget(self.autofit_vorticity_btn)
        row5.addWidget(self.autofit_both_btn)
        row5.addStretch()
        layout.addLayout(row5)

        group.setLayout(layout)
        return group

    def _update_solver_info(self):
        """Update solver information display"""
        # This method is called by the viewer to update solver info
        pass  # InfoPanel doesn't need to display solver info

    # ---------- Preserved original methods ----------
    def set_chord_range_for_domain(self, max_chord: float):
        """Update chord spinbox range based on domain size"""
        if hasattr(self, 'chord_spinbox') and self.chord_spinbox is not None:
            self.chord_spinbox.setRange(0.1, max_chord)

    def show_naca_controls(self, show: bool) -> None:
        """Show/hide NACA controls based on obstacle selection"""
        if hasattr(self, 'naca_widget'):
            self.naca_widget.setVisible(show)

    def _check_naca_availability(self):
        """Check if NACA airfoils are available"""
        try:
            from solver.naca_airfoils import NACA_AIRFOILS
            return True
        except ImportError:
            return False

    def _update_epsilon_label(self, value):
        """Update epsilon label when slider changes"""
        # Show eps_multiplier value directly (not computed epsilon)
        self.epsilon_label.setText(f"{value}")

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


class InfoPanel(QWidget):
    """Information panel showing FPS, solver info, and status"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Solver and status labels
        self.solver_label = QLabel("Solver: Not initialized")
        self.info_label = QLabel("Info: Ready")
        self.div_label = QLabel("Divergence: 0.000")
        self.sim_fps_label = QLabel("Sim FPS: 0")

        # Change metric labels (not error)
        self.l2_error_label = QLabel("L2 Change: 0.000e+00")
        self.max_error_label = QLabel("Max Change: 0.000e+00")
        self.rel_error_label = QLabel("Rel Change: 0.000e+00")
        self.l2_u_error_label = QLabel("L2 U Change: 0.000e+00")
        self.l2_v_error_label = QLabel("L2 V Change: 0.000e+00")
        
        # Airfoil-specific metric labels
        self.cl_label = QLabel("CL: 0.000")
        self.cd_label = QLabel("CD: 0.000")
        self.stagnation_label = QLabel("Stagnation: 0.000")
        self.separation_label = QLabel("Separation: 0.000")
        self.cp_min_label = QLabel("Cp_min: 0.000")
        self.wake_deficit_label = QLabel("Wake Deficit: 0.000")
        
        # Copy buttons
        self.copy_all_btn = QPushButton("Copy All")
        self.copy_airfoil_btn = QPushButton("Copy Airfoil")
        
        # Marker visibility toggles
        self.show_stagnation_marker_cb = QCheckBox("Show Stagnation")
        self.show_stagnation_marker_cb.setChecked(False)
        self.show_separation_marker_cb = QCheckBox("Show Separation")
        self.show_separation_marker_cb.setChecked(False)
        
        # Airfoil metrics calculation toggle
        self.compute_airfoil_metrics_cb = QCheckBox("Compute Airfoil Metrics")
        self.compute_airfoil_metrics_cb.setChecked(False)
        
        # Reference to visualization for controlling markers
        self.visualization = None
        # Reference to solver for controlling airfoil metrics computation
        self.solver = None
        
        # Store current error values for copying
        self.current_l2_error = "0.000e+00"
        self.current_max_error = "0.000e+00"
        self.current_rel_error = "0.000e+00"
        self.current_l2_u_error = "0.000e+00"
        self.current_l2_v_error = "0.000e+00"
        
        # Store current airfoil metrics for copying
        self.current_cl = "0.000"
        self.current_stagnation = "0.000"
        self.current_separation = "0.000"
        self.current_cp_min = "0.000"
        self.current_wake_deficit = "0.000"
        self.current_cd = "0.000"
        
        self.setup_ui()
        self._setup_copy_buttons()
        self._setup_marker_toggles()

    def setup_ui(self):
        """Setup the information panel UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # ========== INFO DISPLAY ==========
        info_group = QGroupBox("Simulation Information")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)

        # First row: Solver info
        solver_row = QHBoxLayout()
        solver_row.addWidget(QLabel("Solver:"))
        solver_row.addWidget(self.solver_label)
        solver_row.addStretch()
        info_layout.addLayout(solver_row)

        # Second row: Status info
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        status_row.addWidget(self.info_label)
        status_row.addStretch()
        info_layout.addLayout(status_row)

        # Third row: Performance info
        perf_row = QHBoxLayout()
        perf_row.addWidget(QLabel("FPS:"))
        perf_row.addWidget(self.sim_fps_label)
        perf_row.addWidget(QLabel("|"))
        perf_row.addWidget(QLabel("Divergence:"))
        perf_row.addWidget(self.div_label)
        perf_row.addStretch()
        info_layout.addLayout(perf_row)

        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        # ========== ERROR METRICS ==========
        error_group = QGroupBox("Error Metrics")
        error_layout = QVBoxLayout()
        error_layout.setSpacing(4)

        # Metrics enable checkbox
        self.diagnostics_checkbox = QCheckBox("Enable Metrics")
        self.diagnostics_checkbox.setChecked(False)
        self.diagnostics_checkbox.setToolTip("Enable error metrics calculations (may slow simulation)")
        error_layout.addWidget(self.diagnostics_checkbox)

        # Metrics frame skip input
        skip_row = QHBoxLayout()
        skip_row.addWidget(QLabel("Compute every N frames:"))
        self.metrics_frame_skip_input = QSpinBox()
        self.metrics_frame_skip_input.setRange(1, 1000)
        self.metrics_frame_skip_input.setValue(100)
        self.metrics_frame_skip_input.setMaximumWidth(80)
        self.metrics_frame_skip_input.setToolTip("Compute metrics only every N-th frame to improve performance")
        skip_row.addWidget(self.metrics_frame_skip_input)
        skip_row.addStretch()
        error_layout.addLayout(skip_row)

        # L2 Error
        l2_row = QHBoxLayout()
        l2_row.addWidget(self.l2_error_label)
        l2_row.addStretch()
        error_layout.addLayout(l2_row)

        # Max Error
        max_row = QHBoxLayout()
        max_row.addWidget(self.max_error_label)
        max_row.addStretch()
        error_layout.addLayout(max_row)

        # Relative Error
        rel_row = QHBoxLayout()
        rel_row.addWidget(self.rel_error_label)
        rel_row.addStretch()
        error_layout.addLayout(rel_row)

        # Component Errors
        comp_row = QHBoxLayout()
        comp_row.addWidget(self.l2_u_error_label)
        comp_row.addWidget(QLabel("|"))
        comp_row.addWidget(self.l2_v_error_label)
        comp_row.addStretch()
        error_layout.addLayout(comp_row)

        # Copy All button
        copy_row = QHBoxLayout()
        copy_row.addWidget(self.copy_all_btn)
        copy_row.addStretch()
        error_layout.addLayout(copy_row)

        error_group.setLayout(error_layout)

        # ========== AIRFOIL METRICS ==========
        airfoil_group = QGroupBox("Airfoil Metrics")
        airfoil_layout = QVBoxLayout()
        airfoil_layout.setSpacing(4)

        # Lift Coefficient
        cl_row = QHBoxLayout()
        cl_row.addWidget(self.cl_label)
        cl_row.addWidget(self.cd_label)
        cl_row.addStretch()
        airfoil_layout.addLayout(cl_row)

        # Stagnation and Separation
        ss_row = QHBoxLayout()
        ss_row.addWidget(self.stagnation_label)
        ss_row.addWidget(QLabel("|"))
        ss_row.addWidget(self.separation_label)
        ss_row.addStretch()
        airfoil_layout.addLayout(ss_row)

        # Cp and Wake Deficit
        cw_row = QHBoxLayout()
        cw_row.addWidget(self.cp_min_label)
        cw_row.addWidget(QLabel("|"))
        cw_row.addWidget(self.wake_deficit_label)
        cw_row.addStretch()
        airfoil_layout.addLayout(cw_row)

        # Copy Airfoil button
        copy_airfoil_row = QHBoxLayout()
        copy_airfoil_row.addWidget(self.copy_airfoil_btn)
        copy_airfoil_row.addStretch()
        airfoil_layout.addLayout(copy_airfoil_row)
        
        # Marker visibility toggles
        marker_toggle_row = QHBoxLayout()
        marker_toggle_row.addWidget(self.show_stagnation_marker_cb)
        marker_toggle_row.addWidget(self.show_separation_marker_cb)
        marker_toggle_row.addStretch()
        airfoil_layout.addLayout(marker_toggle_row)
        
        # Airfoil metrics calculation toggle
        metrics_toggle_row = QHBoxLayout()
        metrics_toggle_row.addWidget(self.compute_airfoil_metrics_cb)
        metrics_toggle_row.addStretch()
        airfoil_layout.addLayout(metrics_toggle_row)

        airfoil_group.setLayout(airfoil_layout)

        # Create horizontal layout for error and airfoil metrics
        metrics_row = QHBoxLayout()
        metrics_row.addWidget(error_group)
        metrics_row.addWidget(airfoil_group)
        main_layout.addLayout(metrics_row)
        
        main_layout.addStretch()

        self.setLayout(main_layout)

    def _create_numerical_group(self):
        """Group for advection scheme, pressure solver, Jacobi iterations"""
        group = QGroupBox("Numerical Schemes")
        layout = QHBoxLayout()
        
        # Advection scheme
        layout.addWidget(QLabel("Advection:"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems(["rk3"])
        layout.addWidget(self.scheme_combo)
        
        self.apply_scheme_btn = QPushButton("Apply Scheme")
        layout.addWidget(self.apply_scheme_btn)
        layout.addWidget(QLabel("|"))
        
        # Pressure solver
        layout.addWidget(QLabel("Pressure:"))
        self.pressure_combo = QComboBox()
        self.pressure_combo.addItems(["fft", "cg", "multigrid"])
        layout.addWidget(self.pressure_combo)
        
        # Jacobi iterations (hidden by default)
        self.jacobi_iter_label = QLabel("Jacobi Iter:")
        self.jacobi_iter_input = QSpinBox()
        self.jacobi_iter_input.setRange(1, 1000)
        self.jacobi_iter_input.setValue(50)
        self.jacobi_iter_input.setSingleStep(10)
        self.jacobi_iter_label.setVisible(False)
        self.jacobi_iter_input.setVisible(False)
        
        layout.addWidget(self.jacobi_iter_label)
        layout.addWidget(self.jacobi_iter_input)
        
        self.apply_pressure_btn = QPushButton("Apply Pressure")
        layout.addWidget(self.apply_pressure_btn)
        
        group.setLayout(layout)
        return group

    def _update_solver_info(self):
        """Update solver information display"""
        # This method is called by the viewer to update solver info
        pass  # InfoPanel doesn't need to display solver info

    def _create_obstacle_group(self):
        """Group for obstacle selection (cylinder / NACA airfoil) and NACA parameters"""
        group = QGroupBox("Obstacle Configuration")
        layout = QVBoxLayout()
        
        # Obstacle type selection
        obstacle_layout = QHBoxLayout()
        obstacle_layout.addWidget(QLabel("Type:"))
        self.obstacle_type_combo = QComboBox()
        self.obstacle_type_combo.addItems(["cylinder", "naca_airfoil"])
        obstacle_layout.addWidget(self.obstacle_type_combo)
        obstacle_layout.addStretch()
        layout.addLayout(obstacle_layout)
        
        # NACA controls (hidden by default)
        naca_layout = QHBoxLayout()
        naca_layout.addWidget(QLabel("Chord:"))
        self.naca_chord_spinbox = QDoubleSpinBox()
        self.naca_chord_spinbox.setRange(0.1, 2.0)
        self.naca_chord_spinbox.setValue(0.3)
        self.naca_chord_spinbox.setDecimals(3)
        self.naca_chord_spinbox.setSingleStep(0.01)
        naca_layout.addWidget(self.naca_chord_spinbox)
        
        naca_layout.addWidget(QLabel("Angle:"))
        self.naca_angle_spinbox = QDoubleSpinBox()
        self.naca_angle_spinbox.setRange(-15.0, 15.0)
        self.naca_angle_spinbox.setValue(5.0)
        self.naca_angle_spinbox.setDecimals(1)
        self.naca_angle_spinbox.setSingleStep(0.5)
        naca_layout.addWidget(self.naca_angle_spinbox)
        
        naca_layout.addWidget(QLabel("X:"))
        self.naca_x_spinbox = QDoubleSpinBox()
        self.naca_x_spinbox.setRange(0.1, 2.0)
        self.naca_x_spinbox.setValue(0.3)
        self.naca_x_spinbox.setDecimals(3)
        self.naca_x_spinbox.setSingleStep(0.01)
        naca_layout.addWidget(self.naca_x_spinbox)
        
        naca_layout.addWidget(QLabel("Y:"))
        self.naca_y_spinbox = QDoubleSpinBox()
        self.naca_y_spinbox.setRange(0.1, 2.0)
        self.naca_y_spinbox.setValue(0.5)
        self.naca_y_spinbox.setDecimals(3)
        self.naca_y_spinbox.setSingleStep(0.01)
        naca_layout.addWidget(self.naca_y_spinbox)
        
        naca_layout.addStretch()
        layout.addLayout(naca_layout)
        
        # Hide NACA controls by default
        self.naca_chord_spinbox.setVisible(False)
        self.naca_angle_spinbox.setVisible(False)
        self.naca_x_spinbox.setVisible(False)
        self.naca_y_spinbox.setVisible(False)
        
        group.setLayout(layout)
        return group

    def _update_solver_info(self):
        """Update solver information display"""
        # This method is called by the viewer to update solver info
        pass  # InfoPanel doesn't need to display solver info

    def _create_time_group(self):
        """Group for time step controls and CFL"""
        group = QGroupBox("Time Stepping")
        layout = QHBoxLayout()
        
        # Time step
        layout.addWidget(QLabel("dt:"))
        self.dt_spinbox = QDoubleSpinBox()
        self.dt_spinbox.setRange(0.0001, 0.1)
        self.dt_spinbox.setValue(0.002)
        self.dt_spinbox.setDecimals(4)
        self.dt_spinbox.setSingleStep(0.0001)
        layout.addWidget(self.dt_spinbox)
        
        # Adaptive dt
        self.adaptive_dt_checkbox = QCheckBox("Adaptive dt")
        self.adaptive_dt_checkbox.setChecked(True)  # Enable adaptive dt by default
        layout.addWidget(self.adaptive_dt_checkbox)
        layout.addWidget(QLabel("|"))
        
        # CFL display
        layout.addWidget(QLabel("CFL:"))
        self.cfl_label = QLabel("0.000")
        layout.addWidget(self.cfl_label)
        
        group.setLayout(layout)
        return group

    def _update_solver_info(self):
        """Update solver information display"""
        # This method is called by the viewer to update solver info
        pass  # InfoPanel doesn't need to display solver info

    def _create_visualization_group(self):
        """Group for all visualization settings (performance, toggles, colormaps, export)"""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()
        
        # Performance controls
        perf_layout = QHBoxLayout()
        perf_layout.addWidget(QLabel("Target FPS:"))
        self.target_fps_spinbox = QSpinBox()
        self.target_fps_spinbox.setRange(1, 120)
        self.target_fps_spinbox.setValue(30)
        perf_layout.addWidget(self.target_fps_spinbox)
        perf_layout.addWidget(QLabel("|"))
        
        self.limit_fps_checkbox = QCheckBox("Limit FPS")
        self.limit_fps_checkbox.setChecked(True)
        perf_layout.addWidget(self.limit_fps_checkbox)
        perf_layout.addStretch()
        layout.addLayout(perf_layout)
        
        # Display toggles
        toggle_layout = QHBoxLayout()
        self.show_velocity_checkbox = QCheckBox("Show Velocity")
        self.show_velocity_checkbox.setChecked(True)
        toggle_layout.addWidget(self.show_velocity_checkbox)
        
        self.show_vorticity_checkbox = QCheckBox("Show Vorticity")
        self.show_vorticity_checkbox.setChecked(True)
        toggle_layout.addWidget(self.show_vorticity_checkbox)
        
        self.show_pressure_checkbox = QCheckBox("Show Pressure")
        self.show_pressure_checkbox.setChecked(True)
        toggle_layout.addWidget(self.show_pressure_checkbox)
        toggle_layout.addStretch()
        layout.addLayout(toggle_layout)
        
        # Colormap controls
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Velocity:"))
        self.velocity_colormap_combo = QComboBox()
        self.velocity_colormap_combo.addItems(["viridis", "plasma", "inferno", "RdBu"])
        self.velocity_colormap_combo.setCurrentText("RdBu")
        colormap_layout.addWidget(self.velocity_colormap_combo)
        
        colormap_layout.addWidget(QLabel("|"))
        
        colormap_layout.addWidget(QLabel("Vorticity:"))
        self.vorticity_colormap_combo = QComboBox()
        self.vorticity_colormap_combo.addItems(["viridis", "plasma", "inferno", "RdBu"])
        self.vorticity_colormap_combo.setCurrentText("RdBu")
        colormap_layout.addWidget(self.vorticity_colormap_combo)
        
        colormap_layout.addWidget(QLabel("|"))
        
        colormap_layout.addWidget(QLabel("Pressure:"))
        self.pressure_colormap_combo = QComboBox()
        self.pressure_colormap_combo.addItems(["viridis", "plasma", "inferno", "RdBu"])
        self.pressure_colormap_combo.setCurrentText("RdBu")
        colormap_layout.addWidget(self.pressure_colormap_combo)
        colormap_layout.addStretch()
        layout.addLayout(colormap_layout)
        
        # Export controls
        export_layout = QHBoxLayout()
        self.export_data_btn = QPushButton("Export Data")
        export_layout.addWidget(self.export_data_btn)
        export_layout.addWidget(QLabel("|"))
        
        self.record_video_btn = QPushButton("Record Video")
        export_layout.addWidget(self.record_video_btn)
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        # Auto-fit buttons
        autofit_layout = QHBoxLayout()
        self.autofit_velocity_btn = QPushButton("Auto-fit Velocity")
        autofit_layout.addWidget(self.autofit_velocity_btn)
        
        autofit_layout.addWidget(QLabel("|"))
        
        self.autofit_vorticity_btn = QPushButton("Auto-fit Vorticity")
        autofit_layout.addWidget(self.autofit_vorticity_btn)
        
        autofit_layout.addWidget(QLabel("|"))
        
        self.autofit_pressure_btn = QPushButton("Auto-fit Pressure")
        autofit_layout.addWidget(self.autofit_pressure_btn)
        autofit_layout.addStretch()
        layout.addLayout(autofit_layout)
        
        group.setLayout(layout)
        return group

    def _update_solver_info(self):
        """Update solver information display"""
        # This method is called by the viewer to update solver info
        pass  # InfoPanel doesn't need to display solver info

        # Add columns to horizontal layout
        columns_layout.addWidget(left_col, 1)  # stretch factor 1 makes them equal width
        columns_layout.addWidget(right_col, 1)

        # Add the two‑column section to the container
        container_layout.addLayout(columns_layout)

        # Add the visualization group (spans full width) below the columns
        container_layout.addWidget(visualization_group)

        # Add the container to the main layout
        main_layout.addWidget(sidebar_container, 1)

        self.setLayout(main_layout)


    def _style_labels(self):
        """Apply styling to info labels"""
        # Style error metric labels
        error_style = "color: #ffffff; padding: 2px 6px; border-radius: 3px; font-family: monospace;"
        self.l2_error_label.setStyleSheet(f"background-color: #8b4513; {error_style}")
        self.max_error_label.setStyleSheet(f"background-color: #2e8b57; {error_style}")
        self.rel_error_label.setStyleSheet(f"background-color: #4682b4; {error_style}")
        self.l2_u_error_label.setStyleSheet(f"background-color: #9370db; {error_style}")
        self.l2_v_error_label.setStyleSheet(f"background-color: #20b2aa; {error_style}")
    
    def clear_error_metrics(self):
        """Clear error metric labels"""
        self.l2_error_label.setText("L2 Change: 0.000e+00")
        self.max_error_label.setText("Max Change: 0.000e+00")
        self.rel_error_label.setText("Rel Change: 0.000e+00")
        self.l2_u_error_label.setText("L2 Change U: 0.000e+00")
        self.l2_v_error_label.setText("L2 Change V: 0.000e+00")

    def update_error_metrics(self, l2_error: float, max_error: float = None,
                           rel_error: float = None, l2_u_error: float = None,
                           l2_v_error: float = None):
        """Update error metric labels with latest nonzero values"""
        # Use the provided value if nonzero, otherwise keep current display
        if l2_error != 0:
            self.l2_error_label.setText(f"L2 Change: {l2_error:.3e}")
        
        # Use L2 error as fallback if other values are None
        max_err = max_error if max_error is not None else l2_error
        rel_err = rel_error if rel_error is not None else l2_error
        l2_u_err = l2_u_error if l2_u_error is not None else l2_error
        l2_v_err = l2_v_error if l2_v_error is not None else l2_error
        
        # Only update labels if values are nonzero
        if max_err != 0:
            self.max_error_label.setText(f"Max Change: {max_err:.3e}")
        if rel_err != 0:
            self.rel_error_label.setText(f"Rel Change: {rel_err * 100:.3f}%")
        if l2_u_err != 0:
            self.l2_u_error_label.setText(f"L2 U Change: {l2_u_err:.3e}")
        if l2_v_err != 0:
            self.l2_v_error_label.setText(f"L2 V Change: {l2_v_err:.3e}")
        
        # Store current values for copying
        self.current_l2_error = f"{l2_error:.3e}"
        self.current_max_error = f"{max_err:.3e}"
        self.current_rel_error = f"{rel_err * 100:.3f}%"
        self.current_l2_u_error = f"{l2_u_err:.3e}"
        self.current_l2_v_error = f"{l2_v_err:.3e}"

        # Style remaining labels
        base_style = "color: white; padding: 2px 6px; border-radius: 3px;"
        self.info_label.setStyleSheet("background-color: black; color: white; padding: 2px 8px; border-radius: 3px;")
    
    def clear_airfoil_metrics(self):
        """Clear airfoil metric labels"""
        self.cl_label.setText("CL: 0.000")
        self.cd_label.setText("CD: 0.000")
        self.stagnation_label.setText("Stagnation: 0.000")
        self.separation_label.setText("Separation: 0.000")
        self.cp_min_label.setText("Cp_min: 0.000")
        self.wake_deficit_label.setText("Wake Deficit: 0.000")

    def update_airfoil_metrics(self, cl: float = None, stagnation: float = None,
                               separation: float = None, cp_min: float = None,
                               wake_deficit: float = None, cd: float = None,
                               airfoil_x: float = 2.5, chord_length: float = 3.0):
        """Update airfoil metric labels with current values"""
        if cl is not None:
            self.cl_label.setText(f"CL: {cl:.3f}")
            self.current_cl = f"{cl:.3f}"
        if cd is not None:
            self.cd_label.setText(f"CD: {cd:.3f}")
            self.current_cd = f"{cd:.3f}"
        if stagnation is not None:
            # Convert from absolute to relative coordinates (x/c)
            stagnation_rel = (stagnation - airfoil_x) / chord_length if chord_length > 0 else 0.0
            self.stagnation_label.setText(f"Stagnation: {stagnation_rel:.3f}c")
            self.current_stagnation = f"{stagnation_rel:.3f}c"
        if separation is not None:
            # Convert from absolute to relative coordinates (x/c)
            separation_rel = (separation - airfoil_x) / chord_length if chord_length > 0 else 0.0
            self.separation_label.setText(f"Separation: {separation_rel:.3f}c")
            self.current_separation = f"{separation_rel:.3f}c"
        if cp_min is not None:
            self.cp_min_label.setText(f"Cp_min: {cp_min:.2f}")
            self.current_cp_min = f"{cp_min:.2f}"
        if wake_deficit is not None:
            self.wake_deficit_label.setText(f"Wake Deficit: {wake_deficit:.3f}")
            self.current_wake_deficit = f"{wake_deficit:.3f}"
    
    def _setup_copy_buttons(self):
        """Setup copy buttons functionality"""
        # Connect copy all button
        self.copy_all_btn.clicked.connect(self._copy_all_to_clipboard)
        
        # Connect copy airfoil button
        self.copy_airfoil_btn.clicked.connect(self._copy_airfoil_to_clipboard)
        
        # Style copy all button
        button_style = "QPushButton { padding: 4px 12px; font-size: 11px; border: 1px solid #666; border-radius: 3px; background-color: #4CAF50; color: white; font-weight: bold; } QPushButton:hover { background-color: #45a049; }"
        self.copy_all_btn.setStyleSheet(button_style)
        
        # Style copy airfoil button
        airfoil_button_style = "QPushButton { padding: 4px 12px; font-size: 11px; border: 1px solid #666; border-radius: 3px; background-color: #2196F3; color: white; font-weight: bold; } QPushButton:hover { background-color: #1976D2; }"
        self.copy_airfoil_btn.setStyleSheet(airfoil_button_style)
    
    def _setup_marker_toggles(self):
        """Setup marker visibility toggle checkboxes"""
        self.show_stagnation_marker_cb.toggled.connect(self._toggle_stagnation_marker)
        self.show_separation_marker_cb.toggled.connect(self._toggle_separation_marker)
        self.compute_airfoil_metrics_cb.toggled.connect(self._toggle_airfoil_metrics)
    
    def set_visualization(self, visualization):
        """Set the visualization reference for controlling markers"""
        self.visualization = visualization
    
    def set_solver(self, solver):
        """Set the solver reference for controlling airfoil metrics computation"""
        self.solver = solver
    
    def _toggle_stagnation_marker(self, checked):
        """Toggle stagnation marker visibility"""
        if self.visualization is not None and hasattr(self.visualization, 'show_stagnation_marker'):
            self.visualization.show_stagnation_marker = checked
    
    def _toggle_separation_marker(self, checked):
        """Toggle separation marker visibility"""
        if self.visualization is not None and hasattr(self.visualization, 'show_separation_marker'):
            self.visualization.show_separation_marker = checked
    
    def _toggle_airfoil_metrics(self, checked):
        """Toggle airfoil metrics calculation"""
        if self.solver is not None and hasattr(self.solver, 'compute_airfoil_metrics'):
            self.solver.compute_airfoil_metrics = checked
            print(f"Airfoil metrics calculation {'enabled' if checked else 'disabled'}")
    
    def _copy_all_to_clipboard(self):
        """Copy all metrics to clipboard"""
        try:
            clipboard = QApplication.clipboard()

            # Get solver configuration if available
            sim_info = ""
            if self.solver is not None:
                # Grid size
                grid_size = f"{self.solver.grid.nx} x {self.solver.grid.ny}"

                # Advection scheme (from actual solver settings)
                advection_scheme = getattr(self.solver.sim_params, 'advection_scheme', 'unknown')

                # Pressure solver (from actual solver settings)
                pressure_solver = getattr(self.solver.sim_params, 'pressure_solver', 'unknown')

                # Current FPS (from info panel label)
                fps_label = self.sim_fps_label.text()
                fps = fps_label.replace("Sim FPS: ", "")

                # NACA airfoil (from actual solver settings)
                naca_airfoil = getattr(self.solver.sim_params, 'naca_airfoil', 'none')

                # Airfoil chord length (from actual solver settings)
                chord_length = getattr(self.solver.sim_params, 'naca_chord', 'N/A')

                # Angle of attack (from actual solver settings)
                aoa = getattr(self.solver.sim_params, 'naca_angle', 'N/A')

                # Simulation time
                sim_time = self.solver.history['time'][-1] if self.solver.history.get('time') else 0.0

                # Current dt
                current_dt = self.solver.dt

                # Adaptive dt
                adaptive_dt = getattr(self.solver.sim_params, 'adaptive_dt', False)

                # LES
                les_used = getattr(self.solver.sim_params, 'use_les', False)

                # Reynolds number
                reynolds = self.solver.flow.Re

                # Smagorinsky (if LES)
                if les_used:
                    smagorinsky = getattr(self.solver.sim_params, 'smagorinsky_constant', 'N/A')
                    smagorinsky_type = getattr(self.solver.sim_params, 'dynamic_smagorinsky', False)
                    smagorinsky_info = f"Dynamic: {smagorinsky_type}, Constant: {smagorinsky}"
                else:
                    smagorinsky_info = "N/A (LES disabled)"

                # Epsilon
                epsilon = getattr(self.solver.sim_params, 'eps', 'N/A')

                # Float precision (read from actual JAX config)
                import jax
                float_precision = "float64" if jax.config.jax_enable_x64 else "float32"

                sim_info = (
                    f"--- Simulation Configuration ---\n"
                    f"Grid Size: {grid_size}\n"
                    f"Advection Scheme: {advection_scheme}\n"
                    f"Pressure Solver: {pressure_solver}\n"
                    f"Sim FPS: {fps}\n"
                    f"NACA Airfoil: {naca_airfoil}\n"
                    f"Chord Length: {chord_length}\n"
                    f"Angle of Attack: {aoa}°\n"
                    f"Simulation Time: {sim_time:.3f} s\n"
                    f"Current dt: {current_dt:.6f}\n"
                    f"Adaptive dt: {adaptive_dt}\n"
                    f"LES Used: {les_used}\n"
                    f"Reynolds Number: {reynolds:.1f}\n"
                    f"Smagorinsky: {smagorinsky_info}\n"
                    f"Epsilon: {epsilon}\n"
                    f"Float Precision: {float_precision}\n\n"
                )

            # Format all error values for copying
            all_values = (
                f"{sim_info}"
                f"--- Metrics ---\n"
                f"L2 Error: {self.current_l2_error}\n"
                f"Max Error: {self.current_max_error}\n"
                f"Rel Error: {self.current_rel_error}\n"
                f"L2 U Error: {self.current_l2_u_error}\n"
                f"L2 V Error: {self.current_l2_v_error}\n"
                f"CL: {self.current_cl}\n"
                f"Stagnation: {self.current_stagnation}\n"
                f"Separation: {self.current_separation}\n"
                f"Cp_min: {self.current_cp_min}\n"
                f"Wake Deficit: {self.current_wake_deficit}"
            )

            clipboard.setText(all_values)

            # Brief visual feedback
            original_text = self.copy_all_btn.text()
            self.copy_all_btn.setText("Copied!")
            self.copy_all_btn.setStyleSheet("QPushButton { padding: 4px 12px; font-size: 11px; border: 1px solid #666; border-radius: 3px; background-color: #90ee90; color: white; font-weight: bold; }")

            # Reset after 1 second
            QTimer.singleShot(1000, lambda: self._reset_copy_all_button())
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
            self.copy_all_btn.setText("Error!")
            QTimer.singleShot(1000, lambda: self._reset_copy_all_button())
    
    def _copy_airfoil_to_clipboard(self):
        """Copy airfoil metrics to clipboard"""
        try:
            clipboard = QApplication.clipboard()

            # Format airfoil values for copying
            airfoil_values = (
                f"CL: {self.current_cl}\n"
                f"Stagnation: {self.current_stagnation}\n"
                f"Separation: {self.current_separation}\n"
                f"Cp_min: {self.current_cp_min}\n"
                f"Wake Deficit: {self.current_wake_deficit}"
            )

            clipboard.setText(airfoil_values)

            # Brief visual feedback
            original_text = self.copy_airfoil_btn.text()
            self.copy_airfoil_btn.setText("Copied!")
            self.copy_airfoil_btn.setStyleSheet("QPushButton { padding: 4px 12px; font-size: 11px; border: 1px solid #666; border-radius: 3px; background-color: #90ee90; color: white; font-weight: bold; }")

            # Reset after 1 second
            QTimer.singleShot(1000, lambda: self._reset_copy_airfoil_button())
        except Exception as e:
            print(f"Error copying airfoil metrics: {e}")
            self.copy_airfoil_btn.setText("Error!")
            QTimer.singleShot(1000, lambda: self._reset_copy_airfoil_button())
    
    def _reset_copy_airfoil_button(self):
        """Reset copy airfoil button appearance after copying"""
        airfoil_button_style = "QPushButton { padding: 4px 12px; font-size: 11px; border: 1px solid #666; border-radius: 3px; background-color: #2196F3; color: white; font-weight: bold; } QPushButton:hover { background-color: #1976D2; }"
        self.copy_airfoil_btn.setStyleSheet(airfoil_button_style)
        self.copy_airfoil_btn.setText("Copy Airfoil")
    
    def _reset_copy_all_button(self):
        """Reset copy all button appearance after copying"""
        button_style = "QPushButton { padding: 4px 12px; font-size: 11px; border: 1px solid #666; border-radius: 3px; background-color: #4CAF50; color: white; font-weight: bold; } QPushButton:hover { background-color: #45a049; }"
        self.copy_all_btn.setText("Copy All")
        self.copy_all_btn.setStyleSheet(button_style)