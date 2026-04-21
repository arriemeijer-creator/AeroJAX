"""
Information panel showing FPS, solver info, error metrics, and airfoil metrics.
"""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QCheckBox, QApplication, QScrollArea
)
from .collapsible_groupbox import CollapsibleGroupBox


class InfoPanel(QWidget):
    """Information panel showing FPS, solver info, and status"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Solver and status labels
        self.solver_label = QLabel("Solver: Not initialized")
        self.info_label = QLabel("Info: Ready")
        self.div_label = QLabel("Divergence: 0.000")
        self.sim_fps_label = QLabel("Sim FPS: 0")

        # Additional solver info labels (hidden, used for copy button)
        self.grid_size_label = QLabel("Grid: 512x96")
        self.reynolds_label = QLabel("Re: 2000")
        self.hyper_viscosity_label = QLabel("Hyper ν: 0%")
        self.mask_epsilon_label = QLabel("Mask ε: 0.5")
        self.dt_mode_label = QLabel("DT Mode: Adaptive")
        self.dt_value_label = QLabel("DT: 0.005859")
        self.les_status_label = QLabel("LES: Off")
        self.les_model_label = QLabel("LES Model: --")
        self.advection_scheme_label = QLabel("Scheme: RK3")

        # Change metric labels
        self.l2_error_label = QLabel("L2 Change: 0.000e+00")
        self.rms_change_label = QLabel("RMS Change: 0.000e+00")
        self.max_error_label = QLabel("Max Change: 0.000e+00")
        self.change_99p_label = QLabel("Change 99p: 0.000e+00")
        self.rel_error_label = QLabel("Rel Change: 0.000e+00")
        self.l2_u_error_label = QLabel("L2 U Change: 0.000e+00")
        self.l2_v_error_label = QLabel("L2 V Change: 0.000e+00")

        # Airfoil-specific metric labels
        self.cl_label = QLabel("CL: 0.000")
        self.cd_label = QLabel("CD: 0.000")
        self.strouhal_label = QLabel("St: --")
        self.stagnation_label = QLabel("Stagnation: 0.000")
        self.separation_label = QLabel("Separation: 0.000")
        self.cp_min_label = QLabel("Cp_min: 0.000")
        self.wake_deficit_label = QLabel("Wake Deficit: 0.000")

        # Obstacle-specific labels (hidden, used for copy button)
        self.obstacle_type_label = QLabel("Obstacle: NACA 0012")
        self.obstacle_aoa_label = QLabel("AoA: 10.0°")
        self.obstacle_chord_label = QLabel("Chord: 3.000")
        self.obstacle_x_label = QLabel("X: 2.500")
        self.obstacle_y_label = QLabel("Y: 1.900")
        self.obstacle_diameter_label = QLabel("Diameter: 0.360")

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

        # References
        self.visualization = None
        self.solver = None

        # Store current values for copying
        self.current_l2_error = "0.000e+00"
        self.current_max_error = "0.000e+00"
        self.current_rel_error = "0.000e+00"
        self.current_l2_u_error = "0.000e+00"
        self.current_l2_v_error = "0.000e+00"
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

        # ========== ERROR METRICS ==========
        error_group = CollapsibleGroupBox("Error Metrics", start_collapsed=True)
        error_layout = QVBoxLayout()
        error_layout.setSpacing(4)

        # Metrics enable checkbox and Save CSV button
        metrics_row = QHBoxLayout()
        self.diagnostics_checkbox = QCheckBox("Enable Metrics")
        self.diagnostics_checkbox.setChecked(False)
        self.diagnostics_checkbox.setToolTip("Enable error metrics calculations (may slow simulation)")
        metrics_row.addWidget(self.diagnostics_checkbox)

        self.save_csv_btn = QPushButton("Save CSV")
        self.save_csv_btn.setMaximumWidth(80)
        self.save_csv_btn.setToolTip("Save all metrics history to a CSV file")
        metrics_row.addWidget(self.save_csv_btn)
        metrics_row.addStretch()
        error_layout.addLayout(metrics_row)

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

        # RMS Change
        rms_row = QHBoxLayout()
        rms_row.addWidget(self.rms_change_label)
        rms_row.addStretch()
        error_layout.addLayout(rms_row)

        # Max Error
        max_row = QHBoxLayout()
        max_row.addWidget(self.max_error_label)
        max_row.addStretch()
        error_layout.addLayout(max_row)

        # Change 99p
        change_99p_row = QHBoxLayout()
        change_99p_row.addWidget(self.change_99p_label)
        change_99p_row.addStretch()
        error_layout.addLayout(change_99p_row)

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
        airfoil_group = CollapsibleGroupBox("Airfoil Metrics", start_collapsed=True)
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

        # ========== SCROLLABLE AREA FOR METRICS ==========
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Scrollable content widget
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Add error metrics group
        scroll_layout.addWidget(error_group)

        # Add airfoil metrics group
        scroll_layout.addWidget(airfoil_group)

        scroll_layout.addStretch()

        # Set scroll content as scroll area widget
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)  # Give stretch factor of 1

        self.setLayout(main_layout)

    def _setup_copy_buttons(self):
        """Setup copy buttons functionality"""
        self.copy_all_btn.clicked.connect(self._copy_all_to_clipboard)
        self.copy_airfoil_btn.clicked.connect(self._copy_airfoil_to_clipboard)

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

    def clear_error_metrics(self):
        """Clear error metric labels"""
        self.l2_error_label.setText("L2 Change: 0.000e+00")
        self.rms_change_label.setText("RMS Change: 0.000e+00")
        self.max_error_label.setText("Max Change: 0.000e+00")
        self.change_99p_label.setText("Change 99p: 0.000e+00")
        self.rel_error_label.setText("Rel Change: 0.000e+00")
        self.l2_u_error_label.setText("L2 Change U: 0.000e+00")
        self.l2_v_error_label.setText("L2 Change V: 0.000e+00")

    def update_error_metrics(self, l2_error: float = None, rms_change: float = None,
                           max_error: float = None, change_99p: float = None,
                           rel_error: float = None, l2_u_error: float = None,
                           l2_v_error: float = None):
        """Update error metric labels with proper None handling"""
        if l2_error is not None and l2_error > 0:
            self.l2_error_label.setText(f"L2 Change: {l2_error:.3e}")
            self.current_l2_error = f"{l2_error:.3e}"

        if rms_change is not None and rms_change > 0:
            self.rms_change_label.setText(f"RMS Change: {rms_change:.3e}")
            self.current_rms_change = f"{rms_change:.3e}"

        if max_error is not None and max_error > 0:
            self.max_error_label.setText(f"Max Change: {max_error:.3e}")
            self.current_max_error = f"{max_error:.3e}"

        if change_99p is not None and change_99p > 0:
            self.change_99p_label.setText(f"Change 99p: {change_99p:.3e}")
            self.current_change_99p = f"{change_99p:.3e}"

        if rel_error is not None and rel_error > 0:
            self.rel_error_label.setText(f"Rel Change: {rel_error * 100:.3f}%")
            self.current_rel_error = f"{rel_error * 100:.3f}%"

        if l2_u_error is not None and l2_u_error > 0:
            self.l2_u_error_label.setText(f"L2 U Change: {l2_u_error:.3e}")
            self.current_l2_u_error = f"{l2_u_error:.3e}"

        if l2_v_error is not None and l2_v_error > 0:
            self.l2_v_error_label.setText(f"L2 V Change: {l2_v_error:.3e}")
            self.current_l2_v_error = f"{l2_v_error:.3e}"

        self.info_label.setStyleSheet("background-color: black; color: white; padding: 2px 8px; border-radius: 3px;")

    def clear_airfoil_metrics(self):
        """Clear airfoil metric labels"""
        self.cl_label.setText("CL: 0.000")
        self.cd_label.setText("CD: 0.000")
        self.strouhal_label.setText("St: --")
        self.stagnation_label.setText("Stagnation: 0.000")
        self.separation_label.setText("Separation: 0.000")
        self.cp_min_label.setText("Cp_min: 0.000")
        self.wake_deficit_label.setText("Wake Deficit: 0.000")

    def update_airfoil_metrics(self, cl: float = None, stagnation: float = None,
                               separation: float = None, cp_min: float = None,
                               wake_deficit: float = None, cd: float = None,
                               avg_cl: float = None, avg_cd: float = None,
                               strouhal: float = None,
                               airfoil_x: float = 2.5, chord_length: float = 3.0,
                               obstacle_type: str = 'naca_airfoil',
                               cylinder_diameter: float = 0.36,
                               cylinder_center_x: float = 5.0):
        """Update airfoil metric labels with current values"""
        if cl is not None:
            if avg_cl is not None:
                self.cl_label.setText(f"CL: {cl:.3f} (avg: {avg_cl:.3f})")
            else:
                self.cl_label.setText(f"CL: {cl:.3f}")
            self.current_cl = f"{cl:.3f}"
        if cd is not None:
            if avg_cd is not None:
                self.cd_label.setText(f"CD: {cd:.3f} (avg: {avg_cd:.3f})")
            else:
                self.cd_label.setText(f"CD: {cd:.3f}")
            self.current_cd = f"{cd:.3f}"
        if stagnation is not None:
            if obstacle_type == 'naca_airfoil':
                stagnation_rel = (stagnation - airfoil_x) / chord_length if chord_length > 0 else 0.0
                self.stagnation_label.setText(f"Stagnation: {stagnation_rel:.3f}c")
                self.current_stagnation = f"{stagnation_rel:.3f}c"
            else:
                # For cylinders, measure from front of cylinder (center - radius)
                cylinder_front = cylinder_center_x - (cylinder_diameter / 2.0)
                stagnation_rel = (stagnation - cylinder_front) / cylinder_diameter if cylinder_diameter > 0 else 0.0
                self.stagnation_label.setText(f"Stagnation: {stagnation_rel:.3f}d")
                self.current_stagnation = f"{stagnation_rel:.3f}d"
        if separation is not None:
            if obstacle_type == 'naca_airfoil':
                separation_rel = (separation - airfoil_x) / chord_length if chord_length > 0 else 0.0
                if separation_rel == 0.0:
                    self.separation_label.setText("Separation: N/A")
                    self.current_separation = "N/A"
                else:
                    self.separation_label.setText(f"Separation: {separation_rel:.3f}c")
                    self.current_separation = f"{separation_rel:.3f}c"
            else:
                # For cylinders, measure from front of cylinder (center - radius)
                cylinder_front = cylinder_center_x - (cylinder_diameter / 2.0)
                separation_rel = (separation - cylinder_front) / cylinder_diameter if cylinder_diameter > 0 else 0.0
                if separation_rel == 0.0:
                    self.separation_label.setText("Separation: N/A")
                    self.current_separation = "N/A"
                else:
                    self.separation_label.setText(f"Separation: {separation_rel:.3f}d")
                    self.current_separation = f"{separation_rel:.3f}d"
        if strouhal is not None:
            self.strouhal_label.setText(f"St: {strouhal:.3f}")
        if cp_min is not None:
            self.cp_min_label.setText(f"Cp_min: {cp_min:.2f}")
            self.current_cp_min = f"{cp_min:.2f}"
        if wake_deficit is not None:
            self.wake_deficit_label.setText(f"Wake Deficit: {wake_deficit:.3f}")
            self.current_wake_deficit = f"{wake_deficit:.3f}"

    def _copy_all_to_clipboard(self):
        """Copy all simulation data to clipboard"""
        try:
            clipboard = QApplication.clipboard()
            all_values = (
                f"=== Solver Info ===\n"
                f"{self.solver_label.text()}\n"
                f"{self.info_label.text()}\n"
                f"{self.grid_size_label.text()}\n"
                f"{self.reynolds_label.text()}\n"
                f"{self.hyper_viscosity_label.text()}\n"
                f"{self.mask_epsilon_label.text()}\n"
                f"{self.dt_mode_label.text()}\n"
                f"{self.dt_value_label.text()}\n"
                f"{self.les_status_label.text()}\n"
                f"{self.les_model_label.text()}\n"
                f"{self.advection_scheme_label.text()}\n"
                f"{self.div_label.text()}\n"
                f"{self.sim_fps_label.text()}\n\n"
                f"=== Change Metrics ===\n"
                f"{self.l2_error_label.text()}\n"
                f"{self.rms_change_label.text()}\n"
                f"{self.max_error_label.text()}\n"
                f"{self.change_99p_label.text()}\n"
                f"{self.rel_error_label.text()}\n"
                f"{self.l2_u_error_label.text()}\n"
                f"{self.l2_v_error_label.text()}\n\n"
                f"=== Obstacle Info ===\n"
                f"{self.obstacle_type_label.text()}\n"
                f"{self.obstacle_aoa_label.text()}\n"
                f"{self.obstacle_chord_label.text()}\n"
                f"{self.obstacle_x_label.text()}\n"
                f"{self.obstacle_y_label.text()}\n"
                f"{self.obstacle_diameter_label.text()}\n\n"
                f"=== Airfoil Metrics ===\n"
                f"{self.cl_label.text()}\n"
                f"{self.cd_label.text()}\n"
                f"{self.strouhal_label.text()}\n"
                f"{self.stagnation_label.text()}\n"
                f"{self.separation_label.text()}\n"
                f"{self.cp_min_label.text()}\n"
                f"{self.wake_deficit_label.text()}\n"
            )
            clipboard.setText(all_values)
            self.copy_all_btn.setText("Copied!")
            QTimer.singleShot(1000, self._reset_copy_all_button)
        except Exception as e:
            print(f"Error copying to clipboard: {e}")

    def _copy_airfoil_to_clipboard(self):
        """Copy airfoil metrics to clipboard"""
        try:
            clipboard = QApplication.clipboard()
            airfoil_values = (
                f"CL: {self.current_cl}\n"
                f"Stagnation: {self.current_stagnation}\n"
                f"Separation: {self.current_separation}\n"
                f"Cp_min: {self.current_cp_min}\n"
                f"Wake Deficit: {self.current_wake_deficit}"
            )
            clipboard.setText(airfoil_values)
            self.copy_airfoil_btn.setText("Copied!")
            QTimer.singleShot(1000, self._reset_copy_airfoil_button)
        except Exception as e:
            print(f"Error copying airfoil metrics: {e}")

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
