"""
UI Components for Differential Backpropagation GUI
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QSpinBox, QDoubleSpinBox, QPushButton, QComboBox, QGroupBox,
    QCheckBox, QTabWidget, QFormLayout, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt
from typing import Optional, Dict, Any


class GoalSettingPanel(QWidget):
    """Panel for setting optimization goals"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Goals Group
        goals_group = QGroupBox("Optimization Goals")
        goals_layout = QFormLayout()
        
        # Cl goal
        self.cl_enabled = QCheckBox("Cl")
        self.cl_enabled.setChecked(True)
        self.cl_value = QDoubleSpinBox()
        self.cl_value.setRange(-2.0, 2.0)
        self.cl_value.setSingleStep(0.1)
        self.cl_value.setValue(0.8)
        self.cl_value.setEnabled(True)
        self.cl_enabled.toggled.connect(self.cl_value.setEnabled)
        
        cl_layout = QHBoxLayout()
        cl_layout.addWidget(self.cl_enabled)
        cl_layout.addWidget(QLabel("Target Cl:"))
        cl_layout.addWidget(self.cl_value)
        goals_layout.addRow(cl_layout)
        
        # Cd goal
        self.cd_enabled = QCheckBox("Cd")
        self.cd_enabled.setChecked(True)
        self.cd_value = QDoubleSpinBox()
        self.cd_value.setRange(0.0, 1.0)
        self.cd_value.setSingleStep(0.01)
        self.cd_value.setValue(0.05)
        self.cd_value.setEnabled(True)
        self.cd_enabled.toggled.connect(self.cd_value.setEnabled)
        
        cd_layout = QHBoxLayout()
        cd_layout.addWidget(self.cd_enabled)
        cd_layout.addWidget(QLabel("Target Cd:"))
        cd_layout.addWidget(self.cd_value)
        goals_layout.addRow(cd_layout)
        
        # Strouhal goal
        self.strouhal_enabled = QCheckBox("Strouhal")
        self.strouhal_enabled.setChecked(False)
        self.strouhal_value = QDoubleSpinBox()
        self.strouhal_value.setRange(0.0, 1.0)
        self.strouhal_value.setSingleStep(0.01)
        self.strouhal_value.setValue(0.2)
        self.strouhal_value.setEnabled(False)
        self.strouhal_enabled.toggled.connect(self.strouhal_value.setEnabled)
        
        strouhal_layout = QHBoxLayout()
        strouhal_layout.addWidget(self.strouhal_enabled)
        strouhal_layout.addWidget(QLabel("Target Strouhal:"))
        strouhal_layout.addWidget(self.strouhal_value)
        goals_layout.addRow(strouhal_layout)
        
        # AoA goal
        self.aoa_enabled = QCheckBox("AoA")
        self.aoa_enabled.setChecked(False)
        self.aoa_value = QDoubleSpinBox()
        self.aoa_value.setRange(-15.0, 15.0)
        self.aoa_value.setSingleStep(0.5)
        self.aoa_value.setValue(0.0)
        self.aoa_value.setEnabled(False)
        self.aoa_enabled.toggled.connect(self.aoa_value.setEnabled)
        
        aoa_layout = QHBoxLayout()
        aoa_layout.addWidget(self.aoa_enabled)
        aoa_layout.addWidget(QLabel("Target AoA (°):"))
        aoa_layout.addWidget(self.aoa_value)
        goals_layout.addRow(aoa_layout)
        
        goals_group.setLayout(goals_layout)
        layout.addWidget(goals_group)
        
        # Weights Group
        weights_group = QGroupBox("Goal Weights")
        weights_layout = QFormLayout()
        
        self.cl_weight = QDoubleSpinBox()
        self.cl_weight.setRange(0.0, 10.0)
        self.cl_weight.setSingleStep(0.1)
        self.cl_weight.setValue(1.0)
        weights_layout.addRow("Cl Weight:", self.cl_weight)
        
        self.cd_weight = QDoubleSpinBox()
        self.cd_weight.setRange(0.0, 10.0)
        self.cd_weight.setSingleStep(0.1)
        self.cd_weight.setValue(1.0)
        weights_layout.addRow("Cd Weight:", self.cd_weight)
        
        self.strouhal_weight = QDoubleSpinBox()
        self.strouhal_weight.setRange(0.0, 10.0)
        self.strouhal_weight.setSingleStep(0.1)
        self.strouhal_weight.setValue(1.0)
        weights_layout.addRow("Strouhal Weight:", self.strouhal_weight)
        
        self.shape_reg = QDoubleSpinBox()
        self.shape_reg.setRange(0.0, 1.0)
        self.shape_reg.setSingleStep(0.01)
        self.shape_reg.setValue(0.1)
        weights_layout.addRow("Shape Regularization:", self.shape_reg)
        
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def get_goals(self) -> Dict[str, Any]:
        """Get current goal settings"""
        return {
            'target_cl': self.cl_value.value() if self.cl_enabled.isChecked() else None,
            'target_cd': self.cd_value.value() if self.cd_enabled.isChecked() else None,
            'target_strouhal': self.strouhal_value.value() if self.strouhal_enabled.isChecked() else None,
            'target_aoa': self.aoa_value.value() if self.aoa_enabled.isChecked() else None,
            'cl_weight': self.cl_weight.value(),
            'cd_weight': self.cd_weight.value(),
            'strouhal_weight': self.strouhal_weight.value(),
            'shape_regularization': self.shape_reg.value()
        }


class AirfoilSelectionPanel(QWidget):
    """Panel for airfoil selection and parameters"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_ui()
        self._populate_airfoils()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Airfoil selection
        airfoil_group = QGroupBox("Airfoil Configuration")
        airfoil_layout = QFormLayout()
        
        self.airfoil_combo = QComboBox()
        airfoil_layout.addRow("Airfoil:", self.airfoil_combo)
        
        self.chord_spin = QDoubleSpinBox()
        self.chord_spin.setRange(0.1, 2.0)
        self.chord_spin.setSingleStep(0.05)
        self.chord_spin.setValue(0.3)
        airfoil_layout.addRow("Chord Length:", self.chord_spin)
        
        self.aoa_spin = QDoubleSpinBox()
        self.aoa_spin.setRange(-15.0, 15.0)
        self.aoa_spin.setSingleStep(0.5)
        self.aoa_spin.setValue(0.0)
        airfoil_layout.addRow("Angle of Attack (°):", self.aoa_spin)
        
        self.pos_x_spin = QDoubleSpinBox()
        self.pos_x_spin.setRange(1.0, 15.0)
        self.pos_x_spin.setSingleStep(0.5)
        self.pos_x_spin.setValue(5.0)
        airfoil_layout.addRow("Position X:", self.pos_x_spin)
        
        self.pos_y_spin = QDoubleSpinBox()
        self.pos_y_spin.setRange(0.5, 2.5)
        self.pos_y_spin.setSingleStep(0.1)
        self.pos_y_spin.setValue(1.5)
        airfoil_layout.addRow("Position Y:", self.pos_y_spin)
        
        airfoil_group.setLayout(airfoil_layout)
        layout.addWidget(airfoil_group)
        
        # Design parameters (for optimization)
        design_group = QGroupBox("Design Parameters (Optimization)")
        design_layout = QFormLayout()
        
        self.camber_spin = QDoubleSpinBox()
        self.camber_spin.setRange(0.0, 0.1)
        self.camber_spin.setSingleStep(0.001)
        self.camber_spin.setValue(0.02)
        self.camber_spin.setDecimals(3)
        design_layout.addRow("Camber:", self.camber_spin)
        
        self.camber_pos_spin = QDoubleSpinBox()
        self.camber_pos_spin.setRange(0.1, 0.9)
        self.camber_pos_spin.setSingleStep(0.05)
        self.camber_pos_spin.setValue(0.4)
        design_layout.addRow("Camber Position:", self.camber_pos_spin)
        
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.05, 0.25)
        self.thickness_spin.setSingleStep(0.01)
        self.thickness_spin.setValue(0.12)
        design_layout.addRow("Thickness:", self.thickness_spin)
        
        design_group.setLayout(design_layout)
        layout.addWidget(design_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _populate_airfoils(self):
        """Populate airfoil dropdown"""
        try:
            from obstacles.naca_airfoils import NACA_AIRFOILS
            self.airfoil_combo.addItems(list(NACA_AIRFOILS.keys()))
            self.airfoil_combo.setCurrentText("NACA 0012")
        except ImportError:
            self.airfoil_combo.addItems(["NACA 0012", "NACA 2412", "NACA 4412"])
    
    def get_airfoil_config(self) -> Dict[str, Any]:
        """Get current airfoil configuration"""
        return {
            'designation': self.airfoil_combo.currentText(),
            'chord_length': self.chord_spin.value(),
            'angle_of_attack': self.aoa_spin.value(),
            'position_x': self.pos_x_spin.value(),
            'position_y': self.pos_y_spin.value()
        }
    
    def get_design_params(self) -> Dict[str, float]:
        """Get current design parameters"""
        return {
            'camber': self.camber_spin.value(),
            'camber_position': self.camber_pos_spin.value(),
            'thickness': self.thickness_spin.value(),
            'angle_of_attack': self.aoa_spin.value()
        }


class VariableSelectionPanel(QWidget):
    """Panel for selecting which variables to optimize (max 2)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Variable selection group
        var_group = QGroupBox("Variables to Optimize (max 2)")
        var_layout = QVBoxLayout()
        
        # Variable checkboxes
        self.aoa_var = QCheckBox("Angle of Attack (AoA)")
        self.aoa_var.setChecked(True)
        
        self.thickness_var = QCheckBox("Thickness")
        self.thickness_var.setChecked(True)
        
        self.camber_pos_var = QCheckBox("Camber Position")
        self.camber_pos_var.setChecked(False)
        
        self.camber_var = QCheckBox("Camber")
        self.camber_var.setChecked(False)
        
        # Connect to enforce max 2 selection
        self.aoa_var.stateChanged.connect(self._enforce_max_2)
        self.thickness_var.stateChanged.connect(self._enforce_max_2)
        self.camber_pos_var.stateChanged.connect(self._enforce_max_2)
        self.camber_var.stateChanged.connect(self._enforce_max_2)
        
        var_layout.addWidget(self.aoa_var)
        var_layout.addWidget(self.thickness_var)
        var_layout.addWidget(self.camber_pos_var)
        var_layout.addWidget(self.camber_var)
        
        var_group.setLayout(var_layout)
        layout.addWidget(var_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _enforce_max_2(self):
        """Enforce maximum 2 variable selection"""
        checkboxes = [self.aoa_var, self.thickness_var, self.camber_pos_var, self.camber_var]
        checked = [cb for cb in checkboxes if cb.isChecked()]
        
        if len(checked) > 2:
            # Uncheck the most recently checked box (the sender)
            sender = self.sender()
            sender.setChecked(False)
    
    def get_selected_variables(self) -> Dict[str, bool]:
        """Get selected variables"""
        return {
            'aoa': self.aoa_var.isChecked(),
            'thickness': self.thickness_var.isChecked(),
            'camber_position': self.camber_pos_var.isChecked(),
            'camber': self.camber_var.isChecked()
        }


class GridConfigPanel(QWidget):
    """Panel for grid configuration"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        grid_group = QGroupBox("Grid Configuration")
        grid_layout = QFormLayout()
        
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(128, 2048)
        self.nx_spin.setSingleStep(64)
        self.nx_spin.setValue(512)
        grid_layout.addRow("Nx:", self.nx_spin)
        
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(32, 512)
        self.ny_spin.setSingleStep(16)
        self.ny_spin.setValue(96)
        grid_layout.addRow("Ny:", self.ny_spin)
        
        self.lx_spin = QDoubleSpinBox()
        self.lx_spin.setRange(5.0, 50.0)
        self.lx_spin.setSingleStep(1.0)
        self.lx_spin.setValue(20.0)
        grid_layout.addRow("Lx:", self.lx_spin)
        
        self.ly_spin = QDoubleSpinBox()
        self.ly_spin.setRange(1.0, 10.0)
        self.ly_spin.setSingleStep(0.5)
        self.ly_spin.setValue(3.0)
        grid_layout.addRow("Ly:", self.ly_spin)
        
        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)
        
        # Flow parameters
        flow_group = QGroupBox("Flow Parameters")
        flow_layout = QFormLayout()
        
        self.u_inlet_spin = QDoubleSpinBox()
        self.u_inlet_spin.setRange(0.1, 10.0)
        self.u_inlet_spin.setSingleStep(0.1)
        self.u_inlet_spin.setValue(1.0)
        flow_layout.addRow("U_inlet:", self.u_inlet_spin)
        
        self.nu_spin = QDoubleSpinBox()
        self.nu_spin.setRange(0.0001, 0.1)
        self.nu_spin.setSingleStep(0.0001)
        self.nu_spin.setDecimals(4)
        self.nu_spin.setValue(0.003)
        flow_layout.addRow("ν:", self.nu_spin)
        
        self.re_label = QLabel("Re: 1000.0")
        flow_layout.addRow("Reynolds:", self.re_label)
        
        # Update Reynolds when U or ν changes
        self.u_inlet_spin.valueChanged.connect(self._update_reynolds)
        self.nu_spin.valueChanged.connect(self._update_reynolds)
        
        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _update_reynolds(self):
        """Update Reynolds number display"""
        u = self.u_inlet_spin.value()
        nu = self.nu_spin.value()
        L = self.lx_spin.value() / 10.0  # Characteristic length
        re = u * L / nu
        self.re_label.setText(f"Re: {re:.1f}")
    
    def get_grid_config(self) -> Dict[str, Any]:
        """Get current grid configuration"""
        return {
            'nx': self.nx_spin.value(),
            'ny': self.ny_spin.value(),
            'Lx': self.lx_spin.value(),
            'Ly': self.ly_spin.value(),
            'u_inlet': self.u_inlet_spin.value(),
            'nu': self.nu_spin.value()
        }


class OptimizationControlPanel(QWidget):
    """Panel for optimization control"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Optimizer selection
        optimizer_group = QGroupBox("Optimizer Method")
        optimizer_layout = QFormLayout()
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("JAX Autodiff", "autodiff")
        self.optimizer_combo.addItem("Finite Differences", "finite_diff")
        self.optimizer_combo.setCurrentIndex(0)  # Default to autodiff
        optimizer_layout.addRow("Method:", self.optimizer_combo)
        
        optimizer_group.setLayout(optimizer_layout)
        layout.addWidget(optimizer_group)
        
        # Optimization settings
        opt_group = QGroupBox("Optimization Settings")
        opt_layout = QFormLayout()
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setSingleStep(10)
        self.max_iter_spin.setValue(100)
        opt_layout.addRow("Max Iterations:", self.max_iter_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.5)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.01)
        opt_layout.addRow("Learning Rate:", self.lr_spin)
        
        self.conv_thresh_spin = QDoubleSpinBox()
        self.conv_thresh_spin.setRange(1e-8, 1e-3)
        self.conv_thresh_spin.setSingleStep(1e-6)
        self.conv_thresh_spin.setDecimals(8)
        self.conv_thresh_spin.setValue(1e-6)
        opt_layout.addRow("Convergence Threshold:", self.conv_thresh_spin)
        
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start Optimization")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white;")
        button_layout.addWidget(self.stop_button)
        
        self.reset_button = QPushButton("Reset")
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get current optimization settings"""
        return {
            'optimizer_method': self.optimizer_combo.currentData(),
            'max_iterations': self.max_iter_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'convergence_threshold': self.conv_thresh_spin.value()
        }
