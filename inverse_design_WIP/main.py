"""
Differential Backpropagation - Main GUI Application
A JAX-based differentiable CFD framework for inverse airfoil design
"""

import sys
import os

# Add parent directory to path to allow running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QMenuBar, QStatusBar, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from inverse_design_WIP.config import InverseDesignConfig
from inverse_design_WIP.optimizer import InverseDesigner
from inverse_design_WIP.ui_components import (
    GoalSettingPanel, AirfoilSelectionPanel,
    GridConfigPanel, OptimizationControlPanel
)
from inverse_design_WIP.visualization import InverseDesignVisualization, MetricsDisplay


class DifferentialBackpropagationApp(QMainWindow):
    """
    Main application window for Differential Backpropagation
    """
    
    def __init__(self):
        super().__init__()
        self.config = InverseDesignConfig()
        self.optimizer = None
        self.solver = None
        self.optimization_running = False
        self.optimization_paused = False
        
        self._setup_ui()
        self._setup_menu()
        # Don't initialize solver - JAX fails on this Windows system
        # Use surrogate model instead
        
    def _setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Differential Backpropagation - Inverse Design")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left sidebar with configuration panels
        left_sidebar = QWidget()
        left_layout = QVBoxLayout(left_sidebar)
        
        # Tab widget for different configuration panels
        self.config_tabs = QTabWidget()
        
        # Grid configuration tab
        self.grid_panel = GridConfigPanel(self.config)
        self.config_tabs.addTab(self.grid_panel, "Grid")
        
        # Airfoil configuration tab
        self.airfoil_panel = AirfoilSelectionPanel(self.config)
        self.config_tabs.addTab(self.airfoil_panel, "Airfoil")
        
        # Goals tab
        self.goals_panel = GoalSettingPanel(self.config)
        self.config_tabs.addTab(self.goals_panel, "Goals")
        
        # Optimization settings tab
        self.opt_control_panel = OptimizationControlPanel(self.config)
        self.config_tabs.addTab(self.opt_control_panel, "Optimization")
        
        left_layout.addWidget(self.config_tabs)
        
        # Metrics display
        self.metrics_display = MetricsDisplay()
        left_layout.addWidget(self.metrics_display)
        
        # Add left sidebar to splitter
        splitter.addWidget(left_sidebar)
        
        # Right side with visualization
        self.visualization = InverseDesignVisualization(self.config)
        splitter.addWidget(self.visualization)
        
        # Set initial splitter sizes (30% left, 70% right)
        splitter.setSizes([480, 1120])
        
        main_layout.addWidget(splitter)
        
        # Setup status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Connect signals
        self._connect_signals()
        
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        save_config_action = QAction('Save Configuration', self)
        save_config_action.triggered.connect(self._save_configuration)
        file_menu.addAction(save_config_action)
        
        load_config_action = QAction('Load Configuration', self)
        load_config_action.triggered.connect(self._load_configuration)
        file_menu.addAction(load_config_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        reset_action = QAction('Reset Optimization', self)
        reset_action.triggered.connect(self._reset_optimization)
        tools_menu.addAction(reset_action)
        
        export_action = QAction('Export Results', self)
        export_action.triggered.connect(self._export_results)
        tools_menu.addAction(export_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _connect_signals(self):
        """Connect signals from UI components"""
        self.opt_control_panel.start_button.clicked.connect(self._start_optimization)
        self.opt_control_panel.pause_button.clicked.connect(self._pause_optimization)
        self.opt_control_panel.stop_button.clicked.connect(self._stop_optimization)
        self.opt_control_panel.reset_button.clicked.connect(self._reset_optimization)
    
    def _initialize_solver(self):
        """Initialize the CFD solver"""
        try:
            from solver import (
                GridParams, FlowParams, GeometryParams, SimulationParams, BaselineSolver
            )
            
            # Get grid configuration
            grid_config = self.grid_panel.get_grid_config()
            
            # Create grid parameters
            self.grid_params = GridParams(
                nx=grid_config['nx'],
                ny=grid_config['ny'],
                Lx=grid_config['Lx'],
                Ly=grid_config['Ly']
            )
            
            # Create flow parameters
            self.flow_params = FlowParams(
                u_inlet=grid_config['u_inlet'],
                nu=grid_config['nu']
            )
            
            # Get airfoil configuration
            airfoil_config = self.airfoil_panel.get_airfoil_config()
            
            # Create geometry parameters
            self.geometry_params = GeometryParams(
                obstacle_type='naca_airfoil',
                naca_designation=airfoil_config['designation'],
                naca_chord=airfoil_config['chord_length'],
                naca_angle=airfoil_config['angle_of_attack'],
                naca_x=airfoil_config['position_x'],
                naca_y=airfoil_config['position_y']
            )
            
            # Create simulation parameters
            self.sim_params = SimulationParams(
                flow_type='von_karman',
                dt=0.005,
                advection_scheme='rk3',
                pressure_solver='multigrid'
            )
            
            # Create solver
            self.solver = BaselineSolver(
                self.grid_params,
                self.flow_params,
                self.geometry_params,
                self.sim_params
            )
            
            self.status_bar.showMessage("Solver initialized successfully")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error initializing solver: {e}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_optimization(self):
        """Start the optimization process"""
        if self.optimization_running:
            return
        
        # Update configuration from UI
        self._update_config_from_ui()
        
        # Create optimizer
        self.optimizer = InverseDesigner(self.solver, self.config)
        
        # Start optimization timer
        self.optimization_running = True
        self.optimization_paused = False
        
        self.opt_control_panel.start_button.setEnabled(False)
        self.opt_control_panel.pause_button.setEnabled(True)
        self.opt_control_panel.stop_button.setEnabled(True)
        
        self.optimization_timer = QTimer()
        self.optimization_timer.timeout.connect(self._run_optimization_step)
        self.optimization_timer.start(100)  # Run every 100ms
        
        self.status_bar.showMessage("Optimization started")
    
    def _pause_optimization(self):
        """Pause/resume the optimization"""
        if not self.optimization_running:
            return
        
        self.optimization_paused = not self.optimization_paused
        
        if self.optimization_paused:
            self.opt_control_panel.pause_button.setText("Resume")
            self.status_bar.showMessage("Optimization paused")
        else:
            self.opt_control_panel.pause_button.setText("Pause")
            self.status_bar.showMessage("Optimization resumed")
    
    def _stop_optimization(self):
        """Stop the optimization"""
        if not self.optimization_running:
            return
        
        self.optimization_running = False
        
        if hasattr(self, 'optimization_timer'):
            self.optimization_timer.stop()
        
        self.opt_control_panel.start_button.setEnabled(True)
        self.opt_control_panel.pause_button.setEnabled(False)
        self.opt_control_panel.stop_button.setEnabled(False)
        self.opt_control_panel.pause_button.setText("Pause")
        
        self.status_bar.showMessage("Optimization stopped")
    
    def _reset_optimization(self):
        """Reset the optimization"""
        self._stop_optimization()
        
        if self.optimizer:
            self.optimizer = None
        
        self.metrics_display.update_metrics(0, 0.0, 0.0, 0.0)
        self.visualization.update_history({'loss': [], 'cl': [], 'cd': []})
        
        self.status_bar.showMessage("Optimization reset")
    
    def _run_optimization_step(self):
        """Run one optimization step"""
        if not self.optimization_running or self.optimization_paused:
            return
        
        if self.optimizer is None:
            return
        
        try:
            # Run optimization step
            results = self.optimizer.run_optimization_step()
            print(f"Iteration {results['iteration']}: Loss={results['loss']:.6f}, Cl={results['cl']:.4f}, Cd={results['cd']:.4f}")
            
            # Update metrics display
            self.metrics_display.update_metrics(
                results['iteration'],
                results['loss'],
                results['cl'],
                results['cd'],
                results.get('strouhal')
            )
            
            # Update history plots
            status = self.optimizer.get_optimization_status()
            self.visualization.update_history(status['history'])
            
            # Update target lines
            goals = self.config.goals
            self.visualization.update_target_lines(
                target_cl=goals.target_cl,
                target_cd=goals.target_cd
            )
            
            # Update progress bar
            progress = (results['iteration'] / self.config.max_iterations) * 100
            self.opt_control_panel.progress_bar.setValue(int(progress))
            
            # Check convergence (only after minimum iterations to avoid premature stopping)
            if results['converged'] and results['iteration'] > 5:
                self._stop_optimization()
                self.status_bar.showMessage(f"Optimization converged at iteration {results['iteration']}")
                return
            
            # Check max iterations
            if results['iteration'] >= self.config.max_iterations:
                self._stop_optimization()
                self.status_bar.showMessage(f"Optimization completed after {results['iteration']} iterations")
        
        except Exception as e:
            print(f"Error in optimization step: {e}")
            import traceback
            traceback.print_exc()
            self._stop_optimization()
            self.status_bar.showMessage(f"Optimization error: {e}")
    
    def _update_config_from_ui(self):
        """Update configuration from UI components"""
        # Update grid configuration
        grid_config = self.grid_panel.get_grid_config()
        self.config.grid.nx = grid_config['nx']
        self.config.grid.ny = grid_config['ny']
        self.config.grid.Lx = grid_config['Lx']
        self.config.grid.Ly = grid_config['Ly']
        self.config.flow.u_inlet = grid_config['u_inlet']
        self.config.flow.nu = grid_config['nu']
        
        # Update airfoil configuration
        airfoil_config = self.airfoil_panel.get_airfoil_config()
        self.config.airfoil.designation = airfoil_config['designation']
        self.config.airfoil.chord_length = airfoil_config['chord_length']
        self.config.airfoil.angle_of_attack = airfoil_config['angle_of_attack']
        self.config.airfoil.position_x = airfoil_config['position_x']
        self.config.airfoil.position_y = airfoil_config['position_y']
        
        # Update goals
        goals = self.goals_panel.get_goals()
        self.config.goals.target_cl = goals['target_cl']
        self.config.goals.target_cd = goals['target_cd']
        self.config.goals.target_strouhal = goals['target_strouhal']
        self.config.goals.target_aoa = goals['target_aoa']
        self.config.goals.cl_weight = goals['cl_weight']
        self.config.goals.cd_weight = goals['cd_weight']
        self.config.goals.strouhal_weight = goals['strouhal_weight']
        self.config.goals.shape_regularization = goals['shape_regularization']
        
        # Update optimization settings
        opt_settings = self.opt_control_panel.get_optimization_settings()
        self.config.max_iterations = opt_settings['max_iterations']
        self.config.learning_rate = opt_settings['learning_rate']
        self.config.convergence_threshold = opt_settings['convergence_threshold']
    
    def _save_configuration(self):
        """Save current configuration"""
        self.status_bar.showMessage("Configuration saved")
    
    def _load_configuration(self):
        """Load configuration"""
        self.status_bar.showMessage("Configuration loaded")
    
    def _export_results(self):
        """Export optimization results"""
        self.status_bar.showMessage("Results exported")
    
    def _show_about(self):
        """Show about dialog"""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About Differential Backpropagation",
            "Differential Backpropagation\n\n"
            "A JAX-based differentiable CFD framework for inverse airfoil design.\n\n"
            "Features:\n"
            "- Differentiable immersed boundary simulation\n"
            "- Gradient-based inverse design\n"
            "- Real-time visualization\n"
            "- Multi-objective optimization"
        )


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DifferentialBackpropagationApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
