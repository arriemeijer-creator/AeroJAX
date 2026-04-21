
"""
Navier-Stokes Flow Simulator - Main Application
A modular, maintainable fluid dynamics visualization tool
"""

import sys
import time
from typing import Optional, Dict, Any
import numpy as np
import jax.numpy as jnp
import jax
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QMenuBar, QDockWidget, QHBoxLayout, QSizePolicy, QSplitter
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

# Application modules
from viewer.ui_components import ControlPanel, InfoPanel, FloatingControlBar
from viewer.visualization import FlowVisualization, ObstacleRenderer, SDFVisualization
from viewer.simulation_controller import SimulationController, RecordingManager, DataExporter
from viewer.config import ConfigManager, PerformanceSettings
from viewer.parameter_handlers import ParameterHandlers
from viewer.display_manager import DisplayManager
from viewer.flow_manager import FlowManager
from viewer.naca_handler import NACAHandler
from viewer.modern_stylesheet import MODERN_DARK_THEME, MODERN_LIGHT_THEME

# Solver imports
from solver import (
    GridParams, FlowParams, FlowConstraints, GeometryParams, SimulationParams, 
    CavityGeometryParams, BaselineSolver, compute_forces
)

# Optional modules - NACA airfoils
# Inverse design GUI integration not available

try:
    from obstacles.naca_airfoils import NACA_AIRFOILS
    NACA_AVAILABLE = True
except ImportError:
    NACA_AVAILABLE = False


class BaselineViewerRefactored(QMainWindow, ParameterHandlers, DisplayManager, FlowManager, NACAHandler):
    """
    Main application window for the Navier-Stokes flow simulator.
    
    Manages the simulation, visualization, and user interface components
    in a modular, maintainable architecture.
    """
    
    def __init__(self, solver: BaselineSolver, config: ConfigManager):
        """Initialize the viewer with a simulation solver instance."""
        super().__init__()
        self.solver = solver
        self.config = config
        
        # Store initial configuration for full reset
        self._store_initial_config()
        
        # Initialize state
        self.is_paused = False
        self.is_dark_theme = False  # Start with light theme
        
        # Initialize all components
        self._initialize_components()
        self._build_user_interface()
        self._connect_event_handlers()
        self._load_initial_state()
        
        # Set up Redux store subscription for unidirectional data flow
        self.setup_store_subscription()
        
        print("Application initialized successfully")
        
        # Connect adaptive dt checkbox manually to prevent automatic triggering
        self.control_panel.adaptive_dt_checkbox.clicked.connect(self.on_adaptive_dt_checkbox_clicked)
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def _store_initial_config(self) -> None:
        """Store initial configuration for full reset capability."""
        self.initial_config = {
            'grid': {
                'nx': self.solver.grid.nx,
                'ny': self.solver.grid.ny,
                'lx': self.solver.grid.lx,
                'ly': self.solver.grid.ly
            },
            'flow': {
                'Re': self.solver.flow.Re,
                'U_inf': self.solver.flow.U_inf,
                'nu': self.solver.flow.nu,
                'L_char': self.solver.flow.L_char,
                'lock_U': self.solver.flow.constraints.lock_U,
                'lock_nu': self.solver.flow.constraints.lock_nu,
                'lock_Re': self.solver.flow.constraints.lock_Re
            },
            'geometry': {
                'center_x': float(self.solver.geom.center_x),
                'center_y': float(self.solver.geom.center_y),
                'radius': float(self.solver.geom.radius)
            },
            'simulation': {
                'eps': self.solver.sim_params.eps,
                'eps_multiplier': self.solver.sim_params.eps_multiplier,
                'flow_type': self.solver.sim_params.flow_type,
                'obstacle_type': self.solver.sim_params.obstacle_type,
                'naca_airfoil': getattr(self.solver.sim_params, 'naca_airfoil', 'NACA 0012'),
                'naca_x': getattr(self.solver.sim_params, 'naca_x', 2.5),
                'naca_y': getattr(self.solver.sim_params, 'naca_y', 1.875),
                'naca_chord': getattr(self.solver.sim_params, 'naca_chord', 2.0),
                'naca_angle': getattr(self.solver.sim_params, 'naca_angle', -10.0),
                'advection_scheme': self.solver.sim_params.advection_scheme,
                'pressure_solver': self.solver.sim_params.pressure_solver,
                'fixed_dt': self.solver.sim_params.fixed_dt,
                'adaptive_dt': self.solver.sim_params.adaptive_dt,
                'use_les': self.solver.sim_params.use_les,
                'les_model': self.solver.sim_params.les_model
            }
        }
        print("Initial configuration stored for full reset")
    
    def _initialize_components(self) -> None:
        """Create all modular components that make up the application."""
        # User interface panels
        self.control_panel = ControlPanel(parent=self)
        self.info_panel = InfoPanel(self)
        self.floating_control_bar = FloatingControlBar(parent=self)
        
        # Visualization components
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.flow_viz = FlowVisualization(self.plot_widget, solver=self.solver, control_panel=self.control_panel)
        
        # Connect visualization to info panel for marker visibility control
        self.info_panel.set_visualization(self.flow_viz)
        # Connect solver to info panel for airfoil metrics computation control
        self.info_panel.set_solver(self.solver)
        
        # Configure colormaps from saved settings
        self.flow_viz.set_initial_colormaps(
            velocity_colormap=self.config.viz_config.default_velocity_colormap,
            vorticity_colormap=self.config.viz_config.default_vorticity_colormap,
            pressure_colormap='RdBu'  # Default pressure colormap
        )
        
        # Visual overlays - restore obstacle renderer with error handling
        try:
            if hasattr(self.flow_viz, 'vel_outline') and hasattr(self.flow_viz, 'vort_outline') and hasattr(self.flow_viz, 'scalar_outline') and hasattr(self.flow_viz, 'pressure_outline'):
                if self.flow_viz.vel_outline is not None and self.flow_viz.vort_outline is not None and self.flow_viz.scalar_outline is not None and self.flow_viz.pressure_outline is not None:
                    self.obstacle_renderer = ObstacleRenderer(
                        self.flow_viz.vel_outline,
                        self.flow_viz.vort_outline,
                        self.flow_viz.scalar_outline,
                        self.flow_viz.pressure_outline
                    )
                else:
                    self.obstacle_renderer = None
            else:
                self.obstacle_renderer = None
        except Exception as e:
            print(f"Warning: Failed to create obstacle renderer: {e}")
            self.obstacle_renderer = None
            
        self.sdf_viz = SDFVisualization(
            self.flow_viz.vel_sdf, 
            self.flow_viz.vort_sdf,
            self.flow_viz
        )
        
        # Simulation management
        self.sim_controller = SimulationController(self.solver, self.control_panel, self.info_panel)
        self.recording_manager = RecordingManager()
        self.data_exporter = DataExporter()
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_update_time = time.time()
        self.current_sim_fps = 0.0
    
    def _build_user_interface(self) -> None:
        """Construct the main application window layout."""
        # Apply modern light theme stylesheet (default)
        self.setStyleSheet(MODERN_LIGHT_THEME)
        
        # Update theme toggle button to reflect light mode
        self.control_panel.theme_toggle_btn.setText("☀️")
        self.control_panel.theme_toggle_btn.setToolTip("Switch to Dark Mode")
        
        # Set plot background for light mode
        self.config.viz_config.plot_background = "#ffffff"

        # Add floating control bar as a dock widget
        self.floating_control_dock = QDockWidget("Quick Controls", self)
        self.floating_control_dock.setWidget(self.floating_control_bar)
        self.floating_control_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                                               QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.floating_control_dock.setAllowedAreas(Qt.DockWidgetArea.TopDockWidgetArea |
                                                   Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.floating_control_dock)
        self.floating_control_dock.setFloating(True)  # Make it float on startup

        # Window properties
        self.setWindowTitle("Baseline Navier-Stokes Solver")
        self.setGeometry(100, 100, 
                       PerformanceSettings.DEFAULT_WINDOW_WIDTH,
                       PerformanceSettings.DEFAULT_WINDOW_HEIGHT)
        
        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left sidebar with control and info panels (30% initial width)
        left_sidebar = QWidget()
        left_sidebar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Control panel (top of left sidebar) - 70% of vertical space
        left_layout.addWidget(self.control_panel, 7)
        
        # Add a horizontal divider line
        from PyQt6.QtWidgets import QFrame
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setLineWidth(1)
        divider.setMinimumHeight(2)
        divider.setStyleSheet("QFrame { background-color: #888; border: none; }")
        left_layout.addWidget(divider)
        
        # Info panel (bottom of left sidebar) - 30% of vertical space
        left_layout.addWidget(self.info_panel, 3)
        
        left_sidebar.setLayout(left_layout)
        
        # Right side with plot widget (70% initial width)
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add widgets to splitter with initial sizes
        splitter.addWidget(left_sidebar)
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([480, 1120])  # Initial 30/70 split for 1600px width
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(splitter)  # Make splitter the central widget
        
        # Visualization styling
        self.plot_widget.setBackground(self.config.viz_config.plot_background)
        self.plot_widget.setMinimumSize(
            PerformanceSettings.MIN_PLOT_WIDTH, 
            PerformanceSettings.MIN_PLOT_HEIGHT
        )
    
    def _connect_event_handlers(self) -> None:
        """Connect all UI controls to their callback functions."""
        # Simulation controls
        self.control_panel.start_btn.clicked.connect(self.start_simulation)
        self.control_panel.pause_btn.clicked.connect(self.pause_simulation)
        self.control_panel.reset_btn.clicked.connect(self.reset_simulation)

        # Floating control bar controls
        self.floating_control_bar.start_btn.clicked.connect(self.start_simulation)
        self.floating_control_bar.pause_btn.clicked.connect(self.pause_simulation)
        self.floating_control_bar.reset_btn.clicked.connect(self.reset_simulation)
        self.floating_control_bar.velocity_colormap_combo.currentTextChanged.connect(self.change_velocity_colormap)
        self.floating_control_bar.vorticity_colormap_combo.currentTextChanged.connect(self.change_vorticity_colormap)
        self.floating_control_bar.pressure_colormap_combo.currentTextChanged.connect(self.change_pressure_colormap)
        self.floating_control_bar.error_metrics_cb.stateChanged.connect(self.on_error_metrics_checkbox_changed)
        self.floating_control_bar.airfoil_metrics_cb.stateChanged.connect(self.on_airfoil_metrics_checkbox_changed)

        # Floating control bar dye controls
        self.floating_control_bar.dye_x_slider.valueChanged.connect(self.update_dye_marker_from_sliders)
        self.floating_control_bar.dye_y_slider.valueChanged.connect(self.update_dye_marker_from_sliders)
        self.floating_control_bar.inject_dye_btn.pressed.connect(self.inject_dye_start)
        self.floating_control_bar.inject_dye_btn.released.connect(self.inject_dye_stop)
        # Sync floating control bar dye sliders to sidebar
        self.floating_control_bar.dye_x_slider.valueChanged.connect(self._sync_dye_x_slider_from_floating)
        self.floating_control_bar.dye_y_slider.valueChanged.connect(self._sync_dye_y_slider_from_floating)

        # Synchronize floating control bar checkboxes with info panel
        self._sync_floating_checkboxes()

        # Connect info panel checkbox changes to sync floating checkboxes
        self.info_panel.diagnostics_checkbox.stateChanged.connect(self._sync_error_metrics_from_info_panel)
        self.info_panel.compute_airfoil_metrics_cb.toggled.connect(self._sync_airfoil_metrics_from_info_panel)

        self.control_panel.theme_toggle_btn.clicked.connect(self.toggle_theme)
        
        # Parameter controls
        self.control_panel.apply_re_btn.clicked.connect(self.update_reynolds_number)
        self.info_panel.metrics_frame_skip_input.valueChanged.connect(self.update_metrics_frame_skip)
        self.info_panel.diagnostics_checkbox.stateChanged.connect(self.on_metrics_checkbox_changed)
        self.control_panel.inject_dye_btn.clicked.connect(self.inject_dye)
        self.info_panel.save_csv_btn.clicked.connect(self.save_csv_dialog)
        
        # Constraint lock controls
        self.control_panel.lock_u_cb.stateChanged.connect(self.on_lock_u_changed)
        self.control_panel.lock_nu_cb.stateChanged.connect(self.on_lock_nu_changed)
        self.control_panel.lock_re_cb.stateChanged.connect(self.on_lock_re_changed)
        self.control_panel.apply_precision_btn.clicked.connect(self.update_precision)
        self.control_panel.apply_grid_btn.clicked.connect(self.update_grid_resolution)
        self.control_panel.apply_cylinder_btn.clicked.connect(self.update_cylinder_radius)
        self.control_panel.apply_cylinder_array_btn.clicked.connect(self.update_cylinder_array_params)
        self.control_panel.apply_epsilon_btn.clicked.connect(self.update_epsilon)
        self.control_panel.apply_dt_btn.clicked.connect(self.update_timestep)
        self.control_panel.apply_frame_skip_btn.clicked.connect(self.update_frame_skip)
        self.control_panel.apply_vis_fps_btn.clicked.connect(self.update_visualization_fps)
        self.control_panel.apply_les_btn.clicked.connect(self.update_les_settings)
        self.control_panel.apply_vcycles_btn.clicked.connect(self.update_vcycles)
        
        # Visualization toggles
        self.control_panel.show_velocity_checkbox.stateChanged.connect(self.toggle_velocity_display)
        self.control_panel.show_vorticity_checkbox.stateChanged.connect(self.toggle_vorticity_display)
        self.control_panel.show_sdf_checkbox.stateChanged.connect(self.toggle_sdf_overlay)
        self.control_panel.show_streamlines_checkbox.stateChanged.connect(self.toggle_streamlines)
        self.control_panel.fast_mode_checkbox.stateChanged.connect(self.toggle_fast_mode)
        self.control_panel.velocity_colormap_combo.currentTextChanged.connect(self.change_velocity_colormap)
        self.control_panel.vorticity_colormap_combo.currentTextChanged.connect(self.change_vorticity_colormap)
        self.control_panel.pressure_colormap_combo.currentTextChanged.connect(self.change_pressure_colormap)
        self.control_panel.log_colorscale_checkbox.stateChanged.connect(self.update_visualization_settings)
        self.control_panel.spatial_colorscale_checkbox.stateChanged.connect(self.update_visualization_settings)
        self.control_panel.upscale_slider.valueChanged.connect(self.change_upscale_factor)
        
        # Dye injection button - continuous injection while held
        self.inject_dye_pressed = False
        self.control_panel.inject_dye_btn.pressed.connect(self.inject_dye_start)
        self.control_panel.inject_dye_btn.released.connect(self.inject_dye_stop)
        
        # Dye slider signals to update marker position
        self.control_panel.dye_x_slider.valueChanged.connect(self.update_dye_marker_from_sliders)
        self.control_panel.dye_y_slider.valueChanged.connect(self.update_dye_marker_from_sliders)
        # Sync sidebar dye sliders to floating control bar
        self.control_panel.dye_x_slider.valueChanged.connect(self._sync_dye_x_slider_from_sidebar)
        self.control_panel.dye_y_slider.valueChanged.connect(self._sync_dye_y_slider_from_sidebar)
        
        # Export and recording
        self.control_panel.export_btn.clicked.connect(self.export_simulation_data)
        self.control_panel.record_btn.clicked.connect(self.toggle_video_recording)
        self.control_panel.save_btn.clicked.connect(self.save_recorded_video)
        
        # Auto-scaling
        self.control_panel.autofit_velocity_btn.clicked.connect(self.auto_scale_velocity_plot)
        self.control_panel.autofit_vorticity_btn.clicked.connect(self.auto_scale_vorticity_plot)
        self.control_panel.autofit_both_btn.clicked.connect(self.auto_scale_both_plots)
        
        # Advanced features
        # Temporarily disconnect adaptive dt checkbox to prevent automatic triggering
        # self.control_panel.adaptive_dt_checkbox.stateChanged.connect(self.toggle_adaptive_timestep)
        
        if hasattr(self.control_panel, 'flow_combo'):
            self.control_panel.flow_combo.currentTextChanged.connect(self.on_flow_type_selected)
        
        if hasattr(self.control_panel, 'obstacle_button_group') and self.control_panel.obstacle_button_group:
            # Radio buttons are connected in ui_components.py via buttonClicked signal
            pass
        
        if hasattr(self.control_panel, 'apply_naca_btn') and self.control_panel.apply_naca_btn:
            self.control_panel.apply_naca_btn.clicked.connect(self.apply_naca_airfoil_settings)
        
        # Freeform drawing button
        if hasattr(self.control_panel, 'draw_custom_btn') and self.control_panel.draw_custom_btn:
            self.control_panel.draw_custom_btn.clicked.connect(self.launch_freeform_drawer)
        
        # Real-time angle slider connection
        if hasattr(self.control_panel, 'angle_slider') and self.control_panel.angle_slider:
            self.control_panel.angle_slider.valueChanged.connect(self.on_angle_slider_changed)
            self.control_panel.angle_slider.sliderReleased.connect(self.on_angle_slider_released)
        
        # Spinbox angle connection for bidirectional sync
        if hasattr(self.control_panel, 'angle_spinbox') and self.control_panel.angle_spinbox:
            self.control_panel.angle_spinbox.valueChanged.connect(self.on_angle_spinbox_changed)
        
        # Simulation callbacks
        self.sim_controller.callbacks = {
            'data_ready': self.handle_simulation_data,
            'fps_update': self.update_simulation_fps_display,
            'metrics_ready': self.handle_metrics_data
        }
        
        # Keyboard shortcuts
        self._setup_keyboard_shortcuts()
    
    def _setup_keyboard_shortcuts(self) -> None:
        """Configure keyboard shortcuts for common actions."""
        try:
            from PyQt6.QtWidgets import QShortcut
            from PyQt6.QtGui import QKeySequence
            
            shortcut_auto_fit = QShortcut(QKeySequence("Ctrl+F"), self)
            shortcut_auto_fit.activated.connect(self.auto_scale_both_plots)
            
            shortcut_reset_ranges = QShortcut(QKeySequence("Ctrl+R"), self)
            shortcut_reset_ranges.activated.connect(self.reset_plot_view)
        except ImportError:
            print("Keyboard shortcuts not available")
    
    def _load_initial_state(self) -> None:
        """Load the initial simulation state and configure UI."""
        # Create obstacle renderer if it wasn't created during initialization
        if self.obstacle_renderer is None:
            try:
                # Try to get outline items from flow_viz
                vel_outline = getattr(self.flow_viz, 'vel_outline', None) if hasattr(self, 'flow_viz') else None
                vort_outline = getattr(self.flow_viz, 'vort_outline', None) if hasattr(self, 'flow_viz') else None
                scalar_outline = getattr(self.flow_viz, 'scalar_outline', None) if hasattr(self, 'flow_viz') else None
                pressure_outline = getattr(self.flow_viz, 'pressure_outline', None) if hasattr(self, 'flow_viz') else None

                if vel_outline is not None and vort_outline is not None and scalar_outline is not None and pressure_outline is not None:
                    self.obstacle_renderer = ObstacleRenderer(vel_outline, vort_outline, scalar_outline, pressure_outline)
                    print("Obstacle renderer created successfully in _load_initial_state")
                else:
                    print(f"Warning: Outline items not available - vel_outline={vel_outline}, vort_outline={vort_outline}, scalar_outline={scalar_outline}, pressure_outline={pressure_outline}")
            except Exception as e:
                print(f"Warning: Failed to create obstacle renderer: {e}")
                self.obstacle_renderer = None
        
        # Update obstacle outlines with initial position
        if self.obstacle_renderer is not None and self.solver is not None:
            try:
                self.obstacle_renderer.update_obstacle_outlines(self.solver, force_update=True)
                print("Obstacle outlines updated with initial position")
            except Exception as e:
                print(f"Warning: Failed to update initial obstacle outlines: {e}")
        
        # Populate UI controls with current solver values
        self.control_panel.re_input.setValue(int(self.solver.flow.Re))
        if hasattr(self.control_panel, 'u_input'):
            self.control_panel.u_input.setValue(float(self.solver.flow.U_inf))
        if hasattr(self.control_panel, 'nu_input'):
            self.control_panel.nu_input.setValue(float(self.solver.flow.nu))
        if hasattr(self.control_panel, 'lock_u_cb'):
            self.control_panel.lock_u_cb.setChecked(self.solver.flow.constraints.lock_U)
        if hasattr(self.control_panel, 'lock_nu_cb'):
            self.control_panel.lock_nu_cb.setChecked(self.solver.flow.constraints.lock_nu)
        if hasattr(self.control_panel, 'lock_re_cb'):
            self.control_panel.lock_re_cb.setChecked(self.solver.flow.constraints.lock_Re)
        self.control_panel.flow_combo.setCurrentText(self.solver.sim_params.flow_type)
        self.control_panel.dt_spinbox.setValue(self.solver.dt)
        
        # Block signal during initial setup to prevent automatic triggering
        self.control_panel.adaptive_dt_checkbox.blockSignals(True)
        # Keep adaptive dt checkbox checked by default (as set in ui_components.py)
        self.control_panel.adaptive_dt_checkbox.blockSignals(False)
        
        # Initialize NACA controls with current solver values
        if hasattr(self.control_panel, 'angle_spinbox') and self.control_panel.angle_spinbox is not None:
            self.control_panel.angle_spinbox.setValue(self.solver.sim_params.naca_angle)
        if hasattr(self.control_panel, 'angle_slider') and self.control_panel.angle_slider is not None:
            # Convert angle to slider value (-20 to +20 degrees to -200 to 200)
            slider_value = int(self.solver.sim_params.naca_angle * 10.0)
            self.control_panel.angle_slider.setValue(slider_value)
        if hasattr(self.control_panel, 'chord_spinbox') and self.control_panel.chord_spinbox is not None:
            self.control_panel.chord_spinbox.setValue(self.solver.sim_params.naca_chord)
        if hasattr(self.control_panel, 'naca_combo') and self.control_panel.naca_combo is not None:
            self.control_panel.naca_combo.setCurrentText(self.solver.sim_params.naca_airfoil)
        if hasattr(self.control_panel, 'obstacle_button_group') and self.control_panel.obstacle_button_group is not None:
            # Update radio button selection based on current obstacle type
            if self.solver.sim_params.obstacle_type == 'cylinder':
                self.control_panel.cylinder_radio.setChecked(True)
            elif self.solver.sim_params.obstacle_type == 'naca_airfoil':
                self.control_panel.naca_radio.setChecked(True)
            elif self.solver.sim_params.obstacle_type == 'cow':
                self.control_panel.cow_radio.setChecked(True)
            elif self.solver.sim_params.obstacle_type == 'three_cylinder_array':
                self.control_panel.cylinder_array_radio.setChecked(True)
        
        # Configure grid display - set spinboxes to current solver values
        self.control_panel.grid_x_spinbox.setValue(self.solver.grid.nx)
        self.control_panel.grid_y_spinbox.setValue(self.solver.grid.ny)
        
        # Initialize cylinder radius with current solver value
        if hasattr(self.control_panel, 'cylinder_radius_spinbox'):
            radius_value = float(self.solver.geom.radius.item()) if hasattr(self.solver.geom.radius, 'item') else float(self.solver.geom.radius)
            self.control_panel.cylinder_radius_spinbox.setValue(radius_value)
        
        # Initialize epsilon slider with current solver value
        if hasattr(self.control_panel, 'epsilon_slider'):
            epsilon_value = float(self.solver.sim_params.eps.item()) if hasattr(self.solver.sim_params.eps, 'item') else float(self.solver.sim_params.eps)
            slider_value = int(epsilon_value * 100)  # Convert epsilon (0.01-0.50) to slider value (1-50)
            self.control_panel.epsilon_slider.setValue(slider_value)
            self.control_panel.epsilon_label.setText(f"{epsilon_value:.2f}")
        
        # Set up visualization scaling - now handled internally in setup_plots
        grid_nx, grid_ny = self.solver.grid.nx, self.solver.grid.ny
        grid_lx, grid_ly = self.solver.grid.lx, self.solver.grid.ly
        
        # Set initial plot boundaries
        self.flow_viz.vel_plot.setXRange(0, grid_lx)
        self.flow_viz.vel_plot.setYRange(0, grid_ly)
        self.flow_viz.vort_plot.setXRange(0, grid_lx)
        self.flow_viz.vort_plot.setYRange(0, grid_ly)
        
        # Start visualization refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        refresh_interval = int(1000 / self.config.viz_config.target_vis_fps)
        self.refresh_timer.start(refresh_interval)
        
        # Activate adaptive dt mode on startup if checkbox is checked (moved here after timer creation)
        if (hasattr(self.control_panel, 'adaptive_dt_checkbox') and 
            self.control_panel.adaptive_dt_checkbox.isChecked()):
            self.toggle_adaptive_timestep(2)  # Checked state
        
        # Update status displays
        self._update_solver_info()
    
    # -------------------------------------------------------------------------
    # Simulation Control
    # -------------------------------------------------------------------------
    
    def start_simulation(self) -> None:
        """Begin or resume the fluid simulation."""
        try:
            # If we're paused, just resume
            if self.is_paused:
                self.sim_controller.resume_simulation()
                self.is_paused = False
            else:
                # Start new simulation (start_simulation handles worker cleanup)
                self.sim_controller.start_simulation({
                    'data_ready': self.handle_simulation_data,
                    'fps_update': self.update_simulation_fps_display,
                    'metrics_ready': self.handle_metrics_data
                })

            self.on_simulation_started()  # Lock x-position slider
            self.control_panel.start_btn.setEnabled(False)
            self.control_panel.pause_btn.setEnabled(True)
            refresh_interval = int(1000 / self.config.viz_config.target_vis_fps)
            self.refresh_timer.start(refresh_interval)

        except Exception as e:
            print(f"ERROR: start_simulation failed: {e}")
            import traceback
            traceback.print_exc()
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def pause_simulation(self) -> None:
        """Pause the fluid simulation."""
        self.sim_controller.pause_simulation()
        self.refresh_timer.stop()
        
        self.on_simulation_stopped()  # Unlock x-position slider
        self.control_panel.start_btn.setEnabled(True)
        self.control_panel.pause_btn.setEnabled(False)
        self.is_paused = True  # Track that we're paused, not stopped
    
    def full_reset_to_initial(self) -> None:
        """Reset entire GUI and solver to initial launch state."""
        print("Performing full reset to initial configuration...")
        
        # Stop simulation
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        
        self.on_simulation_stopped()  # Unlock x-position slider
        
        # Recreate entire solver object to ensure clean state
        ic = self.initial_config
        
        # Create new solver with initial parameters
        new_grid = GridParams(
            nx=ic['grid']['nx'],
            ny=ic['grid']['ny'],
            lx=ic['grid']['lx'],
            ly=ic['grid']['ly']
        )
        
        new_flow = FlowParams(
            Re=ic['flow']['Re'],
            U_inf=ic['flow']['U_inf'],
            nu=ic['flow']['nu'],
            L_char=ic['flow']['L_char']
        )
        new_flow.constraints.lock_U = ic['flow']['lock_U']
        new_flow.constraints.lock_nu = ic['flow']['lock_nu']
        new_flow.constraints.lock_Re = ic['flow']['lock_Re']
        
        new_geom = GeometryParams(
            center_x=ic['geometry']['center_x'],
            center_y=ic['geometry']['center_y'],
            radius=ic['geometry']['radius']
        )
        
        new_sim_params = SimulationParams(
            eps=ic['simulation']['eps'],
            eps_multiplier=ic['simulation'].get('eps_multiplier', 1.0),
            flow_type=ic['simulation']['flow_type'],
            obstacle_type=ic['simulation']['obstacle_type'],
            naca_airfoil=ic['simulation']['naca_airfoil'],
            naca_x=ic['simulation']['naca_x'],
            naca_y=ic['simulation']['naca_y'],
            naca_chord=ic['simulation']['naca_chord'],
            naca_angle=ic['simulation']['naca_angle'],
            advection_scheme=ic['simulation']['advection_scheme'],
            pressure_solver=ic['simulation']['pressure_solver'],
            fixed_dt=ic['simulation']['fixed_dt'],
            adaptive_dt=ic['simulation']['adaptive_dt'],
            use_les=ic['simulation']['use_les'],
            les_model=ic['simulation']['les_model']
        )
        
        # Create new solver instance
        new_solver = BaselineSolver(new_grid, new_flow, new_geom, new_sim_params)
        
        # Replace old solver
        self.solver = new_solver
        
        # Update simulation controller reference
        self.sim_controller.solver = new_solver
        
        # Update metrics worker reference
        if self.sim_controller.metrics_worker:
            self.sim_controller.metrics_worker.solver = new_solver
        
        # Update info_panel solver reference for metrics computation
        if hasattr(self, 'info_panel') and self.info_panel:
            self.info_panel.set_solver(new_solver)
        
        # Update visualization references
        if hasattr(self, 'flow_viz') and self.flow_viz is not None:
            self.flow_viz.solver = new_solver
        if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer is not None:
            self.obstacle_renderer.solver = new_solver
        
        # Restore GUI controls
        self.control_panel.grid_x_spinbox.setValue(ic['grid']['nx'])
        self.control_panel.grid_y_spinbox.setValue(ic['grid']['ny'])
        self.control_panel.u_input.setValue(ic['flow']['U_inf'])
        self.control_panel.nu_input.setValue(ic['flow']['nu'])
        self.control_panel.re_input.setValue(int(ic['flow']['Re']))
        self.control_panel.lock_u_cb.setChecked(ic['flow']['lock_U'])
        self.control_panel.lock_nu_cb.setChecked(ic['flow']['lock_nu'])
        self.control_panel.lock_re_cb.setChecked(ic['flow']['lock_Re'])
        
        # Reset iteration counter
        self.solver.iteration = 0
        self.solver.history = {
            'time': [0.0],
            'dt': [self.solver.dt],
            'l2_change': [0.0],
            'rms_change': [0.0],
            'max_change': [0.0],
            'change_99p': [0.0],
            'rel_change': [0.0],
            'l2_change_u': [0.0],
            'l2_change_v': [0.0],
            'rms_divergence': [0.0],
            'l2_divergence': [0.0],
            'drag': [0.0],
            'lift': [0.0],
            'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 'Cp_min': [], 'wake_deficit': []}
        }
        
        # Clear paused state
        self.is_paused = False
        
        # Update button states
        self.control_panel.start_btn.setEnabled(True)
        self.control_panel.pause_btn.setEnabled(False)
        
        print("Full reset complete - GUI and solver restored to initial state")
    
    def reset_simulation(self, keep_timer_running: bool = True) -> None:
        """Reset the simulation to initial conditions (full GUI reset)."""
        try:
            # Clear error plot visualization data before reset
            if hasattr(self, 'flow_viz'):
                self.flow_viz.clear_error_plot()
            self.full_reset_to_initial()
        except Exception as e:
            print(f"ERROR during reset: {e}")
            import traceback
            traceback.print_exc()
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def _reset_von_karman_flow(self) -> None:
        """Reset von Karman (obstacle) flow configuration."""
        obstacle_type = getattr(self.solver.sim_params, 'obstacle_type', 'NOT_SET')
        
        # Preserve current NACA parameters if they exist
        preserved_naca_angle = getattr(self.solver.sim_params, 'naca_angle', None)
        preserved_naca_chord = getattr(self.solver.sim_params, 'naca_chord', None)
        preserved_naca_airfoil = getattr(self.solver.sim_params, 'naca_airfoil', None)
        preserved_naca_x = getattr(self.solver.sim_params, 'naca_x', None)
        preserved_naca_y = None
        
        if obstacle_type == 'naca_airfoil':
            # Save current NACA parameters
            preserved_naca_angle = getattr(self.solver.sim_params, 'naca_angle', 0.0)
            preserved_naca_chord = getattr(self.solver.sim_params, 'naca_chord', 2.0)
            preserved_naca_airfoil = getattr(self.solver.sim_params, 'naca_airfoil', 'NACA 0012')
            preserved_naca_x = getattr(self.solver.sim_params, 'naca_x', 2.0)
            preserved_naca_y = getattr(self.solver.sim_params, 'naca_y', 2.25)
        
        # Initialize the flow
        self.solver._initialize_von_karman_flow()
        
        # Restore NACA parameters if they were preserved
        if obstacle_type == 'naca_airfoil' and preserved_naca_angle is not None:
            self.solver.sim_params.naca_angle = preserved_naca_angle
            # Re-scale chord to 15% of lx (don't preserve old value)
            chord_percentage = 0.15
            self.solver.sim_params.naca_chord = chord_percentage * self.solver.grid.lx
            self.solver.sim_params.naca_airfoil = preserved_naca_airfoil
            self.solver.sim_params.naca_x = preserved_naca_x
            self.solver.sim_params.naca_y = preserved_naca_y
            
            print(f"Preserved NACA parameters during reset: angle={preserved_naca_angle:.1f}°, chord={self.solver.sim_params.naca_chord:.3f} (15% of lx)")
        
        # Recompute mask with preserved parameters
        self.solver.mask = self.solver._compute_mask()
    
    def _reset_cavity_flow(self) -> None:
        """Reset lid-driven cavity flow configuration."""
        self.solver.sim_params.obstacle_type = 'none'
        self.solver.sim_params.obstacle_x = 0.0
        self.solver.sim_params.obstacle_y = 0.0
        self.solver.sim_params.obstacle_radius = 0.0
        self.solver.sim_params.obstacle_chord = 0.0
        self.solver.sim_params.obstacle_angle = 0.0
        self.solver.sim_params.naca_airfoil = 'none'
        
        self.solver._initialize_cavity_flow()
        self.solver.mask = self.solver._compute_mask()
    
    # -------------------------------------------------------------------------
    # Theme Management
    # -------------------------------------------------------------------------
    
    def toggle_theme(self) -> None:
        """Toggle between light and dark themes."""
        self.is_dark_theme = not self.is_dark_theme
        
        if self.is_dark_theme:
            self.setStyleSheet(MODERN_DARK_THEME)
            self.control_panel.theme_toggle_btn.setText("🌙")
            self.control_panel.theme_toggle_btn.setToolTip("Switch to Light Mode")
            plot_bg = "#1e1e2e"
        else:
            self.setStyleSheet(MODERN_LIGHT_THEME)
            self.control_panel.theme_toggle_btn.setText("☀️")
            self.control_panel.theme_toggle_btn.setToolTip("Switch to Dark Mode")
            plot_bg = "#ffffff"
        
        # Update plot background color
        self.plot_widget.setBackground(plot_bg)
        self.config.viz_config.plot_background = plot_bg
        
        # Update visualization if it exists
        if hasattr(self, 'flow_viz') and self.flow_viz:
            # PlotItem uses setBrush for background
            self.flow_viz.vel_plot.getViewBox().setBackgroundColor(plot_bg)
            self.flow_viz.vort_plot.getViewBox().setBackgroundColor(plot_bg)

    def _sync_floating_checkboxes(self) -> None:
        """Synchronize floating control bar checkboxes with info panel checkboxes."""
        # Sync error metrics checkbox
        self.floating_control_bar.error_metrics_cb.blockSignals(True)
        self.floating_control_bar.error_metrics_cb.setChecked(self.info_panel.diagnostics_checkbox.isChecked())
        self.floating_control_bar.error_metrics_cb.blockSignals(False)

        # Sync airfoil metrics checkbox
        self.floating_control_bar.airfoil_metrics_cb.blockSignals(True)
        self.floating_control_bar.airfoil_metrics_cb.setChecked(self.info_panel.compute_airfoil_metrics_cb.isChecked())
        self.floating_control_bar.airfoil_metrics_cb.blockSignals(False)

    def _sync_error_metrics_from_info_panel(self, state) -> None:
        """Sync floating error metrics checkbox when info panel checkbox changes."""
        self.floating_control_bar.error_metrics_cb.blockSignals(True)
        self.floating_control_bar.error_metrics_cb.setChecked(state == 2)  # Qt.CheckState.Checked
        self.floating_control_bar.error_metrics_cb.blockSignals(False)

    def _sync_airfoil_metrics_from_info_panel(self, checked: bool) -> None:
        """Sync floating airfoil metrics checkbox when info panel checkbox changes."""
        self.floating_control_bar.airfoil_metrics_cb.blockSignals(True)
        self.floating_control_bar.airfoil_metrics_cb.setChecked(checked)
        self.floating_control_bar.airfoil_metrics_cb.blockSignals(False)

    def _sync_dye_x_slider_from_floating(self):
        """Sync dye X slider from floating control bar to sidebar"""
        self.control_panel.dye_x_slider.blockSignals(True)
        self.control_panel.dye_x_slider.setValue(self.floating_control_bar.dye_x_slider.value())
        self.control_panel.dye_x_slider.blockSignals(False)

    def _sync_dye_y_slider_from_floating(self):
        """Sync dye Y slider from floating control bar to sidebar"""
        self.control_panel.dye_y_slider.blockSignals(True)
        self.control_panel.dye_y_slider.setValue(self.floating_control_bar.dye_y_slider.value())
        self.control_panel.dye_y_slider.blockSignals(False)

    def _sync_dye_x_slider_from_sidebar(self):
        """Sync dye X slider from sidebar to floating control bar"""
        self.floating_control_bar.dye_x_slider.blockSignals(True)
        self.floating_control_bar.dye_x_slider.setValue(self.control_panel.dye_x_slider.value())
        self.floating_control_bar.dye_x_slider.blockSignals(False)

    def _sync_dye_y_slider_from_sidebar(self):
        """Sync dye Y slider from sidebar to floating control bar"""
        self.floating_control_bar.dye_y_slider.blockSignals(True)
        self.floating_control_bar.dye_y_slider.setValue(self.control_panel.dye_y_slider.value())
        self.floating_control_bar.dye_y_slider.blockSignals(False)

    def on_airfoil_metrics_checkbox_changed(self, state) -> None:
        """Handle airfoil metrics checkbox state change from floating control bar."""
        is_checked = (state == 2)  # Qt.CheckState.Checked
        # Sync info panel checkbox
        self.info_panel.compute_airfoil_metrics_cb.blockSignals(True)
        self.info_panel.compute_airfoil_metrics_cb.setChecked(is_checked)
        self.info_panel.compute_airfoil_metrics_cb.blockSignals(False)

        # Call the info panel's toggle method to update solver
        self.info_panel._toggle_airfoil_metrics(is_checked)

        # Also toggle stagnation and separation markers when airfoil metrics is activated
        if is_checked:
            self.info_panel.show_stagnation_marker_cb.blockSignals(True)
            self.info_panel.show_stagnation_marker_cb.setChecked(True)
            self.info_panel.show_stagnation_marker_cb.blockSignals(False)
            self.info_panel._toggle_stagnation_marker(True)

            self.info_panel.show_separation_marker_cb.blockSignals(True)
            self.info_panel.show_separation_marker_cb.setChecked(True)
            self.info_panel.show_separation_marker_cb.blockSignals(False)
            self.info_panel._toggle_separation_marker(True)
        else:
            self.info_panel.show_stagnation_marker_cb.blockSignals(True)
            self.info_panel.show_stagnation_marker_cb.setChecked(False)
            self.info_panel.show_stagnation_marker_cb.blockSignals(False)
            self.info_panel._toggle_stagnation_marker(False)

            self.info_panel.show_separation_marker_cb.blockSignals(True)
            self.info_panel.show_separation_marker_cb.setChecked(False)
            self.info_panel.show_separation_marker_cb.blockSignals(False)
            self.info_panel._toggle_separation_marker(False)

    def on_error_metrics_checkbox_changed(self, state) -> None:
        """Handle error metrics checkbox state change from floating control bar."""
        is_checked = (state == 2)  # Qt.CheckState.Checked
        # Sync info panel checkbox
        self.info_panel.diagnostics_checkbox.blockSignals(True)
        self.info_panel.diagnostics_checkbox.setChecked(is_checked)
        self.info_panel.diagnostics_checkbox.blockSignals(False)

        # Call the existing handler to update metrics worker
        self.on_metrics_checkbox_changed(state)

        # Toggle error plot visibility
        if hasattr(self, 'flow_viz') and self.flow_viz and hasattr(self.flow_viz, 'l2_plot') and self.flow_viz.l2_plot:
            self.flow_viz.l2_plot.setVisible(is_checked)
    
    # -------------------------------------------------------------------------
    # Visualization Controls
    # -------------------------------------------------------------------------
    
    def toggle_velocity_display(self, state) -> None:
        """Show or hide velocity magnitude visualization."""
        is_visible = (state == 2)  # Qt.Checked
        self.config.viz_config.show_velocity = is_visible
        self.flow_viz.set_visibility(is_visible, self.config.viz_config.show_vorticity)
        print(f"Velocity display {'on' if is_visible else 'off'}")
    
    def toggle_vorticity_display(self, state) -> None:
        """Show or hide vorticity visualization."""
        is_visible = (state == 2)
        self.config.viz_config.show_vorticity = is_visible
        self.flow_viz.set_visibility(self.config.viz_config.show_velocity, is_visible)
        print(f"Vorticity display {'on' if is_visible else 'off'}")
    
    def toggle_sdf_overlay(self, state) -> None:
        """Show or hide the signed distance field mask."""
        is_visible = (state == 2)
        self.sdf_viz.set_visibility(is_visible)
        print(f"SDF mask {'on' if is_visible else 'off'}")
    
    def inject_dye_at_slider_position(self):
        """Inject dye at the position specified by sliders"""
        # Get slider values (0-100 range)
        x_percent = self.control_panel.dye_x_slider.value() / 100.0
        y_percent = self.control_panel.dye_y_slider.value() / 100.0

        # Map to simulation domain
        x_pos = x_percent * self.solver.grid.lx
        y_pos = y_percent * self.solver.grid.ly

        # Inject dye
        self.solver.inject_dye(x_pos, y_pos, amount=0.5)

    def inject_dye(self):
        """Inject dye at current slider position"""
        # Get slider values (0-100 range)
        x_percent = self.control_panel.dye_x_slider.value() / 100.0
        y_percent = self.control_panel.dye_y_slider.value() / 100.0

        # Map to simulation domain
        x_pos = x_percent * self.solver.grid.lx
        y_pos = y_percent * self.solver.grid.ly

        # Inject dye
        self.solver.inject_dye(x_pos, y_pos, amount=0.5)

    def save_csv_dialog(self):
        """Open file dialog to save CSV with custom filename"""
        from PyQt6.QtWidgets import QFileDialog

        # Get current angle of attack for default filename
        aoa = getattr(self.solver.sim_params, 'naca_angle', 10.0)
        default_filename = f"AoA={aoa:.0f}_metrics.csv"

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Metrics CSV",
            default_filename,
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            # Call save_csv_to_file method (inherited from DisplayManager)
            success = self.save_csv_to_file(file_path)
            if success:
                print(f"CSV saved successfully to {file_path}")
            else:
                print("Failed to save CSV")

    def inject_dye_start(self):
        """Start continuous dye injection when button is pressed"""
        self.inject_dye_pressed = True
        self.inject_dye_at_slider_position()  # Inject immediately
    
    def inject_dye_stop(self):
        """Stop continuous dye injection when button is released"""
        self.inject_dye_pressed = False
    
    def update_dye_marker_from_sliders(self):
        """Update the dye marker position based on slider values"""
        if self.solver is None:
            return
        
        # Get slider values (0-100 range)
        x_percent = self.control_panel.dye_x_slider.value() / 100.0
        y_percent = self.control_panel.dye_y_slider.value() / 100.0
        
        # Map to simulation domain
        x_pos = x_percent * self.solver.grid.lx
        y_pos = y_percent * self.solver.grid.ly
        
        # Update marker
        self.flow_viz.update_dye_marker(x_pos, y_pos)
    
    def launch_freeform_drawer(self):
        """Launch the pygame drawing interface for custom obstacles."""
        try:
            from PyQt6.QtWidgets import QProgressDialog
            from PyQt6.QtCore import Qt
            
            from obstacles.freeform_drawer import FreeformDrawer, create_freeform_mask_smooth
            
            print("Launching freeform obstacle drawer...")
            
            # Show progress dialog while pygame initializes
            progress = QProgressDialog("Initializing Pygame drawing window...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setWindowTitle("Loading")
            progress.setCancelButton(None)
            progress.show()
            
            # Process events to ensure the dialog is displayed
            QApplication.processEvents()
            
            # Create and run the drawer (pygame initializes inside run())
            drawer = FreeformDrawer(width=512, height=512)
            mask = drawer.run()
            
            # Close progress dialog after pygame window closes
            progress.close()
            
            if mask is not None:
                print("Custom obstacle mask created successfully!")
                
                # Store the mask in the solver for later use
                self.solver.custom_obstacle_mask = mask
                
                # Update the solver to use the custom obstacle
                self.solver.sim_params.obstacle_type = 'custom'
                self.solver.sim_params.custom_mask = mask
                
                # Recompute the mask with the custom obstacle
                self.solver.mask = self.solver._compute_mask()
                
                # Update obstacle outlines
                if self.obstacle_renderer is not None:
                    self.obstacle_renderer.update_obstacle_outlines(self.solver, force_update=True)
                
                print("Custom obstacle applied to simulation!")
            else:
                print("Drawing cancelled or no mask saved.")
                
        except ImportError as e:
            print(f"Error: Required dependencies for freeform drawing not available: {e}")
            print("Please ensure pygame and scipy are installed.")
            self.flow_viz.scalar_plot.hide()
            # Hide dye marker when scalar display is disabled
            if hasattr(self.flow_viz, 'dye_marker') and self.flow_viz.dye_marker is not None:
                self.flow_viz.dye_marker.setVisible(False)
            print("Dye injection disabled")
    
    def toggle_streamlines(self, state):
        """Enable/disable streamlines visualization"""
        is_enabled = (state == 2)
        if hasattr(self, 'flow_viz') and self.flow_viz is not None:
            self.flow_viz.toggle_streamlines(is_enabled)
            print(f"Streamlines {'enabled' if is_enabled else 'disabled'}")
    
    def toggle_fast_mode(self, state):
        """Enable/disable fast mode (RK2 vs RK3)"""
        is_enabled = (state == 2)
        if hasattr(self.solver, 'sim_params'):
            # Stop simulation before switching schemes to prevent instability
            self.refresh_timer.stop()
            if hasattr(self, 'sim_controller'):
                self.sim_controller.stop_simulation()
            
            self.solver.sim_params.fast_mode = is_enabled
            print(f"Fast Mode {'enabled (RK2 - Real-Time Interaction)' if is_enabled else 'disabled (RK3 - Scientific/F1 Grade)'}")
            
            # Reset adaptive timestep controller if it exists
            if hasattr(self.solver, 'dt_controller') and self.solver.dt_controller is not None:
                # Reset to initial timestep to account for different stability limits
                from timestepping.adaptivedt import PIDController
                target_div = 0.1 if is_enabled else 0.15  # RK2 needs tighter control
                self.solver.dt_controller = PIDController(
                    dt=self.solver.dt,
                    target_divergence=target_div,
                    kp=0.1, ki=0.01, kd=0.0,
                    dt_min=1e-6, dt_max=self.solver.dt_max
                )
                print(f"Adaptive dt controller reset with target_div={target_div}")
            
            # Clear JAX cache to recompile with new fast_mode setting
            import jax
            jax.clear_caches()
            
            # Clear JIT cache to force recompilation with new fast_mode
            if hasattr(self.solver, '_jit_cache'):
                self.solver._jit_cache.clear()
            
            # Recompile solver with new fast_mode
            if hasattr(self.solver, '_step_jit'):
                self.solver._step_jit = jax.jit(self.solver._step)
            
            # Re-enable start button
            if hasattr(self.control_panel, 'start_btn'):
                self.control_panel.start_btn.setEnabled(True)
                self.control_panel.pause_btn.setEnabled(False)
    
    def change_velocity_colormap(self, colormap_name: str) -> None:
        """Change the color scheme for velocity plots."""
        if hasattr(self, 'flow_viz') and self.flow_viz is not None:
            self.flow_viz.change_velocity_colormap(colormap_name)
            self.config.viz_config.default_velocity_colormap = colormap_name
            # Sync floating control bar
            if hasattr(self, 'floating_control_bar') and self.floating_control_bar is not None:
                self.floating_control_bar.velocity_colormap_combo.blockSignals(True)
                self.floating_control_bar.velocity_colormap_combo.setCurrentText(colormap_name)
                self.floating_control_bar.velocity_colormap_combo.blockSignals(False)
            # Sync control panel
            if hasattr(self, 'control_panel') and self.control_panel is not None:
                self.control_panel.velocity_colormap_combo.blockSignals(True)
                self.control_panel.velocity_colormap_combo.setCurrentText(colormap_name)
                self.control_panel.velocity_colormap_combo.blockSignals(False)

    def change_vorticity_colormap(self, colormap_name: str) -> None:
        """Change vorticity colormap"""
        self.flow_viz.change_vorticity_colormap(colormap_name)
        # Sync floating control bar
        if hasattr(self, 'floating_control_bar') and self.floating_control_bar is not None:
            self.floating_control_bar.vorticity_colormap_combo.blockSignals(True)
            self.floating_control_bar.vorticity_colormap_combo.setCurrentText(colormap_name)
            self.floating_control_bar.vorticity_colormap_combo.blockSignals(False)
        # Sync control panel
        if hasattr(self, 'control_panel') and self.control_panel is not None:
            self.control_panel.vorticity_colormap_combo.blockSignals(True)
            self.control_panel.vorticity_colormap_combo.setCurrentText(colormap_name)
            self.control_panel.vorticity_colormap_combo.blockSignals(False)

    def change_pressure_colormap(self, colormap_name: str) -> None:
        """Change pressure colormap"""
        if hasattr(self, 'flow_viz') and self.flow_viz is not None:
            self.flow_viz.change_pressure_colormap(colormap_name)
            # Sync floating control bar
            if hasattr(self, 'floating_control_bar') and self.floating_control_bar is not None:
                self.floating_control_bar.pressure_colormap_combo.blockSignals(True)
                self.floating_control_bar.pressure_colormap_combo.setCurrentText(colormap_name)
                self.floating_control_bar.pressure_colormap_combo.blockSignals(False)
            # Sync control panel
            if hasattr(self, 'control_panel') and self.control_panel is not None:
                self.control_panel.pressure_colormap_combo.blockSignals(True)
                self.control_panel.pressure_colormap_combo.setCurrentText(colormap_name)
                self.control_panel.pressure_colormap_combo.blockSignals(False)

    def update_visualization_settings(self) -> None:
        """Update visualization settings when color scale checkboxes change"""
        # The visualization loop automatically checks checkbox states each frame
        # No additional action needed - just ensures settings are applied immediately
        pass
    
    def change_upscale_factor(self, value: int) -> None:
        """Change the visualization upscale factor for smooth rendering"""
        if hasattr(self, 'flow_viz') and self.flow_viz is not None:
            self.flow_viz.upscale_factor = value
            self.control_panel.upscale_label.setText(f"{value}x")

    def auto_scale_velocity_plot(self) -> None:
        """Adjust velocity plot scale to fit current data."""
        if hasattr(self, 'flow_viz') and self.flow_viz is not None:
            self.flow_viz.auto_fit_velocity()
    
    def auto_scale_vorticity_plot(self) -> None:
        """Adjust vorticity plot scale to fit current data."""
        if hasattr(self, 'flow_viz') and self.flow_viz is not None:
            self.flow_viz.auto_fit_vorticity()
    
    def auto_scale_both_plots(self) -> None:
        """Adjust both plots to fit their current data."""
        if hasattr(self, 'flow_viz') and self.flow_viz is not None:
            self.flow_viz.auto_fit_both()
    
    def reset_plot_view(self) -> None:
        """Reset plot boundaries to the original domain extents."""
        if hasattr(self, 'flow_viz') and self.flow_viz is not None and hasattr(self, 'solver'):
            self.flow_viz.reset_plot_ranges(
                self.solver.grid.nx, 
                self.solver.grid.ny, 
                self.solver.grid.lx, 
                self.solver.grid.ly
            )
    
    def on_adaptive_dt_checkbox_clicked(self, checked: bool) -> None:
        """Handle adaptive dt checkbox click - only triggered by user interaction."""
        if checked:
            self.toggle_adaptive_timestep(2)  # Checked state
        else:
            self.toggle_adaptive_timestep(0)  # Unchecked state
    
    def toggle_adaptive_timestep(self, state) -> None:
        """Switch between fixed and adaptive timestep modes."""
        is_adaptive = (state == 2)
        
        # Allow adaptive timestep for all flow types including LDC
        if is_adaptive:
            print(f"Enabling adaptive timestep for {self.solver.sim_params.flow_type} flow")
        
        # Stop simulation completely before switching modes
        if hasattr(self, 'sim_controller'):
            self.sim_controller.stop_simulation()
        
        self.refresh_timer.stop()
        
        # Reset UI controls to stopped state
        self.control_panel.start_btn.setEnabled(True)
        self.control_panel.pause_btn.setEnabled(False)
        
        try:
            if is_adaptive:
                print("Switching to adaptive timestep mode...")
                
                # Simple approach - just set adaptive dt and update UI
                self.solver.set_adaptive_dt()
                self.control_panel.dt_spinbox.setEnabled(False)
                self.control_panel.apply_dt_btn.setEnabled(False)
                print("Switched to adaptive timestep mode")
                
                # Only basic recompilation - skip complex operations
                try:
                    self.solver._step_jit = jax.jit(self.solver._step)
                    print("DEBUG: Basic JIT recompilation for adaptive dt")
                except Exception as jit_error:
                    print(f"Warning: JIT recompilation failed: {jit_error}")
                    print("Continuing with existing JIT functions")
                
            else:
                current_dt = self.solver.dt
                self.solver.set_fixed_dt(current_dt)
                self.control_panel.dt_spinbox.setEnabled(True)
                self.control_panel.apply_dt_btn.setEnabled(True)
                self.control_panel.dt_spinbox.setValue(self.solver.dt)
                print(f"Switched to fixed timestep mode: {self.solver.dt:.6f}")
                
        except Exception as e:
            print(f"Error switching timestep mode: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency recovery: try to restore fixed dt mode
            try:
                print("Emergency recovery: restoring fixed dt mode...")
                self.solver.set_fixed_dt(0.001)
                self.control_panel.dt_spinbox.setEnabled(True)
                self.control_panel.apply_dt_btn.setEnabled(True)
                self.control_panel.dt_spinbox.setValue(0.001)
                self.control_panel.adaptive_dt_checkbox.setChecked(False)
                self.solver._step_jit = jax.jit(self.solver._step)
                print("Emergency recovery: fixed dt mode restored")
            except Exception as recovery_error:
                print(f"Emergency recovery failed: {recovery_error}")
        
        # Don't call reset_simulation - it causes crashes with adaptive dt
    
    # -------------------------------------------------------------------------
    # Export and Recording
    # -------------------------------------------------------------------------

    def export_simulation_data(self) -> None:
        """Export simulation results to a file."""
        self.data_exporter.export_simulation_data(self.solver)
    
    def toggle_video_recording(self) -> None:
        """Start or stop recording the visualization as video."""
        new_text = self.recording_manager.toggle_recording()
        self.control_panel.record_btn.setText(new_text)
        self.control_panel.save_btn.setEnabled(self.recording_manager.has_frames())
    
    def save_recorded_video(self) -> None:
        """Save the recorded video to disk."""
        self.recording_manager.save_video(self)
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def closeEvent(self, event) -> None:
        """Clean up resources when closing the application."""
        print("Cleaning up application...")
        
        self.sim_controller.stop_simulation()
        self.refresh_timer.stop()
        if hasattr(self, 'flow_viz'):
            self.flow_viz.clear()
        
        print("Cleanup complete")
        event.accept()


# -----------------------------------------------------------------------------
# Application Entry Point
# -----------------------------------------------------------------------------

def run_visualization(solver: BaselineSolver, config: ConfigManager) -> None:
    """Launch the visualization application (no longer used - splash screen moved to main)."""
    print("Starting Navier-Stokes flow visualization...")
    print(f"Domain: X=[0.0, {solver.grid.lx:.1f}] m, Y=[0.0, {solver.grid.ly:.1f}] m")
    print(f"Grid: {solver.grid.nx} x {solver.grid.ny}")
    print(f"Reynolds number: {solver.flow.Re:.1f}")
    print("Close the window to stop the simulation.")
    
    # Splash screen is now handled in main()
    # This function is kept for backward compatibility but is not used


def main() -> None:
    """Main entry point for the application."""
    try:
        print("=" * 60)
        print("Navier-Stokes Flow Simulator")
        print("=" * 60)
        
        # === CREATE AND SHOW SPLASH SCREEN FIRST ===
        # Create QApplication first (required for splash screen)
        app = QApplication(sys.argv)
        
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWidgets import QSplashScreen
        from PyQt6.QtCore import Qt
        
        splash_pixmap = QPixmap("viewer/ui_components/splash screen.png")
        splash_pixmap = splash_pixmap.scaled(int(splash_pixmap.width() * 0.75), int(splash_pixmap.height() * 0.75), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()  # Force splash to render immediately
        
        # Now do all the heavy initialization while splash is visible
        # Add global exception handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            print(f"CRITICAL ERROR: {exc_type.__name__}: {exc_value}")
            print("Attempting to continue simulation...")
            # Don't crash - just report and continue
        
        sys.excepthook = handle_exception
        
        # Load configuration (this can be slow)
        splash.showMessage("Loading configuration...")
        app.processEvents()
        config = ConfigManager()
        
        # Create solver with default settings
        splash.showMessage("Creating simulation grid...")
        app.processEvents()
        grid = GridParams(
            nx=config.sim_config.default_nx, 
            ny=config.sim_config.default_ny, 
            lx=config.sim_config.default_lx, 
            ly=config.sim_config.default_ly
        )
        
        splash.showMessage("Configuring flow parameters...")
        app.processEvents()
        flow = FlowParams(
            Re=config.sim_config.default_reynolds,
            nu=config.sim_config.default_nu,
            constraints=FlowConstraints(lock_U=False, lock_nu=True, lock_Re=True, lock_L=True)
        )
        
        splash.showMessage("Setting up geometry...")
        app.processEvents()
        geometry = GeometryParams(
            center_x=jnp.array(2.5), 
            center_y=jnp.array(config.sim_config.default_ly / 2.0),  # Center in Y
            radius=jnp.array(0.18)  # Still need radius for compatibility
        )
        
        simulation_params = SimulationParams(
            eps=0.01,
            obstacle_type='naca_airfoil',
            naca_airfoil='NACA 0012',
            naca_x=2.5,
            naca_y=config.sim_config.default_ly / 2.0,  # Center in Y
            naca_chord=0.15 * config.sim_config.default_lx,  # 15% of domain width
            naca_angle=10.0
        )
        
        splash.showMessage("Initializing solver...")
        app.processEvents()

        # Clear JAX caches to force recompilation with updated code
        jax.clear_caches()

        solver = BaselineSolver(
            grid, flow, geometry, simulation_params,
            dt=config.sim_config.default_dt
        )
        
        # Display simulation details
        X, Y = solver.grid.X, solver.grid.Y
        print(f"\nSimulation Configuration:")
        print(f"  Domain: X=[{float(X[0, 0]):.1f}, {float(X[-1, 0]):.1f}] m")
        print(f"          Y=[{float(Y[0, 0]):.1f}, {float(Y[0, -1]):.1f}] m")
        print(f"  Grid spacing: dx={solver.grid.dx:.4f} m, dy={solver.grid.dy:.4f} m")
        print(f"  Initial perturbation applied")
        print(f"  Ready for vortex shedding visualization\n")
        
        # Create and show main window, then close splash
        splash.showMessage("Building user interface...")
        app.processEvents()
        viewer = BaselineViewerRefactored(solver, config)
        viewer.showMaximized()
        splash.finish(viewer)  # This closes the splash screen
        
        # Add global exception handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            print(f"Unhandled exception: {exc_type.__name__}: {exc_value}")
            print("Application continuing despite exception...")
        
        sys.excepthook = handle_exception
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        print("Application cannot continue.")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
