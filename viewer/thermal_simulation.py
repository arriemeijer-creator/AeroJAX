"""
Thermal Simulation Window
Separate window for heat equation simulation with obstacles
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QSlider, QSpinBox, 
                             QDoubleSpinBox, QGroupBox, QFrame, QDialog, QCheckBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
import jax
import jax.numpy as jnp


class ThermalSimulationWindow(QDialog):
    """Window for thermal simulation with obstacles"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thermal Simulation")
        self.resize(1000, 700)
        
        # Simulation parameters
        self.nx = 128
        self.ny = 128
        self.lx = 1.0
        self.ly = 1.0
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.thermal_diffusivity = 0.01
        self.dt = 0.001  # Smaller for stability with buoyancy
        self.ambient_temp = 20.0
        self.surface_temp = 100.0
        
        # Boussinesq parameters
        self.gravity = 9.81
        self.thermal_expansion = 0.003  # Thermal expansion coefficient
        self.kinematic_viscosity = 0.01  # Fluid viscosity
        
        # Obstacle parameters
        self.obstacle_type = 'square'  # square, rectangle, circle
        self.obstacle_size = 0.2
        self.obstacle_x = 0.5
        self.obstacle_y = 0.5
        self.obstacle_width = 0.2  # for rectangle
        self.obstacle_height = 0.1  # for rectangle
        
        # Wall boundary condition type
        self.walls_bounding = True  # True = bounding (closed walls), False = infinite flow (open)
        
        # Simulation state
        self.temperature = None
        self.u = None  # Velocity x
        self.v = None  # Velocity y
        self.mask = None
        self.running = False
        self.iteration = 0
        
        # Timer for simulation loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        
        self._setup_ui()
        self._initialize_simulation()
        
    def _setup_ui(self):
        """Setup the UI layout"""
        main_layout = QHBoxLayout()
        
        # Left panel: controls
        controls_panel = QFrame()
        controls_panel.setFixedWidth(300)
        controls_layout = QVBoxLayout()
        controls_panel.setLayout(controls_layout)
        
        # Grid size controls
        grid_group = QGroupBox("Grid Size")
        grid_layout = QVBoxLayout()
        
        nx_layout = QHBoxLayout()
        nx_layout.addWidget(QLabel("Nx:"))
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(32, 512)
        self.nx_spin.setValue(128)
        self.nx_spin.valueChanged.connect(self.update_grid_size)
        nx_layout.addWidget(self.nx_spin)
        grid_layout.addLayout(nx_layout)
        
        ny_layout = QHBoxLayout()
        ny_layout.addWidget(QLabel("Ny:"))
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(32, 512)
        self.ny_spin.setValue(128)
        self.ny_spin.valueChanged.connect(self.update_grid_size)
        ny_layout.addWidget(self.ny_spin)
        grid_layout.addLayout(ny_layout)
        
        grid_group.setLayout(grid_layout)
        controls_layout.addWidget(grid_group)
        
        # Obstacle controls
        obstacle_group = QGroupBox("Obstacle")
        obstacle_layout = QVBoxLayout()
        
        obstacle_type_layout = QHBoxLayout()
        obstacle_type_layout.addWidget(QLabel("Type:"))
        self.obstacle_combo = QComboBox()
        self.obstacle_combo.addItems(["square", "rectangle", "circle"])
        self.obstacle_combo.currentTextChanged.connect(self.update_obstacle_type)
        obstacle_type_layout.addWidget(self.obstacle_combo)
        obstacle_layout.addLayout(obstacle_type_layout)
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.05, 0.5)
        self.size_spin.setSingleStep(0.05)
        self.size_spin.setValue(0.2)
        self.size_spin.valueChanged.connect(self.update_obstacle)
        size_layout.addWidget(self.size_spin)
        obstacle_layout.addLayout(size_layout)
        
        # Rectangle-specific controls (wrapped in widgets for visibility control)
        from PyQt6.QtWidgets import QWidget
        
        self.width_widget = QWidget()
        width_layout = QHBoxLayout(self.width_widget)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.addWidget(QLabel("Width:"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.05, 0.5)
        self.width_spin.setSingleStep(0.05)
        self.width_spin.setValue(0.2)
        self.width_spin.valueChanged.connect(self.update_obstacle)
        width_layout.addWidget(self.width_spin)
        obstacle_layout.addWidget(self.width_widget)
        self.width_widget.setVisible(False)
        
        self.height_widget = QWidget()
        height_layout = QHBoxLayout(self.height_widget)
        height_layout.setContentsMargins(0, 0, 0, 0)
        height_layout.addWidget(QLabel("Height:"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.05, 0.5)
        self.height_spin.setSingleStep(0.05)
        self.height_spin.setValue(0.1)
        self.height_spin.valueChanged.connect(self.update_obstacle)
        height_layout.addWidget(self.height_spin)
        obstacle_layout.addWidget(self.height_widget)
        self.height_widget.setVisible(False)
        
        # Position sliders
        obstacle_layout.addWidget(QLabel("Position X:"))
        self.x_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_slider.setRange(0, 100)
        self.x_slider.setValue(50)
        self.x_slider.valueChanged.connect(self.update_obstacle_position)
        obstacle_layout.addWidget(self.x_slider)
        
        obstacle_layout.addWidget(QLabel("Position Y:"))
        self.y_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_slider.setRange(0, 100)
        self.y_slider.setValue(50)
        self.y_slider.valueChanged.connect(self.update_obstacle_position)
        obstacle_layout.addWidget(self.y_slider)
        
        obstacle_group.setLayout(obstacle_layout)
        controls_layout.addWidget(obstacle_group)
        
        # Wall boundary condition
        wall_group = QGroupBox("Wall Boundary")
        wall_layout = QVBoxLayout()
        
        self.wall_bounding_cb = QCheckBox("Bounding (Closed Walls)")
        self.wall_bounding_cb.setChecked(True)
        self.wall_bounding_cb.setToolTip("If checked, walls are closed (no-slip). If unchecked, boundaries are open (infinite flow).")
        self.wall_bounding_cb.stateChanged.connect(self.update_wall_bc)
        wall_layout.addWidget(self.wall_bounding_cb)
        
        wall_group.setLayout(wall_layout)
        controls_layout.addWidget(wall_group)
        
        # Temperature controls
        temp_group = QGroupBox("Temperatures")
        temp_layout = QVBoxLayout()
        
        ambient_layout = QHBoxLayout()
        ambient_layout.addWidget(QLabel("Ambient:"))
        self.ambient_spin = QDoubleSpinBox()
        self.ambient_spin.setRange(-100, 500)
        self.ambient_spin.setValue(20.0)
        self.ambient_spin.valueChanged.connect(self.update_temperatures)
        ambient_layout.addWidget(self.ambient_spin)
        temp_layout.addLayout(ambient_layout)
        
        surface_layout = QHBoxLayout()
        surface_layout.addWidget(QLabel("Surface:"))
        self.surface_spin = QDoubleSpinBox()
        self.surface_spin.setRange(-100, 500)
        self.surface_spin.setValue(100.0)
        self.surface_spin.valueChanged.connect(self.update_temperatures)
        surface_layout.addWidget(self.surface_spin)
        temp_layout.addLayout(surface_layout)
        
        diffusivity_layout = QHBoxLayout()
        diffusivity_layout.addWidget(QLabel("Diffusivity:"))
        self.diffusivity_spin = QDoubleSpinBox()
        self.diffusivity_spin.setRange(0.001, 1.0)
        self.diffusivity_spin.setSingleStep(0.01)
        self.diffusivity_spin.setDecimals(4)
        self.diffusivity_spin.setValue(0.01)
        self.diffusivity_spin.valueChanged.connect(self.update_diffusivity)
        diffusivity_layout.addWidget(self.diffusivity_spin)
        temp_layout.addLayout(diffusivity_layout)
        
        # Boussinesq parameters
        expansion_layout = QHBoxLayout()
        expansion_layout.addWidget(QLabel("Thermal Expansion:"))
        self.expansion_spin = QDoubleSpinBox()
        self.expansion_spin.setRange(0.0001, 0.1)
        self.expansion_spin.setSingleStep(0.001)
        self.expansion_spin.setDecimals(4)
        self.expansion_spin.setValue(0.003)
        self.expansion_spin.valueChanged.connect(self.update_boussinesq_params)
        expansion_layout.addWidget(self.expansion_spin)
        temp_layout.addLayout(expansion_layout)
        
        viscosity_layout = QHBoxLayout()
        viscosity_layout.addWidget(QLabel("Kinematic Viscosity:"))
        self.viscosity_spin = QDoubleSpinBox()
        self.viscosity_spin.setRange(0.001, 1.0)
        self.viscosity_spin.setSingleStep(0.01)
        self.viscosity_spin.setDecimals(4)
        self.viscosity_spin.setValue(0.01)
        self.viscosity_spin.valueChanged.connect(self.update_boussinesq_params)
        viscosity_layout.addWidget(self.viscosity_spin)
        temp_layout.addLayout(viscosity_layout)
        
        temp_group.setLayout(temp_layout)
        controls_layout.addWidget(temp_group)
        
        # Simulation controls
        sim_group = QGroupBox("Simulation")
        sim_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_simulation)
        sim_layout.addWidget(self.start_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        sim_layout.addWidget(self.reset_btn)
        
        self.iter_label = QLabel("Iteration: 0")
        sim_layout.addWidget(self.iter_label)
        
        sim_group.setLayout(sim_layout)
        controls_layout.addWidget(sim_group)
        
        controls_layout.addStretch()
        main_layout.addWidget(controls_panel)
        
        # Right panel: visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        # Temperature plot
        self.temp_plot = pg.PlotItem(title="Temperature Field")
        self.temp_plot.setAspectLocked(True)
        self.temp_plot.setLabel('left', 'Y')
        self.temp_plot.setLabel('bottom', 'X')
        
        self.temp_image = pg.ImageItem()
        self.temp_plot.addItem(self.temp_image)
        
        # Set colormap (thermal)
        try:
            self.temp_image.setLookupTable(pg.colormap.get('inferno').getLookupTable())
        except:
            # Fallback to simple gradient
            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]], dtype=np.ubyte)
            self.temp_image.setLookupTable(pg.colormap.ColorMap(pos, color).getLookupTable())
        
        # Graphics layout widget
        gl = pg.GraphicsLayoutWidget()
        gl.setBackground('#f0f0f0')
        gl.addItem(self.temp_plot)
        
        viz_layout.addWidget(gl)
        main_layout.addWidget(viz_widget)
        
        self.setLayout(main_layout)
        
    def _initialize_simulation(self):
        """Initialize the temperature field, velocity field, and obstacle mask"""
        # Create coordinate grid
        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Initialize temperature field to ambient
        self.temperature = np.full((self.nx, self.ny), self.ambient_temp, dtype=np.float32)
        
        # Initialize velocity field to zero
        self.u = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.v = np.zeros((self.nx, self.ny), dtype=np.float32)
        
        # Create obstacle mask
        self._update_mask(X, Y)
        
        # Set obstacle surface temperature
        self.temperature[self.mask == 0] = self.surface_temp
        
        # Update visualization
        self._update_visualization()
        
    def _update_mask(self, X, Y):
        """Update obstacle mask based on current parameters"""
        self.mask = np.ones((self.nx, self.ny), dtype=np.float32)
        
        ox, oy = self.obstacle_x, self.obstacle_y
        
        if self.obstacle_type == 'square':
            half_size = self.obstacle_size / 2
            mask = (X >= ox - half_size) & (X <= ox + half_size) & \
                   (Y >= oy - half_size) & (Y <= oy + half_size)
            self.mask[mask] = 0
            
        elif self.obstacle_type == 'rectangle':
            half_w = self.obstacle_width / 2
            half_h = self.obstacle_height / 2
            mask = (X >= ox - half_w) & (X <= ox + half_w) & \
                   (Y >= oy - half_h) & (Y <= oy + half_h)
            self.mask[mask] = 0
            
        elif self.obstacle_type == 'circle':
            radius = self.obstacle_size / 2
            dist = np.sqrt((X - ox)**2 + (Y - oy)**2)
            mask = dist <= radius
            self.mask[mask] = 0
    
    def _update_visualization(self):
        """Update the temperature visualization"""
        self.temp_image.setImage(self.temperature)
        
        # Set levels based on temperature range
        t_min = min(self.ambient_temp, self.surface_temp)
        t_max = max(self.ambient_temp, self.surface_temp)
        self.temp_image.setLevels([t_min, t_max])
        
        # Set image bounds
        self.temp_image.setRect(0, 0, self.lx, self.ly)
    
    def update_grid_size(self):
        """Update grid size and reinitialize"""
        self.nx = self.nx_spin.value()
        self.ny = self.ny_spin.value()
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self._initialize_simulation()
    
    def update_obstacle_type(self, text):
        """Update obstacle type"""
        self.obstacle_type = text
        
        # Show/hide rectangle-specific controls
        if text == 'rectangle':
            self.width_widget.setVisible(True)
            self.height_widget.setVisible(True)
        else:
            self.width_widget.setVisible(False)
            self.height_widget.setVisible(False)
        
        self._initialize_simulation()
    
    def update_obstacle(self):
        """Update obstacle parameters"""
        self.obstacle_size = self.size_spin.value()
        self.obstacle_width = self.width_spin.value()
        self.obstacle_height = self.height_spin.value()
        self._initialize_simulation()
    
    def update_obstacle_position(self):
        """Update obstacle position from sliders"""
        self.obstacle_x = self.x_slider.value() / 100.0 * self.lx
        self.obstacle_y = self.y_slider.value() / 100.0 * self.ly
        self._initialize_simulation()
    
    def update_temperatures(self):
        """Update temperature parameters"""
        self.ambient_temp = self.ambient_spin.value()
        self.surface_temp = self.surface_spin.value()
        self._initialize_simulation()
    
    def update_diffusivity(self):
        """Update thermal diffusivity"""
        self.thermal_diffusivity = self.diffusivity_spin.value()
    
    def update_boussinesq_params(self):
        """Update Boussinesq parameters"""
        self.thermal_expansion = self.expansion_spin.value()
        self.kinematic_viscosity = self.viscosity_spin.value()
    
    def update_wall_bc(self, state):
        """Update wall boundary condition type"""
        self.walls_bounding = (state == 2)  # 2 = checked
    
    def toggle_simulation(self):
        """Start or stop the simulation"""
        if self.running:
            self.timer.stop()
            self.start_btn.setText("Start")
            self.running = False
        else:
            self.timer.start(16)  # ~60 FPS
            self.start_btn.setText("Stop")
            self.running = True
    
    def reset_simulation(self):
        """Reset the simulation"""
        if self.running:
            self.toggle_simulation()
        self.iteration = 0
        self.iter_label.setText("Iteration: 0")
        self._initialize_simulation()
    
    def simulation_step(self):
        """Perform one simulation step using JAX with Boussinesq buoyancy"""
        # Convert to JAX arrays
        T = jnp.array(self.temperature)
        u = jnp.array(self.u)
        v = jnp.array(self.v)
        mask = jnp.array(self.mask)
        
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        
        # Compute Laplacian of temperature
        T_padded = jnp.pad(T, ((1, 1), (1, 1)), mode='edge')
        laplacian_T = ((T_padded[2:, 1:-1] - 2*T + T_padded[:-2, 1:-1]) / dx2 +
                       (T_padded[1:-1, 2:] - 2*T + T_padded[1:-1, :-2]) / dy2)
        
        # Compute Laplacian of velocity (for viscous diffusion)
        # Use edge padding for bounding walls to preserve boundary values
        if self.walls_bounding:
            u_padded = jnp.pad(u, ((1, 1), (1, 1)), mode='edge')
            v_padded = jnp.pad(v, ((1, 1), (1, 1)), mode='edge')
        else:
            u_padded = jnp.pad(u, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            v_padded = jnp.pad(v, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        laplacian_u = ((u_padded[2:, 1:-1] - 2*u + u_padded[:-2, 1:-1]) / dx2 +
                       (u_padded[1:-1, 2:] - 2*u + u_padded[1:-1, :-2]) / dy2)
        laplacian_v = ((v_padded[2:, 1:-1] - 2*v + v_padded[:-2, 1:-1]) / dx2 +
                       (v_padded[1:-1, 2:] - 2*v + v_padded[1:-1, :-2]) / dy2)
        
        # Boussinesq buoyancy force: F_buoyancy = g * beta * (T - T_ambient)
        # Acts in vertical direction (y)
        buoyancy = self.gravity * self.thermal_expansion * (T - self.ambient_temp)
        
        # Mask buoyancy at boundaries when bounding is enabled
        if self.walls_bounding:
            # Create boundary mask (True at boundaries)
            boundary_mask = jnp.ones_like(T)
            boundary_mask = boundary_mask.at[0, :].set(0.0)
            boundary_mask = boundary_mask.at[-1, :].set(0.0)
            boundary_mask = boundary_mask.at[:, 0].set(0.0)
            boundary_mask = boundary_mask.at[:, -1].set(0.0)
            buoyancy = buoyancy * boundary_mask
        
        # Momentum equations with buoyancy (simplified, no pressure)
        # du/dt = nu * ∇²u
        # dv/dt = nu * ∇²v + buoyancy
        du_dt = self.kinematic_viscosity * laplacian_u
        dv_dt = self.kinematic_viscosity * laplacian_v + buoyancy
        
        # Update velocity
        u_new = u + self.dt * du_dt
        v_new = v + self.dt * dv_dt
        
        # Apply no-slip boundary conditions at obstacle
        u_new = jnp.where(mask == 0, 0.0, u_new)
        v_new = jnp.where(mask == 0, 0.0, v_new)
        
        # Apply boundary conditions at domain boundaries
        if self.walls_bounding:
            # Bounding: no-slip walls (u=0, v=0)
            # Zero boundary cells and adjacent cells for stronger enforcement
            u_new = u_new.at[0, :].set(0.0)
            u_new = u_new.at[1, :].set(0.0)
            u_new = u_new.at[-1, :].set(0.0)
            u_new = u_new.at[-2, :].set(0.0)
            u_new = u_new.at[:, 0].set(0.0)
            u_new = u_new.at[:, 1].set(0.0)
            u_new = u_new.at[:, -1].set(0.0)
            u_new = u_new.at[:, -2].set(0.0)
            v_new = v_new.at[0, :].set(0.0)
            v_new = v_new.at[1, :].set(0.0)
            v_new = v_new.at[-1, :].set(0.0)
            v_new = v_new.at[-2, :].set(0.0)
            v_new = v_new.at[:, 0].set(0.0)
            v_new = v_new.at[:, 1].set(0.0)
            v_new = v_new.at[:, -1].set(0.0)
            v_new = v_new.at[:, -2].set(0.0)
        else:
            # Infinite flow: open boundaries (zero gradient / extrapolation)
            u_new = u_new.at[0, :].set(u_new[1, :])
            u_new = u_new.at[-1, :].set(u_new[-2, :])
            u_new = u_new.at[:, 0].set(u_new[:, 1])
            u_new = u_new.at[:, -1].set(u_new[:, -2])
            v_new = v_new.at[0, :].set(v_new[1, :])
            v_new = v_new.at[-1, :].set(v_new[-2, :])
            v_new = v_new.at[:, 0].set(v_new[:, 1])
            v_new = v_new.at[:, -1].set(v_new[:, -2])
        
        # Heat equation with advection: ∂T/∂t + u·∇T = α∇²T
        # Compute gradients for advection
        dT_dx = (T_padded[2:, 1:-1] - T_padded[:-2, 1:-1]) / (2 * self.dx)
        dT_dy = (T_padded[1:-1, 2:] - T_padded[1:-1, :-2]) / (2 * self.dy)
        
        # Advection term: u·∇T
        advection = u * dT_dx + v * dT_dy
        
        # Heat equation with advection
        dT_dt = self.thermal_diffusivity * laplacian_T - advection
        
        # Update temperature
        T_new = T + self.dt * dT_dt
        
        # Apply boundary conditions
        # Obstacle surface stays at surface temperature
        T_new = jnp.where(mask == 0, self.surface_temp, T_new)
        
        # Domain boundaries: Neumann BC (∂T/∂n = 0) - already handled by edge padding
        
        # Convert back to numpy
        self.temperature = np.array(T_new, dtype=np.float32)
        self.u = np.array(u_new, dtype=np.float32)
        self.v = np.array(v_new, dtype=np.float32)
        
        # Update iteration counter
        self.iteration += 1
        self.iter_label.setText(f"Iteration: {self.iteration}")
        
        # Update visualization
        self._update_visualization()
