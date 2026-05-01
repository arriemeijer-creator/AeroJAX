"""
Flow initialization methods for BaselineSolver.
"""

import jax.numpy as jnp


def _initialize_von_karman_flow(self):
    """Initialize von Karman flow with uniform velocity matching inlet"""
    if self.sim_params.grid_type == 'mac':
        # MAC staggered grid: u at (nx+1, ny), v at (nx, ny+1), p at (nx, ny)
        self.u = jnp.full((self.grid.nx + 1, self.grid.ny), self.flow.U_inf)
        self.v = jnp.zeros((self.grid.nx, self.grid.ny + 1))
    else:
        # Collocated grid
        self.u = jnp.full((self.grid.nx, self.grid.ny), self.flow.U_inf)
        self.v = jnp.zeros((self.grid.nx, self.grid.ny))

    # Reset pressure field to prevent divergence from old pressure state
    self.current_pressure = jnp.zeros((self.grid.nx, self.grid.ny))
    
    # Initialize scalar dye field (always cell-centered)
    self.c = jnp.zeros((self.grid.nx, self.grid.ny))
    
    # Initialize previous velocity fields
    self.u_prev = jnp.copy(self.u)
    self.v_prev = jnp.copy(self.v)

    self.startup_ramp_steps = 0  # Disabled - startup ramp was creating transients
    print(f"Startup: Inlet velocity ramp disabled (startup_ramp_steps={self.startup_ramp_steps}), grid_type={self.sim_params.grid_type}")


def _initialize_cavity_flow(self):
    """Initialize lid-driven cavity flow - simple approach like backup version"""
    lid_velocity = self.flow.U_inf if hasattr(self.flow, 'U_inf') else 1.0
    
    if self.sim_params.grid_type == 'mac':
        # MAC staggered grid: u at (nx+1, ny), v at (nx, ny+1)
        self.u = jnp.zeros((self.grid.nx + 1, self.grid.ny))
        self.v = jnp.zeros((self.grid.nx, self.grid.ny + 1))
        # Apply lid boundary condition at top (u at y-faces)
        self.u = self.u.at[:, -1].set(lid_velocity)
    else:
        # Collocated grid: u and v at (nx, ny)
        self.u = jnp.zeros((self.grid.nx, self.grid.ny))
        self.v = jnp.zeros((self.grid.nx, self.grid.ny))
        # Apply lid boundary condition at top
        self.u = self.u.at[:, -1].set(lid_velocity)
    
    # Initialize scalar dye field (always cell-centered)
    self.c = jnp.zeros((self.grid.nx, self.grid.ny))
    
    # Initialize pressure field (always cell-centered)
    self.current_pressure = jnp.zeros((self.grid.nx, self.grid.ny))
    
    # Initialize previous velocity fields
    self.u_prev = jnp.copy(self.u)
    self.v_prev = jnp.copy(self.v)
    
    print(f"LDC initialized with Re={self.flow.Re}, lid velocity = {lid_velocity:.6f}, nu = {self.flow.nu:.6f}, grid_type={self.sim_params.grid_type}")


def _initialize_taylor_green_flow(self):
    """Initialize Taylor-Green vortex"""
    X, Y = self.grid.X, self.grid.Y
    if self.sim_params.grid_type == 'mac':
        # For MAC grid, initialize at cell centers then interpolate to faces
        u_center = self.flow.U_inf * jnp.sin(X) * jnp.cos(Y)
        v_center = -self.flow.U_inf * jnp.cos(X) * jnp.sin(Y)
        from solver.operators_mac import interpolate_to_x_face, interpolate_to_y_face
        self.u = interpolate_to_x_face(u_center)
        self.v = interpolate_to_y_face(v_center)
    else:
        # Collocated grid
        self.u = self.flow.U_inf * jnp.sin(X) * jnp.cos(Y)
        self.v = -self.flow.U_inf * jnp.cos(X) * jnp.sin(Y)
    
    # Initialize scalar dye field (always cell-centered)
    self.c = jnp.zeros((self.grid.nx, self.grid.ny))
    
    # Initialize pressure field
    self.current_pressure = jnp.zeros((self.grid.nx, self.grid.ny))
    
    # Initialize previous velocity fields
    self.u_prev = jnp.copy(self.u)
    self.v_prev = jnp.copy(self.v)
    
    print(f"Taylor-Green initialized with grid_type={self.sim_params.grid_type}")
