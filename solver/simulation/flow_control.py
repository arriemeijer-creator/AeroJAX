"""
Flow control methods for BaselineSolver.
Handles flow type changes, obstacle type changes, and timestep control.
"""

import jax
import jax.numpy as jnp
from typing import Optional

# Import from local modules
from ..params import GridParams

# Import NACA availability
try:
    from obstacles.naca_airfoils import NACA_AVAILABLE
except ImportError:
    NACA_AVAILABLE = False


def apply_flow_type(self, flow_type: str):
    """Change flow type and reinitialize"""
    valid_flow_types = ['von_karman', 'lid_driven_cavity', 'taylor_green']
    if flow_type not in valid_flow_types:
        raise ValueError(f"Flow type must be one of {valid_flow_types}")
    
    self.sim_params.flow_type = flow_type
    
    # Update dt_controller dt_max for LDC
    if flow_type == 'lid_driven_cavity' and self.dt_controller is not None:
        self.dt_controller.dt_max = 0.001  # Small dt for LDC stability
    
    if flow_type == 'lid_driven_cavity':
        self.grid = GridParams(nx=128, ny=128, lx=1.0, ly=1.0)
        # Set appropriate flow parameters for lid-driven cavity
        # Use moderate lid velocity and viscosity for stability
        self.flow.U_inf = 1.0  # Moderate lid velocity
        self.flow.nu = 0.01  # Viscosity for stability
        self.flow.Re = (self.flow.U_inf * self.flow.L_char) / self.flow.nu
        self.sim_params.fixed_dt = 0.001  # Small timestep for stability
        self.dt = 0.001
    elif flow_type == 'taylor_green':
        self.grid = GridParams(nx=128, ny=128, lx=2*jnp.pi, ly=2*jnp.pi)
    else:
        self.grid = GridParams(nx=512, ny=192, lx=20.0, ly=7.5)  # Increased ny to 192 for proper circulation contour margins
    
    x = jnp.linspace(0, self.grid.lx, self.grid.nx)
    y = jnp.linspace(0, self.grid.ly, self.grid.ny)
    self.grid.X, self.grid.Y = jnp.meshgrid(x, y, indexing='ij')
    
    if hasattr(self, 'u'):
        delattr(self, 'u')
    if hasattr(self, 'v'):
        delattr(self, 'v')
    if hasattr(self, 'u_prev'):
        delattr(self, 'u_prev')
    if hasattr(self, 'v_prev'):
        delattr(self, 'v_prev')
    if hasattr(self, 'current_pressure'):
        delattr(self, 'current_pressure')
    if hasattr(self, 'mask'):
        delattr(self, 'mask')
    
    self._jit_cache = {}
    jax.clear_caches()
    import gc
    gc.collect()
    
    self.current_pressure = jnp.zeros((self.grid.nx, self.grid.ny))
    self.mask = self._compute_mask()
    
    if flow_type == 'lid_driven_cavity':
        self._initialize_cavity_flow()
    elif flow_type == 'taylor_green':
        self._initialize_taylor_green_flow()
    else:
        self._initialize_von_karman_flow()
    
    self._step_jit = jax.jit(self._step)
    
    # Use appropriate divergence function based on flow type
    from ..operators import vorticity_nonperiodic, divergence_nonperiodic, vorticity, divergence
    if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
        self._vorticity = jax.jit(vorticity_nonperiodic, static_argnums=(2, 3))
        self._divergence = jax.jit(divergence_nonperiodic, static_argnums=(2, 3))
    else:
        self._vorticity = jax.jit(vorticity, static_argnums=(2, 3))
        self._divergence = jax.jit(divergence, static_argnums=(2, 3))
    
    self.mask = self._compute_mask()
    
    if hasattr(self, 'u'):
        self.u_prev = jnp.copy(self.u)
    if hasattr(self, 'v'):
        self.v_prev = jnp.copy(self.v)
    
    self.history = {
        'time': [], 'dt': [], 'drag': [], 'lift': [],
        'l2_change': [], 'rms_change': [], 'l2_change_u': [], 'l2_change_v': [], 'max_change': [], 'change_99p': [], 'rel_change': [],
        'rms_divergence': [], 'l2_divergence': [],
        'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 'Cp_min': [], 'wake_deficit': [], 'strouhal': [], 'time': []}
    }
    self.iteration = 0
    
    print(f"Flow type changed to {flow_type}")
    print(f"Grid updated to {self.grid.nx}x{self.grid.ny} ({self.grid.lx}x{self.grid.ly})")


def set_obstacle_type(self, obstacle_type: str, **kwargs):
    """Set obstacle type (cylinder, NACA airfoil, cow, or three_cylinder_array)"""
    if obstacle_type not in ['cylinder', 'naca_airfoil', 'cow', 'three_cylinder_array']:
        raise ValueError("obstacle_type must be 'cylinder', 'naca_airfoil', 'cow', or 'three_cylinder_array'")
    
    self.sim_params.obstacle_type = obstacle_type
    
    if obstacle_type == 'naca_airfoil' and NACA_AVAILABLE:
        for key, value in kwargs.items():
            if hasattr(self.sim_params, f'naca_{key}'):
                setattr(self.sim_params, f'naca_{key}', value)
    
    # Print current position before computing mask
    if obstacle_type == 'cylinder':
        print(f"[SET_OBSTACLE_TYPE] Computing cylinder mask with center_x={self.geom.center_x}, center_y={self.geom.center_y}")
    elif obstacle_type == 'naca_airfoil':
        print(f"[SET_OBSTACLE_TYPE] Computing NACA mask with naca_x={self.sim_params.naca_x}, naca_y={self.sim_params.naca_y}")
    
    # Clear JAX cache to force re-tracing with new obstacle type
    import jax
    jax.clear_caches()
    
    # Clear JIT cache to force recompilation of step function
    self._jit_cache = {}
    
    self.mask = self._compute_mask()
    # Reset velocity fields to inflow velocity before applying new mask
    # This prevents old mask zeros from persisting when multiplying by new mask
    self.u = self.u.at[:].set(self.flow.U_inf)
    self.v = self.v.at[:].set(0.0)
    # Apply mask to velocity fields immediately
    self.u = self.u * self.mask
    self.v = self.v * self.mask
    
    # Force adaptive_dt=False to prevent dt mismatch
    self.sim_params.adaptive_dt = False
    
    # Recompile step function with new mask
    self._step_jit = self.get_step_jit()
    
    # Reset boundary conditions for von_karman flow
    if self.sim_params.flow_type == 'von_karman':
        self.u = self.u.at[0, :].set(self.flow.U_inf)
        self.v = self.v.at[0, :].set(0.0)
        self.u = self.u.at[-1, :].set(self.u.at[-2, :].get())
        self.v = self.v.at[-1, :].set(self.v.at[-2, :].get())
        self.u = self.u.at[:, 0].set(0.0)
        self.u = self.u.at[:, -1].set(0.0)
        self.v = self.v.at[:, 0].set(0.0)
        self.v = self.v.at[:, -1].set(0.0)
    
    print(f"Obstacle type set to {obstacle_type}")


def update_naca_angle(self, angle_of_attack: float, recompute: bool = True, clear_cache: bool = False):
    """Update NACA angle of attack during simulation
    
    Args:
        angle_of_attack: New angle in degrees
        recompute: If True, recompute mask. If False, only update parameter.
        clear_cache: If True, clear JAX cache to force re-tracing (expensive, use on release).
    """
    if self.sim_params.obstacle_type != 'naca_airfoil':
        print("Warning: Cannot update angle - current obstacle is not a NACA airfoil")
        return
    
    self.sim_params.naca_angle = angle_of_attack
    
    if recompute:
        if clear_cache:
            import jax
            jax.clear_caches()
        
        self.mask = self._compute_mask()
        # Apply mask to velocity fields
        self.u = self.u * self.mask
        self.v = self.v * self.mask
    
    if self.sim_params.flow_type == 'von_karman':
        self.u = self.u.at[0, :].set(self.flow.U_inf)
        self.v = self.v.at[0, :].set(0.0)
        self.u = self.u.at[-1, :].set(self.u.at[-2, :].get())
        self.v = self.v.at[-1, :].set(self.v.at[-2, :].get())
        self.u = self.u.at[:, 0].set(0.0)
        self.u = self.u.at[:, -1].set(0.0)
        self.v = self.v.at[:, 0].set(0.0)
        self.v = self.v.at[:, -1].set(0.0)


def inject_dye(self, x_pos: float, y_pos: float, amount: float = 0.5):
    """Inject dye at physical coordinates"""
    # Enable scalar updates when dye is injected
    self.enable_scalar_update = True
    
    x_clamped = max(0.0, min(x_pos, self.grid.lx))
    y_clamped = max(0.0, min(y_pos, self.grid.ly))
    
    ix = int(x_clamped / self.grid.dx)
    iy = int(y_clamped / self.grid.dy)
    
    # Inject into a 5x5 area around the target cell for smoother distribution - vectorized
    radius = 2
    dx_offsets = jnp.arange(-radius, radius + 1)
    dy_offsets = jnp.arange(-radius, radius + 1)
    dx_grid, dy_grid = jnp.meshgrid(dx_offsets, dy_offsets, indexing='ij')
    
    # Compute target indices
    ix_targets = ix + dx_grid
    iy_targets = iy + dy_grid
    
    # Clamp to grid bounds
    ix_targets = jnp.clip(ix_targets, 0, self.grid.nx - 1)
    iy_targets = jnp.clip(iy_targets, 0, self.grid.ny - 1)
    
    # Compute Gaussian falloff
    distances = jnp.sqrt(dx_grid**2 + dy_grid**2)
    falloffs = jnp.exp(-distances**2 / 2.0)
    
    # Update dye concentration vectorized
    current_values = self.c[ix_targets, iy_targets]
    new_values = jnp.minimum(current_values + amount * falloffs, 1.0)
    
    # Apply updates using scatter
    self.c = self.c.at[ix_targets, iy_targets].set(new_values)
    
    print(f"Dye injected at ({x_pos:.2f}, {y_pos:.2f}) -> grid ({ix}, {iy})")


def set_adaptive_dt(self):
    """Enable adaptive timestep control using CFL-based approach"""
    self.sim_params.adaptive_dt = True
    if self.dt_controller is None:
        try:
            from timestepping.cfl_adaptive_dt import CFLAdaptiveController
            self.dt_controller = CFLAdaptiveController(
                cfl_target=self.sim_params.max_cfl,
                dt_min=self.sim_params.dt_min,
                dt_max=self.sim_params.dt_max
            )
            print(f"CFL-based adaptive timestep enabled with cfl_target={self.sim_params.max_cfl}, dt_min={self.sim_params.dt_min}, dt_max={self.sim_params.dt_max}")
        except ImportError:
            print("Warning: CFLAdaptiveController not available, adaptive dt may not work properly")
    jax.clear_caches()
    if hasattr(self, '_step_jit'):
        delattr(self, '_step_jit')
    self._step_jit = self.get_step_jit()  # Recreate JIT function
    # Reinitialize velocity fields with proper initial conditions
    # Call the appropriate flow initialization function to avoid starting from zero
    if self.sim_params.flow_type == 'von_karman':
        self._initialize_von_karman_flow()
    elif self.sim_params.flow_type == 'lid_driven_cavity':
        self._initialize_cavity_flow()
    elif self.sim_params.flow_type == 'taylor_green':
        self._initialize_taylor_green_flow()


def set_fixed_dt(self, dt: float):
    """Set fixed timestep and disable adaptive control"""
    self.sim_params.adaptive_dt = False
    self.dt = dt
    self.sim_params.fixed_dt = dt
    print(f"Fixed timestep set to {dt}")
    jax.clear_caches()
    if hasattr(self, '_step_jit'):
        delattr(self, '_step_jit')
    self._step_jit = self.get_step_jit()  # Recreate JIT function
