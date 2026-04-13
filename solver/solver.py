"""
Main solver class and pure step functions for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Tuple, Optional

# Import from local modules
from .params import SimState, GridParams, FlowParams, GeometryParams, SimulationParams
from .operators import (
    grad_x, grad_y, grad_x_nonperiodic, grad_y_nonperiodic,
    divergence, divergence_nonperiodic,
    vorticity, vorticity_nonperiodic,
    scalar_advection_diffusion_periodic, scalar_advection_diffusion_nonperiodic
)
from .les_models import dynamic_smagorinsky, constant_smagorinsky
from .boundary_conditions import (
    apply_cavity_boundary_conditions, create_cavity_mask,
    apply_taylor_green_boundary_conditions, create_taylor_green_mask,
    apply_backward_step_boundary_conditions
)
from .geometry import sdf_cylinder, smooth_mask, create_mask_from_params
from .brinkman import apply_brinkman_penalization_consistent
from .metrics import compute_forces, get_airfoil_surface_mask, find_stagnation_point, find_separation_point

# Import external modules
try:
    from .naca_airfoils import NACAParams, create_naca_mask, naca_surface_distance, parse_naca_4digit, parse_naca_5digit
    NACA_AVAILABLE = True
except ImportError:
    NACA_AVAILABLE = False

try:
    from timestepping.adaptivedt import DivergencePIDController
except ImportError:
    DivergencePIDController = None

try:
    from advection_schemes import rk3_step, spectral_step
except ImportError:
    rk3_step = spectral_step = None

try:
    from pressure_solvers import poisson_fft, poisson_cg, poisson_multigrid
    from pressure_solvers.sor_masked import poisson_sor_masked, poisson_cg_masked
except ImportError:
    poisson_fft = poisson_cg = poisson_multigrid = poisson_sor_masked = poisson_cg_masked = None


@jax.jit
def update_dt_pure(dt: float, div_max: float, integral: float, prev_error: float,
                   target_div: float = 1e-4, Kp: float = 0.5, Ki: float = 0.05, Kd: float = 0.1,
                   dt_min: float = 1e-5, dt_max: float = 0.01, eta_max: float = None) -> Tuple[float, float, float]:
    """Pure JAX-compatible adaptive timestep update using PID controller"""
    # Safeguard against invalid divergence values
    div_is_valid = jnp.isfinite(div_max) & (div_max >= 0)
    
    # If divergence is invalid, reduce timestep conservatively
    def handle_invalid_div():
        new_dt = dt * 0.5
        new_dt = jnp.maximum(dt_min, jnp.minimum(dt_max, new_dt))
        return new_dt, integral, prev_error
    
    # Normal case with valid divergence
    def handle_valid_div():
        error = div_max - target_div
        integral_new = integral + error * dt
        # Anti-windup: clamp integral
        integral_new = jnp.maximum(-1.0, jnp.minimum(1.0, integral_new))
        
        derivative = (error - prev_error) / (dt + 1e-8)
        correction = Kp * error + Ki * integral_new + Kd * derivative
        factor = jnp.maximum(0.5, jnp.minimum(2.0, 1.0 + correction))
        new_dt = dt * factor
        
        # Brinkman stiffness limiter
        if eta_max is not None and eta_max > 0:
            rho = 1.0
            C_safety = 0.5
            dt_brinkman_limit = C_safety * (rho / (eta_max + 1e-8))
            new_dt = jnp.minimum(new_dt, dt_brinkman_limit)
        
        new_dt = jnp.maximum(dt_min, jnp.minimum(dt_max, new_dt))
        return new_dt, integral_new, error
    
    return jnp.lax.cond(div_is_valid, handle_valid_div, handle_invalid_div)


@jax.jit(static_argnames=['flow_type', 'advection_scheme', 'pressure_solver', 'les_model', 'limiter', 'use_les', 'adaptive_dt'])
def step_pure(state: SimState, mask: jnp.ndarray, dx: float, dy: float, 
               nu: float, U_inf: float, use_les: bool = False,
               smagorinsky_constant: float = 0.1, weno_epsilon: float = 1e-6,
               eps: float = 0.01, adaptive_dt: bool = False,
               dt_min: float = 1e-5, dt_max: float = 0.01, target_div: float = 1e-4,
               Kp: float = 0.5, Ki: float = 0.05, Kd: float = 0.1,
               flow_type: str = 'von_karman',
               advection_scheme: str = 'rk3',
               pressure_solver: str = 'multigrid',
               les_model: str = 'smagorinsky',
               limiter: str = 'minmod') -> SimState:
    """Pure JAX-compatible step function that accepts and returns SimState"""
    u, v, dt, iteration = state.u, state.v, state.dt, state.iteration
    
    # Compute SGS eddy viscosity if LES is enabled
    nu_total = nu
    if use_les:
        delta = (dx * dy) ** 0.5
        if les_model == 'dynamic_smagorinsky':
            nu_sgs, _ = dynamic_smagorinsky(u, v, dx, dy, delta)
            nu_total = nu + nu_sgs
        elif les_model == 'smagorinsky':
            nu_sgs = constant_smagorinsky(u, v, dx, dy, delta, smagorinsky_constant)
            nu_total = nu + nu_sgs
    
    # Select advection scheme
    if advection_scheme == 'rk3' and rk3_step is not None:
        if flow_type == 'von_karman':
            from advection_schemes.rk3_scheme import rk3_step_simple  # Using simple stable RK3
            u_star, v_star = rk3_step_simple(u, v, dt, nu_total, dx, dy, mask, U_inf=U_inf)
        elif flow_type == 'taylor_green':
            from advection_schemes.rk3_scheme import rk3_step_simple  # Using simple stable RK3
            u_star, v_star = rk3_step_simple(u, v, dt, nu_total, dx, dy, mask, U_inf=U_inf)
        elif flow_type == 'lid_driven_cavity':
            # For LDC, use rk3_step with custom boundary conditions (no inlet/outlet, all walls no-slip)
            def ldc_boundary_conditions(field: jnp.ndarray) -> jnp.ndarray:
                # LDC boundary conditions: all walls no-slip (zero velocity)
                field_bc = field.copy()
                field_bc = field_bc.at[:, 0].set(0.0)  # bottom wall
                field_bc = field_bc.at[:, -1].set(0.0)  # top wall (will be set to lid velocity later)
                field_bc = field_bc.at[0, :].set(0.0)  # left wall
                field_bc = field_bc.at[-1, :].set(0.0)  # right wall
                return field_bc
            u_star, v_star = rk3_step(u, v, dt, nu_total, dx, dy, mask, boundary_conditions=ldc_boundary_conditions)
        else:
            u_star, v_star = rk3_step(u, v, dt, nu_total, dx, dy, mask)
    elif advection_scheme == 'spectral' and spectral_step is not None:
        u_star, v_star = spectral_step(u, v, dt, nu_total, dx, dy, mask, dealias=True)
    else:
        # Default to RK3 if scheme not available
        u_star, v_star = rk3_step(u, v, dt, nu_total, dx, dy, mask)
    
    # Compute divergence
    if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
        div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
    else:
        div_star = divergence(u_star, v_star, dx, dy)
    rhs = div_star / dt
    
    # Solve pressure Poisson equation
    if pressure_solver == 'fft' and poisson_fft is not None:
        p = poisson_fft(rhs, mask, dx, dy)
    elif pressure_solver == 'cg' and poisson_cg is not None:
        p = poisson_cg(rhs, mask, dx, dy)
    elif pressure_solver == 'cg_masked' and poisson_cg_masked is not None:
        p = poisson_cg_masked(rhs, mask, dx, dy, max_iter=200, flow_type=flow_type)
    elif pressure_solver == 'sor_masked' and poisson_sor_masked is not None:
        p = poisson_sor_masked(rhs, mask, dx, dy, max_iter=200, flow_type=flow_type)
    elif pressure_solver == 'multigrid' and poisson_multigrid is not None:
        p = poisson_multigrid(rhs, mask, dx, dy, flow_type=flow_type)
    else:
        p = poisson_cg(rhs, mask, dx, dy)
    
    # Pressure gradient correction
    if flow_type == 'von_karman':
        dp_dx = grad_x_nonperiodic(p, dx)
        dp_dy = grad_y_nonperiodic(p, dy)
    else:
        dp_dx = grad_x(p, dx)
        dp_dy = grad_y(p, dy)
    u_new = u_star - dt * dp_dx
    v_new = v_star - dt * dp_dy
    
    # Apply boundary conditions BEFORE mask to ensure they are enforced
    if flow_type == 'von_karman':
        u_new = u_new.at[0, :].set(U_inf)
        v_new = v_new.at[0, :].set(0.0)
        u_new = u_new.at[-1, :].set(u_new.at[-2, :].get())
        v_new = v_new.at[-1, :].set(v_new.at[-2, :].get())
        u_new = u_new.at[:, 0].set(0.0)
        u_new = u_new.at[:, -1].set(0.0)
        v_new = v_new.at[:, 0].set(0.0)
        v_new = v_new.at[:, -1].set(0.0)
    elif flow_type == 'lid_driven_cavity':
        cavity_height = dy * (u.shape[1] - 1)
        lid_velocity = 1.0
        # Set lid velocity (top boundary)
        u_new = u_new.at[:, -1].set(lid_velocity)
        # Set all other walls to zero (no-slip condition)
        u_new = u_new.at[:, 0].set(0.0)  # bottom wall
        u_new = u_new.at[0, :].set(0.0)  # left wall
        u_new = u_new.at[-1, :].set(0.0)  # right wall
        v_new = v_new.at[:, 0].set(0.0)  # bottom wall
        v_new = v_new.at[:, -1].set(0.0)  # top wall
        v_new = v_new.at[0, :].set(0.0)  # left wall
        v_new = v_new.at[-1, :].set(0.0)  # right wall
    elif flow_type == 'taylor_green':
        pass  # Periodic
    
    # Apply mask AFTER boundary conditions
    u_new = u_new * mask
    v_new = v_new * mask
    
    # Update previous velocity
    u_prev = u
    v_prev = v
    
    # Adaptive timestep update if enabled
    if adaptive_dt:
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
        else:
            div_star = divergence(u_star, v_star, dx, dy)
        div_max = jnp.max(jnp.abs(div_star))

        eta_max = None
        if flow_type == 'von_karman':
            chi_max = jnp.max(1.0 - mask)
            rho = 1.0
            N_decay = 2.0
            eta_max = (rho / (N_decay * dt)) * chi_max

        # Adjust dt_max for low viscosity and high velocity cases to prevent instability
        dt_max_adaptive = dt_max
        if nu < 1e-4:
            dt_max_adaptive = min(dt_max, 0.001)  # Much lower dt_max for very low viscosity
        elif nu < 1e-3:
            dt_max_adaptive = min(dt_max, 0.002)  # Lower dt_max for low viscosity
        elif nu < 2e-3:
            dt_max_adaptive = min(dt_max, 0.003)  # Additional reduction for moderate-low viscosity

        # Additional dt_max reduction for high velocities
        # Use inlet velocity (U_inf) instead of current velocity field to avoid startup ramp issues
        # Use jnp.where for JAX compatibility in JIT-compiled function
        dt_max_adaptive = jnp.where(U_inf > 5.0, 0.001, jnp.where(U_inf > 3.0, 0.002, jnp.where(U_inf > 1.5, 0.003, dt_max_adaptive)))

        dt_new, integral_new, error_new = update_dt_pure(
            dt, div_max, state.integral, state.prev_error,
            target_div, Kp, Ki, Kd, dt_min, dt_max_adaptive, eta_max
        )
    else:
        dt_new = dt
        integral_new = state.integral
        error_new = state.prev_error
    
    return SimState(
        u=u_new, v=v_new, p=p, u_prev=u_prev, v_prev=v_prev,
        c=state.c, dt=dt_new, iteration=iteration + 1,
        integral=integral_new, prev_error=error_new
    )


class BaselineSolver:
    """Main Navier-Stokes solver class"""
    
    def __init__(self,
                 grid: GridParams,
                 flow: FlowParams,
                 geom: GeometryParams,
                 sim_params: SimulationParams,
                 dt: float = None,
                 seed: int = 42):
        
        self.grid = grid
        self.flow = flow
        self.geom = geom
        self.sim_params = sim_params
        
        # Compute characteristic length based on flow type and geometry
        from solver.params import compute_characteristic_length
        self.flow.L_char = compute_characteristic_length(sim_params.flow_type, geom, sim_params, sim_params.obstacle_type)
        
        # Resolve flow constraints to ensure consistent physics
        self.flow.resolve()
        print(f"Flow parameters resolved: U={self.flow.U_inf:.3f}, nu={self.flow.nu:.6f}, Re={self.flow.Re:.1f}, L={self.flow.L_char:.3f}")
        
        # Initialize divergence PID controller
        if sim_params.adaptive_dt and DivergencePIDController is not None:
            self.dt_controller = DivergencePIDController(
                dt_min=sim_params.dt_min, dt_max=sim_params.dt_max
            )
        else:
            self.dt_controller = None
        
        # Smart dt initialization
        if self.sim_params.adaptive_dt:
            # Always use CFL-based dt when adaptive dt is enabled for stability
            # Use lower CFL target for high velocities and high Reynolds numbers
            if self.flow.U_inf > 5.0:
                cfl_target = 0.05  # Very conservative for very high velocities
            elif self.flow.U_inf > 3.0:
                cfl_target = 0.1   # Conservative for high velocities
            elif self.flow.U_inf > 1.5:
                cfl_target = 0.15  # Conservative for moderate velocities
            else:
                cfl_target = 0.3   # Normal for low velocities

            # Further reduce CFL target for low viscosities (high Reynolds numbers)
            # Low viscosity means less viscous damping, requiring smaller timesteps for stability
            if self.flow.nu < 1e-4:
                cfl_target = min(cfl_target, 0.02)  # More aggressive reduction for very low viscosity
            elif self.flow.nu < 1e-3:
                cfl_target = min(cfl_target, 0.05)   # Moderate reduction for low viscosity
            elif self.flow.nu < 2e-3:
                cfl_target = min(cfl_target, 0.1)    # Additional reduction for moderate-low viscosity

            # Add Reynolds number awareness - even with moderate velocity, high Re needs small CFL
            if hasattr(self.flow, 'Re') and self.flow.Re > 500:
                cfl_target = min(cfl_target, 0.15)   # Conservative for Re > 500
            if hasattr(self.flow, 'Re') and self.flow.Re > 1000:
                cfl_target = min(cfl_target, 0.1)    # More conservative for Re > 1000
            if hasattr(self.flow, 'Re') and self.flow.Re > 2000:
                cfl_target = min(cfl_target, 0.05)   # Very conservative for Re > 2000

            dx = self.grid.dx
            dy = self.grid.dy
            max_velocity = self.flow.U_inf
            dt_cfl = cfl_target * min(dx, dy) / (max_velocity + 1e-8)
            dt_diffusion = 0.25 * min(dx**2, dy**2) / self.flow.nu

            # Reduce dt_max for low viscosity and high velocity cases to prevent instability
            dt_max = self.sim_params.dt_max
            if self.flow.nu < 1e-4:
                dt_max = min(dt_max, 0.001)  # Much lower dt_max for very low viscosity
            elif self.flow.nu < 1e-3:
                dt_max = min(dt_max, 0.002)  # Lower dt_max for low viscosity
            elif self.flow.nu < 2e-3:
                dt_max = min(dt_max, 0.003)  # Additional reduction for moderate-low viscosity

            # Additional dt_max reduction for high velocities
            if self.flow.U_inf > 5.0:
                dt_max = min(dt_max, 0.001)  # Very conservative for very high velocities
            elif self.flow.U_inf > 3.0:
                dt_max = min(dt_max, 0.002)  # Conservative for high velocities
            elif self.flow.U_inf > 1.5:
                dt_max = min(dt_max, 0.003)  # Conservative for moderate velocities

            self.dt = min(dt_cfl, dt_diffusion, dt_max)
            self.dt = max(self.dt, self.sim_params.dt_min)
            if dt is not None and dt != self.dt:
                print(f"User-specified dt={dt:.6f} overridden by CFL-based dt={self.dt:.6f} for stability (CFL={cfl_target})")
            else:
                print(f"Using adaptive dt, auto-calculated initial dt = {self.dt:.6f} (CFL={cfl_target})")
        elif dt is not None:
            self.dt = dt
            self.sim_params.fixed_dt = dt
            print(f"Using user-specified fixed dt = {dt:.6f}")
        else:
            self.dt = self.sim_params.fixed_dt
            print(f"Using fixed dt = {self.dt:.6f} from simulation parameters")
        
        # Update Y positions based on actual domain height
        domain_center_y = self.grid.ly / 2
        self.geom.center_y = jnp.array(domain_center_y)
        if not hasattr(self.sim_params, 'naca_airfoil') or self.sim_params.naca_airfoil == 'none':
            self.sim_params.naca_y = domain_center_y
        
        # Grid-consistent ε
        from .params import compute_eps_multiplier
        if self.sim_params.auto_eps_multiplier:
            self.sim_params.eps_multiplier = compute_eps_multiplier(self.flow.Re)
            print(f"Auto-computed eps_multiplier = {self.sim_params.eps_multiplier} from Re = {self.flow.Re:.1f}")
        self.sim_params.eps = self.sim_params.eps_multiplier * self.grid.dx
        print(f"Setting ε = {self.sim_params.eps:.4f} ({self.sim_params.eps_multiplier} * dx)")
        
        # Pre-compute mask
        self.mask = self._compute_mask()
        
        # Initialize based on flow type
        if self.sim_params.flow_type == 'lid_driven_cavity':
            self._initialize_cavity_flow()
        elif self.sim_params.flow_type == 'taylor_green':
            self._initialize_taylor_green_flow()
        else:
            self._initialize_von_karman_flow()
        
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        
        # Use appropriate divergence function based on flow type
        if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
            self._vorticity = jax.jit(vorticity_nonperiodic, static_argnums=(2, 3))
            self._divergence = jax.jit(divergence_nonperiodic, static_argnums=(2, 3))
        else:
            self._vorticity = jax.jit(vorticity, static_argnums=(2, 3))
            self._divergence = jax.jit(divergence, static_argnums=(2, 3))
        
        print(f"Initialized with: {self.sim_params.advection_scheme} advection, {self.sim_params.pressure_solver} pressure solver")
        
        self.history = {
            'time': [], 'dt': [], 'drag': [], 'lift': [],
            'l2_change': [], 'l2_change_u': [], 'l2_change_v': [], 'max_change': [], 'rel_change': [],
            'max_divergence': [], 'l2_divergence': [],
            'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 'Cp_min': [], 'wake_deficit': []}
        }
        self.iteration = 0
        
        self.u_prev = jnp.copy(self.u)
        self.v_prev = jnp.copy(self.v)
        self.c = jnp.zeros((self.grid.nx, self.grid.ny))
        self.current_pressure = jnp.zeros((self.grid.nx, self.grid.ny))
        self.compute_airfoil_metrics = False
        self.metrics_frame_skip = 100  # Compute metrics every N frames (reduced from 1 for better performance)
        
        self.state = SimState(
            u=self.u, v=self.v, p=self.current_pressure,
            u_prev=self.u_prev, v_prev=self.v_prev, c=self.c,
            dt=self.dt, iteration=self.iteration,
            integral=self.dt_controller.integral if self.dt_controller else 0.0,
            prev_error=self.dt_controller.prev_error if self.dt_controller else 0.0
        )
    
    def _initialize_von_karman_flow(self):
        """Initialize von Karman flow"""
        self.u = jnp.zeros((self.grid.nx, self.grid.ny))
        self.v = jnp.zeros((self.grid.nx, self.grid.ny))
        self.u = self.u * self.mask
        self.v = self.v * self.mask
        self.u = self.u.at[:, 0].set(0.0)
        self.u = self.u.at[:, -1].set(0.0)
        self.v = self.v.at[:, 0].set(0.0)
        self.v = self.v.at[:, -1].set(0.0)
        self.startup_ramp_steps = 100
        print(f"Startup: Will ramp inlet velocity over {self.startup_ramp_steps} steps")
    
    def _initialize_cavity_flow(self):
        """Initialize lid-driven cavity flow"""
        self.u = jnp.zeros((self.grid.nx, self.grid.ny))
        self.v = jnp.zeros((self.grid.nx, self.grid.ny))
        lid_velocity = 1.0
        self.u = self.u.at[:, -1].set(lid_velocity)
        
        # Add initial perturbation
        key = jax.random.PRNGKey(42)
        perturbation_u = 0.1 * jax.random.uniform(key, (self.grid.nx, self.grid.ny))
        perturbation_v = 0.1 * jax.random.uniform(jax.random.split(key)[1], (self.grid.nx, self.grid.ny))
        self.u = self.u + perturbation_u
        self.v = self.v + perturbation_v
        
        # Reapply boundary conditions
        self.u = self.u.at[:, 0].set(0.0)
        self.u = self.u.at[0, :].set(0.0)
        self.u = self.u.at[-1, :].set(0.0)
        self.u = self.u.at[:, -1].set(lid_velocity)
        self.v = self.v.at[:, 0].set(0.0)
        self.v = self.v.at[0, :].set(0.0)
        self.v = self.v.at[-1, :].set(0.0)
        self.v = self.v.at[:, -1].set(0.0)
        
        print(f"LDC initialized with Re={self.flow.Re}, lid velocity = {lid_velocity:.6f}, nu = {self.flow.nu:.6f}")
    
    def _initialize_taylor_green_flow(self):
        """Initialize Taylor-Green vortex"""
        X, Y = self.grid.X, self.grid.Y
        self.u = self.flow.U_inf * jnp.sin(X) * jnp.cos(Y)
        self.v = -self.flow.U_inf * jnp.cos(X) * jnp.sin(Y)
    
    def _compute_mask(self):
        """Compute mask based on flow type"""
        if self.sim_params.flow_type == 'lid_driven_cavity':
            return jnp.ones((self.grid.nx, self.grid.ny), dtype=float)
        elif self.sim_params.flow_type == 'taylor_green':
            return create_taylor_green_mask(self.grid.X, self.grid.Y, 2*jnp.pi)
        else:  # von_karman
            if NACA_AVAILABLE and hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'naca_airfoil':
                airfoil_type = '4-digit' if len(self.sim_params.naca_airfoil.replace('NACA ', '').replace(' ', '')) == 4 else '5-digit'
                naca_params = NACAParams(
                    airfoil_type=airfoil_type,
                    designation=self.sim_params.naca_airfoil,
                    chord_length=self.sim_params.naca_chord,
                    angle_of_attack=self.sim_params.naca_angle,
                    position_x=self.sim_params.naca_x,
                    position_y=self.sim_params.naca_y
                )
                sharp_eps = self.sim_params.eps * 0.1
                return create_naca_mask(self.grid.X, self.grid.Y, naca_params, sharp_eps)
            else:
                X, Y = self.grid.X, self.grid.Y
                phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
                return jax.nn.sigmoid(phi / self.sim_params.eps)
    
    def _step(self, u: jnp.ndarray, v: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Single time step using projection method"""
        mask = self.mask
        dx, dy = self.grid.dx, self.grid.dy
        
        # Compute SGS eddy viscosity
        nu_total = self.flow.nu
        if self.sim_params.use_les:
            delta = (dx * dy) ** 0.5
            if self.sim_params.les_model == 'dynamic_smagorinsky':
                nu_sgs, _ = dynamic_smagorinsky(u, v, dx, dy, delta)
                nu_total = self.flow.nu + nu_sgs
            elif self.sim_params.les_model == 'smagorinsky':
                nu_sgs = constant_smagorinsky(u, v, dx, dy, delta, self.sim_params.smagorinsky_constant)
                nu_total = self.flow.nu + nu_sgs
        
        # Advection
        if self.sim_params.advection_scheme == 'rk3' and rk3_step is not None:
            if self.sim_params.flow_type == 'von_karman':
                from advection_schemes.rk3_scheme import euler_step  # DEBUG: Using Euler for stability testing
                u_star, v_star = euler_step(u, v, dt, nu_total, dx, dy, mask, U_inf=self.flow.U_inf)
            elif self.sim_params.flow_type == 'taylor_green':
                from advection_schemes.rk3_scheme import euler_step  # DEBUG: Using Euler for stability testing
                u_star, v_star = euler_step(u, v, dt, nu_total, dx, dy, mask, U_inf=self.flow.U_inf)
            elif self.sim_params.flow_type == 'lid_driven_cavity':
                from advection_schemes.rk3_scheme import euler_step
                u_star, v_star = euler_step(u, v, dt, nu_total, dx, dy, mask, U_inf=0.0)
            else:
                u_star, v_star = rk3_step(u, v, dt, nu_total, dx, dy, mask)
        elif self.sim_params.advection_scheme == 'spectral' and spectral_step is not None:
            u_star, v_star = spectral_step(u, v, dt, nu_total, dx, dy, mask, dealias=True)
        else:
            u_star, v_star = rk3_step(u, v, dt, nu_total, dx, dy, mask)
        
        # Divergence
        if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
        else:
            div_star = divergence(u_star, v_star, dx, dy)
        
        rhs = div_star / dt
        
        # Pressure solve
        if self.sim_params.pressure_solver == 'fft' and poisson_fft is not None:
            p = poisson_fft(rhs, mask, dx, dy)
        elif self.sim_params.pressure_solver == 'cg' and poisson_cg is not None:
            p = poisson_cg(rhs, mask, dx, dy, max_iter=self.sim_params.pressure_max_iter, tol=self.sim_params.pressure_tolerance)
        elif self.sim_params.pressure_solver == 'cg_masked' and poisson_cg_masked is not None:
            p = poisson_cg_masked(rhs, mask, dx, dy, max_iter=self.sim_params.pressure_max_iter, flow_type=self.sim_params.flow_type)
        elif self.sim_params.pressure_solver == 'sor_masked' and poisson_sor_masked is not None:
            p = poisson_sor_masked(rhs, mask, dx, dy, max_iter=self.sim_params.pressure_max_iter, flow_type=self.sim_params.flow_type)
        elif self.sim_params.pressure_solver == 'multigrid' and poisson_multigrid is not None:
            p = poisson_multigrid(rhs, mask, dx, dy, flow_type=self.sim_params.flow_type)
        else:
            p = poisson_cg(rhs, mask, dx, dy)
        
        # Pressure correction
        if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
            dp_dx, dp_dy = grad_x_nonperiodic(p, dx), grad_y_nonperiodic(p, dy)
        else:
            dp_dx, dp_dy = grad_x(p, dx), grad_y(p, dy)
        u_corr = u_star - self.dt * dp_dx
        v_corr = v_star - self.dt * dp_dy
        
        # Brinkman penalization for von_karman
        if self.sim_params.flow_type == 'von_karman':
            u_corr, v_corr = apply_brinkman_penalization_consistent(
                u_corr, v_corr, mask, self.dt, nu_total, self.grid.dx
            )
            
            # Startup ramp
            if self.iteration < self.startup_ramp_steps:
                theta = (jnp.pi / 2.0) * (self.iteration / self.startup_ramp_steps)
                ramp_factor = jnp.sin(theta) ** 2
                inlet_velocity = self.flow.U_inf * ramp_factor
            else:
                inlet_velocity = self.flow.U_inf
            
            u_corr = u_corr.at[0, :].set(inlet_velocity)
            v_corr = v_corr.at[0, :].set(0.0)
            u_corr = u_corr.at[-1, :].set(u_corr.at[-2, :].get())
            v_corr = v_corr.at[-1, :].set(v_corr.at[-2, :].get())
            u_corr = u_corr.at[:, 0].set(0.0)
            u_corr = u_corr.at[:, -1].set(0.0)
            v_corr = v_corr.at[:, 0].set(0.0)
            v_corr = v_corr.at[:, -1].set(0.0)
        elif self.sim_params.flow_type == 'lid_driven_cavity':
            cavity_width = self.grid.lx
            cavity_height = self.grid.ly
            lid_velocity = 1.0
            u_corr, v_corr = apply_cavity_boundary_conditions(u_corr, v_corr, lid_velocity, cavity_width, cavity_height, self.grid.nx, self.grid.ny)
        elif self.sim_params.flow_type == 'taylor_green':
            u_corr, v_corr = apply_taylor_green_boundary_conditions(u_corr, v_corr, self.flow.U_inf, 2*jnp.pi, self.grid.nx, self.grid.ny)
        
        # Apply mask after boundary conditions to enforce walls
        u_corr = u_corr * mask
        v_corr = v_corr * mask
        
        return u_corr, v_corr, mask, p
    
    def get_step_jit(self):
        """Get cached JIT function"""
        key = (self.sim_params.advection_scheme, self.sim_params.pressure_solver,
               self.sim_params.pressure_max_iter, self.grid.nx, self.grid.ny)
        if key not in self._jit_cache:
            self._jit_cache[key] = jax.jit(self._step)
        return self._jit_cache[key]
    
    def step_for_visualization(self, compute_vorticity=True, compute_energy=True, compute_drag_lift=True, compute_diagnostics=True):
        """Optimized step for real-time viewing"""
        # Increment iteration counter at start for correct frame skip logic
        self.iteration += 1
        
        u_current = self.u
        v_current = self.v
        
        # Perform step
        u_new, v_new, mask, self.current_pressure = self._step_jit(self.u, self.v, self.dt)
        
        # Update timestep adaptively if enabled
        if self.sim_params.adaptive_dt and self.dt_controller is not None:
            # Compute divergence for adaptive dt
            div = self._divergence(u_new, v_new, self.grid.dx, self.grid.dy)
            div_max = float(jnp.max(jnp.abs(div)))
            
            # Update dt using PID controller (controller.update() returns only new_dt)
            old_dt = self.dt
            new_dt = self.dt_controller.update(self.dt, div_max)
            self.dt = new_dt
            
            if abs(new_dt - old_dt) > 1e-6:
                print(f"Adaptive dt: {old_dt:.6f} -> {new_dt:.6f} (div_max={div_max:.2e})")
        
        # Update scalar field if enabled
        if self.sim_params.use_scalar:
            if self.sim_params.flow_type == 'von_karman':
                self.c = scalar_advection_diffusion_nonperiodic(
                    self.c, u_new, v_new, self.dt, self.grid.dx, self.grid.dy,
                    self.sim_params.scalar_diffusivity
                )
            else:
                self.c = scalar_advection_diffusion_periodic(
                    self.c, u_new, v_new, self.dt, self.grid.dx, self.grid.dy,
                    self.sim_params.scalar_diffusivity
                )
        
        # Compute diagnostics (only every N-th frame if skip > 1)
        should_compute_metrics = (self.iteration % self.metrics_frame_skip == 0) if self.metrics_frame_skip > 1 else True
        if compute_diagnostics and should_compute_metrics:
            delta_u = u_new - u_current
            delta_v = v_new - v_current
            
            l2_delta_u = jnp.sqrt(jnp.sum(delta_u**2) * self.grid.dx * self.grid.dy)
            l2_delta_v = jnp.sqrt(jnp.sum(delta_v**2) * self.grid.dx * self.grid.dy)
            l2_delta_total = jnp.sqrt(l2_delta_u**2 + l2_delta_v**2)
            
            max_delta_u = jnp.max(jnp.abs(delta_u))
            max_delta_v = jnp.max(jnp.abs(delta_v))
            max_delta_total = jnp.maximum(max_delta_u, max_delta_v)
            
            u_rms = jnp.sqrt(jnp.sum(u_current**2) * self.grid.dx * self.grid.dy / (self.grid.nx * self.grid.ny))
            v_rms = jnp.sqrt(jnp.sum(v_current**2) * self.grid.dx * self.grid.dy / (self.grid.nx * self.grid.ny))
            vel_rms = jnp.sqrt(u_rms**2 + v_rms**2) + 1e-8
            
            rel_delta = l2_delta_total / (vel_rms * jnp.sqrt(self.grid.lx * self.grid.ly))
            
            self.history['l2_change'].append(float(l2_delta_total))
            self.history['l2_change_u'].append(float(l2_delta_u))
            self.history['l2_change_v'].append(float(l2_delta_v))
            self.history['max_change'].append(float(max_delta_total))
            self.history['rel_change'].append(float(rel_delta))
            
            div = self._divergence(u_new, v_new, self.grid.dx, self.grid.dy)
            max_div = jnp.max(jnp.abs(div))
            l2_div = jnp.sqrt(jnp.sum(div**2) * self.grid.dx * self.grid.dy)
            
            self.history['max_divergence'].append(float(max_div))
            self.history['l2_divergence'].append(float(l2_div))
        
        # Compute airfoil metrics (only every N-th frame if skip > 1)
        if self.compute_airfoil_metrics and self.sim_params.flow_type == 'von_karman' and should_compute_metrics:
            try:
                stag_x = find_stagnation_point(u_new, v_new, mask, self.grid.X, self.grid.dx)
                sep_x = find_separation_point(u_new, v_new, mask, self.grid.X, self.grid.dx, self.grid.dy)
                
                drag, lift = compute_forces(u_new, v_new, self.current_pressure, mask, 
                                          self.grid.dx, self.grid.dy, self.flow.nu)
                
                chord_length = self.sim_params.naca_chord if hasattr(self.sim_params, 'naca_chord') else 2.0
                rho = 1.0
                dynamic_pressure = 0.5 * rho * self.flow.U_inf**2
                cl = float(lift / (dynamic_pressure * chord_length)) if dynamic_pressure > 0 else 0.0
                cd = float(drag / (dynamic_pressure * chord_length)) if dynamic_pressure > 0 else 0.0
                
                surface = get_airfoil_surface_mask(mask, self.grid.dx, threshold=0.1)
                # Use actual far-field pressure (inlet pressure) instead of hardcoded 0.0
                # For projection method, pressure is defined up to a constant, so we use the mean pressure far from airfoil
                p_inf = float(jnp.mean(self.current_pressure[0:10, :]))  # Inlet region
                q_inf = 0.5 * rho * self.flow.U_inf**2
                cp = (self.current_pressure - p_inf) / q_inf
                cp_surface = jnp.where(surface, cp, jnp.inf)
                cp_min = float(jnp.min(cp_surface))
                
                airfoil_x = self.sim_params.naca_x if hasattr(self.sim_params, 'naca_x') else 2.5
                wake_x = airfoil_x + chord_length
                wake_x_idx = int(wake_x / self.grid.dx)
                if 0 <= wake_x_idx < self.grid.nx:
                    u_wake = u_new[wake_x_idx, :]
                    wake_deficit = float((self.flow.U_inf - jnp.mean(u_wake)) / self.flow.U_inf)
                else:
                    wake_deficit = 0.0
                
                self.history['airfoil_metrics']['stagnation_x'].append(stag_x)
                self.history['airfoil_metrics']['separation_x'].append(sep_x)
                self.history['airfoil_metrics']['CL'].append(cl)
                self.history['airfoil_metrics']['CD'].append(cd)
                self.history['airfoil_metrics']['Cp_min'].append(cp_min)
                self.history['airfoil_metrics']['wake_deficit'].append(wake_deficit)
                self.history['drag'].append(float(drag))
                self.history['lift'].append(float(lift))
            except Exception:
                for key in self.history['airfoil_metrics']:
                    self.history['airfoil_metrics'][key].append(0.0)
                self.history['drag'].append(0.0)
                self.history['lift'].append(0.0)
        
        self.history['dt'].append(self.dt)
        self.history['time'].append(self.iteration * self.dt)
        
        self.u_prev = u_current
        self.v_prev = v_current
        self.u = u_new
        self.v = v_new
        
        self.state = SimState(
            u=self.u, v=self.v, p=self.current_pressure,
            u_prev=self.u_prev, v_prev=self.v_prev, c=self.c,
            dt=self.dt, iteration=self.iteration,
            integral=self.dt_controller.integral if self.dt_controller else 0.0,
            prev_error=self.dt_controller.prev_error if self.dt_controller else 0.0
        )
        
        if compute_vorticity:
            if self.sim_params.flow_type == 'von_karman':
                vort = vorticity_nonperiodic(self.u, self.v, self.grid.dx, self.grid.dy)
            else:
                vort = self._vorticity(self.u, self.v, self.grid.dx, self.grid.dy)
        else:
            vort = jnp.zeros_like(self.u)
        
        return self.u, self.v, vort
    
    def step_with_simstate(self) -> SimState:
        """Step using pure SimState-based function"""
        adaptive_dt = self.sim_params.adaptive_dt
        dt_min = self.sim_params.dt_min if adaptive_dt else 1e-5
        dt_max = self.sim_params.dt_max if adaptive_dt else 0.01
        target_div = self.dt_controller.target if self.dt_controller else 1e-4
        Kp = self.dt_controller.Kp if self.dt_controller else 0.5
        Ki = self.dt_controller.Ki if self.dt_controller else 0.05
        Kd = self.dt_controller.Kd if self.dt_controller else 0.1
        
        self.state = step_pure(
            state=self.state, mask=self.mask, dx=self.grid.dx, dy=self.grid.dy,
            nu=self.flow.nu, flow_type=self.sim_params.flow_type,
            advection_scheme=self.sim_params.advection_scheme,
            pressure_solver=self.sim_params.pressure_solver,
            U_inf=self.flow.U_inf, use_les=self.sim_params.use_les,
            les_model=self.sim_params.les_model,
            smagorinsky_constant=self.sim_params.smagorinsky_constant,
            weno_epsilon=self.sim_params.weno_epsilon, limiter=self.sim_params.limiter,
            eps=self.sim_params.eps, adaptive_dt=adaptive_dt,
            dt_min=dt_min, dt_max=dt_max, target_div=target_div, Kp=Kp, Ki=Ki, Kd=Kd
        )
        
        self.u = self.state.u
        self.v = self.state.v
        self.u_prev = self.state.u_prev
        self.v_prev = self.state.v_prev
        self.dt = self.state.dt
        self.iteration = self.state.iteration
        
        if self.dt_controller:
            self.dt_controller.integral = self.state.integral
            self.dt_controller.prev_error = self.state.prev_error
        
        return self.state
    
    def apply_flow_type(self, flow_type: str):
        """Change flow type and reinitialize"""
        valid_flow_types = ['von_karman', 'lid_driven_cavity', 'taylor_green']
        if flow_type not in valid_flow_types:
            raise ValueError(f"Flow type must be one of {valid_flow_types}")
        
        self.sim_params.flow_type = flow_type
        
        if flow_type == 'lid_driven_cavity':
            self.grid = GridParams(nx=128, ny=128, lx=1.0, ly=1.0)
        elif flow_type == 'taylor_green':
            self.grid = GridParams(nx=128, ny=128, lx=2*jnp.pi, ly=2*jnp.pi)
        else:
            self.grid = GridParams(nx=128, ny=64, lx=4.0, ly=2.0)
        
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
            'l2_change': [], 'l2_change_u': [], 'l2_change_v': [], 'max_change': [], 'rel_change': [],
            'max_divergence': [], 'l2_divergence': [],
            'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 'Cp_min': [], 'wake_deficit': []}
        }
        self.iteration = 0
        
        print(f"Flow type changed to {flow_type}")
        print(f"Grid updated to {self.grid.nx}x{self.grid.ny} ({self.grid.lx}x{self.grid.ly})")
    
    def set_obstacle_type(self, obstacle_type: str, **kwargs):
        """Set obstacle type (cylinder or NACA airfoil)"""
        if obstacle_type not in ['cylinder', 'naca_airfoil']:
            raise ValueError("obstacle_type must be 'cylinder' or 'naca_airfoil'")
        
        self.sim_params.obstacle_type = obstacle_type
        
        if obstacle_type == 'naca_airfoil' and NACA_AVAILABLE:
            for key, value in kwargs.items():
                if hasattr(self.sim_params, f'naca_{key}'):
                    setattr(self.sim_params, f'naca_{key}', value)
        
        self.mask = self._compute_mask()
        print(f"Obstacle type set to {obstacle_type}")
    
    def update_naca_angle(self, angle_of_attack: float):
        """Update NACA angle of attack during simulation"""
        if self.sim_params.obstacle_type != 'naca_airfoil':
            print("Warning: Cannot update angle - current obstacle is not a NACA airfoil")
            return
        
        self.sim_params.naca_angle = angle_of_attack
        self.mask = self._compute_mask()
        jax.clear_caches()
        self._step_jit = jax.jit(self._step)
        
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
        if not self.sim_params.use_scalar:
            return
        
        x_clamped = max(0.0, min(x_pos, self.grid.lx))
        y_clamped = max(0.0, min(y_pos, self.grid.ly))
        
        ix = int(x_clamped / self.grid.dx)
        iy = int(y_clamped / self.grid.dy)
        
        ix = max(0, min(ix, self.grid.nx - 1))
        iy = max(0, min(iy, self.grid.ny - 1))
        
        current = self.c[ix, iy]
        self.c = self.c.at[ix, iy].set(jnp.minimum(current + amount, 1.0))
        
        print(f"Dye injected at ({x_pos:.2f}, {y_pos:.2f}) -> grid ({ix}, {iy})")
    
    def set_adaptive_dt(self):
        """Enable adaptive timestep control"""
        self.sim_params.adaptive_dt = True
        if self.dt_controller is None:
            try:
                from timestepping.adaptivedt import DivergencePIDController
                self.dt_controller = DivergencePIDController(
                    dt_min=self.sim_params.dt_min,
                    dt_max=self.sim_params.dt_max
                )
                print(f"Adaptive timestep enabled with dt_min={self.sim_params.dt_min}, dt_max={self.sim_params.dt_max}")
            except ImportError:
                print("Warning: DivergencePIDController not available, adaptive dt may not work properly")
        jax.clear_caches()
        if hasattr(self, '_step_jit'):
            delattr(self, '_step_jit')
    
    def set_fixed_dt(self, dt: float):
        """Set fixed timestep and disable adaptive control"""
        self.sim_params.adaptive_dt = False
        self.dt = dt
        self.sim_params.fixed_dt = dt
        print(f"Fixed timestep set to {dt}")
        jax.clear_caches()
        if hasattr(self, '_step_jit'):
            delattr(self, '_step_jit')
    
    def differentiable_rollout(self, num_steps: int,
                               checkpoint_every: int = None,
                               return_history: bool = False,
                               use_scan: bool = True):
        """Run differentiable simulation rollout"""
        from inverse.differentiable_rollout import run_rollout
        
        self.state = SimState(
            u=self.u, v=self.v, p=self.current_pressure,
            u_prev=self.u_prev, v_prev=self.v_prev, c=self.c,
            dt=self.dt, iteration=self.iteration,
            integral=self.dt_controller.integral if self.dt_controller else 0.0,
            prev_error=self.dt_controller.prev_error if self.dt_controller else 0.0
        )
        
        adaptive = self.sim_params.adaptive_dt
        dtc = self.dt_controller
        
        final_state, history = run_rollout(
            initial_state=self.state, mask=self.mask, num_steps=num_steps,
            use_scan=use_scan, checkpoint_every=checkpoint_every,
            return_history=return_history, dx=self.grid.dx, dy=self.grid.dy,
            nu=self.flow.nu, flow_type=self.sim_params.flow_type,
            advection_scheme=self.sim_params.advection_scheme,
            pressure_solver=self.sim_params.pressure_solver,
            U_inf=self.flow.U_inf, use_les=self.sim_params.use_les,
            les_model=self.sim_params.les_model,
            smagorinsky_constant=self.sim_params.smagorinsky_constant,
            weno_epsilon=self.sim_params.weno_epsilon, limiter=self.sim_params.limiter,
            eps=self.sim_params.eps, adaptive_dt=adaptive,
            dt_min=self.sim_params.dt_min if adaptive else 1e-5,
            dt_max=self.sim_params.dt_max if adaptive else 0.01,
            target_div=dtc.target if dtc else 1e-4,
            Kp=dtc.Kp if dtc else 0.5, Ki=dtc.Ki if dtc else 0.05, Kd=dtc.Kd if dtc else 0.1
        )
        
        self.u = final_state.u
        self.v = final_state.v
        self.p = final_state.p
        self.current_pressure = final_state.p
        self.u_prev = final_state.u_prev
        self.v_prev = final_state.v_prev
        self.c = final_state.c
        self.dt = final_state.dt
        self.iteration = final_state.iteration
        
        if dtc:
            dtc.integral = final_state.integral
            dtc.prev_error = final_state.prev_error
        
        return final_state, history
