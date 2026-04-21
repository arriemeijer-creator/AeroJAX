"""
Main solver class and pure step functions for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Tuple, Optional

# Pre-import MAC modules for grid type switching
try:
    from advection_schemes.rk3_mac import rk_step_unified_mac
    MAC_ADVECTION_AVAILABLE = True
except ImportError:
    MAC_ADVECTION_AVAILABLE = False

try:
    from solver.operators_mac import (
        divergence_staggered, divergence_nonperiodic_staggered,
        vorticity_staggered, vorticity_nonperiodic_staggered,
        grad_x_staggered, grad_y_staggered,
        grad_x_nonperiodic_staggered, grad_y_nonperiodic_staggered
    )
    MAC_OPERATORS_AVAILABLE = True
except ImportError:
    MAC_OPERATORS_AVAILABLE = False

try:
    from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
    MAC_PRESSURE_AVAILABLE = True
except ImportError:
    MAC_PRESSURE_AVAILABLE = False

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
from .metrics import (
    compute_forces, get_airfoil_surface_mask, find_stagnation_point, find_separation_point,
    detect_vortex_shedding_stability, compute_time_averaged_coefficients, compute_CL_circulation
)

# Import external modules
try:
    from obstacles.naca_airfoils import NACAParams, create_naca_mask, naca_surface_distance, parse_naca_4digit, parse_naca_5digit
    NACA_AVAILABLE = True
except ImportError:
    NACA_AVAILABLE = False

try:
    from timestepping.adaptivedt import DivergencePIDController
except ImportError:
    DivergencePIDController = None

try:
    from pressure_solvers import poisson_multigrid
except ImportError:
    poisson_multigrid = None


@jax.jit
def update_dt_pure(dt: float, div_max: float, integral: float, prev_error: float,
                   target_div: float = 1e-4, Kp: float = 0.5, Ki: float = 0.05, Kd: float = 0.1,
                   dt_min: float = 1e-5, dt_max: float = 0.01, eta_max: float = None,
                   Re: float = None) -> Tuple[float, float, float]:
    """Pure JAX-compatible adaptive timestep update using PID controller"""
    # Safeguard against invalid divergence values
    div_is_valid = jnp.isfinite(div_max) & (div_max >= 0)
    
    # Re-dependent dt_max clamping
    if Re is not None and Re > 10000:
        dt_max = jnp.minimum(dt_max, 0.005 * (10000.0 / Re))
    
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


@jax.jit(static_argnames=['flow_type', 'advection_scheme', 'pressure_solver', 'les_model', 'limiter', 'use_les', 'adaptive_dt', 'v_cycles', 'fast_mode', 'grid_type'])
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
               limiter: str = 'minmod',
               v_cycles: int = 1,
               fast_mode: bool = False,
               brinkman_eta: float = 0.005,
               grid_type: str = 'collocated') -> SimState:
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
    
    # Select advection scheme based on grid type
    if grid_type == 'mac':
        from advection_schemes.rk3_mac import rk_step_unified_mac
        u_star, v_star = rk_step_unified_mac(u, v, dt, nu_total, dx, dy, mask, U_inf=U_inf,
                                             nu_sgs=None, nu_hyper_ratio=0.0, slip_walls=True,
                                             fast_mode=fast_mode, brinkman_eta=brinkman_eta)
        from solver.operators_mac import divergence_staggered, divergence_nonperiodic_staggered
        # Compute divergence for MAC grid
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic_staggered(u_star, v_star, dx, dy)
        else:
            div_star = divergence_staggered(u_star, v_star, dx, dy)
        # Use MAC pressure solver
        from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
        p = poisson_multigrid_mac(rhs, mask, dx, dy, v_cycles=v_cycles, flow_type=flow_type)
    else:
        # Collocated grid
        from advection_schemes.rk3_simple_new import rk_step_unified
        u_star, v_star = rk_step_unified(u, v, dt, nu_total, dx, dy, mask, U_inf=U_inf,
                                         nu_sgs=None, nu_hyper_ratio=0.0, slip_walls=True,
                                         fast_mode=fast_mode, brinkman_eta=brinkman_eta)
        # Compute divergence for collocated grid
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
        else:
            div_star = divergence(u_star, v_star, dx, dy)
        # Use collocated pressure solver
        p = poisson_multigrid(rhs, mask, dx, dy, v_cycles=v_cycles, flow_type=flow_type)
    
    rhs = div_star / dt  # Proper scaling for projection method

    # Pressure diagnostics every 1000 iterations
    if iteration % 1000 == 0:
        print(f"\n=== PRESSURE DIAGNOSTICS (iter {iteration}) ===")
        print(f"P min: {jnp.min(p):.8f}, max: {jnp.max(p):.8f}, mean: {jnp.mean(p):.8f}")
        print(f"RHS min: {jnp.min(rhs):.8f}, max: {jnp.max(rhs):.8f}, mean: {jnp.mean(rhs):.8f}")
        print(f"Div min: {jnp.min(div_star):.8f}, max: {jnp.max(div_star):.8f}, mean: {jnp.mean(div_star):.8f}")

    # Compute pressure gradient based on grid type
    if grid_type == 'mac':
        from solver.operators_mac import grad_x_nonperiodic_staggered, grad_y_nonperiodic_staggered, grad_x_staggered, grad_y_staggered
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            dp_dx = grad_x_nonperiodic_staggered(p, dx)  # Gradient at x-faces
            dp_dy = grad_y_nonperiodic_staggered(p, dy)  # Gradient at y-faces
        else:
            dp_dx = grad_x_staggered(p, dx)
            dp_dy = grad_y_staggered(p, dy)
    else:
        # Collocated grid
        if flow_type == 'von_karman':
            dp_dx = grad_x_nonperiodic(p, dx)
            dp_dy = grad_y_nonperiodic(p, dy)
        else:
            dp_dx = grad_x(p, dx)
            dp_dy = grad_y(p, dy)

    # For sharp mask: enforce zero normal pressure gradient at solid boundary
    # Compute mask gradient to find the interface
    if flow_type == 'von_karman':
        dm_dx = grad_x_nonperiodic(mask, dx)
        dm_dy = grad_y_nonperiodic(mask, dy)
    else:
        dm_dx = grad_x(mask, dx)
        dm_dy = grad_y(mask, dy)
    grad_mag = jnp.sqrt(dm_dx**2 + dm_dy**2)
    is_interface = grad_mag > 0.1 / dx  # Threshold for interface cells

    # At interface cells, set pressure gradient component normal to the interface to zero
    # Normal vector points from solid to fluid (from low mask to high mask)
    nx = dm_dx / (grad_mag + 1e-8)
    ny = dm_dy / (grad_mag + 1e-8)

    # Project pressure gradient onto tangent plane at interface
    dp_dot_n = dp_dx * nx + dp_dy * ny
    dp_dx = dp_dx - dp_dot_n * nx
    dp_dy = dp_dy - dp_dot_n * ny

    # Apply projection only at interface cells (handle both grid types)
    if grid_type == 'mac':
        from solver.operators_mac import grad_x_nonperiodic_staggered, grad_y_nonperiodic_staggered, grad_x_staggered, grad_y_staggered
        dp_dx_base = grad_x_nonperiodic_staggered(p, dx) if flow_type == 'von_karman' else grad_x_staggered(p, dx)
        dp_dy_base = grad_y_nonperiodic_staggered(p, dy) if flow_type == 'von_karman' else grad_y_staggered(p, dy)
    else:
        dp_dx_base = grad_x_nonperiodic(p, dx) if flow_type == 'von_karman' else grad_x(p, dx)
        dp_dy_base = grad_y_nonperiodic(p, dy) if flow_type == 'von_karman' else grad_y(p, dx)
    
    dp_dx = jnp.where(is_interface, dp_dx, dp_dx_base)
    dp_dy = jnp.where(is_interface, dp_dy, dp_dy_base)

    # Now apply the pressure correction (no mask weighting needed with sharp interface)
    u_new = u_star - dt * dp_dx
    v_new = v_star - dt * dp_dy

    # Diagnostic: divergence after pressure correction
    if grid_type == 'mac':
        from solver.operators_mac import divergence_nonperiodic_staggered, divergence_staggered
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_after_pressure = divergence_nonperiodic_staggered(u_new, v_new, dx, dy)
        else:
            div_after_pressure = divergence_staggered(u_new, v_new, dx, dy)
    else:
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_after_pressure = divergence_nonperiodic(u_new, v_new, dx, dy)
        else:
            div_after_pressure = divergence(u_new, v_new, dx, dy)

    # Apply boundary conditions after pressure correction (slip_walls=True by default)
    if flow_type == 'von_karman':
        u_new = u_new.at[0, :].set(U_inf)
        v_new = v_new.at[0, :].set(0.0)
        u_new = u_new.at[-1, :].set(u_new.at[-2, :].get())
        v_new = v_new.at[-1, :].set(v_new.at[-2, :].get())
        # Wall boundary conditions (slip by default)
        u_new = u_new.at[:, 0].set(u_new.at[:, 1].get())  # Bottom: slip
        u_new = u_new.at[:, -1].set(u_new.at[:, -2].get())  # Top: slip
        v_new = v_new.at[:, 0].set(0.0)  # Bottom: v=0
        v_new = v_new.at[:, -1].set(0.0)  # Top: v=0
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

    # Diagnostic: final divergence after boundary conditions
    if grid_type == 'mac':
        from solver.operators_mac import divergence_nonperiodic_staggered, divergence_staggered
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_final = divergence_nonperiodic_staggered(u_new, v_new, dx, dy)
        else:
            div_final = divergence_staggered(u_new, v_new, dx, dy)
    else:
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_final = divergence_nonperiodic(u_new, v_new, dx, dy)
        else:
            div_final = divergence(u_new, v_new, dx, dy)

    # Update previous velocity
    u_prev = u
    v_prev = v
    
    # Adaptive timestep update if enabled
    if adaptive_dt:
        if grid_type == 'mac':
            from solver.operators_mac import divergence_nonperiodic_staggered, divergence_staggered
            if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
                div_star = divergence_nonperiodic_staggered(u_star, v_star, dx, dy)
            else:
                div_star = divergence_staggered(u_star, v_star, dx, dy)
        else:
            if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
                div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
            else:
                div_star = divergence(u_star, v_star, dx, dy)
        div_rms = jnp.sqrt(jnp.mean(div_star**2))  # RMS divergence better than max

        eta_max = None
        if flow_type == 'von_karman':
            chi_max = jnp.max(1.0 - mask)
            rho = 1.0
            N_decay = 2.0
            eta_max = (rho / (N_decay * dt)) * chi_max

        # Adjust dt_max for low viscosity and high velocity cases to prevent instability
        dt_max_adaptive = dt_max
        if flow_type == 'lid_driven_cavity':
            dt_max_adaptive = min(dt_max, 0.005)  # Conservative dt_max for LDC
        if nu < 1e-4:
            dt_max_adaptive = min(dt_max_adaptive, 0.001)  # Much lower dt_max for very low viscosity
        elif nu < 1e-3:
            dt_max_adaptive = min(dt_max_adaptive, 0.002)  # Lower dt_max for low viscosity
        elif nu < 2e-3:
            dt_max_adaptive = min(dt_max_adaptive, 0.003)  # Additional reduction for moderate-low viscosity

        # Additional dt_max reduction for high velocities
        # Use inlet velocity (U_inf) instead of current velocity field to avoid startup ramp issues
        # Use jnp.where for JAX compatibility in JIT-compiled function
        dt_max_adaptive = jnp.where(U_inf > 5.0, 0.001, jnp.where(U_inf > 3.0, 0.002, jnp.where(U_inf > 1.5, 0.003, dt_max_adaptive)))

        dt_new, integral_new, error_new = update_dt_pure(
            dt, div_rms, state.integral, state.prev_error,
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

        # Apply percentage-based scaling for NACA airfoil parameters
        if sim_params.obstacle_type == 'naca_airfoil':
            # Scale x-position as percentage of domain width (25% of lx)
            x_percentage = 0.25  # 25% from left
            sim_params.naca_x = x_percentage * grid.lx

            # Scale y-position as percentage of domain height (50% of ly)
            y_percentage = 0.5  # Centered in Y
            sim_params.naca_y = y_percentage * grid.ly

            # Scale chord length as percentage of domain width (15% of lx)
            chord_percentage = 0.15  # 15% of domain width
            sim_params.naca_chord = chord_percentage * grid.lx
            print(f"DEBUG Solver __init__: obstacle_type={sim_params.obstacle_type}, grid.lx={grid.lx:.3f}, scaled naca_chord={sim_params.naca_chord:.3f}")
        else:
            print(f"DEBUG Solver __init__: obstacle_type={sim_params.obstacle_type}, skipping NACA scaling")

        # Compute characteristic length based on flow type and geometry
        from solver.params import compute_characteristic_length
        self.flow.L_char = compute_characteristic_length(sim_params.flow_type, geom, sim_params, sim_params.obstacle_type)
        
        # Resolve flow constraints to ensure consistent physics
        self.flow.resolve()
        print(f"Flow parameters resolved: U={self.flow.U_inf:.3f}, nu={self.flow.nu:.6f}, Re={self.flow.Re:.1f}, L={self.flow.L_char:.3f}")
        
        # Get Re-dependent parameters for stability
        from .params import get_re_parameters
        re_params = get_re_parameters(self.flow.Re, self.grid.nx)
        
        # Apply Re-dependent parameters
        self.sim_params.smagorinsky_constant = re_params['C_s']
        self.sim_params.nu_h = re_params['nu_h']
        self.sim_params.brinkman_eta = re_params['brinkman_eta']
        self.sim_params.dt_max = re_params['dt_max']
        
        # Store nu_hyper_ratio for use in RK3 scheme
        self.nu_hyper_ratio = re_params['nu_hyper_ratio']
        
        # Wall boundary condition (slip vs no-slip)
        self.slip_walls = True  # Default to slip walls
        
        print(f"Re-dependent parameters: C_s={re_params['C_s']:.3f}, nu_hyper_ratio={re_params['nu_hyper_ratio']:.3f}, "
              f"brinkman_eta={re_params['brinkman_eta']:.3f}, dt_max={re_params['dt_max']:.4f}")
        
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
            if self.sim_params.flow_type == 'lid_driven_cavity':
                cfl_target = 0.1  # Conservative for LDC
            elif self.flow.U_inf > 5.0:
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
            if self.sim_params.flow_type == 'lid_driven_cavity':
                dt_max = min(dt_max, 0.005)  # Conservative dt_max for LDC
            if self.flow.nu < 1e-4:
                dt_max = min(dt_max, 0.0015)  # Much lower dt_max for very low viscosity
            elif self.flow.nu < 1e-3:
                dt_max = min(dt_max, 0.003)  # Lower dt_max for low viscosity
            elif self.flow.nu < 2e-3:
                dt_max = min(dt_max, 0.0045)  # Additional reduction for moderate-low viscosity

            # Additional dt_max reduction for high velocities
            if self.flow.U_inf > 5.0:
                dt_max = min(dt_max, 0.0015)  # Very conservative for very high velocities
            elif self.flow.U_inf > 3.0:
                dt_max = min(dt_max, 0.003)  # Conservative for high velocities
            elif self.flow.U_inf > 1.5:
                dt_max = min(dt_max, 0.0045)  # Conservative for moderate velocities

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
        import jax
        jax.clear_caches()  # Clear JIT cache to ensure new code is compiled
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
        try:
            self._step_jit = self.get_step_jit()
            print(f"Successfully initialized _step_jit")
        except Exception as e:
            print(f"ERROR: Failed to initialize _step_jit: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Use appropriate divergence/vorticity functions based on grid type and flow type
        if self.sim_params.grid_type == 'mac':
            from solver.operators_mac import (
                divergence_staggered, divergence_nonperiodic_staggered,
                vorticity_staggered, vorticity_nonperiodic_staggered
            )
            if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
                self._vorticity = jax.jit(vorticity_nonperiodic_staggered, static_argnums=(2, 3))
                self._divergence = jax.jit(divergence_nonperiodic_staggered, static_argnums=(2, 3))
            else:
                self._vorticity = jax.jit(vorticity_staggered, static_argnums=(2, 3))
                self._divergence = jax.jit(divergence_staggered, static_argnums=(2, 3))
        else:
            # Collocated grid
            if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
                self._vorticity = jax.jit(vorticity_nonperiodic, static_argnums=(2, 3))
                self._divergence = jax.jit(divergence_nonperiodic, static_argnums=(2, 3))
            else:
                self._vorticity = jax.jit(vorticity, static_argnums=(2, 3))
                self._divergence = jax.jit(divergence, static_argnums=(2, 3))
        
        print(f"Initialized with: {self.sim_params.advection_scheme} advection, {self.sim_params.pressure_solver} pressure solver, grid_type={self.sim_params.grid_type}")
        
        self.history = {
            'time': [], 'dt': [], 'drag': [], 'lift': [],
            'l2_change': [], 'rms_change': [], 'l2_change_u': [], 'l2_change_v': [], 'max_change': [], 'change_99p': [], 'rel_change': [],
            'rms_divergence': [], 'l2_divergence': [],
            'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 'Cp_min': [], 'wake_deficit': [], 'time': []}
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
            grid_type=self.sim_params.grid_type,
            integral=self.dt_controller.integral if self.dt_controller else 0.0,
            prev_error=self.dt_controller.prev_error if self.dt_controller else 0.0
        )
    
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
        if hasattr(self, 'current_pressure'):
            self.current_pressure = jnp.zeros((self.grid.nx, self.grid.ny))
        
        # Initialize scalar dye field (always cell-centered)
        self.c = jnp.zeros((self.grid.nx, self.grid.ny))

        self.startup_ramp_steps = 0  # Disabled - startup ramp was creating transients
        print(f"Startup: Inlet velocity ramp disabled (startup_ramp_steps={self.startup_ramp_steps}), grid_type={self.sim_params.grid_type}")
    
    def _initialize_cavity_flow(self):
        """Initialize lid-driven cavity flow using Taylor-Green vortex for smooth initial condition"""
        X, Y = self.grid.X, self.grid.Y
        # Use Taylor-Green vortex as initial condition (smooth and divergence-free)
        if self.sim_params.grid_type == 'mac':
            # For MAC grid, initialize at cell centers then interpolate to faces
            u_center = 0.01 * jnp.sin(X) * jnp.cos(Y)
            v_center = -0.01 * jnp.cos(X) * jnp.sin(Y)
            # Interpolate to faces
            from solver.operators_mac import interpolate_to_x_face, interpolate_to_y_face
            self.u = interpolate_to_x_face(u_center)
            self.v = interpolate_to_y_face(v_center)
        else:
            # Collocated grid
            self.u = 0.01 * jnp.sin(X) * jnp.cos(Y)
            self.v = -0.01 * jnp.cos(X) * jnp.sin(Y)
        
        # Apply lid velocity at top boundary
        lid_velocity = 0.01
        if self.sim_params.grid_type == 'mac':
            # For MAC grid, set u at top y-faces
            self.u = self.u.at[:, -1].set(lid_velocity)
            self.v = self.v.at[:, -1].set(0.0)
        else:
            self.u = self.u.at[:, -1].set(lid_velocity)
            self.v = self.v.at[:, -1].set(0.0)
        
        # Initialize scalar dye field (always cell-centered)
        self.c = jnp.zeros((self.grid.nx, self.grid.ny))
        
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
        
        print(f"Taylor-Green initialized with grid_type={self.sim_params.grid_type}")
    
    def _compute_mask(self) -> jnp.ndarray:
        """Compute the obstacle mask based on geometry"""
        def smooth_mask(mask: jnp.ndarray) -> jnp.ndarray:
            """Simple smoothing using averaging to reduce sharp gradients"""
            # Simple 3x3 averaging kernel
            kernel = jnp.ones((3, 3)) / 9.0
            # Pad the mask
            mask_padded = jnp.pad(mask, ((1, 1), (1, 1)), mode='edge')
            # Manual convolution
            smoothed = jnp.zeros_like(mask)
            for i in range(3):
                for j in range(3):
                    smoothed += kernel[i, j] * mask_padded[i:i+mask.shape[0], j:j+mask.shape[1]]
            return smoothed
        
        if hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'naca_airfoil':
            from obstacles.naca_airfoils import NACAParams, create_naca_mask, parse_naca_4digit, parse_naca_5digit
            
            # Parse NACA designation
            naca_str = self.sim_params.naca_airfoil.upper().replace('NACA', '').strip()
            if len(naca_str) == 4:
                m, p, t = parse_naca_4digit(naca_str)
                airfoil_type = '4-digit'
            elif len(naca_str) == 5:
                cl, p, m, t = parse_naca_5digit(naca_str)
                airfoil_type = '5-digit'
            else:
                raise ValueError(f"Unsupported NACA designation: {self.sim_params.naca_airfoil}")
            
            naca_params = NACAParams(
                airfoil_type=airfoil_type,
                designation=self.sim_params.naca_airfoil,
                chord_length=self.sim_params.naca_chord,
                angle_of_attack=self.sim_params.naca_angle,
                position_x=self.sim_params.naca_x,
                position_y=self.sim_params.naca_y
            )
            # Unified masking: single sigmoid from SDF to χ (solid fraction)
            # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
            epsilon = self.sim_params.eps  # User-controlled via GUI slider
            # Get SDF from NACA function, then apply unified sigmoid
            from obstacles.naca_airfoils import naca_surface_distance
            if airfoil_type == '4-digit':
                sdf = naca_surface_distance(self.grid.X, self.grid.Y, naca_params.chord_length,
                                           naca_params.angle_of_attack, naca_params.position_x,
                                           naca_params.position_y, m, p, t)
            else:  # 5-digit
                sdf = naca_surface_distance(self.grid.X, self.grid.Y, naca_params.chord_length,
                                           naca_params.angle_of_attack, naca_params.position_x,
                                           naca_params.position_y, cl, p, m, t)
            chi = jax.nn.sigmoid(-sdf / epsilon)  # 1 inside solid, 0 outside
            mask = 1.0 - chi  # 1 in fluid, 0 in solid
            return mask
        elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'cow':
            from obstacles.cow import sdf_cow_side
            # Compute cow position relative to grid bounds
            # Use cow_x and cow_y from sim_params if available, otherwise use defaults
            cow_x = getattr(self.sim_params, 'cow_x', self.grid.lx * 0.25)  # 25% of domain width default
            cow_y = getattr(self.sim_params, 'cow_y', self.grid.ly * 0.35)  # 35% of domain height default
            # Compute scale factor based on grid dimensions relative to reference (20x3.75)
            ref_lx = 20.0
            ref_ly = 3.75
            scale_x = self.grid.lx / ref_lx
            scale_y = self.grid.ly / ref_ly
            cow_scale = (scale_x + scale_y) / 2.0  # Average of x and y scaling
            # Unified masking: single sigmoid from SDF to χ (solid fraction)
            # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
            epsilon = self.sim_params.eps  # User-controlled via GUI slider
            sdf = sdf_cow_side(self.grid.X, self.grid.Y, cow_x, cow_y, cow_scale)
            chi = jax.nn.sigmoid(-sdf / epsilon)  # 1 inside solid, 0 outside
            mask = 1.0 - chi  # 1 in fluid, 0 in solid
            return mask
        elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'three_cylinder_array':
            from obstacles.cylinder_array import sdf_three_cylinders
            cylinder_x = getattr(self.sim_params, 'cylinder_x', 5.0)
            cylinder_y = getattr(self.sim_params, 'cylinder_y', self.grid.ly / 2.0)
            cylinder_diameter = getattr(self.sim_params, 'cylinder_diameter', 0.5)
            cylinder_spacing = getattr(self.sim_params, 'cylinder_spacing', 0.5)
            # Unified masking: single sigmoid from SDF to χ (solid fraction)
            # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
            epsilon = self.sim_params.eps  # User-controlled via GUI slider
            sdf = sdf_three_cylinders(self.grid.X, self.grid.Y, cylinder_x, cylinder_y, cylinder_diameter, cylinder_spacing)
            chi = jax.nn.sigmoid(-sdf / epsilon)  # 1 inside solid, 0 outside
            mask = 1.0 - chi  # 1 in fluid, 0 in solid
            return mask
        elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'custom':
            from obstacles.freeform_drawer import create_freeform_mask_smooth
            custom_mask = getattr(self.sim_params, 'custom_mask', None)
            if custom_mask is not None:
                # Use user's epsilon setting from slider
                epsilon = self.sim_params.eps
                # Get obstacle center position from sliders
                center_x = getattr(self.sim_params, 'custom_x', self.grid.lx * 0.25)
                center_y = getattr(self.sim_params, 'custom_y', self.grid.ly * 0.5)
                # Scale the custom obstacle to fit in the domain while preserving aspect ratio
                # Use the smaller dimension to determine scale, so the drawing fits
                mask_height, mask_width = custom_mask.shape
                
                # Calculate scale to fit in domain (use 60% of the smaller dimension)
                domain_min_dim = min(self.grid.lx, self.grid.ly)
                scale = domain_min_dim * 0.6
                
                # Use same scale for both dimensions to preserve aspect ratio
                scale_x = scale
                scale_y = scale
                
                # Calculate offset to center the obstacle at the specified position
                # offset is the center position
                offset_x = center_x
                offset_y = center_y
                
                mask = create_freeform_mask_smooth(self.grid.X, self.grid.Y, custom_mask, 
                                                  scale_x=scale_x, scale_y=scale_y,
                                                  offset_x=offset_x, offset_y=offset_y,
                                                  smooth_width=epsilon)
                return mask
            else:
                # Fallback to cylinder if no custom mask
                X, Y = self.grid.X, self.grid.Y
                phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
                epsilon = self.sim_params.eps
                chi = jax.nn.sigmoid(-phi / epsilon)
                mask = 1.0 - chi
                return mask
        else:
            # Special case for lid_driven_cavity - all fluid (mask = 1 everywhere)
            if self.sim_params.flow_type == 'lid_driven_cavity':
                return jnp.ones_like(self.grid.X)
            
            X, Y = self.grid.X, self.grid.Y
            phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
            # Unified masking: single sigmoid from SDF to χ (solid fraction)
            # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
            epsilon = self.sim_params.eps  # User-controlled via GUI slider
            chi = jax.nn.sigmoid(-phi / epsilon)  # 1 inside solid, 0 outside
            mask = 1.0 - chi  # 1 in fluid, 0 in solid
            return mask
    
    def _step_collocated(self, u: jnp.ndarray, v: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Single time step using projection method for collocated grid"""
        mask = self.mask
        dx, dy = self.grid.dx, self.grid.dy
        
        # Compute SGS eddy viscosity (constant Smagorinsky for stability and performance)
        nu_total = self.flow.nu
        nu_sgs = None
        
        if self.sim_params.use_les:
            delta = (dx * dy) ** 0.5  # Filter width for LES
            if self.sim_params.les_model == 'dynamic_smagorinsky':
                nu_sgs, _ = dynamic_smagorinsky(u, v, dx, dy, delta)
            else:
                nu_sgs = constant_smagorinsky(u, v, dx, dy, delta, self.sim_params.smagorinsky_constant)
            nu_total = self.flow.nu + nu_sgs
        
        # Advection
        fast_mode = getattr(self.sim_params, 'fast_mode', False)
        brinkman_eta = self.sim_params.brinkman_eta if hasattr(self.sim_params, 'brinkman_eta') else 0.01
        advection_mask = mask if self.sim_params.flow_type != 'lid_driven_cavity' else jnp.ones_like(mask)
        
        from advection_schemes.rk3_simple_new import rk_step_unified
        u_star, v_star = rk_step_unified(u, v, dt, nu_total, dx, dy, advection_mask, 
                                         U_inf=self.flow.U_inf, nu_sgs=nu_sgs, 
                                         nu_hyper_ratio=self.nu_hyper_ratio, 
                                         slip_walls=self.slip_walls, 
                                         fast_mode=fast_mode, 
                                         brinkman_eta=brinkman_eta)

        # Divergence
        if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
        else:
            div_star = divergence(u_star, v_star, dx, dy)

        rhs = div_star / dt  # Proper scaling for projection method
        
        # Pressure solve
        if self.sim_params.flow_type == 'lid_driven_cavity':
            from pressure_solvers.multigrid_solver import poisson_jacobi
            p = poisson_jacobi(rhs, mask, dx, dy, max_iter=5000, flow_type=self.sim_params.flow_type)
        else:
            p = poisson_multigrid(rhs, mask, dx, dy, v_cycles=self.sim_params.multigrid_v_cycles, 
                                  flow_type=self.sim_params.flow_type)
        
        # Pressure correction
        if self.sim_params.flow_type == 'von_karman':
            dp_dx, dp_dy = grad_x_nonperiodic(p, dx), grad_y_nonperiodic(p, dy)
        else:
            dp_dx, dp_dy = grad_x(p, dx), grad_y(p, dy)
        
        u_corr = u_star - self.dt * dp_dx
        v_corr = v_star - self.dt * dp_dy

        # Apply boundary conditions (Brinkman is now integrated into RHS computation)
        if self.sim_params.flow_type == 'von_karman':
            # Startup ramp - use cubic (smoothstep) for gentler initial ramp
            if self.iteration >= self.startup_ramp_steps:
                inlet_velocity = self.flow.U_inf
            else:
                t = self.iteration / self.startup_ramp_steps
                ramp_factor = 3 * t**2 - 2 * t**3  # Cubic smoothstep (zero derivative at endpoints)
                inlet_velocity = self.flow.U_inf * ramp_factor

            u_corr = u_corr.at[0, :].set(inlet_velocity)
            v_corr = v_corr.at[0, :].set(0.0)
            u_corr = u_corr.at[-1, :].set(u_corr.at[-2, :].get())
            v_corr = v_corr.at[-1, :].set(v_corr.at[-2, :].get())
            u_corr = u_corr.at[:, 0].set(0.0)
            u_corr = u_corr.at[:, -1].set(0.0)
            v_corr = v_corr.at[:, 0].set(0.0)
            v_corr = v_corr.at[:, -1].set(0.0)
        elif self.sim_params.flow_type == 'lid_driven_cavity':
            # Minimal boundary conditions: only apply lid velocity at top
            lid_velocity = 0.01
            u_corr = u_corr.at[:, -1].set(lid_velocity)
            v_corr = v_corr.at[:, -1].set(0.0)
        elif self.sim_params.flow_type == 'taylor_green':
            u_corr, v_corr = apply_taylor_green_boundary_conditions(u_corr, v_corr, self.flow.U_inf, 2*jnp.pi, self.grid.nx, self.grid.ny)
        
        return u_corr, v_corr, mask, p

    def _step_mac(self, u: jnp.ndarray, v: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Single time step using projection method for MAC staggered grid"""
        mask = self.mask
        dx, dy = self.grid.dx, self.grid.dy
        
        # Compute SGS eddy viscosity (constant Smagorinsky for stability and performance)
        nu_total = self.flow.nu
        nu_sgs = None
        
        if self.sim_params.use_les:
            delta = (dx * dy) ** 0.5  # Filter width for LES
            # Interpolate staggered velocities to cell centers for LES computation
            from solver.operators_mac import interpolate_to_cell_center
            u_center, v_center = interpolate_to_cell_center(u, v)
            if self.sim_params.les_model == 'dynamic_smagorinsky':
                nu_sgs, _ = dynamic_smagorinsky(u_center, v_center, dx, dy, delta)
            else:
                nu_sgs = constant_smagorinsky(u_center, v_center, dx, dy, delta, self.sim_params.smagorinsky_constant)
            nu_total = self.flow.nu + nu_sgs
        
        # Advection
        fast_mode = getattr(self.sim_params, 'fast_mode', False)
        brinkman_eta = self.sim_params.brinkman_eta if hasattr(self.sim_params, 'brinkman_eta') else 0.01
        advection_mask = mask if self.sim_params.flow_type != 'lid_driven_cavity' else jnp.ones_like(mask)
        
        u_star, v_star = rk_step_unified_mac(u, v, dt, nu_total, dx, dy, advection_mask, 
                                             U_inf=self.flow.U_inf, nu_sgs=None, 
                                             nu_hyper_ratio=self.nu_hyper_ratio, 
                                             slip_walls=self.slip_walls, 
                                             fast_mode=fast_mode, 
                                             brinkman_eta=brinkman_eta)

        # Divergence
        if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic_staggered(u_star, v_star, dx, dy)
        else:
            div_star = divergence_staggered(u_star, v_star, dx, dy)

        rhs = div_star / dt  # Proper scaling for projection method
        
        # Pressure solve
        p = poisson_multigrid_mac(rhs, mask, dx, dy, v_cycles=self.sim_params.multigrid_v_cycles, 
                                  flow_type=self.sim_params.flow_type)
        
        # Pressure correction
        if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
            dp_dx = grad_x_nonperiodic_staggered(p, dx)
            dp_dy = grad_y_nonperiodic_staggered(p, dy)
        else:
            dp_dx = grad_x_staggered(p, dx)
            dp_dy = grad_y_staggered(p, dy)
        
        u_corr = u_star - self.dt * dp_dx
        v_corr = v_star - self.dt * dp_dy

        # Apply boundary conditions (Brinkman is now integrated into RHS computation)
        if self.sim_params.flow_type == 'von_karman':
            # Startup ramp - use cubic (smoothstep) for gentler initial ramp
            if self.iteration >= self.startup_ramp_steps:
                inlet_velocity = self.flow.U_inf
            else:
                t = self.iteration / self.startup_ramp_steps
                ramp_factor = 3 * t**2 - 2 * t**3  # Cubic smoothstep (zero derivative at endpoints)
                inlet_velocity = self.flow.U_inf * ramp_factor

            u_corr = u_corr.at[0, :].set(inlet_velocity)
            v_corr = v_corr.at[0, :].set(0.0)
            u_corr = u_corr.at[-1, :].set(u_corr.at[-2, :].get())
            v_corr = v_corr.at[-1, :].set(v_corr.at[-2, :].get())
            u_corr = u_corr.at[:, 0].set(0.0)
            u_corr = u_corr.at[:, -1].set(0.0)
            v_corr = v_corr.at[:, 0].set(0.0)
            v_corr = v_corr.at[:, -1].set(0.0)
        elif self.sim_params.flow_type == 'lid_driven_cavity':
            # Minimal boundary conditions: only apply lid velocity at top
            lid_velocity = 0.01
            u_corr = u_corr.at[:, -1].set(lid_velocity)
            v_corr = v_corr.at[:, -1].set(0.0)
        elif self.sim_params.flow_type == 'taylor_green':
            u_corr, v_corr = apply_taylor_green_boundary_conditions(u_corr, v_corr, self.flow.U_inf, 2*jnp.pi, self.grid.nx, self.grid.ny)
        
        return u_corr, v_corr, mask, p

    def _step(self, u: jnp.ndarray, v: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Single time step using projection method - delegates to grid-specific step function"""
        if self.sim_params.grid_type == 'mac' and MAC_ADVECTION_AVAILABLE and MAC_OPERATORS_AVAILABLE and MAC_PRESSURE_AVAILABLE:
            return self._step_mac(u, v, dt)
        else:
            return self._step_collocated(u, v, dt)
    
    def get_step_jit(self):
        """Get cached JIT function"""
        fast_mode = getattr(self.sim_params, 'fast_mode', False)
        grid_type = self.sim_params.grid_type
        key = (self.sim_params.advection_scheme, self.sim_params.pressure_solver,
               self.sim_params.pressure_max_iter, self.grid.nx, self.grid.ny,
               self.sim_params.use_les, self.sim_params.les_model, fast_mode,
               self.sim_params.multigrid_v_cycles, grid_type)
        if key not in self._jit_cache:
            # JIT-compile the appropriate step function based on grid type
            if grid_type == 'mac' and MAC_ADVECTION_AVAILABLE and MAC_OPERATORS_AVAILABLE and MAC_PRESSURE_AVAILABLE:
                self._step_jit = jax.jit(self._step_mac)
            else:
                self._step_jit = jax.jit(self._step_collocated)
            self._jit_cache[key] = self._step_jit
        else:
            self._step_jit = self._jit_cache[key]
        return self._step_jit
    
    def step_for_visualization(self, compute_vorticity=True, compute_energy=True, compute_drag_lift=True, compute_diagnostics=True):
        """Optimized step for real-time viewing"""
        # Increment iteration counter at start for correct frame skip logic
        self.iteration += 1

        u_current = self.u
        v_current = self.v

        # Perform step
        u_new, v_new, mask, self.current_pressure = self._step_jit(self.u, self.v, self.dt)

        # DEBUG: Check if implicit Brinkman is working (outside JIT)
        if self.iteration == 10:
            import numpy as np
            brinkman_eta = self.sim_params.brinkman_eta if hasattr(self.sim_params, 'brinkman_eta') else 0.01
            dt = self.dt
            chi = 1.0 - np.array(mask)
            max_chi = np.max(chi)
            penalization_factor = dt * max_chi / brinkman_eta
            print(f"\n=== DEBUG BRINKMAN CHECK (iter {self.iteration}) ===")
            print(f"brinkman_eta = {brinkman_eta}")
            print(f"dt = {dt}")
            print(f"max_chi = {max_chi}")
            print(f"penalization_factor = dt*chi/eta = {penalization_factor}")
            print(f"denominator = 1 + penalization_factor = {1.0 + penalization_factor}")
            max_u_inside = np.max(np.where(np.array(mask) < 0.5, np.abs(u_new), 0.0))
            max_v_inside = np.max(np.where(np.array(mask) < 0.5, np.abs(v_new), 0.0))
            print(f"max |u| inside solid = {max_u_inside}")
            print(f"max |v| inside solid = {max_v_inside}")
            print(f"If these are large, Brinkman is not working!")

        # Diagnostic: check Brinkman penalization effectiveness
        # Commented out for performance testing
        # if self.sim_params.flow_type == 'von_karman' and self.iteration % 100 == 0:
        #     import numpy as np
        #     mask_np = np.array(mask)
        #     u_new_np = np.array(u_new)
        #     v_new_np = np.array(v_new)
        #     max_u_inside = np.max(np.where(mask_np < 0.5, np.abs(u_new_np), 0))
        #     max_v_inside = np.max(np.where(mask_np < 0.5, np.abs(v_new_np), 0))
        #     print(f"DEBUG Brinkman: iter={self.iteration}, eta={self.sim_params.brinkman_eta:.1f}, max |u| inside solid={max_u_inside:.6f}, max |v| inside solid={max_v_inside:.6f}")
        
        # Update timestep adaptively if enabled
        if self.sim_params.adaptive_dt and self.dt_controller is not None:
            # Compute divergence for adaptive dt
            div = self._divergence(u_new, v_new, self.grid.dx, self.grid.dy)
            div_rms = float(jnp.sqrt(jnp.mean(div**2)))  # RMS divergence better than max


            # Update dt using PID controller (controller.update() returns only new_dt)
            old_dt = self.dt
            new_dt = self.dt_controller.update(self.dt, div_rms)
            self.dt = new_dt

            if abs(new_dt - old_dt) > 1e-6:
                print(f"Adaptive dt: {old_dt:.6f} -> {new_dt:.6f} (div_rms={div_rms:.2e})")
        
        # Update scalar field
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

            # Calculate velocity change magnitude
            delta_mag = jnp.sqrt(delta_u**2 + delta_v**2)

            u_rms = jnp.sqrt(jnp.sum(u_current**2) * self.grid.dx * self.grid.dy / (self.grid.nx * self.grid.ny))
            v_rms = jnp.sqrt(jnp.sum(v_current**2) * self.grid.dx * self.grid.dy / (self.grid.nx * self.grid.ny))
            vel_rms = jnp.sqrt(u_rms**2 + v_rms**2) + 1e-8

            rel_delta = l2_delta_total / (vel_rms * jnp.sqrt(self.grid.lx * self.grid.ly))

            self.history['l2_change'].append(float(l2_delta_total))
            self.history['rms_change'].append(float(l2_delta_total / jnp.sqrt(self.grid.nx * self.grid.ny)))
            self.history['l2_change_u'].append(float(l2_delta_u))
            self.history['l2_change_v'].append(float(l2_delta_v))
            self.history['max_change'].append(float(max_delta_total))
            self.history['change_99p'].append(float(jnp.percentile(jnp.abs(delta_mag), 99)))
            self.history['rel_change'].append(float(rel_delta))

            div = self._divergence(u_new, v_new, self.grid.dx, self.grid.dy)
            div_rms = jnp.sqrt(jnp.mean(div**2))  # RMS divergence better than max
            l2_div = jnp.sqrt(jnp.sum(div**2) * self.grid.dx * self.grid.dy)

            self.history['rms_divergence'].append(float(div_rms))
            self.history['l2_divergence'].append(float(l2_div))
        
        # Compute airfoil metrics (only every N-th frame if skip > 1)
        if self.compute_airfoil_metrics and self.sim_params.flow_type == 'von_karman' and should_compute_metrics:
            try:
                stag_x = find_stagnation_point(u_new, v_new, mask, self.grid.X, self.grid.dx)
                sep_x = find_separation_point(u_new, v_new, mask, self.grid.X, self.grid.dx, self.grid.dy)
                
                drag, lift, max_grad, surface_area = compute_forces(u_new, v_new, self.current_pressure, mask,
                                                                   self.grid.dx, self.grid.dy, self.flow.nu,
                                                                   chord_length=self.sim_params.naca_chord)

                chord_length = self.sim_params.naca_chord if hasattr(self.sim_params, 'naca_chord') else 2.0
                rho = 1.0
                dynamic_pressure = 0.5 * rho * self.flow.U_inf**2
                cl_pressure = float(lift / (dynamic_pressure * chord_length)) if dynamic_pressure > 0 else 0.0

                # Compute circulation-based CL
                vort = vorticity_nonperiodic(u_new, v_new, self.grid.dx, self.grid.dy)
                cl_circulation = float(compute_CL_circulation(vort, mask, self.grid.dx, self.grid.dy,
                                                             self.flow.U_inf, chord_length, fluid_threshold=0.95))

                # Use circulation-based CL for metrics (more accurate)
                cl = cl_circulation

                # Diagnostic logging every 1000 iterations
                if self.iteration % 1000 == 0:
                    print(f"DEBUG CL diagnostics (iter {self.iteration}):")
                    print(f"  chord={chord_length:.3f}, dynamic_pressure={dynamic_pressure:.4f}")
                    print(f"  lift={lift:.4f}, drag={drag:.4f}")
                    print(f"  CL_pressure={cl_pressure:.4f}, CL_circulation={cl_circulation:.4f}")
                    print(f"  CD={cd:.4f}")
                    print(f"  max_grad={float(max_grad):.4f}, surface_area={float(surface_area):.4f}")

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
                self.history['airfoil_metrics']['time'].append(self.iteration * self.dt)  # Store actual time
                self.history['drag'].append(float(drag))
                self.history['lift'].append(float(lift))

                # Check for vortex shedding stability every 25 iterations (reduced from 50)
                if len(self.history['airfoil_metrics']['CL']) >= 30 and self.iteration % 25 == 0:
                    import numpy as np
                    cl_history = np.array(self.history['airfoil_metrics']['CL'])
                    cd_history = np.array(self.history['airfoil_metrics']['CD'])
                    times = np.array(self.history['time'])

                    is_stable, stable_start, strouhal = detect_vortex_shedding_stability(
                        cl_history, times, self.flow.U_inf, chord_length
                    )

                    # Debug output - commented out for performance
                    # print(f"Stability check at iteration {self.iteration} (t={self.iteration * self.dt:.3f}s): "
                    #       f"is_stable={is_stable}, strouhal={strouhal:.3f}, stable_start={stable_start}, "
                    #       f"CL_samples={len(cl_history)}")

                    # Store stability state
                    if 'stability_state' not in self.history:
                        self.history['stability_state'] = []

                    self.history['stability_state'].append({
                        'iteration': self.iteration,
                        'is_stable': is_stable,
                        'stable_start': stable_start,
                        'strouhal': strouhal
                    })

                    # If stable, compute time-averaged coefficients
                    if is_stable and stable_start > 0:
                        avg_cl, avg_cd, num_samples = compute_time_averaged_coefficients(
                            cl_history, cd_history, stable_start
                        )

                        # Store time-averaged values
                        if 'time_averaged' not in self.history:
                            self.history['time_averaged'] = {
                                'CL': [], 'CD': [], 'stable_start': [], 'strouhal': [], 'num_samples': []
                            }

                        self.history['time_averaged']['CL'].append(avg_cl)
                        self.history['time_averaged']['CD'].append(avg_cd)
                        self.history['time_averaged']['stable_start'].append(stable_start)
                        self.history['time_averaged']['strouhal'].append(strouhal)
                        self.history['time_averaged']['num_samples'].append(num_samples)

                        # Commented out for performance
                        # print(f"Stability detected at iteration {self.iteration} (t={self.iteration * self.dt:.3f}s): "
                        #       f"St={strouhal:.3f}, avg_CL={avg_cl:.3f}, avg_CD={avg_cd:.3f} (n={num_samples})")
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
        if self.sim_params.flow_type == 'lid_driven_cavity':
            dt_max = min(dt_max, 0.005)  # Conservative dt_max for LDC
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
            dt_min=dt_min, dt_max=dt_max, target_div=target_div, Kp=Kp, Ki=Ki, Kd=Kd,
            v_cycles=self.sim_params.multigrid_v_cycles,
            fast_mode=getattr(self.sim_params, 'fast_mode', False),
            brinkman_eta=self.sim_params.brinkman_eta
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
        
        # Update dt_controller dt_max for LDC
        if flow_type == 'lid_driven_cavity' and self.dt_controller is not None:
            self.dt_controller.dt_max = min(self.dt_controller.dt_max, 0.005)
        
        if flow_type == 'lid_driven_cavity':
            self.grid = GridParams(nx=128, ny=128, lx=1.0, ly=1.0)
        elif flow_type == 'taylor_green':
            self.grid = GridParams(nx=128, ny=128, lx=2*jnp.pi, ly=2*jnp.pi)
        else:
            self.grid = GridParams(nx=512, ny=128, lx=20.0, ly=5.0)  # Updated for uniform grid spacing (dx=dy)
        
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
            'l2_change': [], 'rms_change': [], 'l2_change_u': [], 'l2_change_v': [], 'max_change': [], 'change_99p': [], 'rel_change': [],
            'rms_divergence': [], 'l2_divergence': [],
            'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 'Cp_min': [], 'wake_deficit': [], 'time': []}
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
        x_clamped = max(0.0, min(x_pos, self.grid.lx))
        y_clamped = max(0.0, min(y_pos, self.grid.ly))
        
        ix = int(x_clamped / self.grid.dx)
        iy = int(y_clamped / self.grid.dy)
        
        # Inject into a 5x5 area around the target cell for smoother distribution
        radius = 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                ix_target = ix + dx
                iy_target = iy + dy
                ix_target = max(0, min(ix_target, self.grid.nx - 1))
                iy_target = max(0, min(iy_target, self.grid.ny - 1))
                
                # Use Gaussian-like falloff for smoother injection
                distance = jnp.sqrt(dx**2 + dy**2)
                falloff = jnp.exp(-distance**2 / 2.0)
                current = self.c[ix_target, iy_target]
                self.c = self.c.at[ix_target, iy_target].set(jnp.minimum(current + amount * falloff, 1.0))
        
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

