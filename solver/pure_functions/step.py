"""
Pure step function for the Navier-Stokes solver.
This is a stateless, JIT-compilable function that performs one timestep.
"""

import jax
import jax.numpy as jnp
from typing import Optional

# Import from local modules
from ..params import SimState
from ..operators import (
    grad_x, grad_y, grad_x_nonperiodic, grad_y_nonperiodic,
    divergence, divergence_nonperiodic,
    vorticity, vorticity_nonperiodic,
    scalar_advection_diffusion_periodic, scalar_advection_diffusion_nonperiodic
)
from ..les_models import dynamic_smagorinsky, constant_smagorinsky

# Import pressure solvers
try:
    from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
    MAC_PRESSURE_AVAILABLE = True
except ImportError:
    MAC_PRESSURE_AVAILABLE = False

try:
    from pressure_solvers.cg import poisson_cg_solve
    CG_PRESSURE_AVAILABLE = True
except ImportError:
    CG_PRESSURE_AVAILABLE = False

try:
    from pressure_solvers.FFT import poisson_fft_solve
    FFT_PRESSURE_AVAILABLE = True
except ImportError:
    FFT_PRESSURE_AVAILABLE = False

try:
    from pressure_solvers import poisson_multigrid, poisson_jacobi
except ImportError:
    poisson_multigrid = None
    poisson_jacobi = None

# Import from dt_controller
from .dt_controller import update_dt_pure


@jax.jit(static_argnames=['flow_type', 'advection_scheme', 'pressure_solver', 'les_model', 'limiter', 'use_les', 'v_cycles', 'fast_mode', 'grid_type', 'verbose'])
def step_pure(state: SimState, mask: jnp.ndarray, sdf: Optional[jnp.ndarray], dx: float, dy: float,
               nu: float, U_inf: float, use_les: bool = False,
               smagorinsky_constant: float = 0.1, weno_epsilon: float = 1e-6,
               eps: float = 0.01, adaptive_dt: bool = False,
               dt_min: float = 5e-4, dt_max: float = 0.01, target_div: float = 1e-4,
               Kp: float = 0.5, Ki: float = 0.05, Kd: float = 0.1,
               flow_type: str = 'von_karman',
               advection_scheme: str = 'rk3',
               pressure_solver: str = 'multigrid',
               les_model: str = 'smagorinsky',
               limiter: str = 'minmod',
               v_cycles: int = 1,
               fast_mode: bool = False,
               brinkman_eta: float = 0.005,
               grid_type: str = 'collocated',
               verbose: bool = False,
               nn_pressure_model=None) -> SimState:
    """Pure JAX-compatible step function that accepts and returns SimState
    
    Args:
        state: Current simulation state (u, v, p, dt, iteration, etc.)
        mask: Domain mask (1=fluid, 0=solid)
        sdf: Signed distance function for obstacle (can be None for all-fluid cases)
        dx, dy: Grid spacing
        nu: Kinematic viscosity
        U_inf: Freestream velocity
        use_les: Whether to use LES turbulence modeling
        smagorinsky_constant: Smagorinsky constant for LES
        weno_epsilon: WENO epsilon parameter
        eps: Interface smoothing parameter
        adaptive_dt: Whether to use adaptive timestep
        dt_min, dt_max: Timestep bounds
        target_div: Target divergence for adaptive timestep
        Kp, Ki, Kd: PID controller gains for adaptive timestep
        flow_type: Type of flow ('von_karman', 'lid_driven_cavity', 'taylor_green')
        advection_scheme: Advection scheme ('rk3', etc.)
        pressure_solver: Pressure solver ('multigrid', 'cg', 'fft', 'jacobi', 'nn')
        les_model: LES model ('smagorinsky', 'dynamic_smagorinsky')
        limiter: Flux limiter
        v_cycles: Number of V-cycles for multigrid
        fast_mode: Whether to use fast mode optimizations
        brinkman_eta: Brinkman penalization parameter
        grid_type: Grid type ('collocated' or 'mac')
        verbose: Whether to print diagnostic information
        nn_pressure_model: Neural network pressure model (if pressure_solver='nn')
        
    Returns:
        Updated SimState
    """
    u, v, dt, iteration = state.u, state.v, state.dt, state.iteration
    
    # Debugging print statements (only when verbose=True)
    if verbose:
        print(f"=== PRE-STEP DIAGNOSTIC (iter {iteration}) ===")
        print(f"u_input has_nan: {jnp.any(jnp.isnan(u))}, min: {jnp.min(u):.6f}, max: {jnp.max(u):.6f}")
        print(f"v_input has_nan: {jnp.any(jnp.isnan(v))}, min: {jnp.min(v):.6f}, max: {jnp.max(v):.6f}")
        print(f"dt: {dt:.6f}")
    
    # Compute SGS eddy viscosity if LES is enabled
    nu_total = nu
    nu_sgs_field = None
    if use_les:
        delta = (dx * dy) ** 0.5
        # For MAC grid, interpolate velocities to cell centers before LES computation
        if grid_type == 'mac':
            from solver.operators_mac import interpolate_to_cell_center
            u_les, v_les = interpolate_to_cell_center(u, v)
        else:
            u_les, v_les = u, v
        if les_model == 'dynamic_smagorinsky':
            nu_sgs_field, _ = dynamic_smagorinsky(u_les, v_les, dx, dy, delta)
        elif les_model == 'smagorinsky':
            nu_sgs_field = constant_smagorinsky(u_les, v_les, dx, dy, delta, smagorinsky_constant)
    
    # Select advection scheme based on grid type
    if grid_type == 'mac':
        from advection_schemes.rk3_mac import rk_step_unified_mac
        u_star, v_star = rk_step_unified_mac(u, v, dt, nu, dx, dy, mask, U_inf=U_inf,
                                             nu_sgs=nu_sgs_field, nu_hyper_ratio=0.0, slip_walls=True,
                                             fast_mode=fast_mode, brinkman_eta=brinkman_eta,
                                             flow_type=flow_type)
        from solver.operators_mac import divergence_staggered, divergence_nonperiodic_staggered
        
        # Apply explicit inlet BC enforcement before divergence computation
        if flow_type == 'von_karman':
            # Apply inlet BC
            u_star = u_star.at[0, :].set(U_inf)
            v_star = v_star.at[0, :].set(0.0)
            
            # Apply slip walls to predictor velocity
            u_star = u_star.at[:, 0].set(u_star[:, 1])   # bottom
            u_star = u_star.at[:, -1].set(u_star[:, -2]) # top
            v_star = v_star.at[:, 0].set(0.0)
            v_star = v_star.at[:, -1].set(0.0)
        
        # Compute divergence for MAC grid
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic_staggered(u_star, v_star, dx, dy)
        else:
            div_star = divergence_staggered(u_star, v_star, dx, dy)
        # Apply mask to RHS for immersed boundary
        rhs = (div_star * mask) / dt  # Proper scaling for projection method
        # Pressure solve
        if pressure_solver == 'nn' and nn_pressure_model is not None:
            # Use learned pressure operator
            p = nn_pressure_model(rhs, mask)
        elif pressure_solver == 'jacobi':
            # Use Jacobi solver (interpolate to cell centers first)
            u_center, v_center = u[1:, :] + u[:-1, :], v[:, 1:] + v[:, :-1]
            p = poisson_jacobi(rhs, mask, dx, dy, max_iter=1000, tolerance=1e-6, flow_type=flow_type)
        elif pressure_solver == 'cg' and CG_PRESSURE_AVAILABLE:
            # Use Conjugate Gradient solver
            p = poisson_cg_solve(rhs, mask, dx, dy, maxiter=500, tol=1e-6, flow_type=flow_type)
        elif pressure_solver == 'fft' and FFT_PRESSURE_AVAILABLE:
            # Use FFT-based solver
            p = poisson_fft_solve(rhs, mask, dx, dy, flow_type=flow_type)
            # FFT returns None if obstacles are present - fall back to multigrid
            if p is None:
                from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
                p = poisson_multigrid_mac(rhs, mask, dx, dy, v_cycles=v_cycles, flow_type=flow_type)
        else:
            # Use MAC pressure solver
            from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
            p = poisson_multigrid_mac(rhs, mask, dx, dy, v_cycles=v_cycles, flow_type=flow_type)
    else:
        # Collocated grid
        from advection_schemes.rk3_simple_new import rk_step_unified
        u_star, v_star = rk_step_unified(u, v, dt, nu_total, dx, dy, mask, sdf=sdf, U_inf=U_inf,
                                         nu_sgs=None, nu_hyper_ratio=0.0, slip_walls=True,
                                         fast_mode=fast_mode, brinkman_eta=brinkman_eta,
                                         flow_type=flow_type)
        # Apply explicit inlet BC enforcement before divergence computation
        if flow_type == 'von_karman':
            # Apply inlet BC
            u_star = u_star.at[0, :].set(U_inf)
            v_star = v_star.at[0, :].set(0.0)
            
            # Apply slip walls to predictor velocity
            u_star = u_star.at[:, 0].set(u_star[:, 1])   # bottom
            u_star = u_star.at[:, -1].set(u_star[:, -2]) # top
            v_star = v_star.at[:, 0].set(0.0)
            v_star = v_star.at[:, -1].set(0.0)
        
        # Compute divergence for collocated grid
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
        else:
            div_star = divergence(u_star, v_star, dx, dy)
        # Apply mask to RHS for immersed boundary
        rhs = (div_star * mask) / dt  # Proper scaling for projection method
        # Pressure solve
        if pressure_solver == 'nn' and nn_pressure_model is not None:
            # Use learned pressure operator
            p = nn_pressure_model(rhs, mask)
        elif pressure_solver == 'jacobi':
            # Use Jacobi solver
            p = poisson_jacobi(rhs, mask, dx, dy, max_iter=1000, tolerance=1e-6, flow_type=flow_type)
        elif pressure_solver == 'cg' and CG_PRESSURE_AVAILABLE:
            # Use Conjugate Gradient solver
            p = poisson_cg_solve(rhs, mask, dx, dy, maxiter=500, tol=1e-6, flow_type=flow_type)
        elif pressure_solver == 'fft' and FFT_PRESSURE_AVAILABLE:
            # Use FFT-based solver
            p = poisson_fft_solve(rhs, mask, dx, dy, flow_type=flow_type)
            # FFT returns None if obstacles are present - fall back to multigrid
            if p is None:
                p = poisson_multigrid(rhs, mask, dx, dy, v_cycles=v_cycles, flow_type=flow_type)
        else:
            # Use collocated pressure solver
            p = poisson_multigrid(rhs, mask, dx, dy, v_cycles=v_cycles, flow_type=flow_type)

    # Pressure diagnostics every 1000 iterations OR if adaptive_dt is enabled
    if verbose and (iteration % 1000 == 0 or adaptive_dt):
        print(f"\n=== PRESSURE DIAGNOSTICS (iter {iteration}, adaptive_dt={adaptive_dt}) ===")
        print(f"dt: {dt:.8f}")
        print(f"Div_star min: {jnp.min(div_star):.8f}, max: {jnp.max(div_star):.8f}, mean: {jnp.mean(div_star):.8f}, has_nan: {jnp.any(~jnp.isfinite(div_star))}")
        print(f"RHS min: {jnp.min(rhs):.8f}, max: {jnp.max(rhs):.8f}, mean: {jnp.mean(rhs):.8f}, has_nan: {jnp.any(~jnp.isfinite(rhs))}")
        print(f"Mask min: {jnp.min(mask):.8f}, max: {jnp.max(mask):.8f}, mean: {jnp.mean(mask):.8f}, has_nan: {jnp.any(~jnp.isfinite(mask))}")
        print(f"P min: {jnp.min(p):.8f}, max: {jnp.max(p):.8f}, mean: {jnp.mean(p):.8f}, has_nan: {jnp.any(~jnp.isfinite(p))}")
        
        # Compute pressure residual to check convergence
        from ..operators import laplacian_nonperiodic_x
        laplacian_p = laplacian_nonperiodic_x(p, dx, dy)
        pressure_residual = jnp.linalg.norm(laplacian_p - rhs)
        rhs_norm = jnp.linalg.norm(rhs)
        relative_residual = pressure_residual / (rhs_norm + 1e-10)
        print(f"Pressure residual: {pressure_residual:.8e}, relative: {relative_residual:.8e}")
        if relative_residual > 1e-3:
            print(f"WARNING: Relative pressure residual > 1e-3 - solver may not be converging!")

    # Always print pressure residual on first few iterations for debugging
    if verbose and iteration < 5:
        from ..operators import laplacian_nonperiodic_x
        laplacian_p = laplacian_nonperiodic_x(p, dx, dy)
        pressure_residual = jnp.linalg.norm(laplacian_p - rhs)
        rhs_norm = jnp.linalg.norm(rhs)
        relative_residual = pressure_residual / (rhs_norm + 1e-10)
        print(f"Pressure residual (iter {iteration}): {pressure_residual:.8e}, relative: {relative_residual:.8e}")

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
    if grid_type == 'mac':
        # For MAC grid, project pressure at cell centers, then interpolate to faces
        # Compute mask gradient at cell centers
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            dm_dx = grad_x_nonperiodic(mask, dx)
            dm_dy = grad_y_nonperiodic(mask, dy)
        else:
            dm_dx = grad_x(mask, dx)
            dm_dy = grad_y(mask, dy)
        grad_mag = jnp.sqrt(dm_dx**2 + dm_dy**2)
        is_interface = grad_mag > 0.1 / dx  # Threshold for interface cells

        # Normal vector at cell centers
        nx_center = dm_dx / (grad_mag + 1e-8)
        ny_center = dm_dy / (grad_mag + 1e-8)

        # Base pressure gradient at cell centers
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            dp_dx_center = grad_x_nonperiodic(p, dx)
            dp_dy_center = grad_y_nonperiodic(p, dy)
        else:
            dp_dx_center = grad_x(p, dx)
            dp_dy_center = grad_y(p, dy)

        # Project pressure gradient at cell centers
        dp_dot_n_center = dp_dx_center * nx_center + dp_dy_center * ny_center
        dp_dx_proj = dp_dx_center - dp_dot_n_center * nx_center
        dp_dy_proj = dp_dy_center - dp_dot_n_center * ny_center

        # Apply projection only at interface cells
        dp_dx_center = jnp.where(is_interface, dp_dx_proj, dp_dx_center)
        dp_dy_center = jnp.where(is_interface, dp_dy_proj, dp_dy_center)

        # Interpolate projected gradients to faces for MAC grid
        from solver.operators_mac import interpolate_to_x_face, interpolate_to_y_face
        dp_dx = interpolate_to_x_face(dp_dx_center)
        dp_dy = interpolate_to_y_face(dp_dy_center)
    else:
        # Collocated grid projection
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

        # Apply projection only at interface cells
        if flow_type == 'von_karman':
            dp_dx_base = grad_x_nonperiodic(p, dx)
            dp_dy_base = grad_y_nonperiodic(p, dy)
        else:
            dp_dx_base = grad_x(p, dx)
            dp_dy_base = grad_y(p, dy)

        dp_dx = jnp.where(is_interface, dp_dx, dp_dx_base)
        dp_dy = jnp.where(is_interface, dp_dy, dp_dy_base)

    # Enforce zero pressure gradient at inlet for von Karman flow with corner smoothing
    if flow_type == 'von_karman':
        ny = p.shape[1]
        corner_smooth_width = 12  # Increased from 5 for better smoothing
        dp_dx = dp_dx.at[0, :].set(0.0)   # No pressure gradient at inlet

    # Now apply the pressure correction (no mask weighting needed with sharp interface)
    u_new = u_star - dt * dp_dx
    v_new = v_star - dt * dp_dy
    # Hard mask zeroing: force velocities to zero inside solid (mask < 0.5)
    # This is needed because mask has smooth sigmoid transition, so multiplication doesn't fully zero
    u_new = jnp.where(mask > 0.5, u_new, 0.0)
    v_new = jnp.where(mask > 0.5, v_new, 0.0)

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
        # For MAC grid, u is (nx+1, ny), v is (nx, ny+1)
        # Inlet (left)
        u_new = u_new.at[0, :].set(U_inf)
        v_new = v_new.at[0, :].set(0.0)
        # Outlet (right)
        u_new = u_new.at[-1, :].set(u_new[-2, :])
        v_new = v_new.at[-1, :].set(v_new[-2, :])
        # Wall boundary conditions (slip by default)
        u_new = u_new.at[:, 0].set(u_new[:, 1])  # Bottom: slip
        u_new = u_new.at[:, -1].set(u_new[:, -2])  # Top: slip
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

    # Hard mask zeroing: force velocities to zero inside solid (mask < 0.5)
    # This is needed because mask has smooth sigmoid transition, so multiplication doesn't fully zero
    u_new = jnp.where(mask > 0.5, u_new, 0.0)
    v_new = jnp.where(mask > 0.5, v_new, 0.0)

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
                div_star = divergence_nonperiodic_staggered(u_new, v_new, dx, dy)
            else:
                div_star = divergence_staggered(u_new, v_new, dx, dy)
        else:
            if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
                div_star = divergence_nonperiodic(u_new, v_new, dx, dy)
            else:
                div_star = divergence(u_new, v_new, dx, dy)
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
            dt_max_adaptive = min(dt_max, 0.008)  # Relaxed from 0.005
        if nu < 1e-4:
            dt_max_adaptive = min(dt_max_adaptive, 0.003)  # Relaxed from 0.001
        elif nu < 1e-3:
            dt_max_adaptive = min(dt_max_adaptive, 0.005)  # Relaxed from 0.002
        elif nu < 2e-3:
            dt_max_adaptive = min(dt_max_adaptive, 0.006)  # Relaxed from 0.003

        # Additional dt_max reduction for high velocities
        # Use inlet velocity (U_inf) instead of current velocity field to avoid startup ramp issues
        # Use jnp.where for JAX compatibility in JIT-compiled function
        dt_max_adaptive = jnp.where(U_inf > 5.0, 0.003, jnp.where(U_inf > 3.0, 0.005, jnp.where(U_inf > 1.5, 0.006, dt_max_adaptive)))

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
