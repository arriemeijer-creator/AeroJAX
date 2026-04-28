"""
LES/SGS turbulence models for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
from .operators import grad_x, grad_y


@jax.jit
def interpolate_to_cell_center(u: jnp.ndarray, v: jnp.ndarray) -> tuple:
    """
    Interpolate staggered velocities to cell centers.
    u: (nx+1, ny) at x-faces -> u_center: (nx, ny)
    v: (nx, ny+1) at y-faces -> v_center: (nx, ny)
    """
    u_center = 0.5 * (u[1:, :] + u[:-1, :])
    v_center = 0.5 * (v[:, 1:] + v[:, :-1])
    return u_center, v_center


@jax.jit
def compute_strain_rate(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float):
    """Compute strain rate magnitude |S| = sqrt(2*S_ij*S_ij)"""
    # Check if u and v have staggered shapes (MAC grid)
    # For MAC grid: u is (nx+1, ny), v is (nx, ny+1)
    # For collocated: u and v are both (nx, ny)
    u_shape = jnp.shape(u)
    v_shape = jnp.shape(v)
    nx, ny = u_shape[0] - 1, u_shape[1]  # Try to infer from u shape
    is_mac = v_shape[0] == nx and v_shape[1] == (ny + 1) if nx > 0 else False
    
    if is_mac:
        # Interpolate to cell centers for gradient computation
        u_center, v_center = interpolate_to_cell_center(u, v)
        u_grad = u_center
        v_grad = v_center
    else:
        # Collocated grid - use directly
        u_grad = u
        v_grad = v
    
    du_dx = grad_x(u_grad, dx)
    du_dy = grad_y(u_grad, dy)
    dv_dx = grad_x(v_grad, dx)
    dv_dy = grad_y(v_grad, dy)
    
    S_xx = du_dx
    S_yy = dv_dy
    S_xy = 0.5 * (du_dy + dv_dx)
    
    # |S| = sqrt(2 * S_ij * S_ij)
    S_mag = jnp.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2) + 1e-10)
    return S_mag, (du_dx, du_dy, dv_dx, dv_dy)


@jax.jit
def box_filter_2d(f: jnp.ndarray, dx: float, dy: float, filter_ratio: int = 2):
    """Test filter with width = filter_ratio * grid spacing (5-point stencil for 2Δ)"""
    # Generate all offset pairs using JAX meshgrid
    offset_range = jnp.arange(-filter_ratio, filter_ratio + 1)
    dx_offsets, dy_offsets = jnp.meshgrid(offset_range, offset_range, indexing='ij')
    
    # Flatten and stack for iteration
    offsets = jnp.stack([dx_offsets.ravel(), dy_offsets.ravel()], axis=1)
    
    # Use jax.lax.scan to iterate over offsets and accumulate
    def accumulate(acc, offset):
        i, j = offset
        rolled = jnp.roll(jnp.roll(f, i, axis=0), j, axis=1)
        return acc + rolled, None
    
    f_filtered, _ = jax.lax.scan(accumulate, jnp.zeros_like(f), offsets)
    f_filtered = f_filtered / ((2*filter_ratio+1)**2)
    
    return f_filtered


@jax.jit
def dynamic_smagorinsky(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, delta: float, alpha: float = 2.0):
    """
    Compute dynamic Smagorinsky eddy viscosity.
    
    Returns:
        nu_sgs: eddy viscosity field
        C_d: dynamic coefficient field (for debugging)
    """
    # Check if u and v have staggered shapes (MAC grid)
    u_shape = jnp.shape(u)
    v_shape = jnp.shape(v)
    nx, ny = u_shape[0] - 1, u_shape[1]  # Try to infer from u shape
    is_mac = v_shape[0] == nx and v_shape[1] == (ny + 1) if nx > 0 else False
    
    if is_mac:
        # Interpolate to cell centers for all operations
        u_center, v_center = interpolate_to_cell_center(u, v)
        u_work = u_center
        v_work = v_center
    else:
        # Collocated grid - use directly
        u_work = u
        v_work = v
    
    # Step 1: Compute strain rate at grid level
    S_mag, (du_dx, du_dy, dv_dx, dv_dy) = compute_strain_rate(u_work, v_work, dx, dy)
    
    # Step 2: Apply test filter (coarser scale, αΔ)
    u_test = box_filter_2d(u_work, dx, dy)
    v_test = box_filter_2d(v_work, dx, dy)
    
    # Step 3: Compute strain rate at test level (use same dx, dy - filter handles coarsening)
    S_mag_test, _ = compute_strain_rate(u_test, v_test, dx, dy)
    
    # Step 4: Compute Leonard stress L_ij = ũ_iũ_j - (u_i u_j)_test
    uu = u_work * u_work
    uv = u_work * v_work
    vv = v_work * v_work
    
    uu_test = box_filter_2d(uu, dx, dy)
    uv_test = box_filter_2d(uv, dx, dy)
    vv_test = box_filter_2d(vv, dx, dy)
    
    L_11 = u_test * u_test - uu_test
    L_12 = u_test * v_test - uv_test
    L_22 = v_test * v_test - vv_test
    
    # Step 5: Compute M_ij = α²|S̃|S̃_ij - (|S|S_ij)_test
    # Need S_ij components at grid and test levels (use same dx, dy)
    du_dx_test = grad_x(u_test, dx)
    du_dy_test = grad_y(u_test, dy)
    dv_dx_test = grad_x(v_test, dx)
    dv_dy_test = grad_y(v_test, dy)
    
    S_xx_test = du_dx_test
    S_yy_test = dv_dy_test
    S_xy_test = 0.5 * (du_dy_test + dv_dx_test)
    
    # Grid-level S_ij (already computed)
    S_xx = du_dx
    S_yy = dv_dy
    S_xy = 0.5 * (du_dy + dv_dx)
    
    # Filter grid-level |S|S_ij
    SS_xx_test = box_filter_2d(S_mag * S_xx, dx, dy)
    SS_xy_test = box_filter_2d(S_mag * S_xy, dx, dy)
    SS_yy_test = box_filter_2d(S_mag * S_yy, dx, dy)
    
    M_11 = alpha**2 * S_mag_test * S_xx_test - SS_xx_test
    M_12 = alpha**2 * S_mag_test * S_xy_test - SS_xy_test
    M_22 = alpha**2 * S_mag_test * S_yy_test - SS_yy_test
    
    # Step 6: Compute dynamic coefficient via least squares
    # Average over local patch (3x3) to stabilize
    def local_average(f):
        # Simple 3x3 box average
        f_avg = jnp.zeros_like(f)
        f_avg = f_avg.at[1:-1, 1:-1].set(
            (f[1:-1, 1:-1] + 
             f[2:, 1:-1] + f[:-2, 1:-1] +
             f[1:-1, 2:] + f[1:-1, :-2] +
             f[2:, 2:] + f[2:, :-2] + f[:-2, 2:] + f[:-2, :-2]) / 9.0
        )
        return f_avg
    
    LM = local_average(L_11 * M_11 + L_12 * M_12 + L_22 * M_22)
    MM = local_average(M_11 * M_11 + M_12 * M_12 + M_22 * M_22)
    
    # Compute C_d with clipping to prevent negative eddy viscosity
    C_d = LM / (MM + 1e-8)
    C_d = jnp.clip(C_d, 0.0, 0.2)  # Clip to [0, 0.2] for stability
    
    # Step 7: Compute eddy viscosity
    delta_sq = delta * delta
    nu_sgs = C_d * delta_sq * S_mag
    
    return nu_sgs, C_d


@jax.jit
def constant_smagorinsky(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, delta: float, C_s: float = 0.1):
    """Constant coefficient Smagorinsky model (C_s=0.1 for 2D turbulence)"""
    # Check if u and v have staggered shapes (MAC grid)
    u_shape = jnp.shape(u)
    v_shape = jnp.shape(v)
    nx, ny = u_shape[0] - 1, u_shape[1]  # Try to infer from u shape
    is_mac = v_shape[0] == nx and v_shape[1] == (ny + 1) if nx > 0 else False
    
    if is_mac:
        # Interpolate to cell centers for strain rate computation
        u_center, v_center = interpolate_to_cell_center(u, v)
        u_work = u_center
        v_work = v_center
    else:
        # Collocated grid - use directly
        u_work = u
        v_work = v
    
    S_mag, _ = compute_strain_rate(u_work, v_work, dx, dy)
    nu_sgs = (C_s * delta)**2 * S_mag
    return nu_sgs
