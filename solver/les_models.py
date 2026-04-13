"""
LES/SGS turbulence models for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
from .operators import grad_x, grad_y


@jax.jit
def compute_strain_rate(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float):
    """Compute strain rate magnitude |S| = sqrt(2*S_ij*S_ij)"""
    du_dx = grad_x(u, dx)
    du_dy = grad_y(u, dy)
    dv_dx = grad_x(v, dx)
    dv_dy = grad_y(v, dy)
    
    S_xx = du_dx
    S_yy = dv_dy
    S_xy = 0.5 * (du_dy + dv_dx)
    
    # |S| = sqrt(2 * S_ij * S_ij)
    S_mag = jnp.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2) + 1e-10)
    return S_mag, (du_dx, du_dy, dv_dx, dv_dy)


@jax.jit
def box_filter_2d(f: jnp.ndarray, dx: float, dy: float, filter_width: int = 2):
    """Simple box filter (test filter) for dynamic model"""
    # Average over 2x2 stencil (α=2 test filter)
    f_filtered = jnp.zeros_like(f)
    
    # Interior points (2 to end-2 to avoid boundaries)
    f_filtered = f_filtered.at[1:-1, 1:-1].set(
        (f[1:-1, 1:-1] + 
         f[2:, 1:-1] + f[:-2, 1:-1] +
         f[1:-1, 2:] + f[1:-1, :-2]) / 5.0
    )
    
    # Boundaries: copy original
    f_filtered = f_filtered.at[0, :].set(f[0, :])
    f_filtered = f_filtered.at[-1, :].set(f[-1, :])
    f_filtered = f_filtered.at[:, 0].set(f[:, 0])
    f_filtered = f_filtered.at[:, -1].set(f[:, -1])
    
    return f_filtered


@jax.jit
def dynamic_smagorinsky(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, delta: float, alpha: float = 2.0):
    """
    Compute dynamic Smagorinsky eddy viscosity.
    
    Returns:
        nu_sgs: eddy viscosity field
        C_d: dynamic coefficient field (for debugging)
    """
    # Step 1: Compute strain rate at grid level
    S_mag, (du_dx, du_dy, dv_dx, dv_dy) = compute_strain_rate(u, v, dx, dy)
    
    # Step 2: Apply test filter (coarser scale, αΔ)
    u_test = box_filter_2d(u, dx, dy)
    v_test = box_filter_2d(v, dx, dy)
    
    # Step 3: Compute strain rate at test level
    S_mag_test, _ = compute_strain_rate(u_test, v_test, dx * alpha, dy * alpha)
    
    # Step 4: Compute Leonard stress L_ij = ũ_iũ_j - (u_i u_j)_test
    uu = u * u
    uv = u * v
    vv = v * v
    
    uu_test = box_filter_2d(uu, dx, dy)
    uv_test = box_filter_2d(uv, dx, dy)
    vv_test = box_filter_2d(vv, dx, dy)
    
    L_11 = u_test * u_test - uu_test
    L_12 = u_test * v_test - uv_test
    L_22 = v_test * v_test - vv_test
    
    # Step 5: Compute M_ij = α²|S̃|S̃_ij - (|S|S_ij)_test
    # Need S_ij components at grid and test levels
    du_dx_test = grad_x(u_test, dx * alpha)
    du_dy_test = grad_y(u_test, dy * alpha)
    dv_dx_test = grad_x(v_test, dx * alpha)
    dv_dy_test = grad_y(v_test, dy * alpha)
    
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
def constant_smagorinsky(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, delta: float, C_s: float = 0.17):
    """Constant coefficient Smagorinsky model"""
    S_mag, _ = compute_strain_rate(u, v, dx, dy)
    nu_sgs = (C_s * delta)**2 * S_mag
    return nu_sgs
