"""
Force computation and airfoil metrics for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from .operators import grad_x, grad_y, vorticity, grad_x_nonperiodic, grad_y_nonperiodic


@jax.jit
def compute_forces(u: jnp.ndarray, v: jnp.ndarray, p: jnp.ndarray, mask: jnp.ndarray,
                   dx: float, dy: float, nu: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute drag and lift forces on immersed boundaries"""
    dp_dx = grad_x_nonperiodic(p, dx)
    dp_dy = grad_y_nonperiodic(p, dy)
    du_dx = grad_x_nonperiodic(u, dx)
    du_dy = grad_y_nonperiodic(u, dy)
    dv_dx = grad_x_nonperiodic(v, dx)
    dv_dy = grad_y_nonperiodic(v, dy)
    sigma_xx = -p + 2.0 * nu * du_dx
    sigma_yy = -p + 2.0 * nu * dv_dy
    sigma_xy = nu * (du_dy + dv_dx)
    dm_dx = grad_x_nonperiodic(mask, dx)
    dm_dy = grad_y_nonperiodic(mask, dy)

    # Normalize the gradient to get proper normal vector
    mag_grad = jnp.sqrt(dm_dx**2 + dm_dy**2) + 1e-12
    nx = dm_dx / mag_grad
    ny = dm_dy / mag_grad

    # Surface delta function is the magnitude of the gradient
    delta_surface = mag_grad

    # Force exerted by fluid on body = - (Force exerted by body on fluid)
    # Use delta_surface to weigh contribution across the whole epsilon width (standard IBM approach)
    # delta_surface is 0 away from the airfoil, so we sum across the whole domain
    drag_contrib = -(sigma_xx * nx + sigma_xy * ny) * delta_surface
    lift_contrib = -(sigma_xy * nx + sigma_yy * ny) * delta_surface

    drag = jnp.sum(drag_contrib) * dx * dy
    lift = jnp.sum(lift_contrib) * dx * dy
    
    return drag, lift


def get_airfoil_surface_mask(mask: jnp.ndarray, dx: float, threshold: float = 0.1) -> jnp.ndarray:
    """Find cells where mask gradient is large (the interface)"""
    dm_dx = grad_x_nonperiodic(mask, dx)
    dm_dy = grad_y_nonperiodic(mask, dx)
    grad_mag = jnp.sqrt(dm_dx**2 + dm_dy**2)
    return grad_mag > threshold


def find_stagnation_point(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                          grid_X: jnp.ndarray, dx: float, threshold: float = 0.1) -> float:
    """Find stagnation point on airfoil surface, returned in absolute domain coordinates"""
    surface = get_airfoil_surface_mask(mask, dx, threshold)
    u_mag = jnp.sqrt(u**2 + v**2)
    # Only consider surface cells
    surface_mag = jnp.where(surface, u_mag, jnp.inf)
    min_idx = jnp.argmin(surface_mag)
    return float(grid_X.flatten()[min_idx])


def find_separation_point(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                          grid_X: jnp.ndarray, dx: float, dy: float, threshold: float = 0.1) -> float:
    """Find separation by wall shear stress sign change on surface, returned in absolute domain coordinates"""
    surface = get_airfoil_surface_mask(mask, dx, threshold)
    vort = vorticity(u, v, dx, dy)
    # Only consider surface cells
    surface_vort = jnp.where(surface, vort, 0.0)

    # Get x-coordinates of surface cells with positive vs negative vorticity
    surface_x = jnp.where(surface, grid_X, jnp.inf)
    pos_vort_x = jnp.where(surface_vort > 0, surface_x, jnp.inf)
    neg_vort_x = jnp.where(surface_vort < 0, surface_x, -jnp.inf)

    # Separation is where vorticity changes sign
    min_pos = jnp.min(pos_vort_x)
    max_neg = jnp.max(neg_vort_x)

    if min_pos < jnp.inf and max_neg > -jnp.inf:
        return float((min_pos + max_neg) / 2)  # Midpoint of sign change
    return 0.0
