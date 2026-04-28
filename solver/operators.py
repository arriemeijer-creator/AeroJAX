"""
Differential operators for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Gradient in x-direction (periodic)"""
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)


@jax.jit
def grad_x_nonperiodic(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Non-periodic gradient in x-direction using forward/backward differences at boundaries"""
    nx, ny = f.shape
    grad = jnp.zeros_like(f)
    
    # Interior: central difference
    grad = grad.at[1:-1, :].set((f[2:, :] - f[:-2, :]) / (2.0 * dx))
    
    # Left boundary (inlet): forward difference
    grad = grad.at[0, :].set((f[1, :] - f[0, :]) / dx)
    
    # Right boundary (outlet): backward difference
    grad = grad.at[-1, :].set((f[-1, :] - f[-2, :]) / dx)
    
    return grad


@jax.jit
def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    """Gradient in y-direction (periodic)"""
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)


@jax.jit
def grad_y_nonperiodic(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    """Non-periodic gradient in y-direction using forward/backward differences at boundaries"""
    nx, ny = f.shape
    grad = jnp.zeros_like(f)
    
    # Interior: central difference
    grad = grad.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2.0 * dy))
    
    # Bottom boundary: forward difference
    grad = grad.at[:, 0].set((f[:, 1] - f[:, 0]) / dy)
    
    # Top boundary: backward difference
    grad = grad.at[:, -1].set((f[:, -1] - f[:, -2]) / dy)
    
    return grad


@jax.jit
def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Laplacian (periodic)"""
    return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) +
            jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 4 * f) / (dx**2)


@jax.jit
def laplacian_nonperiodic_x(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Laplacian with non-periodic x-direction"""
    f_padded = jnp.pad(f, ((1, 1), (0, 0)), mode='edge')
    f_xx = (f_padded[2:, :] - 2*f + f_padded[:-2, :]) / (dx*dx)
    f_yy = (jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 2*f) / (dy*dy)
    return f_xx + f_yy


@jax.jit
def divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Compute divergence (periodic)"""
    return grad_x(u, dx) + grad_y(v, dy)


@jax.jit
def divergence_nonperiodic(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Compute divergence (non-periodic in both x and y directions)
    
    For von Karman flow, we exclude boundary cells from divergence calculation
    to avoid artificial divergence from BC enforcement.
    """
    div = jnp.zeros_like(u)
    
    # Interior cells only (exclude boundaries)
    div = div.at[1:-1, 1:-1].set(
        (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx) +
        (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
    )
    
    return div


@jax.jit
def vorticity(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Compute vorticity (periodic)"""
    return grad_x(v, dx) - grad_y(u, dy)


@jax.jit
def vorticity_nonperiodic(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Compute vorticity (non-periodic in both x and y directions)"""
    return grad_x_nonperiodic(v, dx) - grad_y_nonperiodic(u, dy)


@jax.jit
def scalar_advection_diffusion_periodic(c: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, dt: float, 
                                       dx: float, dy: float, diffusivity: float) -> jnp.ndarray:
    """Advect and diffuse passive scalar (dye) - periodic"""
    # Advection using same scheme as velocity
    dc_dx = grad_x(c, dx)
    dc_dy = grad_y(c, dy)
    adv_c = u * dc_dx + v * dc_dy
    
    # Diffusion
    diff_c = diffusivity * laplacian(c, dx, dy)
    
    # Update
    c_new = c + dt * (-adv_c + diff_c)
    
    # Clamp to [0, 1]
    c_new = jnp.clip(c_new, 0.0, 1.0)
    
    return c_new


@jax.jit
def scalar_advection_diffusion_nonperiodic(c: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, dt: float, 
                                         dx: float, dy: float, diffusivity: float) -> jnp.ndarray:
    """Advect and diffuse passive scalar (dye) - non-periodic x-direction"""
    # Advection using same scheme as velocity
    dc_dx = grad_x_nonperiodic(c, dx)
    dc_dy = grad_y(c, dy)
    adv_c = u * dc_dx + v * dc_dy
    
    # Diffusion
    diff_c = diffusivity * laplacian_nonperiodic_x(c, dx, dy)
    
    # Update
    c_new = c + dt * (-adv_c + diff_c)
    
    # Clamp to [0, 1]
    c_new = jnp.clip(c_new, 0.0, 1.0)
    
    return c_new
