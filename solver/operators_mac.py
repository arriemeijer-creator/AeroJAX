"""
Staggered (MAC) grid differential operators for the Navier-Stokes solver.

In a MAC (Marker and Cell) staggered grid:
- u-velocity is stored at cell faces (i+1/2, j): shape (nx+1, ny)
- v-velocity is stored at cell faces (i, j+1/2): shape (nx, ny+1)
- Pressure is stored at cell centers (i, j): shape (nx, ny)
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def grad_x_staggered(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Gradient in x-direction on staggered grid.
    Computes gradient at cell faces from cell-centered values.
    Input f: (nx, ny) cell-centered
    Output: (nx+1, ny) at faces
    """
    # Forward difference at left face, backward at right face
    f_padded = jnp.pad(f, ((1, 1), (0, 0)), mode='edge')
    return (f_padded[1:, :] - f_padded[:-1, :]) / dx


@jax.jit
def grad_y_staggered(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    """
    Gradient in y-direction on staggered grid.
    Computes gradient at cell faces from cell-centered values.
    Input f: (nx, ny) cell-centered
    Output: (nx, ny+1) at faces
    """
    f_padded = jnp.pad(f, ((0, 0), (1, 1)), mode='edge')
    return (f_padded[:, 1:] - f_padded[:, :-1]) / dy


@jax.jit
def grad_x_nonperiodic_staggered(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Non-periodic gradient in x-direction on staggered grid.
    """
    f_padded = jnp.pad(f, ((1, 1), (0, 0)), mode='edge')
    return (f_padded[1:, :] - f_padded[:-1, :]) / dx


@jax.jit
def grad_y_nonperiodic_staggered(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    """
    Non-periodic gradient in y-direction on staggered grid.
    """
    f_padded = jnp.pad(f, ((0, 0), (1, 1)), mode='edge')
    return (f_padded[:, 1:] - f_padded[:, :-1]) / dy


@jax.jit
def divergence_staggered(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute divergence on staggered grid.
    u: (nx+1, ny) at x-faces
    v: (nx, ny+1) at y-faces
    Returns: (nx, ny) at cell centers
    """
    # du/dx: difference of u at adjacent faces
    du_dx = (u[1:, :] - u[:-1, :]) / dx
    # dv/dy: difference of v at adjacent faces
    dv_dy = (v[:, 1:] - v[:, :-1]) / dy
    return du_dx + dv_dy


@jax.jit
def divergence_nonperiodic_staggered(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute divergence on staggered grid with non-periodic boundaries.
    """
    du_dx = (u[1:, :] - u[:-1, :]) / dx
    dv_dy = (v[:, 1:] - v[:, :-1]) / dy
    return du_dx + dv_dy


@jax.jit
def vorticity_staggered(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute vorticity on staggered grid.
    Returns vorticity at cell centers (nx, ny).
    """
    # Interpolate u to cell centers for dv/dx
    u_center = 0.5 * (u[1:, :] + u[:-1, :])
    dv_dx = (jnp.roll(u_center, -1, axis=0) - jnp.roll(u_center, 1, axis=0)) / (2.0 * dx)
    
    # Interpolate v to cell centers for du/dy
    v_center = 0.5 * (v[:, 1:] + v[:, :-1])
    du_dy = (jnp.roll(v_center, -1, axis=1) - jnp.roll(v_center, 1, axis=1)) / (2.0 * dy)
    
    return dv_dx - du_dy


@jax.jit
def vorticity_nonperiodic_staggered(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute vorticity on staggered grid with non-periodic boundaries.
    """
    # Interpolate u to cell centers
    u_center = 0.5 * (u[1:, :] + u[:-1, :])
    u_padded = jnp.pad(u_center, ((1, 1), (0, 0)), mode='edge')
    dv_dx = (u_padded[2:, :] - u_padded[:-2, :]) / (2.0 * dx)
    
    # Interpolate v to cell centers
    v_center = 0.5 * (v[:, 1:] + v[:, :-1])
    v_padded = jnp.pad(v_center, ((0, 0), (1, 1)), mode='edge')
    du_dy = (v_padded[:, 2:] - v_padded[:, :-2]) / (2.0 * dy)
    
    return dv_dx - du_dy


@jax.jit
def interpolate_to_cell_center(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Interpolate staggered velocities to cell centers.
    u: (nx+1, ny) at x-faces -> u_center: (nx, ny)
    v: (nx, ny+1) at y-faces -> v_center: (nx, ny)
    """
    u_center = 0.5 * (u[1:, :] + u[:-1, :])
    v_center = 0.5 * (v[:, 1:] + v[:, :-1])
    return u_center, v_center


@jax.jit
def interpolate_to_x_face(u_center: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate from cell centers to x-faces.
    u_center: (nx, ny) -> u_face: (nx+1, ny)
    """
    u_padded = jnp.pad(u_center, ((1, 1), (0, 0)), mode='edge')
    return 0.5 * (u_padded[1:, :] + u_padded[:-1, :])


@jax.jit
def interpolate_to_y_face(v_center: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate from cell centers to y-faces.
    v_center: (nx, ny) -> v_face: (nx, ny+1)
    """
    v_padded = jnp.pad(v_center, ((0, 0), (1, 1)), mode='edge')
    return 0.5 * (v_padded[:, 1:] + v_padded[:, :-1])


@jax.jit
def laplacian_staggered(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Laplacian on staggered grid (for cell-centered quantities).
    """
    f_padded = jnp.pad(f, ((1, 1), (1, 1)), mode='edge')
    lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f) / dx**2
    lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f) / dy**2
    return lap_x + lap_y


@jax.jit
def scalar_advection_diffusion_periodic_staggered(c: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, 
                                                   dt: float, dx: float, dy: float, 
                                                   diffusivity: float) -> jnp.ndarray:
    """
    Advect and diffuse passive scalar on staggered grid - periodic.
    c: (nx, ny) cell-centered
    u: (nx+1, ny) at x-faces
    v: (nx, ny+1) at y-faces
    """
    # Interpolate velocities to cell centers for advection
    u_center, v_center = interpolate_to_cell_center(u, v)
    
    # Advection using upwind
    dc_dx = (jnp.roll(c, -1, axis=0) - jnp.roll(c, 1, axis=0)) / (2.0 * dx)
    dc_dy = (jnp.roll(c, -1, axis=1) - jnp.roll(c, 1, axis=1)) / (2.0 * dy)
    adv_c = u_center * dc_dx + v_center * dc_dy
    
    # Diffusion
    diff_c = diffusivity * laplacian_staggered(c, dx, dy)
    
    # Update
    c_new = c + dt * (-adv_c + diff_c)
    c_new = jnp.clip(c_new, 0.0, 1.0)
    
    return c_new


@jax.jit
def scalar_advection_diffusion_nonperiodic_staggered(c: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray,
                                                      dt: float, dx: float, dy: float,
                                                      diffusivity: float) -> jnp.ndarray:
    """
    Advect and diffuse passive scalar on staggered grid - non-periodic.
    """
    # Interpolate velocities to cell centers
    u_center, v_center = interpolate_to_cell_center(u, v)
    
    # Advection using central differences with edge padding
    c_padded_x = jnp.pad(c, ((1, 1), (0, 0)), mode='edge')
    dc_dx = (c_padded_x[2:, :] - c_padded_x[:-2, :]) / (2.0 * dx)
    
    c_padded_y = jnp.pad(c, ((0, 0), (1, 1)), mode='edge')
    dc_dy = (c_padded_y[:, 2:] - c_padded_y[:, :-2]) / (2.0 * dy)
    
    adv_c = u_center * dc_dx + v_center * dc_dy
    
    # Diffusion
    diff_c = diffusivity * laplacian_staggered(c, dx, dy)
    
    # Update
    c_new = c + dt * (-adv_c + diff_c)
    c_new = jnp.clip(c_new, 0.0, 1.0)
    
    return c_new
