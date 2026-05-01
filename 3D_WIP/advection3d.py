"""
Simple 3D advection scheme using RK3 time stepping.
"""

import jax
import jax.numpy as jnp


def rk3_step_3d(u, v, w, dt, dx, dy, dz):
    """
    RK3 time stepping for 3D advection.
    
    Args:
        u, v, w: Velocity fields (nx, ny, nz)
        dt: Time step
        dx, dy, dz: Grid spacing
        
    Returns:
        u_new, v_new, w_new: Updated velocity fields
    """
    def compute_rhs(u_in, v_in, w_in):
        """Compute RHS using upwind scheme"""
        # Simple upwind advection
        adv_u = -u_in * grad_x(u_in, dx) - v_in * grad_y(u_in, dy) - w_in * grad_z(u_in, dz)
        adv_v = -u_in * grad_x(v_in, dx) - v_in * grad_y(v_in, dy) - w_in * grad_z(v_in, dz)
        adv_w = -u_in * grad_x(w_in, dx) - v_in * grad_y(w_in, dy) - w_in * grad_z(w_in, dz)
        return adv_u, adv_v, adv_w
    
    # RK3 stages
    k1u, k1v, k1w = compute_rhs(u, v, w)
    u2 = u + dt * k1u
    v2 = v + dt * k1v
    w2 = w + dt * k1w
    
    k2u, k2v, k2w = compute_rhs(u2, v2, w2)
    u3 = 0.75 * u + 0.25 * (u2 + dt * k2u)
    v3 = 0.75 * v + 0.25 * (v2 + dt * k2v)
    w3 = 0.75 * w + 0.25 * (w2 + dt * k2w)
    
    k3u, k3v, k3w = compute_rhs(u3, v3, w3)
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u3 + dt * k3u)
    v_new = (1.0/3.0) * v + (2.0/3.0) * (v3 + dt * k3v)
    w_new = (1.0/3.0) * w + (2.0/3.0) * (w3 + dt * k3w)
    
    return u_new, v_new, w_new


def grad_x(f, dx):
    """Central difference gradient in x-direction"""
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)


def grad_y(f, dy):
    """Central difference gradient in y-direction"""
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)


def grad_z(f, dz):
    """Central difference gradient in z-direction"""
    return (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2.0 * dz)


def laplacian_3d(f, dx, dy, dz):
    """3D Laplacian operator"""
    lap_x = (jnp.roll(f, -1, axis=0) - 2*f + jnp.roll(f, 1, axis=0)) / dx**2
    lap_y = (jnp.roll(f, -1, axis=1) - 2*f + jnp.roll(f, 1, axis=1)) / dy**2
    lap_z = (jnp.roll(f, -1, axis=2) - 2*f + jnp.roll(f, 1, axis=2)) / dz**2
    return lap_x + lap_y + lap_z
