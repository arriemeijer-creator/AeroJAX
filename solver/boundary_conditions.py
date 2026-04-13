"""
Boundary condition functions for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def apply_cavity_boundary_conditions(u: jnp.ndarray, v: jnp.ndarray, lid_velocity: float, 
                                    cavity_width: float, cavity_height: float, nx: int, ny: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply boundary conditions for lid-driven cavity"""
    u_bc = u.copy()
    v_bc = v.copy()
    
    # Top wall (moving lid)
    u_bc = u_bc.at[:, -1].set(lid_velocity)
    v_bc = v_bc.at[:, -1].set(0.0)
    
    # Bottom wall (no-slip)
    u_bc = u_bc.at[:, 0].set(0.0)
    v_bc = v_bc.at[:, 0].set(0.0)
    
    # Left wall (no-slip)
    u_bc = u_bc.at[0, :].set(0.0)
    v_bc = v_bc.at[0, :].set(0.0)
    
    # Right wall (no-slip)
    u_bc = u_bc.at[-1, :].set(0.0)
    v_bc = v_bc.at[-1, :].set(0.0)
    
    return u_bc, v_bc


@jax.jit
def create_cavity_mask(X: jnp.ndarray, Y: jnp.ndarray, cavity_width: float, cavity_height: float) -> jnp.ndarray:
    """Create mask for lid-driven cavity (1 inside cavity, 0 outside)"""
    # Simple rectangular cavity
    mask = ((X >= 0) & (X <= cavity_width) & (Y >= 0) & (Y <= cavity_height)).astype(float)
    return mask


@jax.jit
def apply_taylor_green_boundary_conditions(u: jnp.ndarray, v: jnp.ndarray, amplitude: float,
                                         domain_size: float, nx: int, ny: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply boundary conditions for Taylor-Green vortex (periodic)"""
    # Taylor-Green is periodic, so no boundary conditions needed
    # The flow is already periodic through the advection scheme
    return u, v


@jax.jit
def create_taylor_green_mask(X: jnp.ndarray, Y: jnp.ndarray, domain_size: float) -> jnp.ndarray:
    """Create mask for Taylor-Green vortex (full domain)"""
    # Taylor-Green is periodic, so mask is all ones
    return jnp.ones_like(X)


@jax.jit
def apply_backward_step_boundary_conditions(u: jnp.ndarray, v: jnp.ndarray, inlet_velocity: float,
                                          step_height: float, channel_height: float, 
                                          channel_length: float, nx: int, ny: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply boundary conditions for backward-facing step flow"""
    u_bc = u.copy()
    v_bc = v.copy()
    
    # Inlet (left boundary) - parabolic profile above step
    y_indices = jnp.arange(ny)
    y = y_indices * (channel_height / (ny - 1)) if ny > 1 else jnp.array([0.0])
    inlet_height = channel_height - step_height
    parabolic_profile = 6 * inlet_velocity * (y - step_height) * (channel_height - y) / (inlet_height**2)
    # Set only above step height
    inlet_mask = y >= step_height
    u_bc = u_bc.at[0, :].set(jnp.where(inlet_mask, parabolic_profile, 0.0))
    v_bc = v_bc.at[0, :].set(0.0)
    
    # Outlet (right boundary) - zero gradient
    u_bc = u_bc.at[-1, :].set(u_bc.at[-2, :].get())
    v_bc = v_bc.at[-1, :].set(v_bc.at[-2, :].get())
    
    # Top wall (no-slip)
    u_bc = u_bc.at[:, -1].set(0.0)
    v_bc = v_bc.at[:, -1].set(0.0)
    
    # Bottom wall and step (no-slip)
    u_bc = u_bc.at[:, 0].set(0.0)
    v_bc = v_bc.at[:, 0].set(0.0)
    
    # Step vertical face (no-slip)
    step_x_idx = int((channel_length/4) / (channel_length/nx)) if nx > 0 else 0
    if step_x_idx < nx:
        step_y_end = int((step_height / channel_height) * ny) if ny > 0 else 0
        u_bc = u_bc.at[step_x_idx, :step_y_end].set(0.0)
        v_bc = v_bc.at[step_x_idx, :step_y_end].set(0.0)
    
    return u_bc, v_bc
