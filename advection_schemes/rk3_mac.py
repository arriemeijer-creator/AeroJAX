"""
RK3 advection scheme for MAC (staggered) grid.
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from solver.operators_mac import (
    interpolate_to_cell_center, interpolate_to_x_face, interpolate_to_y_face,
    laplacian_staggered
)


@jax.jit(static_argnames=('nu_hyper_ratio', 'slip_walls'))
def rk3_step_mac(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float,
                 dx: float, dy: float, mask: jnp.ndarray, U_inf: float = 1.0,
                 nu_sgs: jnp.ndarray = None, nu_hyper_ratio: float = 0.0,
                 slip_walls: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    RK3 scheme for MAC staggered grid.
    u: (nx+1, ny) at x-faces
    v: (nx, ny+1) at y-faces
    mask: (nx, ny) cell-centered
    """
    nx, ny = mask.shape
    
    def grad_x_face(f_face):
        """Gradient at x-faces from face values"""
        return (f_face[1:, :] - f_face[:-1, :]) / dx
    
    def grad_y_face(f_face):
        """Gradient at y-faces from face values"""
        return (f_face[:, 1:] - f_face[:, :-1]) / dy
    
    def laplacian_face(f_face, axis):
        """Laplacian at face locations"""
        if axis == 0:  # x-faces
            f_padded = jnp.pad(f_face, ((1, 1), (1, 1)), mode='edge')
            lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f_face) / dx**2
            lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f_face) / dy**2
        else:  # y-faces
            f_padded = jnp.pad(f_face, ((1, 1), (1, 1)), mode='edge')
            lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f_face) / dx**2
            lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f_face) / dy**2
        return lap_x + lap_y
    
    def apply_boundary_conditions_mac(u_field, v_field):
        """Apply boundary conditions for MAC grid"""
        # Inlet (left)
        u_field = u_field.at[0, :].set(U_inf)
        v_field = v_field.at[0, :].set(0.0)
        # Outlet (right)
        u_field = u_field.at[-1, :].set(u_field[-2, :])
        v_field = v_field.at[-1, :].set(v_field[-2, :])
        # Walls
        if slip_walls:
            u_field = u_field.at[:, 0].set(u_field[:, 1])
            u_field = u_field.at[:, -1].set(u_field[:, -2])
            v_field = v_field.at[:, 0].set(0.0)
            v_field = v_field.at[:, -1].set(0.0)
        else:
            u_field = u_field.at[:, 0].set(0.0)
            u_field = u_field.at[:, -1].set(0.0)
            v_field = v_field.at[:, 0].set(0.0)
            v_field = v_field.at[:, -1].set(0.0)
        return u_field, v_field
    
    def compute_rhs_mac(u_in, v_in):
        """Compute RHS for MAC grid"""
        # Interpolate velocities to cell centers for advection computation
        u_center, v_center = interpolate_to_cell_center(u_in, v_in)
        
        # Apply mask
        u_center_masked = u_center * mask
        v_center_masked = v_center * mask
        
        # Upwind advection at cell centers, then interpolate back to faces
        def upwind_advection_center(field, vel_u, vel_v):
            f_x = jnp.where(vel_u > 0,
                            vel_u * (field - jnp.roll(field, 1, axis=0)) / dx,
                            vel_u * (jnp.roll(field, -1, axis=0) - field) / dx)
            f_y = jnp.where(vel_v > 0,
                            vel_v * (field - jnp.roll(field, 1, axis=1)) / dy,
                            vel_v * (jnp.roll(field, -1, axis=1) - field) / dy)
            return f_x + f_y
        
        adv_u_center = upwind_advection_center(u_center_masked, u_center_masked, v_center_masked)
        adv_v_center = upwind_advection_center(v_center_masked, u_center_masked, v_center_masked)
        
        # Interpolate advection to faces
        adv_u_face = interpolate_to_x_face(adv_u_center)
        adv_v_face = interpolate_to_y_face(adv_v_center)
        
        # Diffusion at faces
        lap_u = laplacian_face(u_in, 0)
        lap_v = laplacian_face(v_in, 1)
        
        nu_total = nu if nu_sgs is None else nu + nu_sgs
        
        rhs_u = -adv_u_face + nu_total * lap_u
        rhs_v = -adv_v_face + nu_total * lap_v
        
        # Apply mask at faces (interpolate mask to faces)
        mask_u = interpolate_to_x_face(mask)
        mask_v = interpolate_to_y_face(mask)
        
        return rhs_u * mask_u, rhs_v * mask_v
    
    # RK3 stages
    k1u, k1v = compute_rhs_mac(u, v)
    u2 = u + dt * k1u
    v2 = v + dt * k1v
    
    k2u, k2v = compute_rhs_mac(u2, v2)
    u3 = 0.75 * u + 0.25 * (u2 + dt * k2u)
    v3 = 0.75 * v + 0.25 * (v2 + dt * k2v)
    
    k3u, k3v = compute_rhs_mac(u3, v3)
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u3 + dt * k3u)
    v_new = (1.0/3.0) * v + (2.0/3.0) * (v3 + dt * k3v)
    
    u_new, v_new = apply_boundary_conditions_mac(u_new, v_new)
    
    return u_new, v_new


@jax.jit(static_argnames=('nu_hyper_ratio', 'slip_walls', 'fast_mode'))
def rk_step_unified_mac(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float,
                        dx: float, dy: float, mask: jnp.ndarray, U_inf: float = 1.0,
                        nu_sgs: jnp.ndarray = None, nu_hyper_ratio: float = 0.0,
                        slip_walls: bool = True, fast_mode: bool = False, 
                        brinkman_eta: float = 0.01) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unified RK2/RK3 step for MAC grid with Brinkman penalization.
    """
    nx, ny = mask.shape
    chi = 1.0 - mask
    eta = brinkman_eta
    
    def grad_x_face(f_face):
        return (f_face[1:, :] - f_face[:-1, :]) / dx
    
    def grad_y_face(f_face):
        return (f_face[:, 1:] - f_face[:, :-1]) / dy
    
    def laplacian_face(f_face, axis):
        if axis == 0:
            f_padded = jnp.pad(f_face, ((1, 1), (1, 1)), mode='edge')
            lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f_face) / dx**2
            lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f_face) / dy**2
        else:
            f_padded = jnp.pad(f_face, ((1, 1), (1, 1)), mode='edge')
            lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f_face) / dx**2
            lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f_face) / dy**2
        return lap_x + lap_y
    
    def apply_boundary_conditions_mac(u_field, v_field):
        # u_field is (nx+1, ny), v_field is (nx, ny+1)
        # Inlet (left) - u at x=0 face, v at x=0 cell
        u_field = u_field.at[0, :].set(U_inf)
        v_field = v_field.at[0, :].set(0.0)
        # Outlet (right) - u at x=L face, v at x=L cell
        u_field = u_field.at[-1, :].set(u_field[-2, :])
        v_field = v_field.at[-1, :].set(v_field[-2, :])
        # Walls
        if slip_walls:
            # u at y=0 and y=H faces
            u_field = u_field.at[:, 0].set(u_field[:, 1])
            u_field = u_field.at[:, -1].set(u_field[:, -2])
            # v at y=0 and y=H faces
            v_field = v_field.at[:, 0].set(0.0)
            v_field = v_field.at[:, -1].set(0.0)
        else:
            u_field = u_field.at[:, 0].set(0.0)
            u_field = u_field.at[:, -1].set(0.0)
            v_field = v_field.at[:, 0].set(0.0)
            v_field = v_field.at[:, -1].set(0.0)
        return u_field, v_field
    
    def compute_rhs_explicit_mac(u_in, v_in):
        u_center, v_center = interpolate_to_cell_center(u_in, v_in)
        
        def upwind_advection_center(field, vel_u, vel_v):
            f_x = jnp.where(vel_u > 0,
                            vel_u * (field - jnp.roll(field, 1, axis=0)) / dx,
                            vel_u * (jnp.roll(field, -1, axis=0) - field) / dx)
            f_y = jnp.where(vel_v > 0,
                            vel_v * (field - jnp.roll(field, 1, axis=1)) / dy,
                            vel_v * (jnp.roll(field, -1, axis=1) - field) / dy)
            return f_x + f_y
        
        adv_u_center = upwind_advection_center(u_center, u_center, v_center)
        adv_v_center = upwind_advection_center(v_center, u_center, v_center)
        
        adv_u_face = interpolate_to_x_face(adv_u_center)
        adv_v_face = interpolate_to_y_face(adv_v_center)
        
        lap_u = laplacian_face(u_in, 0)
        lap_v = laplacian_face(v_in, 1)
        
        nu_total = nu if nu_sgs is None else nu + nu_sgs
        
        rhs_u = -adv_u_face + nu_total * lap_u
        rhs_v = -adv_v_face + nu_total * lap_v
        
        if nu_hyper_ratio > 0:
            nu_hyper = nu * nu_hyper_ratio * (dx * dy)
            biharmonic_u = laplacian_face(lap_u, 0)
            biharmonic_v = laplacian_face(lap_v, 1)
            rhs_u = rhs_u - nu_hyper * biharmonic_u
            rhs_v = rhs_v - nu_hyper * biharmonic_v
        
        return rhs_u, rhs_v
    
    # Interpolate chi to faces for Brinkman
    # chi is (nx, ny), chi_u should be (nx+1, ny), chi_v should be (nx, ny+1)
    chi_u = interpolate_to_x_face(chi)
    chi_v = interpolate_to_y_face(chi)
    
    def rk3_step():
        k1u, k1v = compute_rhs_explicit_mac(u, v)
        u2 = (u + dt * k1u) / (1 + dt * chi_u / eta)
        v2 = (v + dt * k1v) / (1 + dt * chi_v / eta)
        
        k2u, k2v = compute_rhs_explicit_mac(u2, v2)
        u_star = u + dt * k2u
        v_star = v + dt * k2v
        u3 = (0.75 * u + 0.25 * u_star) / (1 + dt * chi_u / eta)
        v3 = (0.75 * v + 0.25 * v_star) / (1 + dt * chi_v / eta)
        
        k3u, k3v = compute_rhs_explicit_mac(u3, v3)
        u_star = u + dt * k3u
        v_star = v + dt * k3v
        u_new = ((1/3) * u + (2/3) * u_star) / (1 + dt * chi_u / eta)
        v_new = ((1/3) * v + (2/3) * v_star) / (1 + dt * chi_v / eta)
        
        u_new, v_new = apply_boundary_conditions_mac(u_new, v_new)
        return u_new, v_new
    
    def rk2_step():
        k1u, k1v = compute_rhs_explicit_mac(u, v)
        u_star = (u + dt * k1u) / (1 + dt * chi_u / eta)
        v_star = (v + dt * k1v) / (1 + dt * chi_v / eta)
        
        k2u, k2v = compute_rhs_explicit_mac(u_star, v_star)
        u_new = (u + 0.5 * dt * (k1u + k2u)) / (1 + dt * chi_u / eta)
        v_new = (v + 0.5 * dt * (k1v + k2v)) / (1 + dt * chi_v / eta)
        
        u_new, v_new = apply_boundary_conditions_mac(u_new, v_new)
        return u_new, v_new
    
    u_new, v_new = jax.lax.cond(fast_mode, rk2_step, rk3_step)
    return u_new, v_new
