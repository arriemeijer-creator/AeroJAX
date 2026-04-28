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


@jax.jit(static_argnames=('nu_hyper_ratio', 'slip_walls', 'flow_type'))
def rk3_step_mac(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float,
                 dx: float, dy: float, mask: jnp.ndarray, U_inf: float = 1.0,
                 nu_sgs: jnp.ndarray = None, nu_hyper_ratio: float = 0.0,
                 slip_walls: bool = True, flow_type: str = 'von_karman') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    RK3 scheme for MAC staggered grid.
    u: (nx+1, ny) at x-faces
    v: (nx, ny+1) at y-faces
    mask: (nx, ny) cell-centered
    """
    nx, ny = mask.shape
    
    def interpolate_v_to_u_face(v):
        """Interpolate v from y-faces to u-faces for cross-term advection
        v: (nx, ny+1) at y-faces -> v_at_u: (nx+1, ny) at u-faces
        """
        # Average v in y-direction to cell centers, then interpolate to x-faces
        v_center = 0.5 * (v[:, 1:] + v[:, :-1])  # (nx, ny)
        v_padded = jnp.pad(v_center, ((1, 1), (0, 0)), mode='edge')
        return 0.5 * (v_padded[1:, :] + v_padded[:-1, :])  # (nx+1, ny)
    
    def interpolate_u_to_v_face(u):
        """Interpolate u from x-faces to v-faces for cross-term advection
        u: (nx+1, ny) at x-faces -> u_at_v: (nx, ny+1) at v-faces
        """
        # Average u in x-direction to cell centers, then interpolate to y-faces
        u_center = 0.5 * (u[1:, :] + u[:-1, :])  # (nx, ny)
        u_padded = jnp.pad(u_center, ((0, 0), (1, 1)), mode='edge')
        return 0.5 * (u_padded[:, 1:] + u_padded[:, :-1])  # (nx, ny+1)
    
    def smooth_upwind(vel, flux_backward, flux_forward, epsilon=0.05):
        """Smooth upwind blending using tanh for C^1 continuity
        vel: velocity field
        flux_backward: flux when vel > 0 (backward difference)
        flux_forward: flux when vel < 0 (forward difference)
        epsilon: smoothing parameter (increased to 0.05 for stability)
        """
        alpha = 0.5 * (1 + jnp.tanh(vel / epsilon))
        return alpha * flux_forward + (1 - alpha) * flux_backward
    
    def grad_x_face(f_face):
        """Gradient at x-faces from face values"""
        return (f_face[1:, :] - f_face[:-1, :]) / dx
    
    def grad_y_face(f_face):
        """Gradient at y-faces from face values"""
        return (f_face[:, 1:] - f_face[:, :-1]) / dy
    
    def laplacian_face(f_face, axis):
        """Laplacian at face locations with proper ghost cell padding for BCs"""
        if axis == 0:  # x-faces: shape (nx+1, ny)
            if slip_walls:
                # Free-slip: symmetric padding (reflect with same sign)
                f_padded = jnp.pad(f_face, ((1, 1), (1, 1)), mode='symmetric')
            else:
                # No-slip: asymmetric padding (ghost cells with opposite sign for zero at wall)
                # For walls at y=0 and y=H, set ghost cells to -f[0,:] and -f[-1,:]
                f_padded_y = jnp.pad(f_face, ((0, 0), (1, 1)), mode='constant', constant_values=0)
                f_padded_y = f_padded_y.at[:, 0].set(-f_face[:, 0])
                f_padded_y = f_padded_y.at[:, -1].set(-f_face[:, -1])
                # For inlet/outlet, use edge padding
                f_padded = jnp.pad(f_padded_y, ((1, 1), (0, 0)), mode='edge')
            lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f_face) / dx**2
            lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f_face) / dy**2
        else:  # y-faces: shape (nx, ny+1)
            if slip_walls:
                # Free-slip: symmetric padding
                f_padded = jnp.pad(f_face, ((1, 1), (1, 1)), mode='symmetric')
            else:
                # No-slip: asymmetric padding for x-direction walls
                f_padded_x = jnp.pad(f_face, ((1, 1), (0, 0)), mode='constant', constant_values=0)
                f_padded_x = f_padded_x.at[0, :].set(-f_face[0, :])
                f_padded_x = f_padded_x.at[-1, :].set(-f_face[-1, :])
                # For top/bottom, use edge padding
                f_padded = jnp.pad(f_padded_x, ((0, 0), (1, 1)), mode='edge')
            lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f_face) / dx**2
            lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f_face) / dy**2
        return lap_x + lap_y
    
    def apply_boundary_conditions_mac(u_field, v_field):
        """Apply boundary conditions for MAC grid"""
        if flow_type == 'lid_driven_cavity':
            # LDC: Lid at top moves with U_inf, all other walls are no-slip
            # u is at (nx+1, ny) x-faces, v is at (nx, ny+1) y-faces
            u_field = u_field.at[:, -1].set(U_inf)  # Top lid (u at y-faces)
            u_field = u_field.at[:, 0].set(0.0)  # Bottom wall
            u_field = u_field.at[0, :].set(0.0)  # Left wall
            u_field = u_field.at[-1, :].set(0.0)  # Right wall
            v_field = v_field.at[:, 0].set(0.0)  # Bottom wall
            v_field = v_field.at[:, -1].set(0.0)  # Top wall
            v_field = v_field.at[0, :].set(0.0)  # Left wall
            v_field = v_field.at[-1, :].set(0.0)  # Right wall
        else:
            # von_karman: Inlet at left, outlet at right
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
        """Compute RHS for MAC grid directly on staggered faces
        This preserves mass conservation and avoids interpolation error.
        Computes advection on BC-compliant fields, applies mask at end.
        """
        # Interpolate mask to faces (for final masking)
        mask_u = interpolate_to_x_face(mask)
        mask_v = interpolate_to_y_face(mask)
        
        # Use BC-compliant fields for advection (NOT pre-masked)
        u_work, v_work = apply_boundary_conditions_mac(u_in, v_in)
        
        # Interpolate cross-velocity components for staggered-grid advection
        v_at_u = interpolate_v_to_u_face(v_work)  # v at u-faces
        u_at_v = interpolate_u_to_v_face(u_work)  # u at v-faces
        
        # Compute advection directly on faces using smooth upwind with proper padding
        # For u-momentum: u*du/dx + v*du/dy
        
        # du/dx at u-faces using padding (NOT roll)
        u_pad_x = jnp.pad(u_work, ((1, 1), (0, 0)), mode='edge')
        du_dx_back = (u_work - u_pad_x[:-2, :]) / dx
        du_dx_fwd = (u_pad_x[2:, :] - u_work) / dx
        
        # Smooth upwind blending for u*du/dx
        alpha_u = 0.5 * (1 + jnp.tanh(u_work / 0.05))
        adv_u_x = u_work * (alpha_u * du_dx_back + (1 - alpha_u) * du_dx_fwd)
        
        # v*du/dy term (using interpolated v at u-faces)
        # Compute du/dy at u-faces using central difference
        du_dy = jnp.zeros_like(u_work)
        du_dy = du_dy.at[:, 1:-1].set((u_work[:, 2:] - u_work[:, :-2]) / (2.0 * dy))
        du_dy = du_dy.at[:, 0].set((u_work[:, 1] - u_work[:, 0]) / dy)
        du_dy = du_dy.at[:, -1].set((u_work[:, -1] - u_work[:, -2]) / dy)
        adv_u_y = v_at_u * du_dy
        
        adv_u_face = adv_u_x + adv_u_y
        
        # For v-momentum: u*dv/dx + v*dv/dy
        
        # dv/dy at v-faces using padding (NOT roll)
        v_pad_y = jnp.pad(v_work, ((0, 0), (1, 1)), mode='edge')
        dv_dy_back = (v_work - v_pad_y[:, :-2]) / dy
        dv_dy_fwd = (v_pad_y[:, 2:] - v_work) / dy
        
        # Smooth upwind blending for v*dv/dy
        alpha_v = 0.5 * (1 + jnp.tanh(v_work / 0.05))
        adv_v_y = v_work * (alpha_v * dv_dy_back + (1 - alpha_v) * dv_dy_fwd)
        
        # u*dv/dx term (using interpolated u at v-faces)
        # Compute dv/dx at v-faces with proper shape
        dv_dx = jnp.zeros_like(v_work)
        dv_dx = dv_dx.at[1:-1, :].set((v_work[2:, :] - v_work[:-2, :]) / (2.0 * dx))
        dv_dx = dv_dx.at[0, :].set((v_work[1, :] - v_work[0, :]) / dx)
        dv_dx = dv_dx.at[-1, :].set((v_work[-1, :] - v_work[-2, :]) / dx)
        adv_v_x = u_at_v * dv_dx
        
        adv_v_face = adv_v_x + adv_v_y
        
        # Diffusion at faces with proper BC padding
        lap_u = laplacian_face(u_in, 0)
        lap_v = laplacian_face(v_in, 1)
        
        nu_total = nu if nu_sgs is None else nu + nu_sgs
        
        rhs_u = -adv_u_face + nu_total * lap_u
        rhs_v = -adv_v_face + nu_total * lap_v
        
        # Apply mask at the end to zero out RHS inside solids
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


@jax.jit(static_argnames=('nu_hyper_ratio', 'slip_walls', 'fast_mode', 'flow_type'))
def rk_step_unified_mac(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float,
                        dx: float, dy: float, mask: jnp.ndarray, U_inf: float = 1.0,
                        nu_sgs: jnp.ndarray = None, nu_hyper_ratio: float = 0.0,
                        slip_walls: bool = True, fast_mode: bool = False,
                        brinkman_eta: float = 0.01, flow_type: str = 'von_karman') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unified RK2/RK3 step for MAC grid with Brinkman penalization.
    mask: (nx, ny) cell-centered mask
    """
    nx, ny = mask.shape
    chi = 1.0 - mask
    eta = brinkman_eta
    # Avoid division by zero when brinkman_eta is 0 (no penalization)
    eta_safe = jnp.maximum(eta, 1e-10)

    # Determine if we need non-periodic advection
    use_nonperiodic = (flow_type == 'von_karman' or flow_type == 'lid_driven_cavity')
    
    def interpolate_v_to_u_face(v):
        """Interpolate v from y-faces to u-faces for cross-term advection
        v: (nx, ny+1) at y-faces -> v_at_u: (nx+1, ny) at u-faces
        """
        v_center = 0.5 * (v[:, 1:] + v[:, :-1])  # (nx, ny)
        v_padded = jnp.pad(v_center, ((1, 1), (0, 0)), mode='edge')
        return 0.5 * (v_padded[1:, :] + v_padded[:-1, :])  # (nx+1, ny)
    
    def interpolate_u_to_v_face(u):
        """Interpolate u from x-faces to v-faces for cross-term advection
        u: (nx+1, ny) at x-faces -> u_at_v: (nx, ny+1) at v-faces
        """
        u_center = 0.5 * (u[1:, :] + u[:-1, :])  # (nx, ny)
        u_padded = jnp.pad(u_center, ((0, 0), (1, 1)), mode='edge')
        return 0.5 * (u_padded[:, 1:] + u_padded[:, :-1])  # (nx, ny+1)
    
    def smooth_upwind(vel, flux_backward, flux_forward, epsilon=0.05):
        """Smooth upwind blending using tanh for C^1 continuity
        epsilon: increased to 0.05 for stability
        """
        alpha = 0.5 * (1 + jnp.tanh(vel / epsilon))
        return alpha * flux_forward + (1 - alpha) * flux_backward
    
    def grad_x_face(f_face):
        return (f_face[1:, :] - f_face[:-1, :]) / dx
    
    def grad_y_face(f_face):
        return (f_face[:, 1:] - f_face[:, :-1]) / dy
    
    def laplacian_face(f_face, axis):
        """Laplacian at face locations with proper ghost cell padding for BCs"""
        if axis == 0:  # x-faces: shape (nx+1, ny)
            if slip_walls:
                f_padded = jnp.pad(f_face, ((1, 1), (1, 1)), mode='symmetric')
            else:
                f_padded_y = jnp.pad(f_face, ((0, 0), (1, 1)), mode='constant', constant_values=0)
                f_padded_y = f_padded_y.at[:, 0].set(-f_face[:, 0])
                f_padded_y = f_padded_y.at[:, -1].set(-f_face[:, -1])
                f_padded = jnp.pad(f_padded_y, ((1, 1), (0, 0)), mode='edge')
            lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f_face) / dx**2
            lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f_face) / dy**2
        else:  # y-faces: shape (nx, ny+1)
            if slip_walls:
                f_padded = jnp.pad(f_face, ((1, 1), (1, 1)), mode='symmetric')
            else:
                f_padded_x = jnp.pad(f_face, ((1, 1), (0, 0)), mode='constant', constant_values=0)
                f_padded_x = f_padded_x.at[0, :].set(-f_face[0, :])
                f_padded_x = f_padded_x.at[-1, :].set(-f_face[-1, :])
                f_padded = jnp.pad(f_padded_x, ((0, 0), (1, 1)), mode='edge')
            lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f_face) / dx**2
            lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f_face) / dy**2
        return lap_x + lap_y
    
    def apply_boundary_conditions_mac(u_field, v_field):
        # u_field is (nx+1, ny), v_field is (nx, ny+1)
        if flow_type == 'lid_driven_cavity':
            # LDC: Lid at top moves with U_inf, all other walls are no-slip
            u_field = u_field.at[:, -1].set(U_inf)  # Top lid (u at y-faces)
            u_field = u_field.at[:, 0].set(0.0)  # Bottom wall
            u_field = u_field.at[0, :].set(0.0)  # Left wall
            u_field = u_field.at[-1, :].set(0.0)  # Right wall
            v_field = v_field.at[:, 0].set(0.0)  # Bottom wall
            v_field = v_field.at[:, -1].set(0.0)  # Top wall
            v_field = v_field.at[0, :].set(0.0)  # Left wall
            v_field = v_field.at[-1, :].set(0.0)  # Right wall
        else:
            # von_karman: Inlet at left, outlet at right
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
                # v at y=0 and y=H faces - v is at y-faces
                # For slip walls, v=0 at top and bottom walls
                v_field = v_field.at[:, 0].set(0.0)
                v_field = v_field.at[:, -1].set(0.0)
            else:
                # No-slip walls
                u_field = u_field.at[:, 0].set(0.0)
                u_field = u_field.at[:, -1].set(0.0)
            v_field = v_field.at[:, 0].set(0.0)
            v_field = v_field.at[:, -1].set(0.0)
        return u_field, v_field

    def upwind_advection_face_periodic(u_in, v_in):
        """Compute advection directly on faces for periodic boundaries
        Uses BC-compliant fields and proper padding for stability.
        """
        mask_u = interpolate_to_x_face(mask)
        mask_v = interpolate_to_y_face(mask)
        
        # Use BC-compliant fields for advection
        u_work, v_work = apply_boundary_conditions_mac(u_in, v_in)
        
        v_at_u = interpolate_v_to_u_face(v_work)
        u_at_v = interpolate_u_to_v_face(u_work)
        
        # u-momentum advection with proper padding
        u_pad_x = jnp.pad(u_work, ((1, 1), (0, 0)), mode='wrap')
        du_dx_back = (u_work - u_pad_x[:-2, :]) / dx
        du_dx_fwd = (u_pad_x[2:, :] - u_work) / dx
        alpha_u = 0.5 * (1 + jnp.tanh(u_work / 0.05))
        adv_u_x = u_work * (alpha_u * du_dx_back + (1 - alpha_u) * du_dx_fwd)
        
        # v*du/dy term - compute gradient at u-faces
        # Use central difference with roll for periodic boundaries
        du_dy = (jnp.roll(u_work, -1, axis=1) - jnp.roll(u_work, 1, axis=1)) / (2.0 * dy)
        adv_u_y = v_at_u * du_dy
        adv_u_face = adv_u_x + adv_u_y
        
        # v-momentum advection with proper padding
        v_pad_y = jnp.pad(v_work, ((0, 0), (1, 1)), mode='wrap')
        dv_dy_back = (v_work - v_pad_y[:, :-2]) / dy
        dv_dy_fwd = (v_pad_y[:, 2:] - v_work) / dy
        alpha_v = 0.5 * (1 + jnp.tanh(v_work / 0.05))
        adv_v_y = v_work * (alpha_v * dv_dy_back + (1 - alpha_v) * dv_dy_fwd)
        
        # u*dv/dx term - compute gradient at v-faces
        # Use central difference with roll for periodic boundaries
        dv_dx = (jnp.roll(v_work, -1, axis=0) - jnp.roll(v_work, 1, axis=0)) / (2.0 * dx)
        adv_v_x = u_at_v * dv_dx
        adv_v_face = adv_v_x + adv_v_y
        
        return adv_u_face * mask_u, adv_v_face * mask_v

    def upwind_advection_face_nonperiodic(u_in, v_in):
        """Compute advection directly on faces for non-periodic boundaries
        Uses BC-compliant fields and proper padding for stability.
        """
        mask_u = interpolate_to_x_face(mask)
        mask_v = interpolate_to_y_face(mask)
        
        # Use BC-compliant fields for advection
        u_work, v_work = apply_boundary_conditions_mac(u_in, v_in)
        
        v_at_u = interpolate_v_to_u_face(v_work)
        u_at_v = interpolate_u_to_v_face(u_work)
        
        # u-momentum advection with proper padding
        u_pad_x = jnp.pad(u_work, ((1, 1), (0, 0)), mode='edge')
        du_dx_back = (u_work - u_pad_x[:-2, :]) / dx
        du_dx_fwd = (u_pad_x[2:, :] - u_work) / dx
        alpha_u = 0.5 * (1 + jnp.tanh(u_work / 0.05))
        adv_u_x = u_work * (alpha_u * du_dx_back + (1 - alpha_u) * du_dx_fwd)
        
        # v*du/dy term - compute gradient at u-faces
        # Use forward/backward differences at boundaries to maintain shape
        du_dy = jnp.zeros_like(u_work)
        du_dy = du_dy.at[:, 1:-1].set((u_work[:, 2:] - u_work[:, :-2]) / (2.0 * dy))
        du_dy = du_dy.at[:, 0].set((u_work[:, 1] - u_work[:, 0]) / dy)
        du_dy = du_dy.at[:, -1].set((u_work[:, -1] - u_work[:, -2]) / dy)
        adv_u_y = v_at_u * du_dy
        adv_u_face = adv_u_x + adv_u_y
        
        # v-momentum advection with proper padding
        v_pad_y = jnp.pad(v_work, ((0, 0), (1, 1)), mode='edge')
        dv_dy_back = (v_work - v_pad_y[:, :-2]) / dy
        dv_dy_fwd = (v_pad_y[:, 2:] - v_work) / dy
        alpha_v = 0.5 * (1 + jnp.tanh(v_work / 0.05))
        adv_v_y = v_work * (alpha_v * dv_dy_back + (1 - alpha_v) * dv_dy_fwd)
        
        # u*dv/dx term - compute gradient at v-faces with proper shape
        dv_dx = jnp.zeros_like(v_work)
        dv_dx = dv_dx.at[1:-1, :].set((v_work[2:, :] - v_work[:-2, :]) / (2.0 * dx))
        dv_dx = dv_dx.at[0, :].set((v_work[1, :] - v_work[0, :]) / dx)
        dv_dx = dv_dx.at[-1, :].set((v_work[-1, :] - v_work[-2, :]) / dx)
        adv_v_x = u_at_v * dv_dx
        adv_v_face = adv_v_x + adv_v_y
        
        return adv_u_face * mask_u, adv_v_face * mask_v

    # Select advection function based on flow_type
    if use_nonperiodic:
        upwind_fn = upwind_advection_face_nonperiodic
    else:
        upwind_fn = upwind_advection_face_periodic

    def compute_rhs_explicit_mac(u_in, v_in):
        # Compute advection directly on faces (no cell-center interpolation)
        adv_u_face, adv_v_face = upwind_fn(u_in, v_in)

        # Diffusion with proper BC padding
        lap_u = laplacian_face(u_in, 0)
        lap_v = laplacian_face(v_in, 1)
        
        # Mask already applied in upwind_fn, but ensure for safety
        mask_u = interpolate_to_x_face(mask)
        mask_v = interpolate_to_y_face(mask)

        # Handle spatially varying viscosity (LES)
        if nu_sgs is None:
            nu_total_u = nu
            nu_total_v = nu
        else:
            # Interpolate nu_sgs to face locations
            nu_sgs_u = interpolate_to_x_face(nu_sgs)
            nu_sgs_v = interpolate_to_y_face(nu_sgs)
            nu_total_u = nu + nu_sgs_u
            nu_total_v = nu + nu_sgs_v

        rhs_u = -adv_u_face + nu_total_u * lap_u
        rhs_v = -adv_v_face + nu_total_v * lap_v

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
        u2 = (u + dt * k1u) / (1 + dt * chi_u / eta_safe)
        v2 = (v + dt * k1v) / (1 + dt * chi_v / eta_safe)
        
        k2u, k2v = compute_rhs_explicit_mac(u2, v2)
        u_star = u + dt * k2u
        v_star = v + dt * k2v
        u3 = (0.75 * u + 0.25 * u_star) / (1 + dt * chi_u / eta_safe)
        v3 = (0.75 * v + 0.25 * v_star) / (1 + dt * chi_v / eta_safe)
        
        k3u, k3v = compute_rhs_explicit_mac(u3, v3)
        u_star = u + dt * k3u
        v_star = v + dt * k3v
        u_new = ((1/3) * u + (2/3) * u_star) / (1 + dt * chi_u / eta_safe)
        v_new = ((1/3) * v + (2/3) * v_star) / (1 + dt * chi_v / eta_safe)
        # Hard mask zeroing: force velocities to zero inside solid (chi > 0.5)
        # Use interpolated chi for MAC grid (chi_u for u, chi_v for v)
        u_new = jnp.where(chi_u < 0.5, u_new, 0.0)
        v_new = jnp.where(chi_v < 0.5, v_new, 0.0)
        u_new, v_new = apply_boundary_conditions_mac(u_new, v_new)
        return u_new, v_new
    
    def rk2_step():
        k1u, k1v = compute_rhs_explicit_mac(u, v)
        u_star = (u + dt * k1u) / (1 + dt * chi_u / eta_safe)
        v_star = (v + dt * k1v) / (1 + dt * chi_v / eta_safe)
        
        k2u, k2v = compute_rhs_explicit_mac(u_star, v_star)
        u_new = (u + 0.5 * dt * (k1u + k2u)) / (1 + dt * chi_u / eta_safe)
        v_new = (v + 0.5 * dt * (k1v + k2v)) / (1 + dt * chi_v / eta_safe)
        # Hard mask zeroing: force velocities to zero inside solid (chi > 0.5)
        # Use interpolated chi for MAC grid (chi_u for u, chi_v for v)
        u_new = jnp.where(chi_u < 0.5, u_new, 0.0)
        v_new = jnp.where(chi_v < 0.5, v_new, 0.0)
        u_new, v_new = apply_boundary_conditions_mac(u_new, v_new)
        return u_new, v_new
    
    u_new, v_new = jax.lax.cond(fast_mode, rk2_step, rk3_step)
    return u_new, v_new
