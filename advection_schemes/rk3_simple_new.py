import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit(static_argnames=('nu_hyper_ratio', 'slip_walls', 'flow_type'))
def rk3_step_simple_new(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float,
                        dx: float, dy: float, mask: jnp.ndarray, U_inf: float = 1.0,
                        nu_sgs: jnp.ndarray = None, nu_hyper_ratio: float = 0.0,
                        slip_walls: bool = True, flow_type: str = 'von_karman') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Improved RK3 scheme with proper handling of masks and biharmonic operator
    """
    
    nx, ny = u.shape
    
    def grad_x(f):
        f_padded = jnp.pad(f, ((1, 1), (0, 0)), mode='edge')
        return (f_padded[2:, :] - f_padded[:-2, :]) / (2.0 * dx)

    def grad_y(f):
        f_padded = jnp.pad(f, ((0, 0), (1, 1)), mode='edge')
        return (f_padded[:, 2:] - f_padded[:, :-2]) / (2.0 * dy)

    def laplacian(f):
        f_padded = jnp.pad(f, ((1, 1), (1, 1)), mode='edge')
        lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f) / dx**2
        lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f) / dy**2
        return lap_x + lap_y

    def create_sponge_layer(nx, ny, sponge_width_ratio=0.15):
        """Create tapered sponge layer with smoothstep function"""
        sponge_width_x = int(nx * sponge_width_ratio)
        sponge_width_y = int(ny * sponge_width_ratio)
        
        def taper(x, width):
            t = jnp.clip(x / width, 0, 1)
            return 3*t**2 - 2*t**3
        
        sponge_x = jnp.ones(nx)
        if sponge_width_x > 0:
            sponge_x = sponge_x.at[:sponge_width_x].set(
                taper(jnp.arange(sponge_width_x)[::-1], sponge_width_x)
            )
            sponge_x = sponge_x.at[-sponge_width_x:].set(
                taper(jnp.arange(sponge_width_x), sponge_width_x)
            )
        
        sponge_y = jnp.ones(ny)
        if sponge_width_y > 0:
            sponge_y = sponge_y.at[:sponge_width_y].set(
                taper(jnp.arange(sponge_width_y)[::-1], sponge_width_y)
            )
            sponge_y = sponge_y.at[-sponge_width_y:].set(
                taper(jnp.arange(sponge_width_y), sponge_width_y)
            )
        
        return jnp.maximum(sponge_x[:, None], sponge_y[None, :])

    def apply_boundary_conditions(u_field, v_field):
        if flow_type == 'lid_driven_cavity':
            # LDC: Lid at top moves with U_inf, all other walls are no-slip
            u_field = u_field.at[:, -1].set(U_inf)  # Top lid
            u_field = u_field.at[:, 0].set(0.0)  # Bottom wall
            u_field = u_field.at[0, :].set(0.0)  # Left wall
            u_field = u_field.at[-1, :].set(0.0)  # Right wall
            v_field = v_field.at[:, 0].set(0.0)  # Bottom wall
            v_field = v_field.at[:, -1].set(0.0)  # Top wall
            v_field = v_field.at[0, :].set(0.0)  # Left wall
            v_field = v_field.at[-1, :].set(0.0)  # Right wall
        else:
            # von_karman: Inlet at left, outlet at right
            u_field = u_field.at[0, :].set(U_inf)
            v_field = v_field.at[0, :].set(0.0)
            u_field = u_field.at[-1, :].set(u_field[-2, :])
            v_field = v_field.at[-1, :].set(v_field[-2, :])
            # Wall boundary conditions (slip vs no-slip)
            if slip_walls:
                # Slip walls: v=0, u extrapolated from interior
                u_field = u_field.at[:, 0].set(u_field[:, 1])  # Bottom: slip
                u_field = u_field.at[:, -1].set(u_field[:, -2])  # Top: slip
                v_field = v_field.at[:, 0].set(0.0)  # Bottom: v=0
                v_field = v_field.at[:, -1].set(0.0)  # Top: v=0
            else:
                # No-slip walls: u=0, v=0
                u_field = u_field.at[:, 0].set(0.0)
                u_field = u_field.at[:, -1].set(0.0)
                v_field = v_field.at[:, 0].set(0.0)
                v_field = v_field.at[:, -1].set(0.0)
        return u_field, v_field

    # Precompute sponge layer once (avoids recomputation in each RK stage)
    sponge_field = create_sponge_layer(nx, ny, sponge_width_ratio=0.0)  # Disabled
    
    def compute_rhs(u_in, v_in):
        # Use upwind scheme to eliminate Gibbs oscillations
        def upwind_advection(field, vel_u, vel_v):
            # X-direction upwind with non-periodic boundaries
            f_padded = jnp.pad(field, ((1, 1), (0, 0)), mode='edge')
            f_x = jnp.where(vel_u > 0,
                            vel_u * (field - f_padded[:-2, :]) / dx,
                            vel_u * (f_padded[2:, :] - field) / dx)
            # Y-direction upwind with non-periodic boundaries
            f_padded_y = jnp.pad(field, ((0, 0), (1, 1)), mode='edge')
            f_y = jnp.where(vel_v > 0,
                            vel_v * (field - f_padded_y[:, :-2]) / dy,
                            vel_v * (f_padded_y[:, 2:] - field) / dy)
            return f_x + f_y

        # Apply mask to velocity for advection
        u_masked = u_in * mask
        v_masked = v_in * mask

        # Use upwind advection
        advection_u = upwind_advection(u_masked, u_masked, v_masked)
        advection_v = upwind_advection(v_masked, u_masked, v_masked)

        # Viscosity (laplacian on masked velocity - this is OK)
        lap_u = laplacian(u_masked)
        lap_v = laplacian(v_masked)

        # Total viscosity
        nu_total = nu if nu_sgs is None else nu + nu_sgs

        # Base RHS
        rhs_u = -advection_u + nu_total * lap_u
        rhs_v = -advection_v + nu_total * lap_v

        # Add hyperviscosity if needed
        if nu_hyper_ratio > 0:  # Python conditional - fine since nu_hyper_ratio is static
            # Scale hyperviscosity appropriately for grid spacing
            # Note: Biharmonic scales as 1/dx^4, so we multiply by dx*dy for stability
            nu_hyper = nu * nu_hyper_ratio * (dx * dy)

            # Compute biharmonic on masked Laplacian
            lap_u_masked = lap_u * mask
            lap_v_masked = lap_v * mask
            biharmonic_u = laplacian(lap_u_masked)
            biharmonic_v = laplacian(lap_v_masked)

            # Apply spatially-varying hyperviscosity
            rhs_u = rhs_u - nu_hyper * sponge_field * biharmonic_u
            rhs_v = rhs_v - nu_hyper * sponge_field * biharmonic_v

        # Apply mask to RHS
        return rhs_u * mask, rhs_v * mask

    # RK3 stages
    k1u, k1v = compute_rhs(u, v)
    u2 = u + dt * k1u
    v2 = v + dt * k1v

    k2u, k2v = compute_rhs(u2, v2)
    u3 = 0.75 * u + 0.25 * (u2 + dt * k2u)
    v3 = 0.75 * v + 0.25 * (v2 + dt * k2v)

    k3u, k3v = compute_rhs(u3, v3)
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u3 + dt * k3u)
    v_new = (1.0/3.0) * v + (2.0/3.0) * (v3 + dt * k3v)

    # Final mask and BCs
    u_new = u_new * mask
    v_new = v_new * mask
    u_new, v_new = apply_boundary_conditions(u_new, v_new)

    return u_new, v_new


@jax.jit(static_argnames=('nu_hyper_ratio', 'slip_walls', 'fast_mode', 'flow_type'))
def rk_step_unified(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float,
                    dx: float, dy: float, mask: jnp.ndarray, sdf: jnp.ndarray = None, U_inf: float = 1.0,
                    nu_sgs: jnp.ndarray = None, nu_hyper_ratio: float = 0.0,
                    slip_walls: bool = True, fast_mode: bool = False, brinkman_eta: float = 0.01,
                    flow_type: str = 'von_karman') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unified step function that switches between RK3 (Scientific) and RK2 (Fast) using jax.lax.cond.
    Uses implicit Brinkman penalization to avoid upstream artifacts.

    Args:
        fast_mode: If True, uses RK2 (Heun's method) for speed. If False, uses RK3 for accuracy.
        brinkman_eta: Brinkman damping parameter for implicit penalization
        sdf: Signed distance function for Brinkman penalization (if provided, uses SDF-based penalization)
        flow_type: Flow type for boundary conditions ('von_karman' or 'lid_driven_cavity')
    """

    nx, ny = u.shape
    # Use SDF-based penalization if SDF is provided, otherwise fall back to mask-based
    if sdf is not None:
        # Single sigmoid from SDF - eliminates double sigmoid issue
        # Note: SDF is negative inside solid, positive outside
        # We want chi=1 inside solid, 0 outside, so use -sdf
        chi = jax.nn.sigmoid(-sdf / (0.1 * dx))  # Use same epsilon as mask creation
    else:
        chi = 1.0 - mask   # 1 inside solid, 0 outside (fallback)
    eta = brinkman_eta
    
    def grad_x(f):
        f_padded = jnp.pad(f, ((1, 1), (0, 0)), mode='edge')
        return (f_padded[2:, :] - f_padded[:-2, :]) / (2.0 * dx)

    def grad_y(f):
        f_padded = jnp.pad(f, ((0, 0), (1, 1)), mode='edge')
        return (f_padded[:, 2:] - f_padded[:, :-2]) / (2.0 * dy)

    def laplacian(f):
        f_padded = jnp.pad(f, ((1, 1), (1, 1)), mode='edge')
        lap_x = (f_padded[2:, 1:-1] + f_padded[:-2, 1:-1] - 2*f) / dx**2
        lap_y = (f_padded[1:-1, 2:] + f_padded[1:-1, :-2] - 2*f) / dy**2
        return lap_x + lap_y

    def create_sponge_layer(nx, ny, sponge_width_ratio=0.15):
        """Create tapered sponge layer with smoothstep function"""
        sponge_width_x = int(nx * sponge_width_ratio)
        sponge_width_y = int(ny * sponge_width_ratio)
        
        def taper(x, width):
            t = jnp.clip(x / width, 0, 1)
            return 3*t**2 - 2*t**3
        
        sponge_x = jnp.ones(nx)
        if sponge_width_x > 0:
            sponge_x = sponge_x.at[:sponge_width_x].set(
                taper(jnp.arange(sponge_width_x)[::-1], sponge_width_x)
            )
            sponge_x = sponge_x.at[-sponge_width_x:].set(
                taper(jnp.arange(sponge_width_x), sponge_width_x)
            )
        
        sponge_y = jnp.ones(ny)
        if sponge_width_y > 0:
            sponge_y = sponge_y.at[:sponge_width_y].set(
                taper(jnp.arange(sponge_width_y)[::-1], sponge_width_y)
            )
            sponge_y = sponge_y.at[-sponge_width_y:].set(
                taper(jnp.arange(sponge_width_y), sponge_width_y)
            )
        
        return jnp.maximum(sponge_x[:, None], sponge_y[None, :])

    def apply_boundary_conditions(u_field, v_field):
        if flow_type == 'lid_driven_cavity':
            # LDC: Lid at top moves with U_inf, all other walls are no-slip
            u_field = u_field.at[:, -1].set(U_inf)  # Top lid
            u_field = u_field.at[:, 0].set(0.0)  # Bottom wall
            u_field = u_field.at[0, :].set(0.0)  # Left wall
            u_field = u_field.at[-1, :].set(0.0)  # Right wall
            v_field = v_field.at[:, 0].set(0.0)  # Bottom wall
            v_field = v_field.at[:, -1].set(0.0)  # Top wall
            v_field = v_field.at[0, :].set(0.0)  # Left wall
            v_field = v_field.at[-1, :].set(0.0)  # Right wall
        else:
            # von_karman: Inlet at left, outlet at right
            u_field = u_field.at[0, :].set(U_inf)
            v_field = v_field.at[0, :].set(0.0)
            u_field = u_field.at[-1, :].set(u_field[-2, :])
            v_field = v_field.at[-1, :].set(v_field[-2, :])
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

    sponge_field = create_sponge_layer(nx, ny, sponge_width_ratio=0.0)  # Disabled

    def compute_rhs_explicit(u_in, v_in):
        # Advection and diffusion only (no penalization in explicit part)
        # Use upwind scheme to eliminate Gibbs oscillations
        def upwind_advection(field, vel_u, vel_v):
            # X-direction upwind with non-periodic boundaries
            f_padded = jnp.pad(field, ((1, 1), (0, 0)), mode='edge')
            f_x = jnp.where(vel_u > 0,
                            vel_u * (field - f_padded[:-2, :]) / dx,
                            vel_u * (f_padded[2:, :] - field) / dx)
            # Y-direction upwind with non-periodic boundaries
            f_padded_y = jnp.pad(field, ((0, 0), (1, 1)), mode='edge')
            f_y = jnp.where(vel_v > 0,
                            vel_v * (field - f_padded_y[:, :-2]) / dy,
                            vel_v * (f_padded_y[:, 2:] - field) / dy)
            return f_x + f_y

        advection_u = upwind_advection(u_in, u_in, v_in)
        advection_v = upwind_advection(v_in, u_in, v_in)
        lap_u = laplacian(u_in)
        lap_v = laplacian(v_in)
        nu_total = nu if nu_sgs is None else nu + nu_sgs
        rhs_u = -advection_u + nu_total * lap_u
        rhs_v = -advection_v + nu_total * lap_v
        if nu_hyper_ratio > 0:
            nu_hyper = nu * nu_hyper_ratio * (dx * dy)
            biharmonic_u = laplacian(lap_u)
            biharmonic_v = laplacian(lap_v)
            rhs_u = rhs_u - nu_hyper * sponge_field * biharmonic_u
            rhs_v = rhs_v - nu_hyper * sponge_field * biharmonic_v
        return rhs_u, rhs_v

    def rk3_step():
        # Stage 1
        k1u, k1v = compute_rhs_explicit(u, v)
        u2 = (u + dt * k1u) / (1 + dt * chi / eta)
        v2 = (v + dt * k1v) / (1 + dt * chi / eta)
        # Stage 2
        k2u, k2v = compute_rhs_explicit(u2, v2)
        u_star = u + dt * k2u
        v_star = v + dt * k2v
        u3 = (0.75 * u + 0.25 * u_star) / (1 + dt * chi / eta)
        v3 = (0.75 * v + 0.25 * v_star) / (1 + dt * chi / eta)
        # Stage 3
        k3u, k3v = compute_rhs_explicit(u3, v3)
        u_star = u + dt * k3u
        v_star = v + dt * k3v
        u_new = ((1/3) * u + (2/3) * u_star) / (1 + dt * chi / eta)
        v_new = ((1/3) * v + (2/3) * v_star) / (1 + dt * chi / eta)
        # Hard mask zeroing: force velocities to zero inside solid (mask < 0.5)
        # This is needed because mask has smooth sigmoid transition, so multiplication doesn't fully zero
        u_new = jnp.where(mask > 0.5, u_new, 0.0)
        v_new = jnp.where(mask > 0.5, v_new, 0.0)
        u_new, v_new = apply_boundary_conditions(u_new, v_new)
        return u_new, v_new

    def rk2_step():
        # Stage 1
        k1u, k1v = compute_rhs_explicit(u, v)
        u_star = (u + dt * k1u) / (1 + dt * chi / eta)
        v_star = (v + dt * k1v) / (1 + dt * chi / eta)
        # Stage 2
        k2u, k2v = compute_rhs_explicit(u_star, v_star)
        u_new = (u + 0.5 * dt * (k1u + k2u)) / (1 + dt * chi / eta)
        v_new = (v + 0.5 * dt * (k1v + k2v)) / (1 + dt * chi / eta)
        # Hard mask zeroing: force velocities to zero inside solid (mask < 0.5)
        # This is needed because mask has smooth sigmoid transition, so multiplication doesn't fully zero
        u_new = jnp.where(mask > 0.5, u_new, 0.0)
        v_new = jnp.where(mask > 0.5, v_new, 0.0)
        u_new, v_new = apply_boundary_conditions(u_new, v_new)
        return u_new, v_new

    # Use jax.lax.cond to switch between RK3 and RK2 without recompilation
    u_new, v_new = jax.lax.cond(fast_mode, rk2_step, rk3_step)

    return u_new, v_new
