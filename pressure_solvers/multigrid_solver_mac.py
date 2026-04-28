"""
Multigrid pressure solver for MAC (staggered) grid.
Pressure is stored at cell centers (nx, ny) in MAC grid.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Literal


@jax.jit(static_argnames=('flow_type', 'max_iter'))
def poisson_jacobi_mac(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float, 
                       max_iter: int = 1000, tolerance: float = 1e-6, 
                       flow_type: Literal['von_karman', 'lid_driven_cavity', 'taylor_green'] = 'von_karman') -> jnp.ndarray:
    """
    Jacobi iterative solver for Poisson equation on MAC grid.
    Pressure is cell-centered (nx, ny), same as collocated case.
    """
    nx, ny = rhs.shape
    b = rhs
    
    # Zero out RHS at Dirichlet boundaries based on flow type
    if flow_type == 'von_karman':
        # Only outlet is Dirichlet (p=0), inlet is Neumann (∂p/∂x=0)
        b = b.at[-1, :].set(0.0)  # Outlet: p = 0
    elif flow_type == 'lid_driven_cavity':
        b = b - jnp.mean(b)
    
    p = jnp.zeros_like(b)
    
    def jacobi_step(p, i):
        p_new = jnp.zeros_like(p)
        dx2 = dx * dx
        dy2 = dy * dy
        denom = 2.0 * (dx2 + dy2)
        p_new = p_new.at[1:-1, 1:-1].set(
            (dy2 * (p[2:, 1:-1] + p[:-2, 1:-1]) + dx2 * (p[1:-1, 2:] + p[1:-1, :-2]) - b[1:-1, 1:-1] * dx2 * dy2) / denom
        )
        # Apply boundary conditions
        if flow_type == 'von_karman':
            p_new = p_new.at[0, :].set(0.0)
            p_new = p_new.at[-1, :].set(0.0)
        elif flow_type == 'lid_driven_cavity':
            p_new = p_new.at[0, :].set(p_new[1, :])
            p_new = p_new.at[-1, :].set(p_new[-2, :])
            p_new = p_new.at[:, 0].set(p_new[:, 1])
            p_new = p_new.at[:, -1].set(p_new[:, -2])
        return p_new, i + 1
    
    p, _ = jax.lax.scan(jacobi_step, p, jnp.arange(max_iter))
    
    if flow_type == 'lid_driven_cavity':
        center_x, center_y = nx // 2, ny // 2
        p = p - p[center_x, center_y]
    
    return p


@jax.jit(static_argnames=('flow_type', 'v_cycles'))
def poisson_multigrid_mac(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float, 
                          levels: int = 4, v_cycles: int = 5, tolerance: float = 1e-6, 
                          flow_type: Literal['von_karman', 'lid_driven_cavity', 'taylor_green'] = 'von_karman') -> jnp.ndarray:
    """Geometric Multigrid solver for MAC grid.

    Args:
        rhs: Right-hand side (divergence/dt)
        mask: Obstacle mask (1 = fluid, 0 = solid)
        dx, dy: Grid spacing
        levels: Number of multigrid levels
        v_cycles: Number of V-cycles to perform
        tolerance: Convergence tolerance
        flow_type: Flow type for boundary conditions
    """
    nx, ny = rhs.shape
    b = rhs

    # Apply immersed boundary mask to RHS at finest level
    b = b * mask

    # Zero out RHS at Dirichlet boundaries based on flow type
    if flow_type == 'von_karman':
        # Only outlet is Dirichlet (p=0), inlet is Neumann (∂p/∂x=0)
        b = b.at[-1, :].set(0.0)  # Outlet: p = 0
    elif flow_type == 'lid_driven_cavity':
        # For LDC with pure Neumann BCs, enforce compatibility condition
        b = b - jnp.mean(b)
    # For Taylor-Green, periodic BCs (no Dirichlet boundaries)

    # Early exit: check if already converged
    def initial_laplacian(p):
        pad_mode = 'edge' if flow_type in ['von_karman', 'lid_driven_cavity'] else 'wrap'
        p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode=pad_mode)
        ax = 1.0 / (dx * dx)
        ay = 1.0 / (dy * dy)
        laplacian = ax * (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1] - 2 * p) + \
                     ay * (p_padded[1:-1, 2:] + p_padded[1:-1, :-2] - 2 * p)
        if flow_type == 'von_karman':
            laplacian = laplacian.at[-1, :].set(0.0)  # Only outlet: p = 0
        return laplacian
    
    initial_residual = b - initial_laplacian(jnp.zeros_like(b))
    initial_norm_sq = jnp.sum(initial_residual**2) * dx * dy
    
    # Check if grid is suitable for multigrid
    max_levels = 1
    temp_nx, temp_ny = nx, ny
    while temp_nx % 2 == 0 and temp_ny % 2 == 0 and max_levels < levels:
        temp_nx //= 2
        temp_ny //= 2
        max_levels += 1
    
    if max_levels < 2:
        raise ValueError(f"Grid dimensions ({nx}, {ny}) not suitable for multigrid with {levels} levels")
    
    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        """Restriction (full weighting)"""
        nx_fine, ny_fine = fine.shape
        nx_coarse = nx_fine // 2
        ny_coarse = ny_fine // 2
        
        if nx_coarse < 2 or ny_coarse < 2:
            return fine[:1, :1]
        
        coarse = jnp.zeros((nx_coarse, ny_coarse))
        fine_even = fine[:nx_coarse*2, :ny_coarse*2]
        coarse = 0.25 * (
            fine_even[::2, ::2] + 
            fine_even[1::2, ::2] + 
            fine_even[::2, 1::2] + 
            fine_even[1::2, 1::2]
        )
        return coarse
    
    def prolong(coarse: jnp.ndarray) -> jnp.ndarray:
        """Prolongation (bilinear interpolation) - fully vectorized"""
        nx_coarse, ny_coarse = coarse.shape
        nx_fine = nx_coarse * 2
        ny_fine = ny_coarse * 2
        
        # Create fine grid with zeros
        fine = jnp.zeros((nx_fine, ny_fine))
        
        # Set coarse points on even indices
        fine = fine.at[::2, ::2].set(coarse)
        
        # Vectorized bilinear interpolation for all interior points
        # x-interpolation (horizontal)
        fine = jax.lax.cond(nx_fine > 2,
                           lambda f: f.at[1:-1:2, ::2].set(0.5 * (f[0:-2:2, ::2] + f[2::2, ::2])),
                           lambda f: f,
                           fine)
        # y-interpolation (vertical)
        fine = jax.lax.cond(ny_fine > 2,
                           lambda f: f.at[::2, 1:-1:2].set(0.5 * (f[::2, 0:-2:2] + f[::2, 2::2])),
                           lambda f: f,
                           fine)
        # xy-interpolation (diagonal)
        fine = jax.lax.cond(nx_fine > 2 and ny_fine > 2,
                           lambda f: f.at[1:-1:2, 1:-1:2].set(0.25 * (
                               f[0:-2:2, 0:-2:2] + f[2::2, 0:-2:2] +
                               f[0:-2:2, 2::2] + f[2::2, 2::2]
                           )),
                           lambda f: f,
                           fine)
        
        return fine
    
    def smooth(p: jnp.ndarray, b: jnp.ndarray, mask_level: jnp.ndarray, nu: int = 2) -> jnp.ndarray:
        """Gauss-Seidel smoother with mask support for immersed boundaries"""
        def smooth_step(p_state, i):
            p = p_state
            p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode='edge')
            ax = 1.0 / (dx * dx)
            ay = 1.0 / (dy * dy)
            p_new = (ax * (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1]) +
                     ay * (p_padded[1:-1, 2:] + p_padded[1:-1, :-2]) - b) / (2.0 * (ax + ay))
            # Apply mask: zero out pressure in solid cells
            p_new = p_new * mask_level
            return p_new, None

        p_final, _ = jax.lax.scan(smooth_step, p, jnp.arange(nu))
        return p_final

    def apply_laplacian(p: jnp.ndarray) -> jnp.ndarray:
        """Apply discrete Laplacian with flow-specific boundary conditions"""
        # Choose padding mode based on flow type
        pad_mode = 'edge' if flow_type in ['von_karman', 'lid_driven_cavity'] else 'wrap'
        p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode=pad_mode)
        ax = 1.0 / (dx * dx)
        ay = 1.0 / (dy * dy)
        laplacian = ax * (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1] - 2 * p) + \
                     ay * (p_padded[1:-1, 2:] + p_padded[1:-1, :-2] - 2 * p)

        if flow_type == 'von_karman':
            laplacian = laplacian.at[-1, :].set(0.0)  # Only outlet: p = 0
        elif flow_type == 'lid_driven_cavity':
            pass

        return laplacian
    
    def v_cycle(p: jnp.ndarray, b: jnp.ndarray, mask_level: jnp.ndarray, level: int) -> jnp.ndarray:
        if level >= max_levels:
            return smooth(p, b, mask_level, nu=10)
        
        p = smooth(p, b, mask_level, nu=2)
        
        # Apply boundary conditions
        if flow_type == 'von_karman':
            p = p.at[-1, :].set(0.0)  # Outlet: p = 0
        
        r = b - apply_laplacian(p)
        r_coarse = restrict(r)
        mask_coarse = restrict(mask_level)
        mask_coarse = (mask_coarse > 0.5).astype(float)
        e_coarse = v_cycle(jnp.zeros_like(r_coarse), r_coarse, mask_coarse, level+1)
        e = prolong(e_coarse)
        p = p + e
        p = smooth(p, b, mask_level, nu=2)
        
        # Apply boundary conditions after post-smoothing
        if flow_type == 'von_karman':
            p = p.at[-1, :].set(0.0)  # Outlet: p = 0
        
        return p
    
    p = jnp.zeros((nx, ny))
    
    def v_cycle_step(carry, i):
        p, converged = carry
        p_new = jax.lax.cond(converged,
                            lambda: p,
                            lambda: v_cycle(p, b, mask, 0))
        # Compute residual norm squared (avoid sqrt for efficiency)
        residual = b - apply_laplacian(p_new)
        residual_norm_sq = jnp.sum(residual**2) * dx * dy
        converged_new = residual_norm_sq < tolerance**2
        return (p_new, converged_new), None
    
    p_final, converged_final = jax.lax.scan(v_cycle_step, (p, False), jnp.arange(v_cycles))[0]

    # For LDC with pure Neumann BCs, pin pressure at center to remove null space
    if flow_type == 'lid_driven_cavity':
        center_x, center_y = nx // 2, ny // 2
        p_final = p_final - p_final[center_x, center_y]

    return p_final
