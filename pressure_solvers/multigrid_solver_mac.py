"""
Multigrid pressure solver for MAC (staggered) grid.
Pressure is stored at cell centers (nx, ny) in MAC grid.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit(static_argnames=('flow_type', 'max_iter'))
def poisson_jacobi_mac(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float, 
                       max_iter: int = 1000, tolerance: float = 1e-6, 
                       flow_type: str = 'von_karman') -> jnp.ndarray:
    """
    Jacobi iterative solver for Poisson equation on MAC grid.
    Pressure is cell-centered (nx, ny), same as collocated case.
    """
    nx, ny = rhs.shape
    b = rhs
    
    # Zero out RHS at Dirichlet boundaries based on flow type
    if flow_type == 'von_karman':
        b = b.at[0, :].set(0.0)
        b = b.at[-1, :].set(0.0)
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
                          flow_type: str = 'von_karman') -> jnp.ndarray:
    """
    Geometric Multigrid solver for MAC grid.
    Pressure is cell-centered, so discretization is same as collocated case.
    The difference is in how the RHS is computed (from staggered velocities).
    """
    nx, ny = rhs.shape
    b = rhs

    # Zero out RHS at Dirichlet boundaries
    if flow_type == 'von_karman':
        b = b.at[0, :].set(0.0)
        b = b.at[-1, :].set(0.0)
    
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
        """Prolongation (bilinear interpolation)"""
        nx_coarse, ny_coarse = coarse.shape
        nx_fine = nx_coarse * 2
        ny_fine = ny_coarse * 2
        
        fine = jnp.zeros((nx_fine, ny_fine))
        fine = fine.at[::2, ::2].set(coarse)
        
        if nx_fine > 2:
            fine = fine.at[1:-1:2, ::2].set(0.5 * (fine[0:-2:2, ::2] + fine[2::2, ::2]))
        if ny_fine > 2:
            fine = fine.at[::2, 1:-1:2].set(0.5 * (fine[::2, 0:-2:2] + fine[::2, 2::2]))
        if nx_fine > 2 and ny_fine > 2:
            fine = fine.at[1:-1:2, 1:-1:2].set(0.25 * (
                fine[0:-2:2, 0:-2:2] + fine[2::2, 0:-2:2] +
                fine[0:-2:2, 2::2] + fine[2::2, 2::2]
            ))
        
        return fine
    
    def smooth(p: jnp.ndarray, b: jnp.ndarray, nu: int = 2) -> jnp.ndarray:
        """Gauss-Seidel smoother with non-periodic boundary conditions"""
        def smooth_step(p_state, i):
            p = p_state
            p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode='edge')
            ax = 1.0 / (dx * dx)
            ay = 1.0 / (dy * dy)
            p_new = (ax * (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1]) +
                     ay * (p_padded[1:-1, 2:] + p_padded[1:-1, :-2]) - b) / (2.0 * (ax + ay))
            return p_new, None

        p_final, _ = jax.lax.scan(smooth_step, p, jnp.arange(nu))
        return p_final

    def apply_laplacian(p: jnp.ndarray) -> jnp.ndarray:
        """Apply discrete Laplacian with flow-specific boundary conditions"""
        p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode='edge')
        ax = 1.0 / (dx * dx)
        ay = 1.0 / (dy * dy)
        laplacian = ax * (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1] - 2 * p) + \
                     ay * (p_padded[1:-1, 2:] + p_padded[1:-1, :-2] - 2 * p)

        if flow_type == 'von_karman':
            laplacian = laplacian.at[0, :].set(0.0)
            laplacian = laplacian.at[-1, :].set(0.0)
        elif flow_type == 'lid_driven_cavity':
            pass
        
        return laplacian
    
    def v_cycle(p: jnp.ndarray, b: jnp.ndarray, level: int) -> jnp.ndarray:
        if level >= max_levels:
            return smooth(p, b, nu=10)
        
        p = smooth(p, b, nu=2)
        r = b - apply_laplacian(p)
        r_coarse = restrict(r)
        e_coarse = v_cycle(jnp.zeros_like(r_coarse), r_coarse, level+1)
        e = prolong(e_coarse)
        p = p + e
        p = smooth(p, b, nu=2)
        return p
    
    p = jnp.zeros((nx, ny))
    
    def v_cycle_step(carry, i):
        p, converged = carry
        p_new = jax.lax.cond(converged,
                            lambda: p,
                            lambda: v_cycle(p, b, 0))
        residual = b - apply_laplacian(p_new)
        residual_norm = jnp.sqrt(jnp.sum(residual**2) * dx * dy)
        converged_new = residual_norm < tolerance
        return (p_new, converged_new), None
    
    p_final, converged_final = jax.lax.scan(v_cycle_step, (p, False), jnp.arange(v_cycles))[0]

    return p_final
