import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit(static_argnames=('flow_type',))
def poisson_multigrid(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float, levels: int = 4, v_cycles: int = 2, flow_type: str = 'von_karman') -> jnp.ndarray:
    """Geometric Multigrid solver with non-periodic boundary conditions"""
    nx, ny = rhs.shape
    b = rhs  # rhs should already be divergence/dt computed by caller

    # Zero out RHS at Dirichlet boundaries based on flow type
    if flow_type == 'von_karman':
        # Inlet/outlet are Dirichlet BCs for von Karman
        b = b.at[0, :].set(0.0)
        b = b.at[-1, :].set(0.0)
    # For LDC, no Dirichlet boundaries (closed cavity)
    
    # Check if grid is suitable for multigrid (must be divisible by 2^levels)
    max_levels = 1
    temp_nx, temp_ny = nx, ny
    while temp_nx % 2 == 0 and temp_ny % 2 == 0 and max_levels < levels:
        temp_nx //= 2
        temp_ny //= 2
        max_levels += 1
    
    if max_levels < 2:
        # Grid not suitable for multigrid - raise error instead of fallback
        raise ValueError(f"Grid dimensions ({nx}, {ny}) not suitable for multigrid with {levels} levels. Grid must be divisible by 2^{levels}")
    
    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        """Restriction (full weighting) with safe indexing"""
        nx_fine, ny_fine = fine.shape
        nx_coarse = nx_fine // 2
        ny_coarse = ny_fine // 2
        
        # Ensure we don't go out of bounds
        if nx_coarse < 2 or ny_coarse < 2:
            return fine[:1, :1]  # Return minimal array
        
        # Safe restriction with proper bounds
        coarse = jnp.zeros((nx_coarse, ny_coarse))
        
        # Use only even indices from fine grid
        fine_even = fine[:nx_coarse*2, :ny_coarse*2]
        
        # Simple averaging for restriction
        coarse = 0.25 * (
            fine_even[::2, ::2] + 
            fine_even[1::2, ::2] + 
            fine_even[::2, 1::2] + 
            fine_even[1::2, 1::2]
        )
        
        return coarse
    
    def prolong(coarse: jnp.ndarray) -> jnp.ndarray:
        """Prolongation (bilinear interpolation) with safe indexing"""
        nx_coarse, ny_coarse = coarse.shape
        nx_fine = nx_coarse * 2
        ny_fine = ny_coarse * 2
        
        fine = jnp.zeros((nx_fine, ny_fine))
        
        # Coarse grid points
        fine = fine.at[::2, ::2].set(coarse)
        
        # Interpolate edges - safe indexing
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
            # Use ghost cells for non-periodic derivatives
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
        # Use ghost cells for non-periodic derivatives
        p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode='edge')
        ax = 1.0 / (dx * dx)
        ay = 1.0 / (dy * dy)
        laplacian = ax * (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1] - 2 * p) + \
                     ay * (p_padded[1:-1, 2:] + p_padded[1:-1, :-2] - 2 * p)

        # Apply boundary conditions based on flow type
        if flow_type == 'von_karman':
            # Inlet (left): Dirichlet BC (p = 0)
            laplacian = laplacian.at[0, :].set(0.0)
            # Outlet (right): Dirichlet BC (p = 0)
            laplacian = laplacian.at[-1, :].set(0.0)
            # Top/Bottom walls: Neumann BC (∂p/∂y = 0) - handled by edge padding
        elif flow_type == 'lid_driven_cavity':
            # All boundaries: Neumann BC (∂p/∂n = 0) for closed cavity
            # Edge padding already handles Neumann BCs
            pass
        # For Taylor-Green, use periodic (handled by caller)

        return laplacian
    
    # V-cycle implementation with safe levels
    def v_cycle(p: jnp.ndarray, b: jnp.ndarray, level: int) -> jnp.ndarray:
        if level >= max_levels:
            return smooth(p, b, nu=10)  # Direct solve on coarsest grid
        
        # Pre-smooth
        p = smooth(p, b, nu=2)
        
        # Restrict residual
        r = b - apply_laplacian(p)
        r_coarse = restrict(r)
        
        # Coarse grid correction
        e_coarse = v_cycle(jnp.zeros_like(r_coarse), r_coarse, level+1)
        e = prolong(e_coarse)
        
        # Add correction
        p = p + e
        
        # Post-smooth
        p = smooth(p, b, nu=2)
        
        return p
    
    p = jnp.zeros((nx, ny))
    
    def v_cycle_step(p_state, i):
        p = p_state
        return v_cycle(p, b, 0), None
    
    p_final, _ = jax.lax.scan(v_cycle_step, p, jnp.arange(v_cycles))
    return p_final

def simple_gauss_seidel(b: jnp.ndarray, dx: float, dy: float, max_iter: int = 50) -> jnp.ndarray:
    """Simple Gauss-Seidel fallback with non-periodic boundary conditions"""
    nx, ny = b.shape
    p = jnp.zeros((nx, ny))

    def smooth_step(p_state, i):
        p = p_state
        # Use ghost cells for non-periodic derivatives
        p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode='edge')
        p_new = (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1] +
                 p_padded[1:-1, 2:] + p_padded[1:-1, :-2] - dx**2 * b) / 4.0
        return p_new, None

    p_final, _ = jax.lax.scan(smooth_step, p, jnp.arange(max_iter))
    return p_final
