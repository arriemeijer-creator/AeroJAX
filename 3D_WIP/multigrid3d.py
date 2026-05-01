"""
Simple 3D multigrid solver for Poisson equation.
"""

import jax
import jax.numpy as jnp


def poisson_multigrid_3d(rhs, dx, dy, dz, levels=3, v_cycles=5, tolerance=1e-6):
    """
    3D geometric multigrid solver for Poisson equation.
    
    Args:
        rhs: Right-hand side (nx, ny, nz)
        dx, dy, dz: Grid spacing
        levels: Number of multigrid levels
        v_cycles: Number of V-cycles
        tolerance: Convergence tolerance
        
    Returns:
        p: Solution field
    """
    nx, ny, nz = rhs.shape
    b = rhs
    
    def restrict_3d(fine):
        """Restriction (full weighting) in 3D"""
        nx_f, ny_f, nz_f = fine.shape
        nx_c, ny_c, nz_c = nx_f // 2, ny_f // 2, nz_f // 2
        
        if nx_c < 2 or ny_c < 2 or nz_c < 2:
            return fine[:1, :1, :1]
        
        fine_even = fine[:nx_c*2, :ny_c*2, :nz_c*2]
        coarse = 0.125 * (
            fine_even[::2, ::2, ::2] + fine_even[1::2, ::2, ::2] +
            fine_even[::2, 1::2, ::2] + fine_even[1::2, 1::2, ::2] +
            fine_even[::2, ::2, 1::2] + fine_even[1::2, ::2, 1::2] +
            fine_even[::2, 1::2, 1::2] + fine_even[1::2, 1::2, 1::2]
        )
        return coarse
    
    def prolong_3d(coarse):
        """Prolongation (trilinear interpolation) in 3D"""
        nx_c, ny_c, nz_c = coarse.shape
        nx_f, ny_f, nz_f = nx_c * 2, ny_c * 2, nz_c * 2
        
        fine = jnp.zeros((nx_f, ny_f, nz_f))
        fine = fine.at[::2, ::2, ::2].set(coarse)
        
        # Interpolate edges
        if nx_f > 2:
            fine = fine.at[1:-1:2, ::2, ::2].set(0.5 * (fine[0:-2:2, ::2, ::2] + fine[2::2, ::2, ::2]))
        if ny_f > 2:
            fine = fine.at[::2, 1:-1:2, ::2].set(0.5 * (fine[::2, 0:-2:2, ::2] + fine[::2, 2::2, ::2]))
        if nz_f > 2:
            fine = fine.at[::2, ::2, 1:-1:2].set(0.5 * (fine[::2, ::2, 0:-2:2] + fine[::2, ::2, 2::2]))
        
        # Interpolate faces
        if nx_f > 2 and ny_f > 2:
            fine = fine.at[1:-1:2, 1:-1:2, ::2].set(0.25 * (
                fine[0:-2:2, 0:-2:2, ::2] + fine[2::2, 0:-2:2, ::2] +
                fine[0:-2:2, 2::2, ::2] + fine[2::2, 2::2, ::2]
            ))
        if nx_f > 2 and nz_f > 2:
            fine = fine.at[1:-1:2, ::2, 1:-1:2].set(0.25 * (
                fine[0:-2:2, ::2, 0:-2:2] + fine[2::2, ::2, 0:-2:2] +
                fine[0:-2:2, ::2, 2::2] + fine[2::2, ::2, 2::2]
            ))
        if ny_f > 2 and nz_f > 2:
            fine = fine.at[::2, 1:-1:2, 1:-1:2].set(0.25 * (
                fine[::2, 0:-2:2, 0:-2:2] + fine[::2, 2::2, 0:-2:2] +
                fine[::2, 0:-2:2, 2::2] + fine[::2, 2::2, 2::2]
            ))
        
        # Interpolate interior
        if nx_f > 2 and ny_f > 2 and nz_f > 2:
            fine = fine.at[1:-1:2, 1:-1:2, 1:-1:2].set(0.125 * (
                fine[0:-2:2, 0:-2:2, 0:-2:2] + fine[2::2, 0:-2:2, 0:-2:2] +
                fine[0:-2:2, 2::2, 0:-2:2] + fine[2::2, 2::2, 0:-2:2] +
                fine[0:-2:2, 0:-2:2, 2::2] + fine[2::2, 0:-2:2, 2::2] +
                fine[0:-2:2, 2::2, 2::2] + fine[2::2, 2::2, 2::2]
            ))
        
        return fine
    
    def smooth_3d(p, b, nu=2):
        """Gauss-Seidel smoother in 3D"""
        def smooth_step(p_state, i):
            p = p_state
            p_padded = jnp.pad(p, ((1, 1), (1, 1), (1, 1)), mode='wrap')
            ax = 1.0 / (dx * dx)
            ay = 1.0 / (dy * dy)
            az = 1.0 / (dz * dz)
            p_new = (ax * (p_padded[2:, 1:-1, 1:-1] + p_padded[:-2, 1:-1, 1:-1]) +
                     ay * (p_padded[1:-1, 2:, 1:-1] + p_padded[1:-1, :-2, 1:-1]) +
                     az * (p_padded[1:-1, 1:-1, 2:] + p_padded[1:-1, 1:-1, :-2]) - b) / (2.0 * (ax + ay + az))
            return p_new, None
        
        p_final, _ = jax.lax.scan(smooth_step, p, jnp.arange(nu))
        return p_final
    
    def apply_laplacian_3d(p):
        """Apply 3D Laplacian"""
        p_padded = jnp.pad(p, ((1, 1), (1, 1), (1, 1)), mode='wrap')
        ax = 1.0 / (dx * dx)
        ay = 1.0 / (dy * dy)
        az = 1.0 / (dz * dz)
        laplacian = ax * (p_padded[2:, 1:-1, 1:-1] + p_padded[:-2, 1:-1, 1:-1] - 2 * p) + \
                    ay * (p_padded[1:-1, 2:, 1:-1] + p_padded[1:-1, :-2, 1:-1] - 2 * p) + \
                    az * (p_padded[1:-1, 1:-1, 2:] + p_padded[1:-1, 1:-1, :-2] - 2 * p)
        return laplacian
    
    def v_cycle_3d(p, b, level):
        """V-cycle in 3D"""
        if level >= levels:
            return smooth_3d(p, b, nu=10)
        
        # Pre-smooth
        p = smooth_3d(p, b, nu=2)
        
        # Restrict residual
        r = b - apply_laplacian_3d(p)
        r_coarse = restrict_3d(r)
        
        # Coarse grid correction
        e_coarse = v_cycle_3d(jnp.zeros_like(r_coarse), r_coarse, level + 1)
        e = prolong_3d(e_coarse)
        
        # Add correction
        p = p + e
        
        # Post-smooth
        p = smooth_3d(p, b, nu=2)
        
        return p
    
    p = jnp.zeros((nx, ny, nz))
    
    def v_cycle_step(carry, i):
        p, converged = carry
        p_new = jax.lax.cond(converged, lambda: p, lambda: v_cycle_3d(p, b, 0))
        residual = b - apply_laplacian_3d(p_new)
        residual_norm_sq = jnp.sum(residual**2) * dx * dy * dz
        converged_new = residual_norm_sq < tolerance**2
        return (p_new, converged_new), None
    
    p_final, _ = jax.lax.scan(v_cycle_step, (p, False), jnp.arange(v_cycles))[0]
    
    return p_final
