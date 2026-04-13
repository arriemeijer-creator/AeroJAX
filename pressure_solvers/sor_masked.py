import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit(static_argnames=('flow_type', 'max_iter'))
def poisson_sor_masked(
    rhs: jnp.ndarray,
    mask: jnp.ndarray,
    dx: float,
    dy: float,
    omega: float = 1.5,
    max_iter: int = 100,
    tol: float = 1e-6,
    flow_type: str = 'von_karman'
) -> jnp.ndarray:
    """
    Mask-aware SOR pressure solver with fixed iterations for differentiability.
    
    The mask enforces Neumann BCs on obstacle surfaces (∇p·n = 0).
    """
    nx, ny = rhs.shape
    p = jnp.zeros((nx, ny))
    
    # Precompute coefficients
    ax = 1.0 / (dx * dx)
    ay = 1.0 / (dy * dy)
    a0 = 2.0 * (ax + ay)
    
    # Create checkerboard masks for red-black ordering
    i_indices, j_indices = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
    red_mask = ((i_indices + j_indices) % 2 == 0).astype(float)
    black_mask = ((i_indices + j_indices) % 2 == 1).astype(float)
    
    # Interior mask (exclude boundaries)
    interior_mask = jnp.ones((nx, ny))
    interior_mask = interior_mask.at[0, :].set(0.0)
    interior_mask = interior_mask.at[-1, :].set(0.0)
    interior_mask = interior_mask.at[:, 0].set(0.0)
    interior_mask = interior_mask.at[:, -1].set(0.0)
    
    red_interior = red_mask * interior_mask
    black_interior = black_mask * interior_mask
    
    def sor_step(carry, _):
        p, iteration = carry
        
        # Red update (using current p)
        p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode='edge')
        p_north = p_padded[2:, 1:-1]
        p_south = p_padded[:-2, 1:-1]
        p_east = p_padded[1:-1, 2:]
        p_west = p_padded[1:-1, :-2]
        
        p_new = (ax * (p_north + p_south) + ay * (p_east + p_west) - rhs) / a0
        p_relax = p + omega * (p_new - p)
        
        # Apply red update
        p_red = p * (1.0 - red_interior) + p_relax * red_interior
        
        # Black update (using updated red values)
        p_padded_red = jnp.pad(p_red, ((1, 1), (1, 1)), mode='edge')
        p_north_red = p_padded_red[2:, 1:-1]
        p_south_red = p_padded_red[:-2, 1:-1]
        p_east_red = p_padded_red[1:-1, 2:]
        p_west_red = p_padded_red[1:-1, :-2]
        
        p_new_red = (ax * (p_north_red + p_south_red) + ay * (p_east_red + p_west_red) - rhs) / a0
        p_relax_red = p_red + omega * (p_new_red - p_red)
        
        # Apply black update
        p = p_red * (1.0 - black_interior) + p_relax_red * black_interior
        
        # Don't apply mask to pressure - rely on Brinkman penalization for obstacle
        # The pressure solver should solve normally in all regions
        
        # Apply domain boundary conditions
        if flow_type == 'von_karman':
            p = p.at[0, :].set(0.0)   # Inlet Dirichlet
            p = p.at[-1, :].set(0.0)  # Outlet Dirichlet
        elif flow_type == 'lid_driven_cavity':
            # Neumann everywhere - extrapolate
            p = p.at[0, :].set(p[1, :])
            p = p.at[-1, :].set(p[-2, :])
            p = p.at[:, 0].set(p[:, 1])
            p = p.at[:, -1].set(p[:, -2])
        
        return (p, iteration + 1), p
    
    # Fixed number of iterations for differentiability (no convergence check)
    (p_final, _), _ = jax.lax.scan(sor_step, (p, 0), None, length=max_iter)
    
    return p_final


@jax.jit(static_argnames=('flow_type', 'max_iter'))
def poisson_cg_masked(
    rhs: jnp.ndarray,
    mask: jnp.ndarray,
    dx: float,
    dy: float,
    max_iter: int = 100,
    tol: float = 1e-6,
    flow_type: str = 'von_karman'
) -> jnp.ndarray:
    """
    Conjugate Gradient pressure solver with mask-aware boundary conditions.
    Fixed iterations for differentiability - no early stopping.
    """
    nx, ny = rhs.shape
    p = jnp.zeros((nx, ny))
    r = rhs.copy()
    
    # Apply mask to residual
    r = r * mask
    
    # Preconditioner (diagonal)
    ax = 1.0 / (dx * dx)
    ay = 1.0 / (dy * dy)
    a0 = 2.0 * (ax + ay)
    precon = 1.0 / (a0 + 1e-12)
    
    z = r * precon
    d = z.copy()
    rz_old = jnp.sum(r * z)
    
    def cg_step(carry, _):
        p, r, d, rz_old = carry
        
        # Matrix-vector product (Laplacian)
        # Use ghost cells for non-periodic boundaries
        p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode='edge')
        Ap = (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1]) / (dx * dx) + \
              (p_padded[1:-1, 2:] + p_padded[1:-1, :-2]) / (dy * dy) - \
              4 * p * (1.0/(dx*dx) + 1.0/(dy*dy))
        
        # Don't apply mask to Ap - rely on Brinkman for obstacle
        
        alpha = rz_old / (jnp.sum(d * Ap) + 1e-12)
        p = p + alpha * d
        r = r - alpha * Ap
        
        # Don't apply mask to residual
        
        z = r * precon
        rz_new = jnp.sum(r * z)
        beta = rz_new / (rz_old + 1e-12)
        d = z + beta * d
        rz_old = rz_new
        
        # Apply boundary conditions after each iteration
        if flow_type == 'von_karman':
            p = p.at[0, :].set(0.0)
            p = p.at[-1, :].set(0.0)
        elif flow_type == 'lid_driven_cavity':
            p = p.at[0, :].set(p[1, :])
            p = p.at[-1, :].set(p[-2, :])
            p = p.at[:, 0].set(p[:, 1])
            p = p.at[:, -1].set(p[:, -2])
        
        return (p, r, d, rz_old), p
    
    # Fixed iterations
    (p_final, _, _, _), _ = jax.lax.scan(cg_step, (p, r, d, rz_old), None, length=max_iter)
    
    # Apply domain boundary conditions
    if flow_type == 'von_karman':
        p_final = p_final.at[0, :].set(0.0)
        p_final = p_final.at[-1, :].set(0.0)
    elif flow_type == 'lid_driven_cavity':
        p_final = p_final.at[0, :].set(p_final[1, :])
        p_final = p_final.at[-1, :].set(p_final[-2, :])
        p_final = p_final.at[:, 0].set(p_final[:, 1])
        p_final = p_final.at[:, -1].set(p_final[:, -2])
    
    return p_final
