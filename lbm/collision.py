"""
Collision operators for LBM (BGK, MRT)
"""

import jax
import jax.numpy as jnp


def equilibrium(rho: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, 
                cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray, 
                cs_squared: float) -> jnp.ndarray:
    """
    Compute equilibrium distribution function (f_eq)
    
    Args:
        rho: Density field (nx, ny)
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
    
    Returns:
        f_eq: Equilibrium distribution (9, nx, ny)
    """
    # Expand dimensions for broadcasting
    # rho, u, v: (nx, ny) -> (1, nx, ny)
    rho = rho[None, :, :]
    u = u[None, :, :]
    v = v[None, :, :]
    
    # Expand lattice vectors
    # cx, cy, w: (9,) -> (9, 1, 1)
    cx = cx[:, None, None]
    cy = cy[:, None, None]
    w = w[:, None, None]
    
    # Compute velocity squared
    u_sq = u**2 + v**2
    
    # Compute dot product c_i · u
    c_dot_u = cx * u + cy * v
    
    # Compute equilibrium distribution
    # f_eq = w_i * rho * (1 + (c_i·u)/cs² + (c_i·u)²/(2*cs⁴) - u²/(2*cs²))
    term1 = 1.0 + c_dot_u / cs_squared
    term2 = (c_dot_u**2) / (2.0 * cs_squared**2)
    term3 = u_sq / (2.0 * cs_squared)
    
    f_eq = w * rho * (term1 + term2 - term3)
    
    return f_eq


def bgk_collision(f: jnp.ndarray, f_eq: jnp.ndarray, omega: float) -> jnp.ndarray:
    """
    BGK (Bhatnagar-Gross-Krook) collision operator
    
    Args:
        f: Distribution function (9, nx, ny)
        f_eq: Equilibrium distribution (9, nx, ny)
        omega: Collision frequency (1/tau)
    
    Returns:
        f_post: Post-collision distribution (9, nx, ny)
    """
    f_post = f - omega * (f - f_eq)
    return f_post


@jax.jit
def collision_step(f: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray,
                   cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray, 
                   cs_squared: float, omega: float) -> jnp.ndarray:
    """
    Complete collision step: compute equilibrium and apply BGK
    
    Args:
        f: Distribution function (9, nx, ny)
        rho: Density field (nx, ny)
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        omega: Collision frequency
    
    Returns:
        f_post: Post-collision distribution (9, nx, ny)
    """
    f_eq = equilibrium(rho, u, v, cx, cy, w, cs_squared)
    f_post = bgk_collision(f, f_eq, omega)
    return f_post
