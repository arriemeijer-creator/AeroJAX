"""
Macroscopic variable computation for LBM
"""

import jax
import jax.numpy as jnp


def macroscopic_variables(f: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray) -> tuple:
    """
    Compute macroscopic density and velocity from distribution functions
    
    Args:
        f: Distribution function (9, nx, ny)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
    
    Returns:
        rho: Density field (nx, ny)
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
    """
    # Density: sum of all distributions
    rho = jnp.sum(f, axis=0)
    
    # Clamp density to prevent division by zero or negative values
    rho = jnp.clip(rho, 0.1, 10.0)
    
    # Momentum: sum of c_i * f_i
    momentum_x = jnp.sum(f * cx[:, None, None], axis=0)
    momentum_y = jnp.sum(f * cy[:, None, None], axis=0)
    
    # Velocity: momentum / density
    u = momentum_x / rho
    v = momentum_y / rho
    
    # Clamp velocities to reasonable range
    u = jnp.clip(u, -10.0, 10.0)
    v = jnp.clip(v, -10.0, 10.0)
    
    return rho, u, v


def compute_density(f: jnp.ndarray) -> jnp.ndarray:
    """
    Compute density from distribution functions
    
    Args:
        f: Distribution function (9, nx, ny)
    
    Returns:
        rho: Density field (nx, ny)
    """
    return jnp.sum(f, axis=0)


def compute_velocity(f: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray) -> tuple:
    """
    Compute velocity from distribution functions
    
    Args:
        f: Distribution function (9, nx, ny)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
    
    Returns:
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
    """
    rho = compute_density(f)
    
    momentum_x = jnp.sum(f * cx[:, None, None], axis=0)
    momentum_y = jnp.sum(f * cy[:, None, None], axis=0)
    
    u = momentum_x / rho
    v = momentum_y / rho
    
    return u, v


def compute_pressure(rho: jnp.ndarray, cs_squared: float, 
                     rho0: float = 1.0) -> jnp.ndarray:
    """
    Compute pressure from density (equation of state)
    
    Args:
        rho: Density field (nx, ny)
        cs_squared: Speed of sound squared
        rho0: Reference density
    
    Returns:
        p: Pressure field (nx, ny)
    """
    p = cs_squared * (rho - rho0)
    # Clamp pressure to reasonable range to prevent visualization errors
    p = jnp.clip(p, -1.0, 1.0)
    return p
