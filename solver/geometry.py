"""
Geometry and SDF (Signed Distance Function) utilities for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
from typing import Dict


@jax.jit
def sdf_cylinder(x: jnp.ndarray, y: jnp.ndarray, center_x: float, center_y: float, radius: float) -> jnp.ndarray:
    """Signed distance function for a cylinder"""
    return jnp.sqrt((x - center_x)**2 + (y - center_y)**2) - radius


@jax.jit
def smooth_mask(phi: jnp.ndarray, eps: float = 0.05) -> jnp.ndarray:
    """Convert SDF to smooth mask using sigmoid"""
    return jax.nn.sigmoid(phi / eps)


def create_mask_from_params(X: jnp.ndarray, Y: jnp.ndarray, params: Dict, eps: float = 0.05) -> jnp.ndarray:
    """Create mask from geometry parameters"""
    phi = sdf_cylinder(X, Y, params['center_x'], params['center_y'], params['radius'])
    return smooth_mask(phi, eps)
