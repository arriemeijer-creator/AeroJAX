"""
Corner smoothing functions for inlet boundary conditions.
"""

import jax.numpy as jnp
from typing import Tuple


def apply_corner_smooth_inlet(u: jnp.ndarray, v: jnp.ndarray, U_inf: float, ny: int,
                                corner_smooth_width: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply smooth corner transition to inlet velocity using sigmoid taper.
    
    This eliminates corner singularities by smoothly transitioning velocity
    near top and bottom walls using a sigmoid function for C^2 continuity.
    Not JIT-compiled because it uses dynamic array sizes.
    """
    # Create normalized distance from wall (0 at wall, 1 at smooth_width)
    indices_bottom = jnp.arange(corner_smooth_width)
    dist_bottom = indices_bottom / corner_smooth_width
    
    # Smooth sigmoid-like taper using sin^2 for smooth monotonic transition
    # sin^2(pi/2 * dist) goes from 0 to 1 smoothly
    sigmoid_factor = jnp.sin(jnp.pi / 2 * dist_bottom) ** 2
    
    # Base inlet velocity
    u_inlet = jnp.ones((ny,)) * U_inf
    v_inlet = jnp.zeros((ny,))
    
    # Apply smooth taper near bottom wall
    u_inlet = u_inlet.at[indices_bottom].set(U_inf * sigmoid_factor)
    
    # Apply smooth taper near top wall
    indices_top = ny - 1 - jnp.arange(corner_smooth_width)
    u_inlet = u_inlet.at[indices_top].set(U_inf * sigmoid_factor)
    
    return u_inlet, v_inlet


def apply_corner_smooth_pressure_gradient(dp_dx: jnp.ndarray, ny: int,
                                          corner_smooth_width: int = 10) -> jnp.ndarray:
    """Apply smooth corner transition to pressure gradient at inlet.
    
    Smoothly transitions from zero gradient at inlet corners to full gradient
    in the interior, preventing pressure singularities at wall-inlet corners.
    Not JIT-compiled because it uses dynamic array sizes.
    """
    # Create normalized distance from wall
    indices_bottom = jnp.arange(corner_smooth_width)
    dist_bottom = indices_bottom / corner_smooth_width
    
    # Smooth sigmoid-like taper using sin^2 for smooth monotonic transition
    sigmoid_factor = jnp.sin(jnp.pi / 2 * dist_bottom) ** 2
    
    # Apply to bottom corner: dp_dx = 0 at wall, gradually increases to interior value
    dp_dx = dp_dx.at[0, indices_bottom].set(dp_dx[0, indices_bottom] * sigmoid_factor)
    
    # Apply to top corner
    indices_top = ny - 1 - jnp.arange(corner_smooth_width)
    dp_dx = dp_dx.at[0, indices_top].set(dp_dx[0, indices_top] * sigmoid_factor)
    
    return dp_dx
