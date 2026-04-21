"""
Brinkman penalization functions for immersed boundary method.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def apply_brinkman_penalization(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                                dt: float, nu: float, dx: float, eta_max: float = 50.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Brinkman penalization for immersed boundaries with improved accuracy.
    """
    # Sigmoid trick: sharper boundary transition with defined gradient everywhere
    beta = 20.0  # Sharpness parameter; higher = sharper wall
    chi = jax.nn.sigmoid(beta * (0.5 - mask))

    # Increased penalization strength
    eta = eta_max * chi

    # Apply with stability limit
    u_penalized = u / (1.0 + dt * eta)
    v_penalized = v / (1.0 + dt * eta)

    # Don't let penalization change velocity by more than 50% per step
    u_penalized = jnp.where(jnp.abs(u_penalized) > 1.5 * jnp.abs(u),
                            1.5 * u, u_penalized)
    v_penalized = jnp.where(jnp.abs(v_penalized) > 1.5 * jnp.abs(v),
                            1.5 * v, v_penalized)

    # Hard-zero the interior: force velocity to zero inside airfoil
    u_final = u_penalized * jnp.where(mask > 0.01, 1.0, 0.0)
    v_final = v_penalized * jnp.where(mask > 0.01, 1.0, 0.0)

    return u_final, v_final


@jax.jit
def apply_brinkman_penalization_ramped(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                                       dt: float, nu: float, dx: float,
                                       iteration: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Brinkman penalization with gradual strength ramping.
    """
    # Ramp eta_max from 10 to 50 over 5000 iterations
    ramp_iterations = 5000
    eta_base = 10.0
    eta_target = 50.0
    eta_max = eta_base + (eta_target - eta_base) * jnp.minimum(1.0, iteration / ramp_iterations)

    # Sigmoid trick: sharper boundary transition with defined gradient everywhere
    beta = 20.0  # Sharpness parameter; higher = sharper wall
    chi = jax.nn.sigmoid(beta * (0.5 - mask))
    eta = eta_max * chi

    u_penalized = u / (1.0 + dt * eta)
    v_penalized = v / (1.0 + dt * eta)

    # Keep the 50% cap for safety
    u_penalized = jnp.where(jnp.abs(u_penalized) > 1.5 * jnp.abs(u), 1.5 * u, u_penalized)
    v_penalized = jnp.where(jnp.abs(v_penalized) > 1.5 * jnp.abs(v), 1.5 * v, v_penalized)

    # Hard-zero the interior: force velocity to zero inside airfoil
    u_final = u_penalized * jnp.where(mask > 0.01, 1.0, 0.0)
    v_final = v_penalized * jnp.where(mask > 0.01, 1.0, 0.0)

    return u_final, v_final


@jax.jit
def apply_brinkman_penalization_mild(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                                     dt: float, nu: float, dx: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Ultra-mild Brinkman - barely stronger than mask multiplication.
    """
    # Sigmoid trick: sharper boundary transition with defined gradient everywhere
    beta = 20.0  # Sharpness parameter; higher = sharper wall
    chi = jax.nn.sigmoid(beta * (0.5 - mask))

    # Very weak penalization
    eta_max = 2.0  # Much smaller than 10.0!
    eta = eta_max * chi

    u_penalized = u / (1.0 + dt * eta)
    v_penalized = v / (1.0 + dt * eta)

    # Hard-zero the interior: force velocity to zero inside airfoil
    u_final = u_penalized * jnp.where(mask > 0.01, 1.0, 0.0)
    v_final = v_penalized * jnp.where(mask > 0.01, 1.0, 0.0)

    return u_final, v_final


@jax.jit
def apply_brinkman_penalization_consistent(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                                           dt: float, nu: float, dx: float, brinkman_eta: float = 1000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Minimal penalization - just apply mask.
    The mask itself provides the obstacle boundary.
    """
    # For now, Brinkman just applies the mask
    # The mask itself provides the obstacle boundary
    u_final = u * mask
    v_final = v * mask

    return u_final, v_final
