"""
Adaptive timestep controller using PID logic.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def update_dt_pure(dt: float, div_max: float, integral: float, prev_error: float,
                   target_div: float = 1e-4, Kp: float = 0.5, Ki: float = 0.05, Kd: float = 0.1,
                   dt_min: float = 5e-4, dt_max: float = 0.01, eta_max: float = None,
                   Re: float = None) -> Tuple[float, float, float]:
    """Pure JAX-compatible adaptive timestep update using PID controller"""
    # Safeguard against invalid divergence values
    div_is_valid = jnp.isfinite(div_max) & (div_max >= 0)
    
    # Re-dependent dt_max clamping (relaxed)
    if Re is not None and Re > 10000:
        dt_max = jnp.minimum(dt_max, 0.008 * (10000.0 / Re))
    
    # If divergence is invalid, reduce timestep conservatively
    def handle_invalid_div():
        new_dt = dt * 0.5
        new_dt = jnp.maximum(dt_min, jnp.minimum(dt_max, new_dt))
        return new_dt, integral, prev_error
    
    # Normal case with valid divergence
    def handle_valid_div():
        error = div_max - target_div
        integral_new = integral + error * dt
        # Anti-windup: clamp integral
        integral_new = jnp.maximum(-1.0, jnp.minimum(1.0, integral_new))
        
        derivative = (error - prev_error) / (dt + 1e-8)
        correction = Kp * error + Ki * integral_new + Kd * derivative
        factor = jnp.maximum(0.5, jnp.minimum(2.0, 1.0 + correction))
        new_dt = dt * factor
        
        # Brinkman stiffness limiter
        if eta_max is not None and eta_max > 0:
            rho = 1.0
            C_safety = 0.5
            dt_brinkman_limit = C_safety * (rho / (eta_max + 1e-8))
            new_dt = jnp.minimum(new_dt, dt_brinkman_limit)
        
        new_dt = jnp.maximum(dt_min, jnp.minimum(dt_max, new_dt))
        return new_dt, integral_new, error
    
    return jnp.lax.cond(div_is_valid, handle_valid_div, handle_invalid_div)
