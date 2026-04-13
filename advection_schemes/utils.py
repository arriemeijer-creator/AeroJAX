import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass

@dataclass
class AdvectionParams:
    scheme: str = 'tvd'
    limiter: str = 'minmod'  # for TVD
    weno_epsilon: float = 1e-6  # for WENO
    dealias: bool = True  # for spectral
    max_cfl: float = 0.5  # maximum CFL number

@jax.jit
def check_cfl(u: jnp.ndarray, v: jnp.ndarray, dt: float, dx: float, dy: float) -> float:
    """Check CFL condition for stability"""
    max_vel = jnp.max(jnp.sqrt(u**2 + v**2))
    cfl_x = max_vel * dt / dx
    cfl_y = max_vel * dt / dy
    return max(cfl_x, cfl_y)

@jax.jit
def adaptive_dt(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, max_cfl: float = 0.5) -> float:
    """Compute adaptive timestep based on CFL condition"""
    max_vel = jnp.max(jnp.sqrt(u**2 + v**2))
    dt_cfl = max_cfl * min(dx, dy) / (max_vel + 1e-10)
    return dt_cfl

@jax.jit
def spectral_dealias_2_3(f_hat: jnp.ndarray, kx: jnp.ndarray, ky: jnp.ndarray) -> jnp.ndarray:
    """2/3 dealiasing rule for spectral methods"""
    kx_max = jnp.max(jnp.abs(kx))
    ky_max = jnp.max(jnp.abs(ky))
    k_cutoff = (2.0/3.0) * min(kx_max, ky_max)
    
    mask = (jnp.abs(kx) <= k_cutoff) & (jnp.abs(ky) <= k_cutoff)
    return f_hat * mask
