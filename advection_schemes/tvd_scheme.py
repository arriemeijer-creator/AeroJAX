import jax
import jax.numpy as jnp
from typing import Tuple

def flux_limiter_minmod(r: jnp.ndarray) -> jnp.ndarray:
    """Minmod flux limiter"""
    return jnp.maximum(0.0, jnp.minimum(1.0, r))

def flux_limiter_superbee(r: jnp.ndarray) -> jnp.ndarray:
    """Superbee flux limiter"""
    return jnp.maximum(0.0, jnp.maximum(jnp.minimum(1.0, 2*r), jnp.minimum(2.0, r)))

def flux_limiter_van_leer(r: jnp.ndarray) -> jnp.ndarray:
    """Van Leer flux limiter"""
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))

@jax.jit(static_argnums=(7,))
def _tvd_step_impl(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, limiter_type: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """TVD scheme implementation with flux limiter function"""
    
    def compute_flux(f: jnp.ndarray, axis: int, dt: float, dx: float, dy: float) -> jnp.ndarray:
        if axis == 0:
            f_upwind = jnp.roll(f, 1, axis=0)
            f_lw = 0.5 * (f + jnp.roll(f, -1, axis=0)) - \
                   0.5 * dt/dx * (f - jnp.roll(f, 1, axis=0))
        else:
            f_upwind = jnp.roll(f, 1, axis=1)
            f_lw = 0.5 * (f + jnp.roll(f, -1, axis=1)) - \
                   0.5 * dt/dy * (f - jnp.roll(f, 1, axis=1))
        
        # Flux limiter - inlined with better numerical stability
        denominator = (jnp.roll(f, -1, axis=axis) - f)
        r = (f - jnp.roll(f, 1, axis=axis)) / jnp.where(jnp.abs(denominator) > 1e-12, denominator, 1e-12)
        if limiter_type == 'minmod':
            phi = jnp.maximum(0.0, jnp.minimum(1.0, r))
        elif limiter_type == 'superbee':
            phi = jnp.maximum(0.0, jnp.maximum(jnp.minimum(1.0, 2*r), jnp.minimum(2.0, r)))
        elif limiter_type == 'van_leer':
            phi = (r + jnp.abs(r)) / (1.0 + jnp.abs(r))
        else:  # default to minmod
            phi = jnp.maximum(0.0, jnp.minimum(1.0, r))
        
        return f_upwind + phi * (f_lw - f_upwind)
    
    # Compute TVD fluxes
    flux_x = compute_flux(u, 0, dt, dx, dy)
    flux_y = compute_flux(v, 1, dt, dx, dy)
    
    adv_x = (flux_x - jnp.roll(flux_x, 1, axis=0)) / dx
    adv_y = (flux_y - jnp.roll(flux_y, 1, axis=1)) / dy
    
    # Diffusion
    def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) - 2*f) / (dx**2) + \
               (jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 2*f) / (dy**2)
    
    diff_x = nu * laplacian(u, dx, dy)
    diff_y = nu * laplacian(v, dx, dy)
    
    u_star = u + dt * (-adv_x + diff_x)
    v_star = v + dt * (-adv_y + diff_y)
    
    return u_star, v_star

@jax.jit
def tvd_step_minmod(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """TVD scheme with minmod limiter"""
    return _tvd_step_impl(u, v, dt, nu, dx, dy, mask, 'minmod')

@jax.jit
def tvd_step_superbee(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """TVD scheme with superbee limiter"""
    return _tvd_step_impl(u, v, dt, nu, dx, dy, mask, 'superbee')

@jax.jit
def tvd_step_van_leer(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """TVD scheme with van Leer limiter"""
    return _tvd_step_impl(u, v, dt, nu, dx, dy, mask, 'van_leer')

def tvd_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, limiter: str = 'minmod') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Total Variation Diminishing scheme with flux limiters"""
    print(f"[DEBUG] TVD scheme called with dt={dt}, limiter={limiter}")
    
    # Simple TVD scheme with upwind flux limitinger"""
    if limiter == 'minmod':
        return tvd_step_minmod(u, v, dt, nu, dx, dy, mask)
    elif limiter == 'superbee':
        return tvd_step_superbee(u, v, dt, nu, dx, dy, mask)
    elif limiter == 'van_leer':
        return tvd_step_van_leer(u, v, dt, nu, dx, dy, mask)
    else:  # default to minmod
        return tvd_step_minmod(u, v, dt, nu, dx, dy, mask)
