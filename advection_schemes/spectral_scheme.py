import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def spectral_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, dealias: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Spectral method using FFT for periodic domains"""
    print(f"[DEBUG] Spectral scheme called with dt={dt}, dealias={dealias}")
    
    nx, ny = u.shape
    
    # Wave numbers
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, dy)
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    # Transform to spectral space
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    
    # Apply dealiasing if requested (2/3 rule) - JAX-compatible
    def apply_dealias(u_hat: jnp.ndarray, v_hat: jnp.ndarray, dealias: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Use jax.lax.cond instead of Python if
        def dealias_true(hat_tuple):
            u_h, v_h = hat_tuple
            kx_max = jnp.max(jnp.abs(kx))
            ky_max = jnp.max(jnp.abs(ky))
            k_cutoff = (2.0/3.0) * jnp.minimum(kx_max, ky_max)
            dealias_mask = (jnp.abs(KX) <= k_cutoff) & (jnp.abs(KY) <= k_cutoff)
            return u_h * dealias_mask, v_h * dealias_mask
        
        def dealias_false(hat_tuple):
            return hat_tuple
        
        return jax.lax.cond(dealias, dealias_true, dealias_false, (u_hat, v_hat))
    
    u_hat, v_hat = apply_dealias(u_hat, v_hat, dealias)
    
    # Compute advection in physical space (dealias)
    u_phys = jnp.real(jnp.fft.ifft2(u_hat))
    v_phys = jnp.real(jnp.fft.ifft2(v_hat))
    
    # Gradients for advection
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    adv_x_phys = u_phys * grad_x(u_phys, dx) + v_phys * grad_y(u_phys, dy)
    adv_y_phys = u_phys * grad_x(v_phys, dx) + v_phys * grad_y(v_phys, dy)
    
    # Transform advection to spectral space
    adv_x_hat = jnp.fft.fft2(adv_x_phys)
    adv_y_hat = jnp.fft.fft2(adv_y_phys)
    
    # Apply dealiasing to advection terms - JAX-compatible
    def apply_dealias_adv(adv_x_hat: jnp.ndarray, adv_y_hat: jnp.ndarray, dealias: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        def dealias_true(hat_tuple):
            adv_x_h, adv_y_h = hat_tuple
            kx_max = jnp.max(jnp.abs(kx))
            ky_max = jnp.max(jnp.abs(ky))
            k_cutoff = (2.0/3.0) * jnp.minimum(kx_max, ky_max)
            dealias_mask = (jnp.abs(KX) <= k_cutoff) & (jnp.abs(KY) <= k_cutoff)
            return adv_x_h * dealias_mask, adv_y_h * dealias_mask
        
        def dealias_false(hat_tuple):
            return hat_tuple
        
        return jax.lax.cond(dealias, dealias_true, dealias_false, (adv_x_hat, adv_y_hat))
    
    adv_x_hat, adv_y_hat = apply_dealias_adv(adv_x_hat, adv_y_hat, dealias)
    
    # Apply diffusion in spectral space (exact integration)
    diffusion_factor = 1.0 / (1.0 + dt * nu * K2)
    u_star_hat = (u_hat + dt * (-adv_x_hat)) * diffusion_factor
    v_star_hat = (v_hat + dt * (-adv_y_hat)) * diffusion_factor
    
    # Transform back to physical space
    u_star = jnp.real(jnp.fft.ifft2(u_star_hat))
    v_star = jnp.real(jnp.fft.ifft2(v_star_hat))
    
    return u_star, v_star
