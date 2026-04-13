import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def poisson_fft(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    FFT-based pressure Poisson solver for periodic domains.
    
    Note: FFT assumes periodic boundary conditions. For non-periodic
    domains (von Karman with inlet/outlet), use multigrid solver instead.
    """
    nx, ny = rhs.shape
    
    # Wave numbers
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, dy)
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    # Avoid division by zero
    K2 = K2.at[0, 0].set(1.0)
    
    # Solve in spectral space
    rhs_hat = jnp.fft.fft2(rhs)
    p_hat = -rhs_hat / K2
    
    # Set mean pressure to zero
    p_hat = p_hat.at[0, 0].set(0.0)
    
    # Transform back
    p = jnp.real(jnp.fft.ifft2(p_hat))
    return p
