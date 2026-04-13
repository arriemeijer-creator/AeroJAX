import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def weno5_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, weno_epsilon: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """5th order WENO scheme for shock-capturing"""
    print(f"[DEBUG] WENO5 scheme called with dt={dt}, epsilon={weno_epsilon}")
    
    def weno5_reconstruction(f: jnp.ndarray, axis: int, direction: int) -> jnp.ndarray:
        # Simplified WENO5 reconstruction
        if axis == 0:  # x-direction
            if direction > 0:  # positive flux
                f_im2 = jnp.roll(f, 2, axis=0)
                f_im1 = jnp.roll(f, 1, axis=0)
                f_i = f
                f_ip1 = jnp.roll(f, -1, axis=0)
                f_ip2 = jnp.roll(f, -2, axis=0)
            else:  # negative flux
                f_im2 = jnp.roll(f, 3, axis=0)
                f_im1 = jnp.roll(f, 2, axis=0)
                f_i = jnp.roll(f, 1, axis=0)
                f_ip1 = f
                f_ip2 = jnp.roll(f, -1, axis=0)
        else:  # y-direction
            if direction > 0:
                f_im2 = jnp.roll(f, 2, axis=1)
                f_im1 = jnp.roll(f, 1, axis=1)
                f_i = f
                f_ip1 = jnp.roll(f, -1, axis=1)
                f_ip2 = jnp.roll(f, -2, axis=1)
            else:
                f_im2 = jnp.roll(f, 3, axis=1)
                f_im1 = jnp.roll(f, 2, axis=1)
                f_i = jnp.roll(f, 1, axis=1)
                f_ip1 = f
                f_ip2 = jnp.roll(f, -1, axis=1)
        
        # Smoothness indicators
        beta1 = (13.0/12.0) * (f_im2 - 2*f_im1 + f_i)**2 + 0.25 * (f_im2 - 4*f_im1 + 3*f_i)**2
        beta2 = (13.0/12.0) * (f_im1 - 2*f_i + f_ip1)**2 + 0.25 * (f_im1 - f_ip1)**2
        beta3 = (13.0/12.0) * (f_i - 2*f_ip1 + f_ip2)**2 + 0.25 * (3*f_i - 4*f_ip1 + f_ip2)**2
        
        # Optimal weights
        d1 = 1.0/10.0
        d2 = 3.0/5.0
        d3 = 3.0/10.0
        
        # Nonlinear weights with configurable epsilon
        alpha1 = d1 / (beta1 + weno_epsilon)**2
        alpha2 = d2 / (beta2 + weno_epsilon)**2
        alpha3 = d3 / (beta3 + weno_epsilon)**2
        
        alpha_sum = alpha1 + alpha2 + alpha3
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum
        w3 = alpha3 / alpha_sum
        
        # Candidate stencils
        p1 = (1.0/3.0) * f_im2 - (7.0/6.0) * f_im1 + (11.0/6.0) * f_i
        p2 = (-1.0/6.0) * f_im1 + (5.0/6.0) * f_i + (1.0/3.0) * f_ip1
        p3 = (1.0/3.0) * f_i + (5.0/6.0) * f_ip1 - (1.0/6.0) * f_ip2
        
        # Reconstructed value
        return w1 * p1 + w2 * p2 + w3 * p3
    
    # Compute WENO fluxes
    u_plus = weno5_reconstruction(u, 0, 1)
    u_minus = weno5_reconstruction(u, 0, -1)
    v_plus = weno5_reconstruction(v, 1, 1)
    v_minus = weno5_reconstruction(v, 1, -1)
    
    # Advection terms (simplified)
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    du_dx = grad_x(u, dx)
    du_dy = grad_y(u, dy)
    dv_dx = grad_x(v, dx)
    dv_dy = grad_y(v, dy)
    
    # Use WENO-reconstructed values for advection
    adv_x = u_plus * du_dx + v_plus * du_dy
    adv_y = u_plus * dv_dx + v_plus * dv_dy
    
    # Diffusion
    def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) +
                jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 4 * f) / (dx**2)
    
    diff_x = nu * laplacian(u, dx, dy)
    diff_y = nu * laplacian(v, dx, dy)
    
    # Update velocities
    u_star = u + dt * (-adv_x + diff_x)
    v_star = v + dt * (-adv_y + diff_y)
    
    return u_star, v_star
