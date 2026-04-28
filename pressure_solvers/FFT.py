"""
FFT-based Poisson solver for rectangular domains with mixed boundary conditions.

Supports:
- von Karman: Neumann at inlet (x=0) and walls (y=0,H), Dirichlet at outlet (x=L, p=0)
- lid_driven_cavity: Neumann everywhere (∂p/∂n=0)
- taylor_green: Periodic in both directions
"""

import jax
import jax.numpy as jnp
from typing import Literal


def poisson_fft_dirichlet_neumann(
    rhs: jnp.ndarray,
    dx: float,
    dy: float,
    flow_type: Literal['von_karman', 'lid_driven_cavity', 'taylor_green'] = 'von_karman'
) -> jnp.ndarray:
    """
    FFT-based Poisson solver for mixed boundary conditions.
    
    For von Karman (Dirichlet at outlet, Neumann at inlet/walls):
        Uses Discrete Sine Transform in x (Dirichlet) and Discrete Cosine Transform in y (Neumann)
    
    For lid_driven_cavity (Neumann everywhere):
        Uses Discrete Cosine Transform in both directions
    
    For taylor_green (periodic):
        Uses standard FFT
    
    Args:
        rhs: Right-hand side (divergence/dt)
        dx, dy: Grid spacing
        flow_type: Flow type for boundary conditions
    
    Returns:
        p: Pressure field
    """
    nx, ny = rhs.shape
    
    # Remove mean from RHS for Neumann cases (ensures compatibility)
    if flow_type == 'lid_driven_cavity':
        rhs = rhs - jnp.mean(rhs)
    
    # ========================================================================
    # Case 1: von Karman - Dirichlet at x=L, Neumann at x=0, y=0, y=H
    # ========================================================================
    if flow_type == 'von_karman':
        # Use DST (sine) in x for Dirichlet BC at x=L (p=0)
        # Use DCT (cosine) in y for Neumann BC at y=0,H (∂p/∂y=0)
        
        # Step 1: Apply DST in x (type II, orthonormal)
        # DST handles anti-symmetric extension internally for Dirichlet BC
        rhs_dst = apply_dst(rhs, axis=0)
        
        # Step 2: Apply DCT in y (type II, orthonormal)
        rhs_dct = apply_dct(rhs_dst, axis=1)
        
        # Step 3: Compute eigenvalues
        # x-direction: Dirichlet eigenvalues (kπ/L)
        kx = jnp.pi * jnp.arange(1, nx+1) / (nx * dx)
        kx2 = kx**2
        
        # y-direction: Neumann eigenvalues (mπ/H)
        ky = jnp.pi * jnp.arange(ny) / (ny * dy)
        ky2 = ky**2
        
        # Create 2D wavenumber grid
        K2 = kx2[:, None] + ky2[None, :]
        K2 = K2.at[0, 0].set(1.0)  # Avoid division by zero
        
        # Step 4: Solve in spectral space
        p_hat = -rhs_dct / K2
        p_hat = p_hat.at[0, 0].set(0.0)  # Set mean to zero
        
        # Step 5: Inverse transforms
        p = apply_idct(p_hat, axis=1)
        p = apply_idst(p, axis=0)
        
        # Enforce outlet Dirichlet BC (p=0 at x=L)
        p = p.at[-1, :].set(0.0)
        
        return p
    
    # ========================================================================
    # Case 2: Lid-driven cavity - Neumann everywhere
    # ========================================================================
    elif flow_type == 'lid_driven_cavity':
        # Use DCT (cosine) in both directions for Neumann BC (∂p/∂n=0)
        
        # Apply DCT in x and y
        rhs_dct = apply_dct(rhs, axis=0)
        rhs_dct = apply_dct(rhs_dct, axis=1)
        
        # Compute eigenvalues
        kx = jnp.pi * jnp.arange(nx) / (nx * dx)
        ky = jnp.pi * jnp.arange(ny) / (ny * dy)
        kx2 = kx**2
        ky2 = ky**2
        
        # Create 2D wavenumber grid
        K2 = kx2[:, None] + ky2[None, :]
        K2 = K2.at[0, 0].set(1.0)
        
        # Solve in spectral space
        p_hat = -rhs_dct / K2
        p_hat = p_hat.at[0, 0].set(0.0)  # Pin mean pressure
        
        # Inverse transforms
        p = apply_idct(p_hat, axis=1)
        p = apply_idct(p, axis=0)
        
        return p
    
    # ========================================================================
    # Case 3: Taylor-Green - Periodic everywhere
    # ========================================================================
    else:  # taylor_green
        from jax.numpy.fft import rfft2, irfft2
        
        # Compute wavenumbers
        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, dx)
        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, dy)
        
        # Create 2D wavenumber grid
        K2 = kx[:, None]**2 + ky[None, :]**2
        K2 = K2.at[0, 0].set(1.0)  # Avoid division by zero
        
        # Solve in Fourier space
        rhs_hat = rfft2(rhs)
        p_hat = -rhs_hat / K2
        p_hat = p_hat.at[0, 0].set(0.0)  # Set mean to zero
        
        # Inverse FFT
        p = irfft2(p_hat, s=(nx, ny))
        
        return p


# ============================================================================
# Helper functions for DCT/DST using FFT
# ============================================================================

def apply_dct(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    Type II Discrete Cosine Transform (orthonormal) using FFT.
    
    DCT-II: X_k = sqrt(2/N) * Σ x_n * cos(π/N * (n + 0.5) * k)
    """
    n = a.shape[axis]
    
    # Pad and fold for DCT via FFT
    def dct_1d(x):
        # Create symmetric extension: [x[0], x[1], ..., x[n-1], x[n-1], ..., x[1], x[0]]
        x_sym = jnp.concatenate([x, jnp.flip(x)])
        x_fft = jnp.fft.rfft(x_sym, n=2*n)
        factor = jnp.exp(-1j * jnp.pi * jnp.arange(n) / (2*n))
        dct = 2 * jnp.real(factor * x_fft[:n])
        dct = dct.at[0].set(dct[0] / jnp.sqrt(2))
        return dct / jnp.sqrt(n)
    
    # Apply along specified axis
    if axis == 0:
        result = jax.vmap(dct_1d, in_axes=1, out_axes=1)(a)
    else:
        result = jax.vmap(dct_1d, in_axes=0, out_axes=0)(a)
    
    return result


def apply_idct(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    Type II Inverse Discrete Cosine Transform (orthonormal) using FFT.
    """
    n = a.shape[axis]
    
    def idct_1d(x):
        x = x * jnp.sqrt(n)
        x = x.at[0].set(x[0] * jnp.sqrt(2))
        x_pad = jnp.zeros(2*n, dtype=x.dtype)
        x_pad = x_pad.at[:n].set(x)
        x_pad = x_pad.at[n:].set(jnp.zeros(n))
        
        # Complex FFT
        x_fft = jnp.fft.fft(x_pad, n=2*n)
        
        # Extract and scale
        idct = 0.5 * jnp.real(x_fft[:n] * jnp.exp(1j * jnp.pi * jnp.arange(n) / (2*n)))
        return idct
    
    # Apply along specified axis
    if axis == 0:
        result = jax.vmap(idct_1d, in_axes=1, out_axes=1)(a)
    else:
        result = jax.vmap(idct_1d, in_axes=0, out_axes=0)(a)
    
    return result


def apply_dst(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    Type II Discrete Sine Transform (orthonormal) using FFT.
    
    DST-II: X_k = sqrt(2/N) * Σ x_n * sin(π/N * (n + 0.5) * k)
    """
    n = a.shape[axis]
    
    def dst_1d(x):
        # Extend anti-symmetrically: [x[0], x[1], ..., x[n-1], -x[n-1], ..., -x[1], -x[0]]
        x_pad = jnp.concatenate([x, -jnp.flip(x)])
        x_fft = jnp.fft.rfft(x_pad)
        dst = -jnp.imag(x_fft[:n])
        dst = dst.at[0].set(0.0)  # Zero at k=0
        dst = dst * jnp.sqrt(2/n)
        return dst
    
    # Apply along specified axis
    if axis == 0:
        result = jax.vmap(dst_1d, in_axes=1, out_axes=1)(a)
    else:
        result = jax.vmap(dst_1d, in_axes=0, out_axes=0)(a)
    
    return result


def apply_idst(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    Type II Inverse Discrete Sine Transform (orthonormal) using FFT.
    """
    n = a.shape[axis]
    
    def idst_1d(x):
        # Prepare for IDFT using concatenation
        x_scaled = 1j * x / jnp.sqrt(2*n)
        x_pad = jnp.concatenate([jnp.array([0.0]), x_scaled, jnp.zeros(n-1), -jnp.flip(x_scaled)[1:]])
        
        # Inverse FFT
        idst = jnp.fft.ifft(x_pad, n=2*n)
        idst = 2 * jnp.imag(idst[:n])
        return idst
    
    # Apply along specified axis
    if axis == 0:
        result = jax.vmap(idst_1d, in_axes=1, out_axes=1)(a)
    else:
        result = jax.vmap(idst_1d, in_axes=0, out_axes=0)(a)
    
    return result

# Simplified wrapper for direct use in your solver
# ============================================================================

def poisson_fft_solve(
    rhs: jnp.ndarray,
    mask: jnp.ndarray,
    dx: float,
    dy: float,
    flow_type: Literal['von_karman', 'lid_driven_cavity', 'taylor_green'] = 'von_karman'
) -> jnp.ndarray:
    """
    Wrapper for FFT-based Poisson solver that matches the interface of your existing solvers.
    
    Args:
        rhs: Right-hand side (divergence/dt)
        mask: Obstacle mask (used to check if FFT is applicable)
        dx, dy: Grid spacing
        flow_type: Flow type for boundary conditions
    
    Returns:
        p: Pressure field
    
    Note: FFT solver is only applicable for simple domains without complex obstacles.
    If mask indicates obstacles (not all ones), returns None to signal fallback to multigrid.
    """
    # Check if there are obstacles (mask not all ones)
    if mask is not None and jnp.any(mask < 0.99):
        # Obstacles present - FFT solver not applicable
        # Return None to signal fallback to multigrid
        return None
    
    return poisson_fft_dirichlet_neumann(rhs, dx, dy, flow_type)


# ============================================================================
# Test/validation function
# ============================================================================

def test_poisson_solver():
    """
    Quick test to verify the FFT solver works correctly.
    """
    # Create a small test grid
    nx, ny = 64, 64
    dx = 20.0 / nx
    dy = 7.5 / ny
    
    # Create a simple RHS (sinusoidal)
    x = jnp.linspace(0, 20, nx)
    y = jnp.linspace(0, 7.5, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    rhs = jnp.sin(2*jnp.pi*X/20) * jnp.cos(jnp.pi*Y/7.5)
    
    # Solve
    p = poisson_fft_solve(rhs, None, dx, dy, flow_type='von_karman')
    
    # Verify BCs
    print(f"p[0,:] (inlet): {p[0, 0]:.6f}, {p[0, ny//2]:.6f}, {p[0, -1]:.6f}")
    print(f"p[-1,:] (outlet): {p[-1, 0]:.6f}, {p[-1, ny//2]:.6f}, {p[-1, -1]:.6f}")
    print(f"p[:,0] (bottom wall): {p[0, 0]:.6f}, {p[nx//2, 0]:.6f}, {p[-1, 0]:.6f}")
    
    return p


if __name__ == "__main__":
    p = test_poisson_solver()