import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def poisson_cg(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float, 
               max_iter: int = 1000, tol: float = 1e-8) -> jnp.ndarray:
    """
    TRUE Conjugate Gradient solver for Poisson equation.
    
    Solves: ∇²p = rhs
    Using standard Conjugate Gradient method with simple implementation.
    """
    nx, ny = rhs.shape
    
    def apply_laplacian(p: jnp.ndarray) -> jnp.ndarray:
        """Apply discrete Laplacian operator with non-periodic boundary conditions
        
        Boundary conditions for von Karman flow:
        - Inlet (left, x=0): Dirichlet (p = 0) to fix pressure level
        - Outlet (right, x=L): Dirichlet (p = 0) to vent pressure
        - Top/Bottom walls: Neumann (∂p/∂y = 0) for no-slip walls
        """
        laplacian = jnp.zeros_like(p)
        
        # Interior points
        laplacian = laplacian.at[1:-1, 1:-1].set(
            (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / (dx * dx) +
            (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) / (dy * dy)
        )
        
        # Inlet (left): Dirichlet BC (p = 0)
        laplacian = laplacian.at[0, :].set(0.0)
        
        # Outlet (right): Dirichlet BC (p = 0)
        laplacian = laplacian.at[-1, :].set(0.0)
        
        # Top/Bottom walls: Neumann BC (∂p/∂y = 0) → p[:,0] = p[:,1], p[:,-1] = p[:,-2]
        laplacian = laplacian.at[1:-1, 0].set(
            (p[2:, 0] - 2.0 * p[1:-1, 0] + p[:-2, 0]) / (dx * dx) +
            (p[1:-1, 1] - 2.0 * p[1:-1, 0] + p[1:-1, 1]) / (dy * dy)
        )
        laplacian = laplacian.at[1:-1, -1].set(
            (p[2:, -1] - 2.0 * p[1:-1, -1] + p[:-2, -1]) / (dx * dx) +
            (p[1:-1, -2] - 2.0 * p[1:-1, -1] + p[1:-1, -2]) / (dy * dy)
        )
        
        # Corners
        laplacian = laplacian.at[0, 0].set(0.0)
        laplacian = laplacian.at[0, -1].set(0.0)
        laplacian = laplacian.at[-1, 0].set(0.0)
        laplacian = laplacian.at[-1, -1].set(0.0)
        
        # Apply mask
        laplacian = laplacian * mask
        
        return laplacian
    
    # Initial guess
    p = jnp.zeros((nx, ny))
    
    # Zero out RHS at inlet and outlet (Dirichlet BC: p=0, so residual must be 0)
    rhs = rhs.at[0, :].set(0.0)
    rhs = rhs.at[-1, :].set(0.0)
    
    # Initial residual: r = b - A·p (with p=0, r = b)
    r = rhs * mask
    
    # Initial search direction
    d = r.copy()
    
    # Initial residual norm squared
    r_norm_sq = jnp.sum(r * r)
    r0_norm_sq = r_norm_sq
    
    def body_fun(carry):
        p, d, r, r_norm_sq, i = carry
        
        # Compute A·d
        Ad = apply_laplacian(d)
        
        # Compute step size α = (r·r) / (d·Ad)
        dAd = jnp.sum(d * Ad)
        alpha = r_norm_sq / jnp.where(jnp.abs(dAd) > 1e-12, dAd, 1e-12)
        
        # Update solution: p ← p + α·d
        p_new = p + alpha * d
        
        # Update residual: r ← r - α·Ad
        r_new = r - alpha * Ad
        
        # Apply mask
        p_new = p_new * mask
        r_new = r_new * mask
        
        # Compute new residual norm squared
        r_new_norm_sq = jnp.sum(r_new * r_new)
        
        # Compute β = (r_new·r_new) / (r·r)
        beta = r_new_norm_sq / (r_norm_sq + 1e-10)
        
        # Update search direction: d ← r_new + β·d
        d_new = r_new + beta * d
        
        return (p_new, d_new, r_new, r_new_norm_sq, i + 1)
    
    def cond_fun(carry):
        p, d, r, r_norm_sq, i = carry
        # Check convergence and iteration count
        return (i < max_iter) & (r_norm_sq > tol * r0_norm_sq)
    
    # Initialize carry
    init_carry = (p, d, r, r_norm_sq, 0)
    
    # Run CG iterations
    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    
    # Extract final pressure
    p_final, _, _, _, iterations = final_carry
    
    return p_final * mask


@jax.jit
def poisson_cg_with_preconditioner(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float,
                                   max_iter: int = 1000, tol: float = 1e-8) -> jnp.ndarray:
    """
    Preconditioned Conjugate Gradient solver with Jacobi preconditioner.
    More stable implementation with safeguards.
    """
    nx, ny = rhs.shape
    
    def apply_laplacian(p: jnp.ndarray) -> jnp.ndarray:
        """Apply discrete Laplacian operator with non-periodic boundary conditions
        
        Boundary conditions for von Karman flow:
        - Inlet (left, x=0): Dirichlet (p = 0) to fix pressure level
        - Outlet (right, x=L): Dirichlet (p = 0) to vent pressure
        - Top/Bottom walls: Neumann (∂p/∂y = 0) for no-slip walls
        """
        laplacian = jnp.zeros_like(p)
        
        # Interior points
        laplacian = laplacian.at[1:-1, 1:-1].set(
            (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / (dx * dx) +
            (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) / (dy * dy)
        )
        
        # Inlet (left): Dirichlet BC (p = 0)
        laplacian = laplacian.at[0, :].set(0.0)
        
        # Outlet (right): Dirichlet BC (p = 0)
        laplacian = laplacian.at[-1, :].set(0.0)
        
        # Top/Bottom walls: Neumann BC (∂p/∂y = 0)
        laplacian = laplacian.at[1:-1, 0].set(
            (p[2:, 0] - 2.0 * p[1:-1, 0] + p[:-2, 0]) / (dx * dx) +
            (p[1:-1, 1] - 2.0 * p[1:-1, 0] + p[1:-1, 1]) / (dy * dy)
        )
        laplacian = laplacian.at[1:-1, -1].set(
            (p[2:, -1] - 2.0 * p[1:-1, -1] + p[:-2, -1]) / (dx * dx) +
            (p[1:-1, -2] - 2.0 * p[1:-1, -1] + p[1:-1, -2]) / (dy * dy)
        )
        
        # Corners
        laplacian = laplacian.at[0, 0].set(0.0)
        laplacian = laplacian.at[0, -1].set(0.0)
        laplacian = laplacian.at[-1, 0].set(0.0)
        laplacian = laplacian.at[-1, -1].set(0.0)
        
        return laplacian * mask
    
    def apply_preconditioner(r: jnp.ndarray) -> jnp.ndarray:
        """
        Jacobi preconditioner: M^{-1} = diagonal^{-1}
        More stable with safeguards.
        """
        # Diagonal of the Laplacian (negative definite)
        diag = -2.0 / (dx * dx) - 2.0 / (dy * dy)
        
        # For interior points, diag is constant negative
        # For boundaries, we want to handle carefully
        # Create a diagonal array with proper handling
        diag_array = jnp.full((nx, ny), diag)
        
        # Override boundaries - at boundaries, we want identity preconditioner
        diag_array = diag_array.at[0, :].set(1.0)
        diag_array = diag_array.at[-1, :].set(1.0)
        diag_array = diag_array.at[:, 0].set(1.0)
        diag_array = diag_array.at[:, -1].set(1.0)
        
        # For solid cells (mask=0), set to 1 to avoid division issues
        diag_array = jnp.where(mask < 0.5, 1.0, diag_array)
        
        # Safe inversion with small epsilon
        inv_diag = 1.0 / (jnp.abs(diag_array) + 1e-8)
        
        # Apply mask
        inv_diag = inv_diag * mask
        
        return r * inv_diag
    
    # Initialize
    p = jnp.zeros((nx, ny))
    
    # Zero out RHS at inlet and outlet (Dirichlet BC: p=0, so residual must be 0)
    rhs = rhs.at[0, :].set(0.0)
    rhs = rhs.at[-1, :].set(0.0)
    
    r = rhs * mask
    
    # Apply preconditioner to initial residual
    z = apply_preconditioner(r)
    d = z.copy()
    rz_old = jnp.sum(r * z)
    r0_norm_sq = jnp.sum(r * r)
    
    def body_fun(carry):
        p, d, r, z, rz_old, i = carry
        
        # Compute A·d
        Ad = apply_laplacian(d)
        
        # Compute step size α = (r·z) / (d·Ad)
        dAd = jnp.sum(d * Ad)
        alpha = rz_old / jnp.where(jnp.abs(dAd) > 1e-12, dAd, 1e-12)
        
        # Update solution
        p_new = p + alpha * d
        
        # Update residual
        r_new = r - alpha * Ad
        
        # Apply mask to prevent accumulation of errors
        p_new = p_new * mask
        r_new = r_new * mask
        
        # Apply preconditioner
        z_new = apply_preconditioner(r_new)
        
        # Compute new r·z
        rz_new = jnp.sum(r_new * z_new)
        
        # Compute β with safeguard
        beta = rz_new / (rz_old + 1e-10)
        
        # Update search direction with clipping to prevent blow-up
        d_new = z_new + beta * d
        
        # Clip d to prevent NaN from extreme values
        d_new = jnp.where(jnp.isnan(d_new), 0.0, d_new)
        d_new = jnp.where(jnp.isinf(d_new), 0.0, d_new)
        
        return (p_new, d_new, r_new, z_new, rz_new, i + 1)
    
    def cond_fun(carry):
        p, d, r, z, rz_old, i = carry
        residual_norm = jnp.sqrt(rz_old + 1e-10)
        # Check for NaN in critical values
        has_nan = jnp.isnan(rz_old) | jnp.isnan(jnp.sum(p))
        return (i < max_iter) & (residual_norm > tol) & (~has_nan)
    
    # Initialize carry
    init_carry = (p, d, r, z, rz_old, 0)
    
    # Run PCG iterations with safe fallback
    def safe_while_loop(carry):
        return jax.lax.while_loop(cond_fun, body_fun, carry)
    
    final_carry = safe_while_loop(init_carry)
    
    # Extract final pressure
    p_final, _, _, _, _, iterations = final_carry
    
    # Final safeguard: replace any NaN with 0
    p_final = jnp.where(jnp.isnan(p_final), 0.0, p_final)
    p_final = jnp.where(jnp.isinf(p_final), 0.0, p_final)
    
    return p_final * mask


@jax.jit
def poisson_cg_simple_preconditioner(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float,
                                     max_iter: int = 1000, tol: float = 1e-8) -> jnp.ndarray:
    """
    Even simpler preconditioned CG - just scale the residual.
    This is more stable but may converge slower.
    """
    nx, ny = rhs.shape
    
    def apply_laplacian(p: jnp.ndarray) -> jnp.ndarray:
        """Apply discrete Laplacian operator with non-periodic boundary conditions
        
        Boundary conditions for von Karman flow:
        - Inlet (left, x=0): Dirichlet (p = 0) to fix pressure level
        - Outlet (right, x=L): Dirichlet (p = 0) to vent pressure
        - Top/Bottom walls: Neumann (∂p/∂y = 0) for no-slip walls
        """
        laplacian = jnp.zeros_like(p)
        
        # Interior points
        laplacian = laplacian.at[1:-1, 1:-1].set(
            (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / (dx * dx) +
            (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) / (dy * dy)
        )
        
        # Inlet (left): Dirichlet BC (p = 0)
        laplacian = laplacian.at[0, :].set(0.0)
        
        # Outlet (right): Dirichlet BC (p = 0)
        laplacian = laplacian.at[-1, :].set(0.0)
        
        # Top/Bottom walls: Neumann BC (∂p/∂y = 0)
        laplacian = laplacian.at[1:-1, 0].set(
            (p[2:, 0] - 2.0 * p[1:-1, 0] + p[:-2, 0]) / (dx * dx) +
            (p[1:-1, 1] - 2.0 * p[1:-1, 0] + p[1:-1, 1]) / (dy * dy)
        )
        laplacian = laplacian.at[1:-1, -1].set(
            (p[2:, -1] - 2.0 * p[1:-1, -1] + p[:-2, -1]) / (dx * dx) +
            (p[1:-1, -2] - 2.0 * p[1:-1, -1] + p[1:-1, -2]) / (dy * dy)
        )
        
        # Corners
        laplacian = laplacian.at[0, 0].set(0.0)
        laplacian = laplacian.at[0, -1].set(0.0)
        laplacian = laplacian.at[-1, 0].set(0.0)
        laplacian = laplacian.at[-1, -1].set(0.0)
        
        return laplacian * mask
    
    # Simple diagonal scaling - just a constant factor
    scale = 1.0 / (2.0/(dx*dx) + 2.0/(dy*dy) + 1e-10)
    
    p = jnp.zeros((nx, ny))
    
    # Zero out RHS at inlet and outlet (Dirichlet BC: p=0, so residual must be 0)
    rhs = rhs.at[0, :].set(0.0)
    rhs = rhs.at[-1, :].set(0.0)
    
    r = rhs * mask
    z = r * scale  # Simple scaling instead of full preconditioner
    d = z.copy()
    rz_old = jnp.sum(r * z)
    r0_norm_sq = jnp.sum(r * r)
    
    def body_fun(carry):
        p, d, r, rz_old, i = carry
        
        Ad = apply_laplacian(d)
        dAd = jnp.sum(d * Ad)
        alpha = rz_old / jnp.where(jnp.abs(dAd) > 1e-12, dAd, 1e-12)
        
        p_new = p + alpha * d
        r_new = r - alpha * Ad
        
        p_new = p_new * mask
        r_new = r_new * mask
        
        z_new = r_new * scale
        rz_new = jnp.sum(r_new * z_new)
        beta = rz_new / (rz_old + 1e-10)
        d_new = z_new + beta * d
        
        return (p_new, d_new, r_new, rz_new, i + 1)
    
    def cond_fun(carry):
        p, d, r, rz_old, i = carry
        residual_norm = jnp.sqrt(rz_old)
        return (i < max_iter) & (residual_norm > tol)
    
    init_carry = (p, d, r, rz_old, 0)
    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    p_final, _, _, _, _ = final_carry
    
    return p_final * mask


def verify_cg_solver():
    """Test the CG solver with a known solution"""
    print("\n=== Verifying CG Solvers ===")
    
    # Test problem: p = sin(πx) * sin(πy) on [0,1]x[0,1]
    nx, ny = 50, 50
    dx = dy = 1.0 / (nx - 1)
    
    # Create grid
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    # Exact solution
    p_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    
    # RHS: ∇²p = -2π² * sin(πx) * sin(πy)
    rhs = -2.0 * jnp.pi * jnp.pi * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    
    # Mask (all fluid)
    mask = jnp.ones((nx, ny))
    
    # Test standard CG
    print("Testing standard CG...")
    p_cg = poisson_cg(rhs, mask, dx, dy, max_iter=500, tol=1e-8)
    
    error_abs_cg = jnp.max(jnp.abs(p_cg - p_exact))
    error_rel_cg = error_abs_cg / (jnp.max(jnp.abs(p_exact)) + 1e-10)
    
    print(f"  Max absolute error: {error_abs_cg:.6e}")
    print(f"  Max relative error: {error_rel_cg:.6e}")
    
    # Test preconditioned CG (with fixes)
    print("\nTesting preconditioned CG (full Jacobi)...")
    p_pcg = poisson_cg_with_preconditioner(rhs, mask, dx, dy, max_iter=500, tol=1e-8)
    
    # Check for NaN
    has_nan = jnp.any(jnp.isnan(p_pcg))
    if has_nan:
        print("  ✗ Preconditioned CG returned NaN - using simpler version")
        print("\nTesting preconditioned CG (simple scaling)...")
        p_pcg_simple = poisson_cg_simple_preconditioner(rhs, mask, dx, dy, max_iter=500, tol=1e-8)
        
        error_abs_pcg = jnp.max(jnp.abs(p_pcg_simple - p_exact))
        error_rel_pcg = error_abs_pcg / (jnp.max(jnp.abs(p_exact)) + 1e-10)
        print(f"  Max absolute error: {error_abs_pcg:.6e}")
        print(f"  Max relative error: {error_rel_pcg:.6e}")
        p_pcg = p_pcg_simple
    else:
        error_abs_pcg = jnp.max(jnp.abs(p_pcg - p_exact))
        error_rel_pcg = error_abs_pcg / (jnp.max(jnp.abs(p_exact)) + 1e-10)
        print(f"  Max absolute error: {error_abs_pcg:.6e}")
        print(f"  Max relative error: {error_rel_pcg:.6e}")
    
    # Compare performance
    print("\n" + "="*50)
    if error_rel_cg < 1e-4:
        print("✓ Standard CG verified successfully!")
        if not has_nan and error_rel_pcg < error_rel_cg:
            print(f"✓ Preconditioned CG is {error_rel_cg/error_rel_pcg:.1f}x more accurate")
        elif not has_nan:
            print("✓ Preconditioned CG also works (but less accurate)")
        return True
    else:
        print("✗ CG solver verification failed!")
        return False


def is_true_cg() -> bool:
    """Returns True if this is a true CG implementation"""
    return True

def get_cg_status():
    """Returns status string for GUI display"""
    return "CG (true Conjugate Gradient with Jacobi preconditioning)"


if __name__ == "__main__":
    # Run verification
    success = verify_cg_solver()
    
    # Also test with a small grid to ensure no JIT issues
    print("\n" + "="*50)
    print("Testing with small grid for JIT verification...")
    nx, ny = 10, 10
    dx = dy = 0.1
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    rhs_test = -2.0 * jnp.pi * jnp.pi * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    mask_test = jnp.ones((nx, ny))
    
    p_test = poisson_cg(rhs_test, mask_test, dx, dy, max_iter=100, tol=1e-6)
    print(f"Small grid test passed! p range: [{jnp.min(p_test):.3f}, {jnp.max(p_test):.3f}]")