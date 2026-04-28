"""
Conjugate Gradient pressure solver for Poisson equation with mixed boundary conditions.
Works for both collocated and MAC grids.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Optional, Literal


@jax.jit(static_argnames=['flow_type'])
def laplacian_operator(p: jnp.ndarray, dx: float, dy: float, 
                       flow_type: Literal['von_karman', 'lid_driven_cavity', 'taylor_green'] = 'von_karman') -> jnp.ndarray:
    """
    Apply Laplacian operator with boundary conditions:
    - von Karman: Neumann at inlet (x=0) and walls (y=0,H), Dirichlet at outlet (x=L)
    - LDC: Neumann everywhere (pure Neumann)
    - Taylor-Green: Periodic
    """
    nx, ny = p.shape
    pad_mode = 'edge' if flow_type in ['von_karman', 'lid_driven_cavity'] else 'wrap'
    p_padded = jnp.pad(p, ((1, 1), (1, 1)), mode=pad_mode)
    
    ax = 1.0 / (dx * dx)
    ay = 1.0 / (dy * dy)
    
    laplacian = ax * (p_padded[2:, 1:-1] + p_padded[:-2, 1:-1] - 2 * p) + \
                ay * (p_padded[1:-1, 2:] + p_padded[1:-1, :-2] - 2 * p)
    
    if flow_type == 'von_karman':
        # Outlet (x = L): Dirichlet, p = 0 (already enforced in solver)
        # No need to modify laplacian here; BCs are applied in the solver via projection
        pass
    elif flow_type == 'lid_driven_cavity':
        # Pure Neumann: no modification needed, pad_mode='edge' gives ∂p/∂n=0
        pass
    # For Taylor-Green: periodic, already handled by 'wrap' padding
    
    return laplacian


@jax.jit(static_argnames=['flow_type', 'maxiter'])
def poisson_cg(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float,
               maxiter: int = 500, tol: float = 1e-6,
               flow_type: Literal['von_karman', 'lid_driven_cavity', 'taylor_green'] = 'von_karman',
               atol: float = 0.0) -> jnp.ndarray:
    """
    Conjugate Gradient solver for Poisson equation with mixed boundary conditions.
    
    Solves: ∇²p = rhs
    
    Boundary conditions:
    - von Karman: Neumann at inlet (x=0) and walls (y=0,H), Dirichlet at outlet (x=L, p=0)
    - lid_driven_cavity: Neumann everywhere (∂p/∂n=0)
    - taylor_green: Periodic
    
    Args:
        rhs: Right-hand side (divergence/dt)
        mask: Domain mask (1=fluid, 0=solid) - not used in this CG solver, but kept for API compatibility
        dx, dy: Grid spacing
        maxiter: Maximum iterations
        tol: Relative residual tolerance
        flow_type: Flow type for boundary conditions
        atol: Absolute residual tolerance (if > 0, used as well)
    
    Returns:
        p: Pressure field
    """
    nx, ny = rhs.shape
    rhs_flat = rhs.ravel()
    
    # For von Karman: modify RHS to satisfy compatibility? No, because Dirichlet at outlet fixes null space.
    # But we still need to ensure the system is consistent.
    if flow_type == 'lid_driven_cavity':
        # Pure Neumann: enforce compatibility by subtracting mean
        rhs_mean = jnp.mean(rhs)
        rhs_flat = rhs_flat - rhs_mean
    
    # Define matrix-vector product (A * x)
    def matvec(x_flat):
        x = x_flat.reshape(nx, ny)
        Ax = laplacian_operator(x, dx, dy, flow_type)
        
        # Apply boundary conditions for von Karman
        if flow_type == 'von_karman':
            # Outlet: enforce p=0 (Dirichlet) by setting rows to identity
            # This is done implicitly through the operator, but we need to zero out
            # the residual at outlet points. Instead, we modify the operator output.
            # Simpler: Set Ax at outlet to x itself (makes it an identity for those points)
            Ax = Ax.at[-1, :].set(x[-1, :])
            # Also enforce p=0 at outlet in the solution after solve
        elif flow_type == 'lid_driven_cavity':
            # Neumann: already handled by padding
            pass
        
        return Ax.ravel()
    
    # Initial guess: zero
    x0 = jnp.zeros_like(rhs_flat)
    
    # Solve using CG
    # JAX doesn't have a built-in CG that works with LinearOperator, so we implement it manually
    def cg_step(carry, _):
        x, r, p, r_norm_sq, r_init_norm = carry
        
        # Compute A*p
        Ap = matvec(p)
        
        # Compute alpha = r·r / p·Ap
        pAp = jnp.dot(p, Ap)
        alpha = r_norm_sq / (pAp + 1e-12)
        
        # Update x and r
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        
        # Compute new residual norm
        r_norm_sq_new = jnp.dot(r_new, r_new)
        
        # Compute beta for next iteration
        beta = r_norm_sq_new / (r_norm_sq + 1e-12)
        
        # Update search direction
        p_new = r_new + beta * p
        
        # Check convergence
        r_norm = jnp.sqrt(r_norm_sq_new / (nx * ny))
        
        # For simplicity, use absolute residual norm
        converged = jnp.logical_or(r_norm < tol, r_norm < atol)
        
        # Keep r_init_norm unchanged (it's set before the loop)
        return (x_new, r_new, p_new, r_norm_sq_new, r_init_norm), converged
    
    def cond_fun(carry):
        _, _, _, r_norm_sq, _ = carry
        r_norm = jnp.sqrt(r_norm_sq / (nx * ny))
        return jnp.logical_and(r_norm > tol, r_norm > atol)
    
    # Initialize
    r0 = rhs_flat - matvec(x0)
    r_norm_sq0 = jnp.dot(r0, r0)
    p0 = r0.copy()
    r_init_norm = jnp.sqrt(r_norm_sq0)
    initial_carry = (x0, r0, p0, r_norm_sq0, r_init_norm)
    
    # Run CG iterations
    final_carry, _ = jax.lax.scan(
        lambda carry, _: cg_step(carry, _),
        initial_carry,
        None,
        length=maxiter
    )
    
    x_final, _, _, _, _ = final_carry
    p = x_final.reshape(nx, ny)
    
    # Apply outlet Dirichlet BC for von Karman
    if flow_type == 'von_karman':
        p = p.at[-1, :].set(0.0)
    
    # Pin pressure to zero mean for LDC
    if flow_type == 'lid_driven_cavity':
        p = p - jnp.mean(p)
    
    return p


# A simpler, more efficient CG using scipy.sparse.linalg.cg via jax.pure_callback
# This is useful if you're on CPU and want faster convergence
try:
    from functools import partial
    from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator, cg as scipy_cg
    import scipy.sparse
    
    def poisson_cg_scipy(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float,
                         maxiter: int = 500, tol: float = 1e-6,
                         flow_type: Literal['von_karman', 'lid_driven_cavity', 'taylor_green'] = 'von_karman') -> jnp.ndarray:
        """
        CG solver using scipy's implementation via jax.pure_callback.
        Faster than manual implementation, but only works on CPU.
        """
        nx, ny = rhs.shape
        size = nx * ny
        
        def matvec_callback(x_np):
            x = jnp.array(x_np).reshape(nx, ny)
            Ax = laplacian_operator(x, dx, dy, flow_type)
            if flow_type == 'von_karman':
                Ax = Ax.at[-1, :].set(x[-1, :])
            return Ax.ravel()
        
        # Create a jax-callable wrapper
        def matvec_jax(x):
            # Use pure_callback to call numpy/scipy
            return jax.pure_callback(
                lambda x: matvec_callback(x),
                jax.ShapeDtypeStruct((size,), rhs.dtype),
                x,
                vectorized=False
            )
        
        # Solve using scipy CG (via callback)
        def solve_callback(rhs_np):
            A_op = ScipyLinearOperator((size, size), matvec=lambda x: matvec_callback(x).__array__())
            p_np, info = scipy_cg(A_op, rhs_np, maxiter=maxiter, tol=tol)
            return p_np
        
        p_flat = jax.pure_callback(
            solve_callback,
            jax.ShapeDtypeStruct((size,), rhs.dtype),
            np.array(rhs.ravel())
        )
        
        p = p_flat.reshape(nx, ny)
        
        if flow_type == 'von_karman':
            p = p.at[-1, :].set(0.0)
        elif flow_type == 'lid_driven_cavity':
            p = p - jnp.mean(p)
        
        return p
    
    POISSON_CG_AVAILABLE = True
except ImportError:
    POISSON_CG_AVAILABLE = False
    print("Warning: scipy not available, using manual CG implementation")


# Wrapper function that automatically selects the best solver
@jax.jit(static_argnames=['flow_type', 'maxiter'])
def poisson_cg_solve(rhs: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float,
                     maxiter: int = 50, tol: float = 1e-6,
                     flow_type: Literal['von_karman', 'lid_driven_cavity', 'taylor_green'] = 'von_karman') -> jnp.ndarray:
    """
    Conjugate Gradient solver for Poisson equation - automatically picks best implementation.
    """
    # Always use manual CG implementation for stability
    # scipy CG callback has compatibility issues
    return poisson_cg(rhs, mask, dx, dy, maxiter, tol, flow_type)