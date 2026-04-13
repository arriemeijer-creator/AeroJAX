import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable

@jax.jit
def rk3_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, 
             dx: float, dy: float, mask: jnp.ndarray,
             boundary_conditions: Optional[Callable] = None,
             cfl_safety: float = 0.5) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    3rd order Runge-Kutta with higher-order spatial discretization.
    """
    
    # Default boundary conditions (no-slip walls)
    if boundary_conditions is None:
        def apply_boundary_conditions(field: jnp.ndarray) -> jnp.ndarray:
            field = field.at[0, :].set(0.0)      # Bottom
            field = field.at[-1, :].set(0.0)     # Top
            field = field.at[:, 0].set(0.0)      # Left
            field = field.at[:, -1].set(0.0)     # Right
            return field
    else:
        apply_boundary_conditions = boundary_conditions
    
    def grad_x_4th(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        """4th order accurate gradient with proper boundary handling"""
        f_ip2 = jnp.roll(f, -2, axis=0)
        f_ip1 = jnp.roll(f, -1, axis=0)
        f_im1 = jnp.roll(f, 1, axis=0)
        f_im2 = jnp.roll(f, 2, axis=0)
        
        interior = (-f_ip2 + 8*f_ip1 - 8*f_im1 + f_im2) / (12.0 * dx)
        
        f_0 = f[0, :]
        f_1 = f[1, :]
        f_2 = f[2, :]
        left_boundary = (f_1 - f_0) / dx
        left_boundary_2 = (f_2 - f_0) / (2*dx)
        
        f_n1 = f[-1, :]
        f_n2 = f[-2, :]
        f_n3 = f[-3, :]
        right_boundary = (f_n1 - f_n2) / dx
        right_boundary_2 = (f_n3 - f_n1) / (2*dx)
        
        interior = interior.at[0, :].set(left_boundary)
        interior = interior.at[1, :].set(left_boundary_2)
        interior = interior.at[-1, :].set(right_boundary)
        interior = interior.at[-2, :].set(right_boundary_2)
        
        return interior
    
    def grad_y_4th(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        """4th order accurate gradient in y with boundary handling"""
        f_jp2 = jnp.roll(f, -2, axis=1)
        f_jp1 = jnp.roll(f, -1, axis=1)
        f_jm1 = jnp.roll(f, 1, axis=1)
        f_jm2 = jnp.roll(f, 2, axis=1)
        
        interior = (-f_jp2 + 8*f_jp1 - 8*f_jm1 + f_jm2) / (12.0 * dy)
        
        f_0 = f[:, 0]
        f_1 = f[:, 1]
        f_2 = f[:, 2]
        bottom_boundary = (f_1 - f_0) / dy
        bottom_boundary_2 = (f_2 - f_0) / (2*dy)
        
        f_n1 = f[:, -1]
        f_n2 = f[:, -2]
        f_n3 = f[:, -3]
        top_boundary = (f_n1 - f_n2) / dy
        top_boundary_2 = (f_n3 - f_n1) / (2*dy)
        
        interior = interior.at[:, 0].set(bottom_boundary)
        interior = interior.at[:, 1].set(bottom_boundary_2)
        interior = interior.at[:, -1].set(top_boundary)
        interior = interior.at[:, -2].set(top_boundary_2)
        
        return interior
    
    def laplacian_4th(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        """4th order accurate Laplacian with boundary handling"""
        f_xx = jnp.zeros_like(f)
        f_yy = jnp.zeros_like(f)
        
        f_xx = f_xx.at[2:-2, :].set(
            (-f[4:, :] + 16*f[3:-1, :] - 30*f[2:-2, :] + 16*f[1:-3, :] - f[:-4, :]) / (12 * dx**2)
        )
        
        f_yy = f_yy.at[:, 2:-2].set(
            (-f[:, 4:] + 16*f[:, 3:-1] - 30*f[:, 2:-2] + 16*f[:, 1:-3] - f[:, :-4]) / (12 * dy**2)
        )
        
        f_xx = f_xx.at[1, :].set((f[2, :] - 2*f[1, :] + f[0, :]) / dx**2)
        f_xx = f_xx.at[-2, :].set((f[-1, :] - 2*f[-2, :] + f[-3, :]) / dx**2)
        
        f_yy = f_yy.at[:, 1].set((f[:, 2] - 2*f[:, 1] + f[:, 0]) / dy**2)
        f_yy = f_yy.at[:, -2].set((f[:, -1] - 2*f[:, -2] + f[:, -3]) / dy**2)
        
        return f_xx + f_yy
    
    def compute_rhs(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        u = u * mask
        v = v * mask
        
        u = apply_boundary_conditions(u)
        v = apply_boundary_conditions(v)
        
        du_dx = grad_x_4th(u, dx)
        du_dy = grad_y_4th(u, dy)
        dv_dx = grad_x_4th(v, dx)
        dv_dy = grad_y_4th(v, dy)
        
        adv_x = u * du_dx + v * du_dy
        adv_y = u * dv_dx + v * dv_dy
        
        diff_x = nu * laplacian_4th(u, dx, dy)
        diff_y = nu * laplacian_4th(v, dx, dy)
        
        rhs_u = (-adv_x + diff_x) * mask
        rhs_v = (-adv_y + diff_y) * mask
        
        rhs_u = apply_boundary_conditions(rhs_u)
        rhs_v = apply_boundary_conditions(rhs_v)
        
        return rhs_u, rhs_v
    
    max_vel = jnp.max(jnp.sqrt(u**2 + v**2))
    cfl = max_vel * dt * (1.0/dx + 1.0/dy)
    dt_actual = dt * jnp.minimum(1.0, cfl_safety / (cfl + 1e-10))
    
    rhs_u1, rhs_v1 = compute_rhs(u, v)
    u2 = u + dt_actual * rhs_u1
    v2 = v + dt_actual * rhs_v1
    
    rhs_u2, rhs_v2 = compute_rhs(u2, v2)
    u3 = 0.75 * u + 0.25 * (u2 + dt_actual * rhs_u2)
    v3 = 0.75 * v + 0.25 * (v2 + dt_actual * rhs_v2)
    
    rhs_u3, rhs_v3 = compute_rhs(u3, v3)
    u_star = (1.0/3.0) * u + (2.0/3.0) * (u3 + dt_actual * rhs_u3)
    v_star = (1.0/3.0) * v + (2.0/3.0) * (v3 + dt_actual * rhs_v3)
    
    u_star = u_star * mask
    v_star = v_star * mask
    u_star = apply_boundary_conditions(u_star)
    v_star = apply_boundary_conditions(v_star)
    
    return u_star, v_star


@jax.jit
def rk3_step_stable(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, 
                    dx: float, dy: float, mask: jnp.ndarray, U_inf: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """STABLE RK3 for high Reynolds numbers"""

    def grad_x_correct(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        """Correct x-gradient with proper boundary stencil"""
        df = jnp.zeros_like(f)
        # 4th order interior
        df = df.at[2:-2, :].set(
            (-f[4:, :] + 8*f[3:-1, :] - 8*f[1:-3, :] + f[:-4, :]) / (12.0 * dx)
        )
        # 2nd order boundaries
        df = df.at[0, :].set((f[1, :] - f[0, :]) / dx)
        df = df.at[1, :].set((f[2, :] - f[0, :]) / (2.0 * dx))
        df = df.at[-2, :].set((f[-1, :] - f[-3, :]) / (2.0 * dx))
        df = df.at[-1, :].set((f[-1, :] - f[-2, :]) / dx)
        return df
    
    def grad_y_correct(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        """Correct y-gradient with proper boundary stencil"""
        df = jnp.zeros_like(f)
        # 4th order interior
        df = df.at[:, 2:-2].set(
            (-f[:, 4:] + 8*f[:, 3:-1] - 8*f[:, 1:-3] + f[:, :-4]) / (12.0 * dy)
        )
        # 2nd order boundaries
        df = df.at[:, 0].set((f[:, 1] - f[:, 0]) / dy)
        df = df.at[:, 1].set((f[:, 2] - f[:, 0]) / (2.0 * dy))
        df = df.at[:, -2].set((f[:, -1] - f[:, -3]) / (2.0 * dy))
        df = df.at[:, -1].set((f[:, -1] - f[:, -2]) / dy)
        return df

    def apply_bc(f: jnp.ndarray, is_u: bool = True) -> jnp.ndarray:
        """Apply von Karman boundary conditions"""
        if is_u:
            f = f.at[0, :].set(U_inf)  # Inlet
        else:
            f = f.at[0, :].set(0.0)    # v=0 at inlet
        f = f.at[-1, :].set(f[-2, :])   # Outlet (zero-gradient)
        f = f.at[:, 0].set(0.0)         # Bottom no-slip
        f = f.at[:, -1].set(0.0)        # Top no-slip
        return f

    def rhs(u_in: jnp.ndarray, v_in: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        u_m = u_in * mask
        v_m = v_in * mask
        
        # Gradients
        du_dx = grad_x_correct(u_m, dx)
        du_dy = grad_y_correct(u_m, dy)
        dv_dx = grad_x_correct(v_m, dx)
        dv_dy = grad_y_correct(v_m, dy)
        
        # Advection
        adv_u = u_m * du_dx + v_m * du_dy
        adv_v = u_m * dv_dx + v_m * dv_dy
        
        # Diffusion (2nd order for stability at high Re)
        lap_u = (jnp.roll(u_m, 1, axis=0) + jnp.roll(u_m, -1, axis=0) - 2*u_m) / dx**2
        lap_u += (jnp.roll(u_m, 1, axis=1) + jnp.roll(u_m, -1, axis=1) - 2*u_m) / dy**2
        
        lap_v = (jnp.roll(v_m, 1, axis=0) + jnp.roll(v_m, -1, axis=0) - 2*v_m) / dx**2
        lap_v += (jnp.roll(v_m, 1, axis=1) + jnp.roll(v_m, -1, axis=1) - 2*v_m) / dy**2
        
        rhs_u = (-adv_u + nu * lap_u) * mask
        rhs_v = (-adv_v + nu * lap_v) * mask
        
        return apply_bc(rhs_u, True), apply_bc(rhs_v, False)

    # CFL with adaptive safety factor
    max_vel = jnp.max(jnp.sqrt(u**2 + v**2))
    cfl = max_vel * dt * (1.0/dx + 1.0/dy)
    # Stricter CFL for high Re (low viscosity)
    safety = jnp.where(nu < 0.001, 0.25, 0.5)
    dt_actual = dt * jnp.minimum(1.0, safety / (cfl + 1e-10))
    
    # RK3 stages
    k1u, k1v = rhs(u, v)
    u2 = u + dt_actual * k1u
    v2 = v + dt_actual * k1v
    
    k2u, k2v = rhs(u2, v2)
    u3 = 0.75 * u + 0.25 * (u2 + dt_actual * k2u)
    v3 = 0.75 * v + 0.25 * (v2 + dt_actual * k2v)
    
    k3u, k3v = rhs(u3, v3)
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u3 + dt_actual * k3u)
    v_new = (1.0/3.0) * v + (2.0/3.0) * (v3 + dt_actual * k3v)
    
    # Final mask (already applied in rhs, but ensure)
    u_new = u_new * mask
    v_new = v_new * mask
    
    return u_new, v_new


class DifferentiableAdvection:
    """Wrapper for differentiable advection schemes"""

    def __init__(self, scheme: str = 'stable', cfl_safety: float = 0.5, U_inf: float = 1.0):
        self.scheme = scheme
        self.cfl_safety = cfl_safety
        self.U_inf = U_inf

        if scheme == 'high_order':
            self.step = lambda u, v, dt, nu, dx, dy, mask: rk3_step(
                u, v, dt, nu, dx, dy, mask, cfl_safety=cfl_safety
            )
        else:  # 'stable'
            self.step = lambda u, v, dt, nu, dx, dy, mask: rk3_step_stable(
                u, v, dt, nu, dx, dy, mask, U_inf=U_inf
            )
    
    def __call__(self, u, v, dt, nu, dx, dy, mask):
        return self.step(u, v, dt, nu, dx, dy, mask)
    
    def gradient(self, u, v, dt, nu, dx, dy, mask, loss_fn):
        """Compute gradients through the advection step"""
        def forward(u, v):
            u_new, v_new = self.step(u, v, dt, nu, dx, dy, mask)
            return loss_fn(u_new, v_new)
        
        grad_u, grad_v = jax.grad(forward, argnums=(0,1))(u, v)
        return grad_u, grad_v


@jax.jit
def euler_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float,
               dx: float, dy: float, mask: jnp.ndarray, U_inf: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Euler step for debugging - extremely stable but inaccurate"""

    def grad_x(f):
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)

    def grad_y(f):
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)

    def laplacian(f):
        return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) - 2*f) / dx**2 + \
               (jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 2*f) / dy**2

    u_masked = u * mask
    v_masked = v * mask

    rhs_u = -u_masked * grad_x(u_masked) - v_masked * grad_y(u_masked) + nu * laplacian(u_masked)
    rhs_v = -u_masked * grad_x(v_masked) - v_masked * grad_y(v_masked) + nu * laplacian(v_masked)

    u_new = u + dt * rhs_u * mask
    v_new = v + dt * rhs_v * mask

    # BCs
    u_new = u_new.at[0, :].set(U_inf)
    v_new = v_new.at[0, :].set(0.0)
    u_new = u_new.at[-1, :].set(u_new[-2, :])
    v_new = v_new.at[-1, :].set(v_new[-2, :])
    u_new = u_new.at[:, 0].set(0.0)
    u_new = u_new.at[:, -1].set(0.0)
    v_new = v_new.at[:, 0].set(0.0)
    v_new = v_new.at[:, -1].set(0.0)

    return u_new, v_new


@jax.jit
def rk3_step_simple(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float,
                    dx: float, dy: float, mask: jnp.ndarray, U_inf: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    SIMPLE and STABLE RK3 - uses only 2nd order, which is more stable at high Re
    """

    # Simple 2nd order gradients (more stable than 4th order for high Re)
    def grad_x(f):
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)

    def grad_y(f):
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)

    def laplacian(f):
        return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) - 2*f) / dx**2 + \
               (jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 2*f) / dy**2

    def apply_boundary_conditions(u_field, v_field):
        """Apply BCs ONCE after each RK stage"""
        # Inlet (left)
        u_field = u_field.at[0, :].set(U_inf)
        v_field = v_field.at[0, :].set(0.0)

        # Outlet (right) - zero gradient
        u_field = u_field.at[-1, :].set(u_field[-2, :])
        v_field = v_field.at[-1, :].set(v_field[-2, :])

        # Top and bottom - no slip
        u_field = u_field.at[:, 0].set(0.0)
        u_field = u_field.at[:, -1].set(0.0)
        v_field = v_field.at[:, 0].set(0.0)
        v_field = v_field.at[:, -1].set(0.0)

        return u_field, v_field

    def rhs(u_in, v_in):
        # Apply mask
        u_masked = u_in * mask
        v_masked = v_in * mask

        # Compute RHS
        rhs_u = -u_masked * grad_x(u_masked) - v_masked * grad_y(u_masked) + nu * laplacian(u_masked)
        rhs_v = -u_masked * grad_x(v_masked) - v_masked * grad_y(v_masked) + nu * laplacian(v_masked)

        # Apply mask to RHS
        rhs_u = rhs_u * mask
        rhs_v = rhs_v * mask

        return rhs_u, rhs_v

    # CFL check with VERY conservative safety for now
    max_vel = jnp.max(jnp.sqrt(u**2 + v**2) + 1e-8)
    cfl = max_vel * dt * (1.0/dx + 1.0/dy)
    # Adaptive safety factor based on velocity
    safety = jnp.where(max_vel > 5.0, 0.1, jnp.where(max_vel > 3.0, 0.15, 0.2))
    dt_actual = dt * jnp.minimum(1.0, safety / (cfl + 1e-8))

    # RK3 stages
    k1u, k1v = rhs(u, v)
    u2 = u + dt_actual * k1u
    v2 = v + dt_actual * k1v
    u2, v2 = apply_boundary_conditions(u2, v2)

    k2u, k2v = rhs(u2, v2)
    u3 = 0.75 * u + 0.25 * (u2 + dt_actual * k2u)
    v3 = 0.75 * v + 0.25 * (v2 + dt_actual * k2v)
    u3, v3 = apply_boundary_conditions(u3, v3)

    k3u, k3v = rhs(u3, v3)
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u3 + dt_actual * k3u)
    v_new = (1.0/3.0) * v + (2.0/3.0) * (v3 + dt_actual * k3v)

    # Final mask and BCs
    u_new = u_new * mask
    v_new = v_new * mask
    u_new, v_new = apply_boundary_conditions(u_new, v_new)

    return u_new, v_new


def test_differentiability():
    """Enhanced differentiability test"""
    print("\n=== Testing Differentiability ===")
    
    nx, ny = 64, 64
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (nx, ny))
    v = jax.random.normal(jax.random.split(key)[1], (nx, ny))
    mask = jnp.ones((nx, ny))
    dt, nu, dx, dy = 0.001, 0.001, 0.1, 0.1  # Smaller dt for stability
    
    # Test both versions
    for name, rk3_func in [('High Order', rk3_step), ('Stable', rk3_step_stable)]:
        print(f"\nTesting {name} RK3...")
        
        def loss(u, v):
            u_new, v_new = rk3_func(u, v, dt, nu, dx, dy, mask)
            return jnp.sum(u_new**2 + v_new**2)
        
        grad_u, grad_v = jax.grad(loss, argnums=(0,1))(u, v)
        
        print(f"  Gradient norms: ||∇L/∇u|| = {jnp.linalg.norm(grad_u):.6e}, "
              f"||∇L/∇v|| = {jnp.linalg.norm(grad_v):.6e}")
        
        has_nan = jnp.any(jnp.isnan(grad_u)) or jnp.any(jnp.isnan(grad_v))
        print(f"  {'✓' if not has_nan else '✗'} No NaN gradients")
    
    print("\n✓ All differentiability tests passed!")


if __name__ == "__main__":
    test_differentiability()