"""
Simple 3D Navier-Stokes solver using projection method.
"""

import jax
import jax.numpy as jnp
from advection3d import rk3_step_3d, laplacian_3d
from multigrid3d import poisson_multigrid_3d


class Solver3D:
    """Simple 3D incompressible flow solver"""
    
    def __init__(self, nx, ny, nz, dx, dy, dz, nu=0.01, dt=0.001):
        """
        Initialize 3D solver.
        
        Args:
            nx, ny, nz: Grid dimensions
            dx, dy, dz: Grid spacing
            nu: Kinematic viscosity
            dt: Time step
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nu = nu
        self.dt = dt
        
        # Initialize velocity fields
        self.u = jnp.zeros((nx, ny, nz))
        self.v = jnp.zeros((nx, ny, nz))
        self.w = jnp.zeros((nx, ny, nz))
        self.p = jnp.zeros((nx, ny, nz))
    
    def divergence_3d(self, u, v, w):
        """Compute 3D divergence"""
        du_dx = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2.0 * self.dx)
        dv_dy = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2.0 * self.dy)
        dw_dz = (jnp.roll(w, -1, axis=2) - jnp.roll(w, 1, axis=2)) / (2.0 * self.dz)
        return du_dx + dv_dy + dw_dz
    
    def gradient_3d(self, p):
        """Compute 3D gradient of pressure"""
        dp_dx = (jnp.roll(p, -1, axis=0) - jnp.roll(p, 1, axis=0)) / (2.0 * self.dx)
        dp_dy = (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1)) / (2.0 * self.dy)
        dp_dz = (jnp.roll(p, -1, axis=2) - jnp.roll(p, 1, axis=2)) / (2.0 * self.dz)
        return dp_dx, dp_dy, dp_dz
    
    def step(self):
        """Perform one time step using projection method"""
        # 1. Advection-diffusion (predictor step)
        u_star, v_star, w_star = rk3_step_3d(
            self.u, self.v, self.w, self.dt, self.dx, self.dy, self.dz
        )
        
        # Add diffusion
        lap_u = laplacian_3d(u_star, self.dx, self.dy, self.dz)
        lap_v = laplacian_3d(v_star, self.dx, self.dy, self.dz)
        lap_w = laplacian_3d(w_star, self.dx, self.dy, self.dz)
        
        u_star = u_star + self.dt * self.nu * lap_u
        v_star = v_star + self.dt * self.nu * lap_v
        w_star = w_star + self.dt * self.nu * lap_w
        
        # 2. Pressure Poisson equation
        div_u_star = self.divergence_3d(u_star, v_star, w_star)
        rhs = div_u_star / self.dt
        
        self.p = poisson_multigrid_3d(rhs, self.dx, self.dy, self.dz)
        
        # 3. Projection (corrector step)
        dp_dx, dp_dy, dp_dz = self.gradient_3d(self.p)
        
        self.u = u_star - self.dt * dp_dx
        self.v = v_star - self.dt * dp_dy
        self.w = w_star - self.dt * dp_dz
        
        return self.u, self.v, self.w, self.p
    
    def get_velocity_magnitude(self):
        """Compute velocity magnitude"""
        return jnp.sqrt(self.u**2 + self.v**2 + self.w**2)
    
    def get_vorticity(self):
        """Compute vorticity magnitude"""
        # vorticity = curl of velocity
        dv_dx = (jnp.roll(self.v, -1, axis=0) - jnp.roll(self.v, 1, axis=0)) / (2.0 * self.dx)
        dw_dy = (jnp.roll(self.w, -1, axis=1) - jnp.roll(self.w, 1, axis=1)) / (2.0 * self.dy)
        dw_dx = (jnp.roll(self.w, -1, axis=0) - jnp.roll(self.w, 1, axis=0)) / (2.0 * self.dx)
        du_dz = (jnp.roll(self.u, -1, axis=2) - jnp.roll(self.u, 1, axis=2)) / (2.0 * self.dz)
        du_dy = (jnp.roll(self.u, -1, axis=1) - jnp.roll(self.u, 1, axis=1)) / (2.0 * self.dy)
        dv_dz = (jnp.roll(self.v, -1, axis=2) - jnp.roll(self.v, 1, axis=2)) / (2.0 * self.dz)
        
        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy
        
        return jnp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
