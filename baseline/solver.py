"""
Clean Baseline Navier-Stokes Solver
Optimized for cylinder flow with vortex shedding

Author: Arno Meijer
Version: 1.0 (Baseline Only)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict
import time
from dataclasses import dataclass, field

# Performance optimizations
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_debug_nans', False)

# ----------------------------------------------------------------------
# 1. Data Structures
# ----------------------------------------------------------------------

@dataclass
class GridParams:
    """Grid parameters container."""
    nx: int
    ny: int
    lx: float
    ly: float
    dx: float = field(init=False)
    dy: float = field(init=False)
    x: jnp.ndarray = field(init=False)
    y: jnp.ndarray = field(init=False)
    X: jnp.ndarray = field(init=False)
    Y: jnp.ndarray = field(init=False)
    
    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x = jnp.linspace(0, self.lx, self.nx)
        self.y = jnp.linspace(0, self.ly, self.ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')

@dataclass
class FlowParams:
    """Flow parameters container."""
    Re: float
    U_inf: float
    nu: float = field(init=False)
    
    def __post_init__(self):
        self.nu = self.U_inf * (2.0 * 0.18) / self.Re  # Cylinder diameter = 0.36

@dataclass
class GeometryParams:
    """Geometry parameters container."""
    center_x: jnp.ndarray
    center_y: jnp.ndarray
    radius: jnp.ndarray
    
    def to_dict(self) -> Dict:
        return {
            'center_x': self.center_x,
            'center_y': self.center_y,
            'radius': self.radius
        }

@dataclass
class EGCEParams:
    """EGCE parameters container (minimal for baseline)."""
    Cs: float = 0.17
    eps: float = 0.05

# ----------------------------------------------------------------------
# 2. Differential Operators (JIT-compiled)
# ----------------------------------------------------------------------

@jax.jit
def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Central difference in x-direction."""
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)

@jax.jit
def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    """Central difference in y-direction."""
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)

@jax.jit
def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """2D Laplacian using central differences (matches working baseline)."""
    return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) +
            jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 4 * f) / (dx**2)

@jax.jit
def divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Compute divergence of velocity field."""
    return grad_x(u, dx) + grad_y(v, dy)

@jax.jit
def vorticity(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Compute vorticity (scalar) from velocity."""
    return grad_x(v, dx) - grad_y(u, dy)

# ----------------------------------------------------------------------
# 3. SGS Model (Smagorinsky)
# ----------------------------------------------------------------------

@jax.jit
def sgs_stress_divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, Cs: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute divergence of SGS stress tensor using Smagorinsky model."""
    du_dx = grad_x(u, dx)
    du_dy = grad_y(u, dy)
    dv_dx = grad_x(v, dx)
    dv_dy = grad_y(v, dy)
    
    Sxx = du_dx
    Syy = dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    
    mag_S = jnp.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2))
    Delta = jnp.sqrt(dx * dy)
    nu_sgs = (Cs * Delta)**2 * mag_S
    
    tau_xx = -2.0 * nu_sgs * Sxx
    tau_yy = -2.0 * nu_sgs * Syy
    tau_xy = -2.0 * nu_sgs * Sxy
    
    div_tau_x = grad_x(tau_xx, dx) + grad_y(tau_xy, dy)
    div_tau_y = grad_x(tau_xy, dx) + grad_y(tau_yy, dy)
    
    return div_tau_x, div_tau_y

# ----------------------------------------------------------------------
# 4. Geometry Module
# ----------------------------------------------------------------------

@jax.jit
def sdf_cylinder(x: jnp.ndarray, y: jnp.ndarray, center_x: float, center_y: float, radius: float) -> jnp.ndarray:
    """Signed distance function for a cylinder."""
    return jnp.sqrt((x - center_x)**2 + (y - center_y)**2) - radius

@jax.jit
def smooth_mask(phi: jnp.ndarray, eps: float = 0.05) -> jnp.ndarray:
    """Smooth indicator function using sigmoid."""
    return jax.nn.sigmoid(phi / eps)

def create_mask_from_params(X: jnp.ndarray, Y: jnp.ndarray, params: Dict, eps: float = 0.05) -> jnp.ndarray:
    """Compute smooth mask from geometry parameters."""
    phi = sdf_cylinder(X, Y, params['center_x'], params['center_y'], params['radius'])
    return smooth_mask(phi, eps)

# ----------------------------------------------------------------------
# 5. Predictor Step (Simple Euler)
# ----------------------------------------------------------------------

@jax.jit
def predictor_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, Cs: float = 0.17) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute predictor velocity with SGS stress (matches working baseline)."""
    # Advection
    du_dx = grad_x(u, dx)
    du_dy = grad_y(u, dy)
    dv_dx = grad_x(v, dx)
    dv_dy = grad_y(v, dy)
    adv_x = u * du_dx + v * du_dy
    adv_y = u * dv_dx + v * dv_dy
    
    # Diffusion
    diff_x = nu * laplacian(u, dx, dy)
    diff_y = nu * laplacian(v, dx, dy)
    
    # SGS stress divergence
    div_tau_x, div_tau_y = sgs_stress_divergence(u, v, dx, dy, Cs)
    
    # Predictor update
    u_star = u + dt * (-adv_x + diff_x + div_tau_x)
    v_star = v + dt * (-adv_y + diff_y + div_tau_y)
    
    return u_star, v_star

# ----------------------------------------------------------------------
# 6. Pressure Poisson Solver
# ----------------------------------------------------------------------

@jax.jit
def solve_pressure_poisson(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                           dx: float, dy: float, dt: float = 0.001, max_iter: int = 15) -> jnp.ndarray:
    """JAX-compatible pressure Poisson solver using lax.while_loop (matches working baseline)."""
    div = divergence(u, v, dx, dy)
    b = div / dt  # CRITICAL: Use dt scaling like working baseline!
    
    def body_fun(carry):
        p, i = carry
        # Vectorized Jacobi iteration using roll (matches baseline exactly)
        p_new = 0.25 * (jnp.roll(p, 1, axis=0) + jnp.roll(p, -1, axis=0) + 
                        jnp.roll(p, 1, axis=1) + jnp.roll(p, -1, axis=1) - dx**2 * b)
        
        # Boundary Conditions (Neumann at walls, Dirichlet at outlet) - matches baseline
        p_new = p_new.at[0, :].set(p_new[1, :])    # Inlet
        p_new = p_new.at[-1, :].set(0.0)           # Outlet
        p_new = p_new.at[:, 0].set(p_new[:, 1])    # Bottom
        p_new = p_new.at[:, -1].set(p_new[:, -2])  # Top
        return p_new, i + 1

    def cond_fun(carry):
        p, i = carry
        return i < max_iter

    p_init = jnp.zeros((u.shape[0], u.shape[1]))
    p_final, _ = jax.lax.while_loop(cond_fun, body_fun, (p_init, 0))
    return p_final

# ----------------------------------------------------------------------
# 7. Force Calculation
# ----------------------------------------------------------------------

@jax.jit
def compute_forces(u: jnp.ndarray, v: jnp.ndarray, p: jnp.ndarray, mask: jnp.ndarray,
                   dx: float, dy: float, nu: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute drag and lift forces using momentum integration."""
    # Pressure gradient (for stress)
    dp_dx = grad_x(p, dx)
    dp_dy = grad_y(p, dy)
    
    # Velocity gradients for viscous stress
    du_dx = grad_x(u, dx)
    du_dy = grad_y(u, dy)
    dv_dx = grad_x(v, dx)
    dv_dy = grad_y(v, dy)
    
    # Stress tensor components
    sigma_xx = -p + 2.0 * nu * du_dx
    sigma_yy = -p + 2.0 * nu * dv_dy
    sigma_xy = nu * (du_dy + dv_dx)
    
    # Boundary indicator (gradient of mask)
    dm_dx = grad_x(mask, dx)
    dm_dy = grad_y(mask, dy)
    boundary_norm = jnp.sqrt(dm_dx**2 + dm_dy**2)
    
    # Integrate stress over boundary
    drag = jnp.sum((sigma_xx * dm_dx + sigma_xy * dm_dy) * boundary_norm) * dx * dy
    lift = jnp.sum((sigma_xy * dm_dx + sigma_yy * dm_dy) * boundary_norm) * dx * dy
    
    return drag, lift

# ----------------------------------------------------------------------
# 8. Main Solver Class
# ----------------------------------------------------------------------

class BaselineSolver:
    """Clean baseline Navier-Stokes solver for cylinder flow with vortex shedding."""
    
    def __init__(self,
                 grid: GridParams,
                 flow: FlowParams,
                 geom: GeometryParams,
                 egce: EGCEParams,
                 dt: float = 0.001,
                 seed: int = 42):
        
        self.grid = grid
        self.flow = flow
        self.geom = geom
        self.egce = egce
        self.dt = dt
        
        # Initialize velocity field
        self.u = jnp.ones((grid.nx, grid.ny)) * flow.U_inf
        self.v = jnp.zeros((grid.nx, grid.ny))
        self._add_initial_perturbation()
        
        # JIT-compiled step function
        self._step_jit = jax.jit(self._step)
        
        # JIT-compiled diagnostic functions
        self._vorticity = jax.jit(vorticity, static_argnums=(2, 3))
        self._divergence = jax.jit(divergence, static_argnums=(2, 3))
        
        # History storage
        self.history = {'time': [], 'ke': [], 'enstrophy': [], 'drag': [], 'lift': []}
        self.iteration = 0
        
    def _add_initial_perturbation(self):
        """Add small perturbation to trigger instability."""
        X, Y = self.grid.X, self.grid.Y
        cylinder_x = float(self.geom.center_x)
        cylinder_radius = float(self.geom.radius)
        perturbation = 0.05 * jnp.sin(2 * jnp.pi * Y / self.grid.ly) * \
                       jnp.exp(-((X - cylinder_x - 2*cylinder_radius)**2) / (2 * cylinder_radius**2))
        self.u = self.u + perturbation
        
    def _step(self, u, v):
        """Single timestep with pressure projection."""
        dx, dy = self.grid.dx, self.grid.dy
        geom_dict = self.geom.to_dict()
        
        # Geometry mask
        mask = create_mask_from_params(self.grid.X, self.grid.Y, geom_dict, self.egce.eps)
        
        # 1. Predictor with SGS (NO mask inside predictor_step)
        u_star, v_star = predictor_step(u, v, self.dt, self.flow.nu, dx, dy, mask, self.egce.Cs)
        
        # 2. Pressure projection (matches working baseline)
        div_star = divergence(u_star, v_star, dx, dy)
        p = solve_pressure_poisson(u_star, v_star, mask, dx, dy, self.dt)
        dp_dx, dp_dy = grad_x(p, dx), grad_y(p, dy)
        u_corr = u_star - self.dt * dp_dx
        v_corr = v_star - self.dt * dp_dy
        
        # 3. Apply geometry mask (once, after pressure correction)
        u_corr = u_corr * mask
        v_corr = v_corr * mask
        
        # 4. Apply boundary conditions (matches working baseline exactly)
        # Left boundary (inlet)
        u_corr = u_corr.at[0, :].set(1.0)  # U_inf = 1.0
        v_corr = v_corr.at[0, :].set(0.0)
        
        # Right boundary (outlet) - zero gradient
        u_corr = u_corr.at[-1, :].set(u_corr.at[-2, :].get())
        v_corr = v_corr.at[-1, :].set(v_corr.at[-2, :].get())
        
        # Top and bottom boundaries - slip (no penetration, free slip tangentially)
        v_corr = v_corr.at[:, 0].set(0.0)
        v_corr = v_corr.at[:, -1].set(0.0)
        # No condition on u at top/bottom - free slip
        
        return u_corr, v_corr, mask
    
    def step(self):
        """Perform one timestep and update fields."""
        self.u, self.v, mask = self._step_jit(self.u, self.v)
        self.iteration += 1
        
        # Compute diagnostics
        vort = self._vorticity(self.u, self.v, self.grid.dx, self.grid.dy)
        ke = 0.5 * jnp.sum(mask * (self.u**2 + self.v**2)) * self.grid.dx * self.grid.dy
        enst = 0.5 * jnp.sum(mask * vort**2) * self.grid.dx * self.grid.dy
        
        # Pressure for force calculation
        p = solve_pressure_poisson(self.u, self.v, mask, self.grid.dx, self.grid.dy, self.dt)
        drag, lift = compute_forces(self.u, self.v, p, mask, self.grid.dx, self.grid.dy, self.flow.nu)
        
        # Store history
        self.history['time'].append(self.iteration * self.dt)
        self.history['ke'].append(float(ke))
        self.history['enstrophy'].append(float(enst))
        self.history['drag'].append(float(drag))
        self.history['lift'].append(float(lift))
        
        return self.u, self.v, vort, ke, enst, drag, lift
    
    def run_simulation(self, n_steps: int = 10000, verbose: bool = True):
        """Run simulation for specified number of steps."""
        print(f"\n=== Running Baseline Simulation ({n_steps} steps) ===")
        t0 = time.time()
        
        for step in range(n_steps):
            u, v, vort, ke, enst, drag, lift = self.step()
            
            if verbose and step % 500 == 0:
                elapsed = time.time() - t0
                speed = step / elapsed if elapsed > 0 else 0
                print(f"Step {step:6d}, Time={step*self.dt:.3f}, "
                      f"KE={ke:.6f}, Drag={drag:.4f}, Lift={lift:.4f}, "
                      f"Speed={speed:.1f} steps/sec")
        
        elapsed = time.time() - t0
        print(f"\nSimulation completed: {n_steps} steps in {elapsed:.1f}s "
              f"({n_steps/elapsed:.1f} steps/sec)")
        
        return u, v

# ----------------------------------------------------------------------
# 9. Main Execution
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Clean Baseline Navier-Stokes Solver")
    print("=" * 70)
    
    # Initialize parameters
    grid = GridParams(nx=512, ny=96, lx=20.0, ly=4.5)  # High resolution for vortex shedding
    flow = FlowParams(Re=150.0, U_inf=1.0)  # Optimal Reynolds number
    geom = GeometryParams(center_x=jnp.array(2.5), center_y=jnp.array(2.25), radius=jnp.array(0.18))
    egce = EGCEParams(Cs=0.17, eps=0.05)
    
    print(f"\nConfiguration:")
    print(f"  Grid: {grid.nx} × {grid.ny}")
    print(f"  Domain: {grid.lx} × {grid.ly}")
    print(f"  Re = {flow.Re:.1f}, U_inf = {flow.U_inf}")
    print(f"  dt = {0.001}, ν = {flow.nu:.6f}")
    print(f"  Cylinder: center=({float(geom.center_x):.1f}, {float(geom.center_y):.1f}), "
          f"radius={float(geom.radius):.3f}")
    
    # Create solver
    solver = BaselineSolver(grid, flow, geom, egce, dt=0.001)
    
    # Run simulation
    u, v = solver.run_simulation(n_steps=20000, verbose=True)
    
    # Print final results
    print("\n" + "=" * 70)
    print("Final Results:")
    print("=" * 70)
    print(f"Final drag coefficient: {solver.history['drag'][-1]:.4f}")
    print(f"Final lift coefficient: {solver.history['lift'][-1]:.4f}")
    print(f"Kinetic energy: {solver.history['ke'][-1]:.6f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
