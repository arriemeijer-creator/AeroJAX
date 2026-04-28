import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def compute_cfl_timestep(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dx: float,
    dy: float,
    nu: float,
    mask: jnp.ndarray = None,
    cfl_target: float = 0.3,
    dt_min: float = 1e-4,
    dt_max: float = 0.01
) -> float:
    """
    Compute adaptive timestep based on CFL condition (convective) and diffusion limit (viscous).
    This is a simplified version of the JAXFluids approach adapted for incompressible Navier-Stokes.

    Args:
        u: x-velocity field
        v: y-velocity field
        dx: grid spacing in x
        dy: grid spacing in y
        nu: kinematic viscosity
        mask: obstacle mask (1 in fluid, 0 in solid)
        cfl_target: target CFL number
        dt_min: minimum timestep
        dt_max: maximum timestep

    Returns:
        dt: computed timestep
    """
    EPS = 1e-12

    # Convective contribution (CFL condition)
    # dt_conv = min(dx, dy) / max(|u| + |v|)
    min_cell_size = jnp.minimum(dx, dy)

    # Handle staggered (MAC) grid where u and v have different shapes
    if u.shape != v.shape:
        # For MAC grid: u is (nx+1, ny), v is (nx, ny+1)
        # Compute max velocity separately and take the maximum
        max_u = jnp.max(jnp.abs(u))
        max_v = jnp.max(jnp.abs(v))
        max_velocity = jnp.maximum(max_u, max_v)
    else:
        # Collocated grid
        max_velocity = jnp.max(jnp.sqrt(u**2 + v**2))

    dt_convective = min_cell_size / (max_velocity + EPS)

    # Viscous contribution (diffusion limit)
    # dt_viscous = const * min(dx, dy)^2 / nu
    # For explicit diffusion, the stability limit is dt <= dx^2 / (4*nu) in 2D
    const = 0.25  # Stability constant for explicit diffusion
    min_cell_size_squared = min_cell_size * min_cell_size
    dt_viscous = const * min_cell_size_squared / (nu + EPS)

    # Take the minimum of convective and viscous limits
    dt = jnp.minimum(dt_convective, dt_viscous)

    # Apply CFL target
    dt *= cfl_target

    # Clamp to [dt_min, dt_max]
    dt = jnp.maximum(dt_min, jnp.minimum(dt_max, dt))

    return dt


class CFLAdaptiveController:
    """
    Simple CFL-based adaptive timestep controller.
    Unlike the PID-based approach, this computes timestep directly from physical constraints
    (CFL and diffusion limits) without feedback loops, avoiding JIT compilation issues.
    """

    def __init__(self, cfl_target: float = 0.3, dt_min: float = 1e-4, dt_max: float = 0.01):
        self.cfl_target = cfl_target
        self.dt_min = dt_min
        self.dt_max = dt_max
        # Dummy attributes for compatibility with PID controller interface
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float,
               nu: float, mask: jnp.ndarray = None) -> float:
        """
        Compute new timestep based on current velocity field.

        Args:
            u: x-velocity field
            v: y-velocity field
            dx: grid spacing in x
            dy: grid spacing in y
            nu: kinematic viscosity
            mask: obstacle mask (optional)

        Returns:
            dt: computed timestep
        """
        # Convert to JAX arrays if needed
        u = jnp.asarray(u)
        v = jnp.asarray(v)

        # Compute timestep using CFL condition
        dt = compute_cfl_timestep(
            u, v, dx, dy, nu, mask,
            self.cfl_target, self.dt_min, self.dt_max
        )

        return float(dt)
