"""
Particle system for Lagrangian tracer particles.
Particles move with the velocity field and are rendered as scatter points.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional


@jax.jit
def interpolate_velocity(x: jnp.ndarray, y: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray,
                        X: jnp.ndarray, Y: jnp.ndarray, dx: float, dy: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Bilinear interpolation of velocity at particle positions.
    
    Args:
        x, y: Particle positions (arrays)
        u, v: Velocity fields on grid
        X, Y: Grid coordinate arrays
        dx, dy: Grid spacing
    
    Returns:
        u_interp, v_interp: Interpolated velocities at particle positions
    """
    nx, ny = u.shape
    lx = X[0, -1] - X[0, 0]
    ly = Y[-1, 0] - Y[0, 0]
    
    # Convert physical coordinates to grid indices
    i = (x - X[0, 0]) / dx
    j = (y - Y[0, 0]) / dy
    
    # Clamp to grid bounds
    i = jnp.clip(i, 0, nx - 1.001)
    j = jnp.clip(j, 0, ny - 1.001)
    
    # Integer indices
    i0 = jnp.floor(i).astype(int)
    j0 = jnp.floor(j).astype(int)
    i1 = jnp.minimum(i0 + 1, nx - 1)
    j1 = jnp.minimum(j0 + 1, ny - 1)
    
    # Fractional parts
    fx = i - i0
    fy = j - j0
    
    # Bilinear interpolation for u
    u00 = u[i0, j0]
    u01 = u[i0, j1]
    u10 = u[i1, j0]
    u11 = u[i1, j1]
    u_interp = (1 - fx) * (1 - fy) * u00 + fx * (1 - fy) * u10 + (1 - fx) * fy * u01 + fx * fy * u11
    
    # Bilinear interpolation for v
    v00 = v[i0, j0]
    v01 = v[i0, j1]
    v10 = v[i1, j0]
    v11 = v[i1, j1]
    v_interp = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11
    
    return u_interp, v_interp


@jax.jit
def advect_particles(particles: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray,
                     X: jnp.ndarray, Y: jnp.ndarray, dx: float, dy: float, dt: float) -> jnp.ndarray:
    """
    Advect particles using RK2 integration.
    
    Args:
        particles: Array of shape (N, 2) with particle positions [x, y]
        u, v: Velocity fields
        X, Y: Grid coordinate arrays
        dx, dy: Grid spacing
        dt: Time step
    
    Returns:
        Updated particle positions
    """
    # RK2 integration
    # First half-step
    u1, v1 = interpolate_velocity(particles[:, 0], particles[:, 1], u, v, X, Y, dx, dy)
    x_half = particles[:, 0] + 0.5 * dt * u1
    y_half = particles[:, 1] + 0.5 * dt * v1
    
    # Full step
    u2, v2 = interpolate_velocity(x_half, y_half, u, v, X, Y, dx, dy)
    x_new = particles[:, 0] + dt * u2
    y_new = particles[:, 1] + dt * v2
    
    return jnp.stack([x_new, y_new], axis=1)


class ParticleSystem:
    """Manages Lagrangian tracer particles for flow visualization."""
    
    def __init__(self, max_particles: int = 1000):
        """
        Initialize particle system.
        
        Args:
            max_particles: Maximum number of particles to track
        """
        self.max_particles = max_particles
        self.particles = np.zeros((0, 2), dtype=np.float32)  # Current particle positions
        self.particle_ages = np.zeros(0, dtype=np.float32)  # Particle ages for lifetime
        self.max_age = 10.0  # Maximum particle lifetime in seconds
        self.enabled = False
        
    def inject_particles(self, x: float, y: float, count: int = 10, spread: float = 0.1):
        """
        Inject particles at a location with random spread.
        
        Args:
            x, y: Center position for injection
            count: Number of particles to inject
            spread: Random spread radius
        """
        if not self.enabled:
            return
            
        # Generate random positions around center
        new_x = x + np.random.uniform(-spread, spread, count)
        new_y = y + np.random.uniform(-spread, spread, count)
        new_particles = np.stack([new_x, new_y], axis=1)
        new_ages = np.zeros(count, dtype=np.float32)
        
        # Add to existing particles
        if len(self.particles) + count > self.max_particles:
            # Remove oldest particles if at capacity
            remove_count = len(self.particles) + count - self.max_particles
            self.particles = self.particles[remove_count:]
            self.particle_ages = self.particle_ages[remove_count:]
        
        self.particles = np.vstack([self.particles, new_particles])
        self.particle_ages = np.concatenate([self.particle_ages, new_ages])
        
    def update(self, u: np.ndarray, v: np.ndarray, X: np.ndarray, Y: np.ndarray,
               dx: float, dy: float, dt: float, domain_bounds: Tuple[float, float, float, float]):
        """
        Update particle positions using velocity field.
        
        Args:
            u, v: Velocity fields
            X, Y: Grid coordinate arrays
            dx, dy: Grid spacing
            dt: Time step
            domain_bounds: (x_min, x_max, y_min, y_max)
        """
        if not self.enabled or len(self.particles) == 0:
            return
            
        x_min, x_max, y_min, y_max = domain_bounds
        
        # Convert to JAX arrays for JIT compilation
        particles_jax = jnp.array(self.particles)
        u_jax = jnp.array(u)
        v_jax = jnp.array(v)
        X_jax = jnp.array(X)
        Y_jax = jnp.array(Y)
        
        # Advect particles
        new_particles = advect_particles(particles_jax, u_jax, v_jax, X_jax, Y_jax, dx, dy, dt)
        
        # Convert back to numpy
        self.particles = np.array(new_particles)
        
        # Update ages
        self.particle_ages += dt
        
        # Remove particles that are out of bounds or too old
        in_bounds = (self.particles[:, 0] >= x_min) & (self.particles[:, 0] <= x_max) & \
                    (self.particles[:, 1] >= y_min) & (self.particles[:, 1] <= y_max)
        not_too_old = self.particle_ages < self.max_age
        valid = in_bounds & not_too_old
        
        self.particles = self.particles[valid]
        self.particle_ages = self.particle_ages[valid]
        
    def clear(self):
        """Clear all particles."""
        self.particles = np.zeros((0, 2), dtype=np.float32)
        self.particle_ages = np.zeros(0, dtype=np.float32)
        
    def get_positions(self) -> np.ndarray:
        """Get current particle positions."""
        return self.particles
