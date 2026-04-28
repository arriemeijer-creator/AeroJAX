"""
Lattice definitions for LBM (D2Q9)
"""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class D2Q9Lattice:
    """D2Q9 lattice (2D, 9 velocities)"""
    
    # Number of discrete velocities
    n_velocities: int = 9
    
    # Discrete velocities (c_i)
    # Format: [x, y] for each direction
    c: jnp.ndarray = None
    
    # Weights (w_i)
    w: jnp.ndarray = None
    
    # Opposite directions (for bounce-back)
    opposite: jnp.ndarray = None
    
    # Speed of sound squared
    cs_squared: float = 1.0 / 3.0
    
    def __post_init__(self):
        """Initialize lattice vectors and weights"""
        # D2Q9 velocities: center, E, N, W, S, NE, NW, SW, SE
        self.c = jnp.array([
            [0, 0],   # 0: center
            [1, 0],   # 1: east
            [0, 1],   # 2: north
            [-1, 0],  # 3: west
            [0, -1],  # 4: south
            [1, 1],   # 5: northeast
            [-1, 1],  # 6: northwest
            [-1, -1], # 7: southwest
            [1, -1],  # 8: southeast
        ])
        
        # D2Q9 weights
        self.w = jnp.array([
            4.0 / 9.0,   # 0: center
            1.0 / 9.0,   # 1: east
            1.0 / 9.0,   # 2: north
            1.0 / 9.0,   # 3: west
            1.0 / 9.0,   # 4: south
            1.0 / 36.0,  # 5: northeast
            1.0 / 36.0,  # 6: northwest
            1.0 / 36.0,  # 7: southwest
            1.0 / 36.0,  # 8: southeast
        ])
        
        # Opposite directions for bounce-back
        self.opposite = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    
    def get_cx(self) -> jnp.ndarray:
        """Get x-components of lattice velocities"""
        return self.c[:, 0]
    
    def get_cy(self) -> jnp.ndarray:
        """Get y-components of lattice velocities"""
        return self.c[:, 1]
