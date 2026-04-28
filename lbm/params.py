"""
LBM-specific parameters and configuration
"""

import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LBMSimulationParams:
    """Parameters specific to LBM simulation"""
    
    # Lattice parameters
    lattice_type: str = "D2Q9"  # Lattice type (D2Q9, D2Q7, etc.)
    tau: float = 0.6  # Relaxation time (must be > 0.5 for stability)
    omega: float = 1.0 / 0.6  # Collision frequency (1/tau)
    
    # Boundary conditions
    boundary_type: str = "bounce_back"  # bounce_back, specular, etc.
    inlet_type: str = "equilibrium"  # equilibrium, zou-he, etc.
    outlet_type: str = "equilibrium"  # equilibrium, open, etc.
    
    # Collision model
    collision_model: str = "BGK"  # BGK, MRT, TRT
    
    # Force terms (optional)
    force_x: float = 0.0
    force_y: float = 0.0
    
    # Initialization
    initial_density: float = 1.0
    initial_velocity_x: float = 0.0
    initial_velocity_y: float = 0.0
    
    # Passive scalar (dye) for visualization
    enable_dye: bool = True
    dye_diffusivity: float = 0.01
    
    def __post_init__(self):
        """Update omega when tau is set"""
        self.omega = 1.0 / self.tau
    
    @property
    def viscosity(self) -> float:
        """Compute kinematic viscosity from relaxation time"""
        cs_squared = 1.0 / 3.0  # Speed of sound squared for D2Q9
        return cs_squared * (self.tau - 0.5)
    
    @viscosity.setter
    def viscosity(self, nu: float):
        """Set relaxation time from kinematic viscosity"""
        cs_squared = 1.0 / 3.0
        self.tau = 0.5 + nu / cs_squared
        self.omega = 1.0 / self.tau
