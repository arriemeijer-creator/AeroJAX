"""
Lattice Boltzmann Method (LBM) Solver Package

This package contains the LBM implementation as an alternative to the
Navier-Stokes solver, with a compatible interface for seamless switching.
"""

from .params import LBMSimulationParams
from .lattice import D2Q9Lattice
from .collision import bgk_collision, equilibrium
from .streaming import stream
from .boundary import apply_bounce_back, apply_inlet_outlet
from .operators import macroscopic_variables, compute_velocity, compute_density
from .solver import LBMSolver

__all__ = [
    'LBMSimulationParams',
    'D2Q9Lattice',
    'bgk_collision',
    'equilibrium',
    'stream',
    'apply_bounce_back',
    'apply_inlet_outlet',
    'macroscopic_variables',
    'compute_velocity',
    'compute_density',
    'LBMSolver',
]
