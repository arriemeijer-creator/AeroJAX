"""
Pure functions for the Navier-Stokes solver.
These functions are stateless and JIT-compilable.
"""

from .step import step_pure
from .dt_controller import update_dt_pure
from .corner_smoothing import apply_corner_smooth_inlet, apply_corner_smooth_pressure_gradient

__all__ = [
    'step_pure',
    'update_dt_pure',
    'apply_corner_smooth_inlet',
    'apply_corner_smooth_pressure_gradient'
]
