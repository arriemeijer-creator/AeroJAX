"""
Refactored Navier-Stokes Solver Package

This package contains the modular components of the CFD solver,
separated by concern for better maintainability.
"""

from .config import configure_jax
from .params import (
    SimState, GridParams, FlowParams, FlowConstraints, GeometryParams,
    CavityGeometryParams, ChannelGeometryParams, BackwardStepGeometryParams,
    TaylorGreenGeometryParams, SimulationParams
)
from .operators import (
    grad_x, grad_y, grad_x_nonperiodic,
    laplacian, laplacian_nonperiodic_x,
    divergence, divergence_nonperiodic,
    vorticity, vorticity_nonperiodic,
    scalar_advection_diffusion_periodic, scalar_advection_diffusion_nonperiodic
)
from .les_models import (
    compute_strain_rate, box_filter_2d,
    dynamic_smagorinsky, constant_smagorinsky
)
from .boundary_conditions import (
    apply_cavity_boundary_conditions, create_cavity_mask,
    apply_taylor_green_boundary_conditions, create_taylor_green_mask,
    apply_backward_step_boundary_conditions
)
from .geometry import (
    sdf_cylinder, smooth_mask, create_mask_from_params
)
from .brinkman import (
    apply_brinkman_penalization,
    apply_brinkman_penalization_ramped,
    apply_brinkman_penalization_mild,
    apply_brinkman_penalization_consistent
)
from .metrics import (
    compute_forces, get_airfoil_surface_mask,
    find_stagnation_point, find_separation_point
)
from .solver import BaselineSolver, update_dt_pure, step_pure

__all__ = [
    # Configuration
    'configure_jax',
    # Parameters
    'SimState', 'GridParams', 'FlowParams', 'FlowConstraints', 'GeometryParams',
    'CavityGeometryParams', 'ChannelGeometryParams', 'BackwardStepGeometryParams',
    'TaylorGreenGeometryParams', 'SimulationParams',
    # Operators
    'grad_x', 'grad_y', 'grad_x_nonperiodic',
    'laplacian', 'laplacian_nonperiodic_x',
    'divergence', 'divergence_nonperiodic',
    'vorticity', 'vorticity_nonperiodic',
    'scalar_advection_diffusion_periodic', 'scalar_advection_diffusion_nonperiodic',
    # LES Models
    'compute_strain_rate', 'box_filter_2d',
    'dynamic_smagorinsky', 'constant_smagorinsky',
    # Boundary Conditions
    'apply_cavity_boundary_conditions', 'create_cavity_mask',
    'apply_taylor_green_boundary_conditions', 'create_taylor_green_mask',
    'apply_backward_step_boundary_conditions',
    # Geometry
    'sdf_cylinder', 'smooth_mask', 'create_mask_from_params',
    # Brinkman
    'apply_brinkman_penalization',
    'apply_brinkman_penalization_ramped',
    'apply_brinkman_penalization_mild',
    'apply_brinkman_penalization_consistent',
    # Metrics
    'compute_forces', 'get_airfoil_surface_mask',
    'find_stagnation_point', 'find_separation_point',
    # Solver
    'BaselineSolver', 'update_dt_pure', 'step_pure',
]
