"""
Main solver module - imports BaselineSolver from simulation submodule.
Standalone differentiable functions (differentiable_rollout, design_loss) remain here.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

# Import from local modules
from .params import SimState, GridParams, FlowParams, GeometryParams, SimulationParams
from .pure_functions import step_pure, update_dt_pure, apply_corner_smooth_inlet, apply_corner_smooth_pressure_gradient

# Import BaselineSolver from simulation submodule
from .simulation import BaselineSolver

# Check availability of pressure solvers
try:
    from pressure_solvers.cg import poisson_cg_solve
    CG_PRESSURE_AVAILABLE = True
except ImportError:
    CG_PRESSURE_AVAILABLE = False

try:
    from pressure_solvers.FFT import poisson_fft_solve
    FFT_PRESSURE_AVAILABLE = True
except ImportError:
    FFT_PRESSURE_AVAILABLE = False

# Re-export for backward compatibility
__all__ = ['BaselineSolver', 'differentiable_rollout', 'design_loss', 'step_pure', 'update_dt_pure', 'CG_PRESSURE_AVAILABLE', 'FFT_PRESSURE_AVAILABLE']


def differentiable_rollout(initial_state: SimState, mask: jnp.ndarray, params: dict, num_steps: int = 200):
    """
    Differentiable rollout using step_pure for inverse design optimization.
    
    Args:
        initial_state: Starting simulation state
        mask: Domain mask (1=fluid, 0=solid)
        params: Dictionary of parameters (could contain design variables)
        num_steps: Number of timesteps to simulate
        
    Returns:
        final_state: Final simulation state
        history: Array of intermediate states for gradient computation
    """
    # Extract parameters from dict with defaults
    dx = params.get('dx', 0.01)
    dy = params.get('dy', 0.01)
    nu = params.get('nu', 0.003)
    U_inf = params.get('U_inf', 2.0)
    use_les = params.get('use_les', False)
    smagorinsky_constant = params.get('smagorinsky_constant', 0.1)
    weno_epsilon = params.get('weno_epsilon', 1e-6)
    eps = params.get('eps', 0.01)
    adaptive_dt = params.get('adaptive_dt', False)
    
    def step(carry, _):
        state = carry
        # Call the pure step function with parameters
        nn_model = params.get('nn_pressure_model', None)
        sdf = params.get('sdf', None)  # Get SDF from params if available
        new_state = step_pure(
            state, mask, sdf=sdf, dx=dx, dy=dy, nu=nu, U_inf=U_inf,
            use_les=use_les, smagorinsky_constant=smagorinsky_constant,
            weno_epsilon=weno_epsilon, eps=eps, adaptive_dt=adaptive_dt,
            nn_pressure_model=nn_model
        )
        return new_state, (new_state.u, new_state.v)
    
    # Run differentiable scan
    final_state, history = jax.lax.scan(step, initial_state, None, length=num_steps)
    return final_state, history


def design_loss(params: dict, initial_state: SimState, mask: jnp.ndarray, target_cl: float = 1.0):
    """
    Loss function for inverse design optimization.
    
    Args:
        params: Design parameters (e.g., airfoil shape parameters)
        initial_state: Starting simulation state
        mask: Domain mask
        target_cl: Target coefficient of lift
        
    Returns:
        loss: Mean squared error in coefficient of lift
    """
    # Run differentiable rollout
    final_state, _ = differentiable_rollout(initial_state, mask, params, num_steps=200)
    
    # Compute lift coefficient from final state
    # Use existing force computation if available
    try:
        from .metrics import compute_forces_circulation
        chord = params.get('chord', 3.0)
        cl, cd = compute_forces_circulation(
            final_state.u, final_state.v, mask, 
            params.get('dx', 0.01), params.get('dy', 0.01),
            U_inf, chord
        )
    except ImportError:
        # Fallback: simple momentum-based lift estimate
        u_avg = jnp.mean(final_state.u[mask == 1])
        cl = u_avg / U_inf  # Normalized lift estimate
    
    # Compute loss
    loss = (cl - target_cl)**2
    return loss


# Gradient computation for optimization
grad_design_loss = jax.grad(design_loss)
