"""
Inverse Design Optimizer using JAX gradients
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from .config import InverseDesignConfig, OptimizationGoals, AirfoilConfig

# Set CPU-only before importing JAX to avoid cloud_tpu_init DLL issues on Windows
os.environ['JAX_PLATFORMS'] = 'cpu'

# Lazy imports to avoid DLL initialization issues
jax = None
jnp = None
SOLVER_AVAILABLE = False

def _ensure_jax_loaded():
    """Lazy load JAX only when needed"""
    global jax, jnp, SOLVER_AVAILABLE
    if jax is None:
        try:
            import jax as _jax
            import jax.numpy as _jnp
            jax = _jax
            jnp = _jnp

            # Import solver components for differentiable rollout
            try:
                from solver import step_pure, SimState
                from solver.geometry import create_mask_from_params
                from solver.metrics import compute_forces
                SOLVER_AVAILABLE = True
            except ImportError:
                SOLVER_AVAILABLE = False
        except ImportError as e:
            print(f"Warning: JAX import failed: {e}")
            SOLVER_AVAILABLE = False


@dataclass
class OptimizationConfig:
    """Configuration for optimization"""
    max_iterations: int = 100
    learning_rate: float = 0.01
    convergence_threshold: float = 1e-6
    use_adaptive_lr: bool = True
    lr_decay: float = 0.99
    num_simulation_steps: int = 50  # Number of CFD steps per optimization iteration


class InverseDesigner:
    """
    Differentiable inverse design optimizer for airfoil shape optimization
    """
    
    def __init__(self, solver, config: InverseDesignConfig):
        self.solver = solver
        self.config = config
        self.optimization_config = OptimizationConfig(
            max_iterations=config.max_iterations,
            learning_rate=config.learning_rate,
            convergence_threshold=config.convergence_threshold
        )
        
        # Optimization state
        self.iteration = 0
        self.best_loss = float('inf')
        self.best_params = None
        self.history = {
            'loss': [],
            'cl': [],
            'cd': [],
            'strouhal': [],
            'cl_error': [],
            'cd_error': [],
            'strouhal_error': []
        }
        
        # Design parameters (to be optimized) - stored as numpy arrays initially
        self.design_params = np.array([
            0.02,      # camber
            0.4,       # camber_position
            0.12,      # thickness
            0.0        # angle_of_attack
        ])
        
        # Store solver state for gradient computation
        self.initial_state = None
        self.grad_loss_fn = None
        self.loss_fn = None
        
    def _get_initial_state(self):
        """Extract initial state from solver for gradient computation"""
        if self.solver is None:
            return None
        
        _ensure_jax_loaded()
        
        # Create SimState from solver's current state
        try:
            return SimState(
                u=jnp.array(self.solver.state.u),
                v=jnp.array(self.solver.state.v),
                p=jnp.array(self.solver.state.p),
                u_prev=jnp.array(self.solver.state.u_prev),
                v_prev=jnp.array(self.solver.state.v_prev),
                c=jnp.array(self.solver.state.c),
                dt=float(self.solver.state.dt),
                iteration=int(self.solver.state.iteration),
                integral=float(self.solver.state.integral),
                prev_error=float(self.solver.state.prev_error)
            )
        except Exception as e:
            print(f"Warning: Could not extract initial state: {e}")
            return None
    
    def _params_to_geometry(self, design_params):
        """Convert design parameters to geometry parameters for mask generation"""
        camber, camber_pos, thickness, aoa = design_params
        
        # Create geometry params based on airfoil parameters
        # For NACA airfoil: camber, camber_position, thickness, angle_of_attack
        from solver.params import GeometryParams
        
        # Update solver's simulation parameters with new design params
        if hasattr(self.solver, 'sim_params'):
            self.solver.sim_params.naca_camber = float(camber)
            self.solver.sim_params.naca_camber_position = float(camber_pos)
            self.solver.sim_params.naca_thickness = float(thickness)
            self.solver.sim_params.naca_angle = float(aoa)
        
        return GeometryParams(
            center_x=self.config.airfoil.position_x,
            center_y=self.config.airfoil.position_y,
            radius=0.18,  # Default cylinder radius as fallback
            obstacle_type='naca_airfoil'
        )
    
    def _differentiable_rollout(self, design_params, initial_state):
        """
        Differentiable CFD rollout that computes Cl and Cd from design parameters.
        This function is JIT-compiled and differentiable with respect to design_params.
        """
        _ensure_jax_loaded()
        
        # Convert to JAX array if needed
        if not isinstance(design_params, jnp.ndarray):
            design_params = jnp.array(design_params)
        
        num_steps = self.optimization_config.num_simulation_steps
        
        # Convert design params to geometry
        camber, camber_pos, thickness, aoa = design_params
        
        # Create mask based on design parameters
        # Note: In a full implementation, this would generate NACA airfoil mask
        # For now, we use a simplified parameterized mask
        nx, ny = self.config.grid.nx, self.config.grid.ny
        lx, ly = self.config.grid.Lx, self.config.grid.Ly
        dx, dy = lx / nx, ly / ny
        
        # Create a parameterized mask (simplified for differentiability)
        x = jnp.linspace(0, lx, nx)
        y = jnp.linspace(0, ly, ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        # Simple parameterized obstacle mask (ellipse-like shape)
        # This is a placeholder - real implementation would use NACA airfoil generation
        center_x = self.config.airfoil.position_x
        center_y = self.config.airfoil.position_y
        
        # Ellipse parameters derived from design params
        a = 0.3 + thickness  # semi-major axis
        b = 0.1 + thickness * 0.5  # semi-minor axis
        
        # Rotate ellipse based on angle of attack
        angle = jnp.radians(aoa)
        X_rot = (X - center_x) * jnp.cos(angle) - (Y - center_y) * jnp.sin(angle)
        Y_rot = (X - center_x) * jnp.sin(angle) + (Y - center_y) * jnp.cos(angle)
        
        # Create mask (1 = fluid, 0 = solid)
        mask = 1.0 - jnp.exp(-((X_rot**2) / (a**2) + (Y_rot**2) / (b**2)))
        mask = jnp.where(mask > 0.5, 1.0, mask)
        
        # Run simulation steps
        state = initial_state
        nu = self.config.flow.nu
        U_inf = self.config.flow.u_inlet
        
        for _ in range(num_steps):
            state = step_pure(
                state=state,
                mask=mask,
                dx=dx,
                dy=dy,
                nu=nu,
                U_inf=U_inf,
                flow_type=self.config.flow.flow_type,
                advection_scheme='rk3',
                pressure_solver='cg_masked',
                use_les=False,
                eps=0.01,
                adaptive_dt=False
            )
        
        # Compute forces (Cl, Cd) from final state
        # Simplified force computation for differentiability
        u, v = state.u, state.v
        
        # Compute momentum flux around obstacle (simplified)
        # Integrate pressure and viscous forces
        # This is a placeholder - real implementation would use proper force integration
        
        # Simplified Cl and Cd based on velocity field
        # Cl ~ circulation (integral of vorticity)
        # Cd ~ drag (momentum deficit)
        
        # Compute vorticity
        dv_dx = (jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0)) / (2 * dx)
        du_dy = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * dy)
        vorticity = dv_dx - du_dy
        
        # Circulation (integral of vorticity)
        circulation = jnp.sum(vorticity * mask) * dx * dy
        
        # Cl = circulation / (0.5 * U_inf * chord_length)
        chord_length = self.config.airfoil.chord_length
        cl = circulation / (0.5 * U_inf * chord_length)
        
        # Cd ~ momentum deficit in wake
        u_deficit = U_inf - u
        momentum_deficit = jnp.sum(u_deficit * mask) * dx * dy
        cd = momentum_deficit / (0.5 * U_inf**2 * chord_length)
        
        return cl, cd
    
    def _differentiable_loss(self, design_params):
        """
        Differentiable loss function for JAX autodiff.
        Computes loss by running CFD simulation and comparing to targets.
        """
        _ensure_jax_loaded()
        
        # Convert to JAX array if needed
        if not isinstance(design_params, jnp.ndarray):
            design_params = jnp.array(design_params)
        
        if not SOLVER_AVAILABLE or self.initial_state is None:
            # Fallback to simple analytical loss if solver not available
            camber, camber_pos, thickness, aoa = design_params
            goals = self.config.goals
            
            # Simple surrogate model for Cl and Cd
            cl = 0.5 + 2.0 * camber * 100 + 0.1 * jnp.sin(jnp.radians(aoa))
            cd = 0.01 + 0.5 * thickness * 100 + 0.01 * (camber_pos - 0.4)**2
            
            loss = 0.0
            if goals.target_cl is not None:
                loss += goals.cl_weight * (cl - goals.target_cl) ** 2
            if goals.target_cd is not None:
                loss += goals.cd_weight * (cd - goals.target_cd) ** 2
            
            # Shape regularization
            if goals.shape_regularization > 0:
                loss += goals.shape_regularization * (
                    jnp.abs(camber) + 
                    jnp.abs(thickness - 0.12) + 
                    jnp.abs(camber_pos - 0.4)
                )
            
            return loss
        
        # Run differentiable rollout
        cl, cd = self._differentiable_rollout(design_params, self.initial_state)
        
        # Compute loss
        goals = self.config.goals
        loss = 0.0
        
        if goals.target_cl is not None:
            cl_error = (cl - goals.target_cl) ** 2
            loss += goals.cl_weight * cl_error
        
        if goals.target_cd is not None:
            cd_error = (cd - goals.target_cd) ** 2
            loss += goals.cd_weight * cd_error
        
        # Shape regularization
        camber, camber_pos, thickness, aoa = design_params
        if goals.shape_regularization > 0:
            shape_penalty = (
                jnp.abs(camber) +
                jnp.abs(thickness - 0.12) +
                jnp.abs(camber_pos - 0.4)
            )
            loss += goals.shape_regularization * shape_penalty
        
        return loss
    
    def compute_loss(self, cl: float, cd: float, strouhal: Optional[float] = None) -> float:
        """
        Compute loss based on optimization goals (non-differentiable version for reporting)
        """
        goals = self.config.goals
        loss = 0.0
        
        if goals.target_cl is not None:
            cl_error = (cl - goals.target_cl) ** 2
            loss += goals.cl_weight * cl_error
            self.history['cl_error'].append(float(cl_error))
        
        if goals.target_cd is not None:
            cd_error = (cd - goals.target_cd) ** 2
            loss += goals.cd_weight * cd_error
            self.history['cd_error'].append(float(cd_error))
        
        if goals.target_strouhal is not None and strouhal is not None:
            strouhal_error = (strouhal - goals.target_strouhal) ** 2
            loss += goals.strouhal_weight * strouhal_error
            self.history['strouhal_error'].append(float(strouhal_error))
        
        # Shape regularization (penalize extreme shapes)
        if goals.shape_regularization > 0:
            camber, camber_pos, thickness, aoa = self.design_params
            shape_penalty = (
                jnp.abs(camber) +
                jnp.abs(thickness - 0.12) +
                jnp.abs(camber_pos - 0.4)
            )
            loss += goals.shape_regularization * float(shape_penalty)
        
        return loss
    
    def run_optimization_step(self) -> Dict:
        """
        Run one optimization step using gradient descent
        """
        # Try to load JAX and compile functions on first run
        if self.loss_fn is None and self.solver is not None:
            try:
                _ensure_jax_loaded()
                # Convert to JAX array
                design_params_jax = jnp.array(self.design_params)
                # Compile functions
                self.loss_fn = jax.jit(self._differentiable_loss)
                self.grad_loss_fn = jax.jit(jax.grad(self._differentiable_loss))
                # Get initial state
                if self.solver is not None:
                    self.initial_state = self._get_initial_state()
                print("JAX loaded successfully - using differentiable CFD")
            except ImportError as e:
                print(f"JAX import failed: {e} - using surrogate model")
                SOLVER_AVAILABLE = False
        
        # Compute current loss
        if self.loss_fn is not None and self.initial_state is not None:
            design_params_jax = jnp.array(self.design_params)
            current_loss = self.loss_fn(design_params_jax)
            current_loss_float = float(current_loss)
            
            # Compute gradients using JAX autodiff
            gradients = self.grad_loss_fn(design_params_jax)
            print(f"Using JAX autodiff - gradients: {[float(g) for g in gradients]}")
        else:
            # Use surrogate model
            camber, camber_pos, thickness, aoa = self.design_params
            goals = self.config.goals
            cl = 0.5 + 2.0 * camber * 100 + 0.1 * np.sin(np.radians(aoa))
            cd = 0.01 + 0.5 * thickness * 100 + 0.01 * (camber_pos - 0.4)**2
            
            current_loss = 0.0
            if goals.target_cl is not None:
                current_loss += goals.cl_weight * (cl - goals.target_cl) ** 2
            if goals.target_cd is not None:
                current_loss += goals.cd_weight * (cd - goals.target_cd) ** 2
            current_loss_float = float(current_loss)
            
            # Compute gradients using finite differences for surrogate model
            gradients = self._compute_finite_difference_gradients_surrogate()
            print(f"Using surrogate model - gradients: {[float(g) for g in gradients]}")
        
        # Update parameters using gradient descent
        lr = self.optimization_config.learning_rate * 10.0  # Higher learning rate for surrogate model
        if self.optimization_config.use_adaptive_lr:
            lr = lr * (self.optimization_config.lr_decay ** self.iteration)
        
        # Apply gradient update with clipping
        self.design_params = self.design_params - lr * gradients
        
        # Clip parameters to reasonable bounds
        self.design_params = self._clip_parameters(self.design_params)
        
        # Get current Cl and Cd for reporting
        camber, camber_pos, thickness, aoa = self.design_params
        cl_float = 0.5 + 2.0 * float(camber) * 100 + 0.1 * np.sin(np.radians(float(aoa)))
        cd_float = 0.01 + 0.5 * float(thickness) * 100 + 0.01 * (float(camber_pos) - 0.4)**2
        
        # Update best solution
        if current_loss_float < self.best_loss:
            self.best_loss = current_loss_float
            self.best_params = self.design_params.copy()
        
        # Record history
        self.history['loss'].append(current_loss_float)
        self.history['cl'].append(cl_float)
        self.history['cd'].append(cd_float)
        
        self.iteration += 1
        
        # Convert params to dict for reporting
        params_dict = {
            'camber': float(self.design_params[0]),
            'camber_position': float(self.design_params[1]),
            'thickness': float(self.design_params[2]),
            'angle_of_attack': float(self.design_params[3])
        }
        
        return {
            'iteration': self.iteration,
            'loss': current_loss_float,
            'cl': cl_float,
            'cd': cd_float,
            'strouhal': None,  # Not implemented yet
            'params': params_dict,
            'gradients': [float(g) for g in gradients],
            'converged': current_loss_float < self.optimization_config.convergence_threshold
        }
    
    def _compute_finite_difference_gradients(self):
        """
        Compute gradients using finite differences (fallback when solver not available)
        """
        epsilon = 1e-6
        gradients = np.zeros_like(self.design_params)
        
        for i in range(len(self.design_params)):
            # Forward difference
            params_plus = self.design_params.copy()
            params_plus[i] += epsilon
            loss_plus = self.loss_fn(params_plus)
            
            # Backward difference
            params_minus = self.design_params.copy()
            params_minus[i] -= epsilon
            loss_minus = self.loss_fn(params_minus)
            
            # Central difference
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients
    
    def _compute_finite_difference_gradients_surrogate(self):
        """
        Compute gradients for surrogate model using finite differences
        """
        epsilon = 1e-6
        gradients = np.zeros_like(self.design_params)
        
        for i in range(len(self.design_params)):
            # Forward difference
            params_plus = self.design_params.copy()
            params_plus[i] += epsilon
            loss_plus = self._compute_surrogate_loss(params_plus)
            
            # Backward difference
            params_minus = self.design_params.copy()
            params_minus[i] -= epsilon
            loss_minus = self._compute_surrogate_loss(params_minus)
            
            # Central difference
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients
    
    def _compute_surrogate_loss(self, params):
        """Compute loss using surrogate model for finite difference gradients"""
        camber, camber_pos, thickness, aoa = params
        goals = self.config.goals
        
        # Simple surrogate model for Cl and Cd
        cl = 0.5 + 2.0 * camber * 100 + 0.1 * np.sin(np.radians(aoa))
        cd = 0.01 + 0.5 * thickness * 100 + 0.01 * (camber_pos - 0.4)**2
        
        loss = 0.0
        if goals.target_cl is not None:
            loss += goals.cl_weight * (cl - goals.target_cl) ** 2
        if goals.target_cd is not None:
            loss += goals.cd_weight * (cd - goals.target_cd) ** 2
        
        # Shape regularization
        if goals.shape_regularization > 0:
            shape_penalty = (
                np.abs(camber) + 
                np.abs(thickness - 0.12) + 
                np.abs(camber_pos - 0.4)
            )
            loss += goals.shape_regularization * shape_penalty
        
        return loss
    
    def _clip_parameters(self, params):
        """Clip parameters to reasonable physical bounds"""
        # Use numpy for clipping since jnp might not be loaded
        # camber: [0.0, 0.1]
        params[0] = np.clip(params[0], 0.0, 0.1)
        # camber_position: [0.1, 0.9]
        params[1] = np.clip(params[1], 0.1, 0.9)
        # thickness: [0.05, 0.25]
        params[2] = np.clip(params[2], 0.05, 0.25)
        # angle_of_attack: [-15.0, 15.0]
        params[3] = np.clip(params[3], -15.0, 15.0)
        
        return params
    
    def get_optimization_status(self) -> Dict:
        """
        Get current optimization status
        """
        # Convert JAX arrays to regular Python types for reporting
        current_params_dict = {
            'camber': float(self.design_params[0]),
            'camber_position': float(self.design_params[1]),
            'thickness': float(self.design_params[2]),
            'angle_of_attack': float(self.design_params[3])
        }
        
        best_params_dict = None
        if self.best_params is not None:
            best_params_dict = {
                'camber': float(self.best_params[0]),
                'camber_position': float(self.best_params[1]),
                'thickness': float(self.best_params[2]),
                'angle_of_attack': float(self.best_params[3])
            }
        
        return {
            'iteration': self.iteration,
            'best_loss': self.best_loss,
            'current_params': current_params_dict,
            'best_params': best_params_dict,
            'history': self.history,
            'converged': self.iteration >= self.optimization_config.max_iterations or
                        self.best_loss < self.optimization_config.convergence_threshold,
            'jax_autodiff_enabled': SOLVER_AVAILABLE and self.initial_state is not None
        }
    
    def reset_optimization(self):
        """
        Reset optimization state to initial values
        """
        self.iteration = 0
        self.best_loss = float('inf')
        self.best_params = None
        self.history = {
            'loss': [],
            'cl': [],
            'cd': [],
            'strouhal': [],
            'cl_error': [],
            'cd_error': [],
            'strouhal_error': []
        }
        self.design_params = jnp.array([0.02, 0.4, 0.12, 0.0])
