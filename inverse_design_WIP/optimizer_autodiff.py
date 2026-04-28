
"""
Inverse Design Optimizer using JAX gradients
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, lax
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from .config import InverseDesignConfig, OptimizationGoals, AirfoilConfig

# JAX enabled for backpropagation optimization
SOLVER_AVAILABLE = True


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
    
    def __init__(self, solver, config: InverseDesignConfig, initial_params=None, selected_variables=None):
        self.solver = solver
        self.config = config
        
        # Store which variables to optimize (all by default)
        if selected_variables is None:
            self.selected_variables = {
                'camber': True,
                'camber_position': True,
                'thickness': True,
                'aoa': True
            }
        else:
            self.selected_variables = selected_variables
        self.optimization_config = OptimizationConfig(
            max_iterations=config.max_iterations,
            learning_rate=config.learning_rate,
            convergence_threshold=config.convergence_threshold
        )
        
        # Optimization state
        self.iteration = 0
        self.best_loss = float('inf')
        self.best_params = None
        
        # History tracking
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
        if initial_params is not None:
            self.design_params = np.array(initial_params)
        else:
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
        
        # Flag to use real solver instead of surrogate
        self.use_real_solver = solver is not None
        if self.use_real_solver:
            print("Using real CFD solver")
        else:
            print("Using surrogate model")
        
    def _get_initial_state(self):
        """Extract initial state from solver for gradient computation"""
        if not self.use_real_solver:
            return None
        
        from solver.params import SimState
        return SimState(
            u=jnp.array(self.solver.u),
            v=jnp.array(self.solver.v),
            p=jnp.array(self.solver.current_pressure),
            u_prev=jnp.array(self.solver.u_prev),
            v_prev=jnp.array(self.solver.v_prev),
            c=jnp.array(self.solver.c),
            dt=jnp.array(self.solver.dt),
            iteration=0,
            grid_type='collocated',
            integral=0.0,
            prev_error=0.0
        )
    
    def _make_differentiable_loss(self, initial_state, grid_x, grid_y, dx, dy, dt, nu, U_inf,
                                  chord_length, airfoil_x, airfoil_y, brinkman_eta,
                                  num_steps):
        """
        Returns a JIT-compiled loss function using JAX autodiff.
        Uses partial wrapper to bind static string parameters.
        """
        from solver.solver import step_pure
        from solver.metrics import compute_forces_ibm
        from obstacles.naca_airfoils import naca_surface_distance
        from solver.operators import vorticity
        from solver.params import SimState
        
        # Static simulation parameters (bind these before JIT)
        flow_type = self.solver.sim_params.flow_type
        advection_scheme = self.solver.sim_params.advection_scheme
        pressure_solver = self.solver.sim_params.pressure_solver
        use_les = self.solver.sim_params.use_les
        les_model = self.solver.sim_params.les_model
        smagorinsky_constant = self.solver.sim_params.smagorinsky_constant
        weno_epsilon = self.solver.sim_params.weno_epsilon
        eps = self.solver.sim_params.eps
        grid_type = self.solver.sim_params.grid_type
        v_cycles = self.solver.sim_params.multigrid_v_cycles
        limiter = self.solver.sim_params.limiter
        
        # Use solver's _step_jit which is already JIT-compiled
        # This bypasses the SimState string parameter issue entirely
        step_jit = self.solver._step_jit
        
        # Ensure initial_state arrays are JAX arrays
        u0 = jnp.array(initial_state.u)
        v0 = jnp.array(initial_state.v)
        p0 = jnp.array(initial_state.p)
        u_prev0 = jnp.array(initial_state.u_prev)
        v_prev0 = jnp.array(initial_state.v_prev)
        c0 = jnp.array(initial_state.c)
        
        @jax.jit
        def loss_fn(params):
            """Compute loss for given design parameters using JAX autodiff"""
            # Unpack design parameters
            camber, camber_pos, thickness, aoa = params
            
            # Compute mask from NACA SDF (differentiable)
            sdf = naca_surface_distance(grid_x, grid_y, chord_length, aoa,
                                        airfoil_x, airfoil_y, camber, camber_pos, thickness)
            mask = 1.0 / (1.0 + jnp.exp(sdf / eps))  # sigmoid using JAX
            
            # Use individual arrays instead of SimState to avoid string tracing issues
            # The partial wrapper already has grid_type bound as static
            carry_init = (u0, v0, p0, u_prev0, v_prev0, c0, dt)
            
            # Use lax.scan for efficient rollout with individual arrays
            def scan_fn(carry, _):
                u, v, p, u_prev, v_prev, c, dt_cur = carry
                
                # Call solver's _step_jit (already JIT-compiled, no SimState needed)
                u_new, v_new, mask_new, p_new = step_jit(u, v, dt_cur)
                
                # Update previous fields
                u_prev_new = u
                v_prev_new = v
                
                return (u_new, v_new, p_new, u_prev_new, v_prev_new, c, dt_cur), None
            
            final_carry, _ = jax.lax.scan(scan_fn, carry_init, None, length=num_steps)
            u_f, v_f, p_f, u_prev_f, v_prev_f, c_f, dt_f = final_carry
            
            # Compute forces from final state
            w_f = vorticity(u_f, v_f, dx, dy)
            # Use a simpler force computation that doesn't have print statements
            # For now, use velocity-based proxy for lift/drag
            # Compute momentum flux through a plane downstream
            x_idx = int(grid_x.shape[0] * 0.6)  # 60% downstream
            u_plane = u_f[x_idx, :]
            v_plane = v_f[x_idx, :]
            # Simple proxy: mean velocity deviation
            cl_proxy = jnp.mean(v_plane) / U_inf  # Vertical momentum as lift proxy
            cd_proxy = (U_inf - jnp.mean(u_plane)) / U_inf  # Drag proxy
            
            # Compute loss using optimization goals
            loss = 0.0
            if self.config.goals.target_cl_enabled:
                loss += self.config.goals.cl_weight * (cl_proxy - self.config.goals.target_cl)**2
            if self.config.goals.target_cd_enabled:
                loss += self.config.goals.cd_weight * (cd_proxy - self.config.goals.target_cd)**2
            # Add shape regularization
            loss += 0.01 * camber**2  # Penalize large camber
            loss += 0.01 * thickness**2  # Penalize large thickness
            return loss
        
        return loss_fn
    
    def _params_to_geometry(self, design_params):
        """Convert design parameters to geometry parameters for mask generation"""
        camber, camber_pos, thickness, aoa = design_params
        
        # This method is not used in surrogate mode
        # Real solver integration disabled due to JAX circular import on Windows
        return None
    
    def _run_solver_with_params(self, design_params, num_steps=5):
        """
        Run CFD solver with given design parameters and extract Cl/Cd.
        This uses the real solver instead of surrogate model.
        """
        if self.solver is None:
            raise RuntimeError("Solver is required - please provide a solver instance")
        
        camber, camber_pos, thickness, aoa = design_params
        
        # Update solver parameters
        self.solver.sim_params.naca_camber = float(camber)
        self.solver.sim_params.naca_camber_position = float(camber_pos)
        self.solver.sim_params.naca_thickness = float(thickness)
        self.solver.sim_params.naca_angle = float(aoa)
        
        # Recompute mask with new airfoil parameters
        self.solver.mask = self.solver._compute_mask()
        
        # Reinitialize flow field to match new airfoil geometry
        # This prevents visualization artifacts from mask/flow mismatch
        self.solver.u = self.solver.u * self.solver.mask
        self.solver.v = self.solver.v * self.solver.mask
        self.solver.current_pressure = np.zeros_like(self.solver.current_pressure)
        
        # Run simulation for num_steps
        for _ in range(num_steps):
            _, _, _, _ = self.solver.step_for_visualization()
        
        # Extract Cl/Cd from solver using same method as main solver
        from solver.metrics import compute_forces_ibm
        from solver.operators import vorticity_nonperiodic
        
        # Compute vorticity on the fly since solver doesn't store current_vorticity
        vort = vorticity_nonperiodic(self.solver.u, self.solver.v, self.solver.grid.dx, self.solver.grid.dy)
        
        cl, cd = compute_forces_ibm(
            self.solver.u, self.solver.v, vort,
            self.solver.grid.X, self.solver.grid.Y,
            self.solver.mask, self.solver.grid.dx, self.solver.grid.dy,
            self.solver.flow.U_inf, self.solver.sim_params.naca_chord,
            self.solver.sim_params.naca_x, self.solver.sim_params.naca_y,
            self.solver.grid.lx, grid_type='collocated'
        )
        
        return cl, cd
    
    
    def _differentiable_rollout(self, design_params, initial_state):
        """
        Differentiable CFD rollout that computes Cl and Cd from design parameters.
        This function is JIT-compiled and differentiable with respect to design_params.
        STUBBED OUT for surrogate mode - not used when JAX is disabled.
        """
        # This method is not used in surrogate mode
        raise NotImplementedError("Differentiable rollout requires JAX - use surrogate mode instead")
    
    def _differentiable_loss(self, design_params):
        """
        Differentiable loss function for JAX autodiff.
        Computes loss by running CFD simulation and comparing to targets.
        STUBBED OUT for surrogate mode - not used when JAX is disabled.
        """
        # This method is not used in surrogate mode
        raise NotImplementedError("Differentiable loss requires JAX - use surrogate mode instead")
    
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
                np.abs(camber) +
                np.abs(thickness - 0.12) +
                np.abs(camber_pos - 0.4)
            )
            loss += goals.shape_regularization * float(shape_penalty)
        
        return loss
    
    def run_optimization_step(self) -> Dict:
        """
        Run one optimization step using JAX autodiff
        """
        num_steps = self.optimization_config.num_simulation_steps
        
        # If JAX is available and we have a real solver, use autodiff
        if self.use_real_solver and SOLVER_AVAILABLE:
            # Extract needed data from solver once
            grid_x = self.solver.grid.X
            grid_y = self.solver.grid.Y
            dx = self.solver.grid.dx
            dy = self.solver.grid.dy
            dt = self.solver.dt
            nu = self.solver.flow.nu
            U_inf = self.solver.flow.U_inf
            chord = self.solver.sim_params.naca_chord
            airfoil_x = self.solver.sim_params.naca_x
            airfoil_y = self.solver.sim_params.naca_y
            brinkman_eta = self.solver.sim_params.brinkman_eta
            
            # Capture current initial state
            init_state = self._get_initial_state()
            
            # Build the differentiable loss function
            loss_fn = self._make_differentiable_loss(
                init_state, grid_x, grid_y, dx, dy, dt, nu, U_inf,
                chord, airfoil_x, airfoil_y, brinkman_eta,
                num_steps
            )
            
            # Compute loss and gradient using JAX autodiff
            params_jax = jnp.array(self.design_params)
            grad_loss = grad(loss_fn)
            loss_val = loss_fn(params_jax)
            grad_val = grad_loss(params_jax)
            
            # Update parameters (numpy conversion)
            lr = self.optimization_config.learning_rate
            if self.optimization_config.use_adaptive_lr:
                lr *= (self.optimization_config.lr_decay ** self.iteration)
            new_params = self.design_params - lr * np.array(grad_val)
            self.design_params = self._clip_parameters(new_params)
            
            # Update solver's actual parameters for the next iteration
            camber, camber_pos, thickness, aoa = self.design_params
            self.solver.sim_params.naca_camber = float(camber)
            self.solver.sim_params.naca_camber_position = float(camber_pos)
            self.solver.sim_params.naca_thickness = float(thickness)
            self.solver.sim_params.naca_angle = float(aoa)
            self.solver.mask = self.solver._compute_mask()
            
            current_loss_float = float(loss_val)
            gradients = np.array(grad_val)
            
            # Compute Cl/Cd for reporting (single solver run)
            cl, cd = self._run_solver_with_params(self.design_params, num_steps)
            print(f"Using JAX Autodiff - Loss: {current_loss_float:.6f}, Cl: {cl:.4f}, Cd: {cd:.4f}")
            print(f"Gradients: {[float(g) for g in gradients]}")
        else:
            # Fallback to finite differences
            cl, cd = self._run_solver_with_params(self.design_params, num_steps)
            print(f"Using real solver - Cl: {cl:.4f}, Cd: {cd:.4f}")
            
            # Compute loss (only for enabled targets)
            goals = self.config.goals
            current_loss = 0.0
            if goals.target_cl is not None and goals.target_cl_enabled:
                cl_error = (cl - goals.target_cl) ** 2
                current_loss += goals.cl_weight * cl_error
                self.history['cl_error'].append(float(cl_error))
            
            if goals.target_cd is not None and goals.target_cd_enabled:
                cd_error = (cd - goals.target_cd) ** 2
                current_loss += goals.cd_weight * cd_error
                self.history['cd_error'].append(float(cd_error))
            
            # Shape regularization
            camber, camber_pos, thickness, aoa = self.design_params
            if goals.shape_regularization > 0:
                shape_penalty = (
                    np.abs(camber) +
                    np.abs(thickness - 0.12) +
                    np.abs(camber_pos - 0.4)
                )
                current_loss += goals.shape_regularization * shape_penalty
            
            current_loss_float = float(current_loss)
            
            # Compute gradients using finite differences with real solver
            gradients = self._compute_finite_difference_gradients_solver(num_steps)
            print(f"Using finite difference gradients: {[float(g) for g in gradients]}")
            
            # Update parameters using gradient descent
            lr = self.optimization_config.learning_rate
            if self.optimization_config.use_adaptive_lr:
                lr = lr * (self.optimization_config.lr_decay ** self.iteration)
            
            # Apply gradient update with clipping
            self.design_params = self.design_params - lr * gradients
            
            # Clip parameters to reasonable bounds
            self.design_params = self._clip_parameters(self.design_params)
        
        # Update best solution
        if current_loss_float < self.best_loss:
            self.best_loss = current_loss_float
            self.best_params = self.design_params.copy()
        
        # Record history
        self.history['loss'].append(current_loss_float)
        self.history['cl'].append(float(cl))
        self.history['cd'].append(float(cd))
        
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
            'cl': float(cl),
            'cd': float(cd),
            'strouhal': None,  # Not implemented yet
            'params': params_dict,
            'gradients': [float(g) for g in gradients],
            'converged': current_loss_float < self.optimization_config.convergence_threshold
        }
    
    def _compute_finite_difference_gradients_solver(self, num_steps=50):
        """
        Compute gradients using finite differences with real solver
        Only computes gradients for selected variables
        """
        epsilon = 1e-6
        gradients = np.zeros_like(self.design_params)
        
        # Variable names corresponding to indices
        var_names = ['camber', 'camber_position', 'thickness', 'aoa']
        
        for i in range(len(self.design_params)):
            # Only compute gradient if this variable is selected for optimization
            if not self.selected_variables[var_names[i]]:
                gradients[i] = 0.0
                continue
            
            # Forward difference
            params_plus = self.design_params.copy()
            params_plus[i] += epsilon
            cl_plus, cd_plus = self._run_solver_with_params(params_plus, num_steps)
            loss_plus = self._compute_loss_from_cl_cd(cl_plus, cd_plus)
            
            # Backward difference
            params_minus = self.design_params.copy()
            params_minus[i] -= epsilon
            cl_minus, cd_minus = self._run_solver_with_params(params_minus, num_steps)
            loss_minus = self._compute_loss_from_cl_cd(cl_minus, cd_minus)
            
            # Central difference
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients
    
    def _compute_loss_from_cl_cd(self, cl, cd):
        """Compute loss from Cl and Cd values"""
        goals = self.config.goals
        loss = 0.0
        
        if goals.target_cl is not None:
            loss += goals.cl_weight * (cl - goals.target_cl) ** 2
        
        if goals.target_cd is not None:
            loss += goals.cd_weight * (cd - goals.target_cd) ** 2
        
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
        self.design_params = np.array([0.02, 0.4, 0.12, 0.0])