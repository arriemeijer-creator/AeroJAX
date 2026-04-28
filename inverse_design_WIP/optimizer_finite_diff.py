"""
Inverse Design Optimizer using Finite Differences
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from .config import InverseDesignConfig, OptimizationGoals, AirfoilConfig


@dataclass
class OptimizationConfig:
    """Configuration for optimization"""
    max_iterations: int = 100
    learning_rate: float = 0.01
    convergence_threshold: float = 1e-6
    use_adaptive_lr: bool = True
    lr_decay: float = 0.99
    num_simulation_steps: int = 5  # Number of CFD steps per optimization iteration


class InverseDesigner:
    """
    Finite differences inverse design optimizer for airfoil shape optimization
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
        
        # Momentum to help escape local minima
        self.momentum = 0.9
        self.velocity = np.zeros(4)
        
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
        
        # Flag to use real solver instead of surrogate
        self.use_real_solver = solver is not None
        if self.use_real_solver:
            print("Using real CFD solver")
        else:
            print("Using surrogate model")
    
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
    
    def run_optimization_step(self) -> Dict:
        """
        Run one optimization step using gradient descent with momentum
        """
        num_steps = self.optimization_config.num_simulation_steps
        
        # Compute Cl and Cd using real solver
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
        print(f"=== GRADIENT DEBUG: AoA={self.design_params[3]:.4f}, Target Cl={self.config.goals.target_cl:.4f} ===")
        print(f"  Current params: camber={self.design_params[0]:.4f}, thickness={self.design_params[2]:.4f}")
        print(f"  Shape penalty: {goals.shape_regularization * (np.abs(self.design_params[0]) + np.abs(self.design_params[2] - 0.12) + np.abs(self.design_params[1] - 0.4)):.6f}")
        gradients = self._compute_finite_difference_gradients_solver(num_steps)
        print(f"=== GRADIENTS: {[float(g) for g in gradients]} ===")
        
        # Clip gradients per parameter based on their ranges (prevents oscillation)
        # Parameter ranges: camber[0,0.1], camber_pos[0.1,0.9], thickness[0.05,0.25], aoa[-15,15]
        param_ranges = np.array([0.1, 0.8, 0.2, 30.0])  # Range widths
        # Use different percentages per parameter: thickness needs smaller steps
        grad_percentages = np.array([0.1, 0.1, 0.05, 0.5])  # 5% of thickness range, 50% of AoA range
        max_grad = grad_percentages * param_ranges
        gradients = np.clip(gradients, -max_grad, max_grad)
        print(f"Clipped gradients: {[float(g) for g in gradients]}")
        
        # Update parameters using gradient descent with momentum
        lr = self.optimization_config.learning_rate * 50.0  # Much higher learning rate
        if self.optimization_config.use_adaptive_lr:
            lr = lr * (self.optimization_config.lr_decay ** self.iteration)
        
        # Debug parameter update
        print(f"  Update: lr={lr:.4f}, aoa_grad={gradients[3]:.4f}, aoa_vel={self.velocity[3]:.4f}")
        print(f"  Before: aoa={self.design_params[3]:.4f}")
        
        # Update velocity with momentum
        self.velocity = self.momentum * self.velocity - lr * gradients
        
        # Apply gradient update with momentum
        self.design_params = self.design_params + self.velocity
        
        print(f"  After: aoa={self.design_params[3]:.4f}")
        
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
            if var_names[i] == 'aoa':
                print(f"  Testing AoA+={params_plus[i]:.6f}")
            elif var_names[i] == 'thickness':
                print(f"  Testing thickness+={params_plus[i]:.6f}")
            cl_plus, cd_plus = self._run_solver_with_params(params_plus, num_steps)
            if var_names[i] == 'aoa':
                print(f"    -> Cl={cl_plus:.6f}")
            elif var_names[i] == 'thickness':
                print(f"    -> Cl={cl_plus:.6f}, Cd={cd_plus:.6f}")
            loss_plus = self._compute_loss_from_cl_cd(cl_plus, cd_plus)
            
            # Backward difference
            params_minus = self.design_params.copy()
            params_minus[i] -= epsilon
            if var_names[i] == 'aoa':
                print(f"  Testing AoA-={params_minus[i]:.6f}")
            elif var_names[i] == 'thickness':
                print(f"  Testing thickness-={params_minus[i]:.6f}")
            cl_minus, cd_minus = self._run_solver_with_params(params_minus, num_steps)
            if var_names[i] == 'aoa':
                print(f"    -> Cl={cl_minus:.6f}")
            elif var_names[i] == 'thickness':
                print(f"    -> Cl={cl_minus:.6f}, Cd={cd_minus:.6f}")
            loss_minus = self._compute_loss_from_cl_cd(cl_minus, cd_minus)
            
            # Central difference
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)

            # Debug gradient direction
            if var_names[i] == 'aoa':
                target_cl = self.config.goals.target_cl
                print(f"  AoA gradient: target_cl={target_cl:.4f}, aoa={self.design_params[i]:.4f}, "
                      f"grad={gradients[i]:.6f}")
            elif var_names[i] == 'thickness':
                print(f"  Thickness gradient: thickness={self.design_params[i]:.4f}, grad={gradients[i]:.6f}")
        
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
            'jax_autodiff_enabled': False
        }
    
    def reset_optimization(self):
        """
        Reset optimization state to initial values
        """
        self.iteration = 0
        self.best_loss = float('inf')
        self.best_params = None
        self.velocity = np.zeros(4)  # Reset momentum
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
