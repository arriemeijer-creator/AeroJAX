"""
LBM Solver class with interface compatible with BaselineSolver
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple
from solver.params import GridParams, FlowParams, GeometryParams, SimulationParams
from .params import LBMSimulationParams
from .lattice import D2Q9Lattice
from .collision import collision_step, equilibrium
from .streaming import streaming_step
from .boundary import apply_boundary_conditions, apply_inlet_outlet, apply_bounce_back
from .operators import macroscopic_variables, compute_pressure


class LBMSolver:
    """Lattice Boltzmann Method solver with BaselineSolver-compatible interface"""
    
    def __init__(self,
                 grid: GridParams,
                 flow: FlowParams,
                 geom: GeometryParams,
                 sim_params: SimulationParams,
                 dt: float = None,
                 seed: int = 42):
        
        # Store parameters
        self.grid = grid
        self.flow = flow
        self.geom = geom
        self.sim_params = sim_params
        self.seed = seed
        
        # Initialize LBM-specific parameters
        self.lbm_params = LBMSimulationParams()
        
        # For LBM stability, we need to ensure:
        # 1. tau > 0.5 (stability condition)
        # 2. Mach number < 0.1 (incompressibility)
        # 3. Velocity in lattice units should be small
        
        # Better mapping from physical to lattice units to preserve Reynolds number
        cs_squared = 1.0 / 3.0
        
        # Choose target lattice Reynolds number for aggressive diffusion reduction
        # Much higher Re_lattice = much less diffusion, push stability limits
        # For vortex shedding, aim for Re_lattice > 1000 for strong vortices
        target_Re_lattice = min(5000.0, flow.Re * 10.0)  # Much more aggressive scaling
        
        # Calculate lattice velocity based on physical velocity
        # Scale physical velocity to lattice units while maintaining stability
        physical_velocity = flow.U_inf
        
        # Calculate characteristic length scale (chord length or obstacle diameter)
        if hasattr(sim_params, 'naca_chord'):
            L_char = sim_params.naca_chord
        else:
            L_char = geom.radius * 2  # Diameter for cylinder
        
        # Calculate lattice velocity scaling with aggressive approach for less diffusion
        dx = grid.lx / grid.nx
        
        # Target much higher lattice velocity for less diffusion
        # Push to higher Mach number for better flow physics
        target_U_lattice = 0.25 * jnp.sqrt(cs_squared)  # Mach ~0.25
        
        # Scale physical velocity relative to a reference (e.g., 2.0 m/s)
        reference_velocity = 2.0  # Reference physical velocity
        velocity_scale_factor = physical_velocity / reference_velocity
        
        # Apply scaling to target lattice velocity
        self.U_lattice = target_U_lattice * velocity_scale_factor
        
        # Apply more relaxed stability constraints (Mach < 0.35 for aggressive flow)
        max_U_lattice = 0.35 * jnp.sqrt(cs_squared)
        min_U_lattice = 0.08  # Higher minimum for better flow
        self.U_lattice = jnp.clip(self.U_lattice, min_U_lattice, max_U_lattice)
        
        # Use grid characteristic length as L_lattice
        L_lattice = min(grid.nx, grid.ny)
        
        # Calculate required lattice viscosity
        nu_lattice = self.U_lattice * L_lattice / target_Re_lattice
        
        # Convert to tau
        tau_target = 0.5 + nu_lattice / cs_squared
        
        # Ensure stability bounds
        tau_min = 0.51  # Just above stability limit
        tau_max = 2.0   # Upper bound for stability
        self.lbm_params.tau = jnp.clip(tau_target, tau_min, tau_max)
        self.lbm_params.omega = 1.0 / self.lbm_params.tau
        
        # Initialize lattice
        self.lattice = D2Q9Lattice()
        
        # Initialize timestep (LBM typically uses dt = 1 in lattice units)
        if dt is not None:
            self.dt = dt
        else:
            # For LBM, we can use dt = 1 (lattice units) or scale to physical units
            # For compatibility with NS, we'll use the same dt
            self.dt = sim_params.fixed_dt if sim_params.fixed_dt else 0.001
        
        # Initialize distribution function f
        # f has shape (9, nx, ny)
        nx, ny = grid.nx, grid.ny
        self.f = jnp.zeros((9, nx, ny))
        
        # Initialize macroscopic variables
        self.u = jnp.zeros((nx, ny))
        self.v = jnp.zeros((nx, ny))
        self.rho = jnp.ones((nx, ny))  # Density
        self.p = jnp.zeros((nx, ny))   # Pressure (computed from density)
        
        # Initialize previous velocity (for metrics)
        self.u_prev = jnp.zeros((nx, ny))
        self.v_prev = jnp.zeros((nx, ny))
        
        # Initialize dye/scalar field
        self.c = jnp.zeros((nx, ny))
        
        # Compute mask
        self.mask = self._compute_mask()
        
        # Initialize flow based on flow type
        self._initialize_flow()
        
        # Initialize distribution function from equilibrium
        from .collision import equilibrium
        self.f = equilibrium(self.rho, self.u, self.v, 
                            self.lattice.get_cx(), self.lattice.get_cy(),
                            self.lattice.w, self.lattice.cs_squared)
        
        # JIT cache
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        
        # History for metrics
        self.history = {
            'time': [], 'dt': [], 'drag': [], 'lift': [],
            'l2_change': [], 'rms_change': [], 'l2_change_u': [], 'l2_change_v': [],
            'max_change': [], 'change_99p': [], 'rel_change': [],
            'rms_divergence': [], 'l2_divergence': [],
            'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 
                               'Cp_min': [], 'wake_deficit': [], 'strouhal': [], 'time': []}
        }
        
        self.iteration = 0
        self.compute_airfoil_metrics = False
        self.metrics_frame_skip = 100
        
        # Store current pressure for compatibility
        self.current_pressure = self.p
        
        print(f"LBM Solver initialized: tau={self.lbm_params.tau:.4f}, omega={self.lbm_params.omega:.4f}, "
              f"nu={flow.nu:.6f}, Re={flow.Re:.1f}")
    
    def _initialize_flow(self):
        """Initialize flow based on flow type"""
        nx, ny = self.grid.nx, self.grid.ny
        U_lattice = self.U_lattice  # Use scaled lattice velocity
        X, Y = self.grid.X, self.grid.Y
        
        if self.sim_params.flow_type == 'lid_driven_cavity':
            # Lid-driven cavity: zero velocity, lid at top
            self.u = jnp.zeros((nx, ny))
            self.v = jnp.zeros((nx, ny))
            self.rho = jnp.ones((nx, ny))
        elif self.sim_params.flow_type == 'taylor_green':
            # Taylor-Green vortex
            self.u = U_lattice * jnp.sin(X) * jnp.cos(Y)
            self.v = -U_lattice * jnp.cos(X) * jnp.sin(Y)
            self.rho = jnp.ones((nx, ny))
        else:
            # von Karman / channel flow: uniform inlet velocity
            self.u = jnp.full((nx, ny), U_lattice)
            self.v = jnp.zeros((nx, ny))
            self.rho = jnp.ones((nx, ny))
        
        # Apply mask to initial velocity
        self.u = self.u * self.mask
        self.v = self.v * self.mask
        
        # Initialize previous velocity
        self.u_prev = jnp.copy(self.u)
        self.v_prev = jnp.copy(self.v)
    
    def _compute_mask(self) -> jnp.ndarray:
        """Compute obstacle mask (1 = fluid, 0 = solid)"""
        # Special case for lid_driven_cavity and taylor_green - all fluid (mask = 1 everywhere)
        if self.sim_params.flow_type == 'lid_driven_cavity':
            return jnp.ones_like(self.grid.X)
        if self.sim_params.flow_type == 'taylor_green':
            return jnp.ones_like(self.grid.X)
        
        # For obstacles, use the geometry module
        if self.sim_params.obstacle_type == 'naca_airfoil':
            from obstacles.naca_airfoils import NACAParams, create_naca_mask, parse_naca_4digit, parse_naca_5digit
            
            # Parse NACA designation
            naca_str = self.sim_params.naca_airfoil.upper().replace('NACA', '').strip()
            if len(naca_str) == 4:
                m, p, t = parse_naca_4digit(naca_str)
                airfoil_type = '4-digit'
            elif len(naca_str) == 5:
                cl, p, m, t = parse_naca_5digit(naca_str)
                airfoil_type = '5-digit'
            else:
                raise ValueError(f"Unsupported NACA designation: {self.sim_params.naca_airfoil}")
            
            naca_params = NACAParams(
                airfoil_type=airfoil_type,
                designation=self.sim_params.naca_airfoil,
                chord_length=self.sim_params.naca_chord,
                angle_of_attack=self.sim_params.naca_angle,
                position_x=self.sim_params.naca_x,
                position_y=self.sim_params.naca_y
            )
            
            mask = create_naca_mask(self.grid.X, self.grid.Y, naca_params, self.sim_params.eps)
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        elif self.sim_params.obstacle_type == 'cylinder':
            from solver.geometry import sdf_cylinder, smooth_mask
            phi = sdf_cylinder(self.grid.X, self.grid.Y, self.geom.center_x, self.geom.center_y, self.geom.radius)
            mask = smooth_mask(phi, self.sim_params.eps)
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        elif self.sim_params.obstacle_type == 'cow':
            from obstacles.cow import create_cow_mask
            mask = create_cow_mask(self.grid.X, self.grid.Y, self.sim_params.cow_x, self.sim_params.cow_y, self.sim_params.eps)
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        elif self.sim_params.obstacle_type == 'three_cylinder_array':
            from obstacles.cylinder_array import create_cylinder_array_mask
            mask = create_cylinder_array_mask(self.grid.X, self.grid.Y, 
                                            self.sim_params.cylinder_x, self.sim_params.cylinder_y,
                                            self.sim_params.eps)
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        else:
            # Default: all fluid
            mask = jnp.ones_like(self.grid.X)
        
        return mask
    
    def set_obstacle_type(self, obstacle_type: str):
        """Change obstacle type and recompute mask"""
        self.sim_params.obstacle_type = obstacle_type
        self.mask = self._compute_mask()
        # Clear JIT cache since mask changed
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        print(f"LBM: Obstacle type changed to {obstacle_type}")
    
    def apply_flow_type(self, flow_type: str):
        """Change flow type and reinitialize"""
        self.sim_params.flow_type = flow_type
        # Recompute mask for new flow type
        self.mask = self._compute_mask()
        self._initialize_flow()
        from .collision import equilibrium
        self.f = equilibrium(self.rho, self.u, self.v, 
                            self.lattice.get_cx(), self.lattice.get_cy(),
                            self.lattice.w, self.lattice.cs_squared)
        self.iteration = 0
        # Clear JIT cache since flow type changed
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        print(f"LBM: Flow type changed to {flow_type}")
    
    def get_step_jit(self):
        """Return JIT-compiled step function"""
        if 'step_jit' not in self._jit_cache:
            # Extract lattice data as static arrays
            cx = self.lattice.get_cx()
            cy = self.lattice.get_cy()
            w = self.lattice.w
            opposite = self.lattice.opposite
            cs_squared = self.lattice.cs_squared
            
            # Create JIT function with static lattice data and grid dimensions
            self._jit_cache['step_jit'] = jax.jit(
                self._step_pure,
                static_argnames=['flow_type', 'nx', 'ny']
            )
            # Store lattice data for use in step
            self._lattice_data = {
                'cx': cx,
                'cy': cy,
                'w': w,
                'opposite': opposite,
                'cs_squared': cs_squared
            }
        return self._jit_cache['step_jit']
    
    def _step_pure(self, f: jnp.ndarray, mask: jnp.ndarray, lattice_data: dict, omega: float,
                   flow_type: str, u_inlet: float, nx: int, ny: int) -> Tuple:
        """
        Pure JAX-compatible LBM step (collision + streaming + all boundary conditions)
        
        Args:
            f: Distribution function (9, nx, ny)
            mask: Obstacle mask (nx, ny)
            lattice_data: Dictionary with lattice arrays (cx, cy, w, opposite, cs_squared)
            omega: Collision frequency
            flow_type: Type of flow ('von_karman', 'lid_driven_cavity', 'taylor_green')
            u_inlet: Inlet velocity for channel flows
            nx: Grid size x
            ny: Grid size y
        
        Returns:
            f_final: Distribution after collision, streaming, and all boundary conditions
            rho: Density field
            u: Velocity x-component
            v: Velocity y-component
        """
        from .boundary import apply_bounce_back, apply_boundary_conditions
        
        cx = lattice_data['cx']
        cy = lattice_data['cy']
        w = lattice_data['w']
        opposite = lattice_data['opposite']
        cs_squared = lattice_data['cs_squared']
        
        # Collision
        rho, u, v = macroscopic_variables(f, cx, cy)
        
        # Apply mask to velocities: set velocity to zero inside obstacles
        u = u * mask
        v = v * mask
        
        f_post = collision_step(f, rho, u, v, cx, cy, w, cs_squared, omega)
        
        # Streaming
        f_streamed = streaming_step(f_post, cx, cy)
        
        # Apply bounce-back for obstacles (inside JIT)
        f_bc = apply_bounce_back(f_streamed, mask, opposite)
        
        # Apply flow-specific boundary conditions (inside JIT)
        f_final = apply_boundary_conditions(
            f_bc, mask, opposite, flow_type, u_inlet, nx, ny, cx, cy, w, cs_squared
        )
        
        # Compute macroscopic variables for output
        rho_new, u_new, v_new = macroscopic_variables(f_final, cx, cy)
        
        # Apply mask to final velocities to ensure zero inside obstacles
        u_new = u_new * mask
        v_new = v_new * mask
        
        return f_final, rho_new, u_new, v_new
    
    def step(self) -> Tuple:
        """Perform one LBM step"""
        # Run complete LBM step (collision + streaming + all boundary conditions) in JIT
        f_final, rho_new, u_new, v_new = self._step_jit(
            self.f, self.mask, self._lattice_data, self.lbm_params.omega,
            self.sim_params.flow_type, self.U_lattice,
            self.grid.nx, self.grid.ny
        )
        
        # Compute pressure
        from .operators import compute_pressure
        p_new = compute_pressure(rho_new, self._lattice_data['cs_squared'], rho0=1.0)
        
        # Update state
        self.u_prev = self.u
        self.v_prev = self.v
        self.f = f_final
        self.u = u_new
        self.v = v_new
        self.rho = rho_new
        self.p = p_new
        self.current_pressure = p_new
        
        # Advect dye (passive scalar) using finite difference
        if self.lbm_params.enable_dye:
            from solver.operators import scalar_advection_diffusion_nonperiodic
            dx = self.grid.lx / self.grid.nx
            dy = self.grid.ly / self.grid.ny
            self.c = scalar_advection_diffusion_nonperiodic(
                self.c, self.u, self.v, self.dt, dx, dy, self.lbm_params.dye_diffusivity
            )
        
        self.iteration += 1
        
        # Update history
        self.history['time'].append(self.iteration * self.dt)
        self.history['dt'].append(self.dt)
        
        return self.u, self.v, self.p
    
    def step_for_visualization(self, compute_divergence: bool = False,
                               compute_drag_lift: bool = False,
                               compute_diagnostics: bool = False) -> Tuple:
        """
        Step and return data for visualization (compatible with BaselineSolver)
        
        Returns:
            u, v, vort, div (divergence may be None)
        """
        # Perform step
        u, v, p = self.step()
        
        # Compute vorticity
        from solver.operators import vorticity_nonperiodic
        vort = vorticity_nonperiodic(u, v, self.grid.dx, self.grid.dy)
        
        # Compute divergence if requested
        div = None
        if compute_divergence:
            from solver.operators import divergence_nonperiodic
            div = divergence_nonperiodic(u, v, self.grid.dx, self.grid.dy)
        
        # Compute airfoil metrics if enabled (same as baseline solver)
        if self.compute_airfoil_metrics and hasattr(self.sim_params, 'flow_type') and self.sim_params.flow_type == 'von_karman':
            try:
                import numpy as np
                from solver.metrics import (
                    find_stagnation_point, find_separation_point, compute_forces_ibm,
                    get_airfoil_surface_mask, compute_CL_circulation, compute_drag_momentum_deficit
                )
                
                # Convert JAX arrays to numpy for metrics computation
                u_np = np.array(u)
                v_np = np.array(v)
                p_np = np.array(p)
                
                # Get mask
                mask_np = np.array(self.mask)
                
                # Grid parameters
                X_np = np.array(self.grid.X)
                dx = self.grid.dx
                dy = self.grid.dy
                
                # Find stagnation and separation points
                stag_x = find_stagnation_point(u_np, v_np, mask_np, p_np, X_np, dx)
                sep_x = find_separation_point(u_np, v_np, mask_np, X_np, dx, dy)
                
                # Get obstacle parameters
                chord_length = getattr(self.sim_params, 'naca_chord', 2.0)
                airfoil_x = getattr(self.sim_params, 'naca_x', 2.5)
                airfoil_y = getattr(self.sim_params, 'naca_y', 1.875)
                
                # Compute forces using circulation-based method
                cl, cd = compute_forces_ibm(u_np, v_np, vort, X_np, np.array(self.grid.Y), mask_np,
                                              dx, dy, self.flow.U_inf, chord_length, airfoil_x, airfoil_y,
                                              self.grid.lx, grid_type='collocated')
                
                # Compute pressure coefficient
                rho = 1.0
                surface = get_airfoil_surface_mask(mask_np, dx, threshold=0.1)
                p_inf = 0.0
                q_inf = 0.5 * rho * self.flow.U_inf**2
                cp = (p_np - p_inf) / q_inf
                cp_surface = np.where(surface, cp, np.inf)
                cp_min = float(np.min(cp_surface))
                
                # Compute wake deficit
                wake_x = airfoil_x + chord_length
                wake_x_idx = int(wake_x / dx)
                wake_deficit = 0.0
                if 0 <= wake_x_idx < self.grid.nx:
                    u_wake = u_np[wake_x_idx, :]
                    wake_deficit = float(self.flow.U_inf - np.mean(u_wake[mask_np[wake_x_idx, :] > 0.5]))
                
                # Store metrics in history (same format as baseline solver)
                self.history['airfoil_metrics']['stagnation_x'].append(stag_x)
                self.history['airfoil_metrics']['separation_x'].append(sep_x)
                self.history['airfoil_metrics']['CL'].append(cl)
                self.history['airfoil_metrics']['CD'].append(cd)
                self.history['airfoil_metrics']['Cp_min'].append(cp_min)
                self.history['airfoil_metrics']['wake_deficit'].append(wake_deficit)
                self.history['airfoil_metrics']['strouhal'].append(0.0)  # Initialize with .0, updated when stable
                self.history['airfoil_metrics']['time'].append(self.iteration * self.dt)
                
                # Store drag and lift for compatibility
                self.history['drag'].append(float(cd))
                self.history['lift'].append(float(cl))
                
                # Check for vortex shedding stability (same as baseline solver)
                if len(self.history['airfoil_metrics']['CL']) >= 30 and self.iteration % 25 == 0:
                    from solver.metrics import detect_vortex_shedding_stability, compute_time_averaged_coefficients
                    
                    cl_history = np.array(self.history['airfoil_metrics']['CL'])
                    cd_history = np.array(self.history['airfoil_metrics']['CD'])
                    times = np.array(self.history['time'])
                    
                    is_stable, stable_start, strouhal = detect_vortex_shedding_stability(
                        cl_history, times, self.flow.U_inf, chord_length
                    )
                    
                    # Store stability state
                    if 'stability_state' not in self.history:
                        self.history['stability_state'] = []
                    
                    self.history['stability_state'].append({
                        'iteration': self.iteration,
                        'is_stable': is_stable,
                        'stable_start': stable_start,
                        'strouhal': strouhal
                    })
                    
                    # If stable, compute time-averaged coefficients
                    if is_stable and stable_start > 0:
                        avg_cl, avg_cd, num_samples = compute_time_averaged_coefficients(
                            cl_history, cd_history, stable_start
                        )
                        
                        # Store time-averaged values
                        if 'time_averaged' not in self.history:
                            self.history['time_averaged'] = {
                                'CL': [], 'CD': [], 'stable_start': [], 'strouhal': [], 'num_samples': []
                            }
                        
                        self.history['time_averaged']['CL'].append(avg_cl)
                        self.history['time_averaged']['CD'].append(avg_cd)
                        self.history['time_averaged']['stable_start'].append(stable_start)
                        self.history['time_averaged']['strouhal'].append(strouhal)
                        self.history['time_averaged']['num_samples'].append(num_samples)
                        
                        # Update recent Strouhal values in airfoil metrics
                        if len(self.history['airfoil_metrics']['strouhal']) > 0:
                            num_updates = min(10, len(self.history['airfoil_metrics']['strouhal']))
                            for i in range(num_updates):
                                self.history['airfoil_metrics']['strouhal'][-(i+1)] = strouhal
                        
            except Exception as e:
                print(f"LBM Error computing airfoil metrics: {e}")
                # Append zeros to maintain history consistency
                for key in self.history['airfoil_metrics']:
                    self.history['airfoil_metrics'][key].append(0.0)
                self.history['drag'].append(0.0)
                self.history['lift'].append(0.0)
        
        return u, v, vort, div
    
    def update_flow_parameters(self):
        """Update LBM parameters when flow parameters change (Re, U, nu)"""
        print("LBM: Updating flow parameters...")
        
        # Recalculate LBM parameters based on new flow parameters
        cs_squared = 1.0 / 3.0
        
        # Choose target lattice Reynolds number for aggressive diffusion reduction
        # Much higher Re_lattice = much less diffusion, push stability limits
        # For vortex shedding, aim for Re_lattice > 1000 for strong vortices
        target_Re_lattice = min(5000.0, self.flow.Re * 10.0)  # Much more aggressive scaling
        
        # Calculate lattice velocity based on physical velocity
        # Scale physical velocity to lattice units while maintaining stability
        physical_velocity = self.flow.U_inf
        
        # Calculate characteristic length scale (chord length or obstacle diameter)
        if hasattr(self.sim_params, 'naca_chord'):
            L_char = self.sim_params.naca_chord
        else:
            L_char = self.geom.radius * 2  # Diameter for cylinder
        
        # Calculate lattice velocity scaling with aggressive approach for less diffusion
        dx = self.grid.lx / self.grid.nx
        
        # Target much higher lattice velocity for less diffusion
        # Push to higher Mach number for better flow physics
        target_U_lattice = 0.25 * jnp.sqrt(cs_squared)  # Mach ~0.25
        
        # Scale physical velocity relative to a reference (e.g., 2.0 m/s)
        reference_velocity = 2.0  # Reference physical velocity
        velocity_scale_factor = physical_velocity / reference_velocity
        
        # Apply scaling to target lattice velocity
        self.U_lattice = target_U_lattice * velocity_scale_factor
        
        # Apply more relaxed stability constraints (Mach < 0.35 for aggressive flow)
        max_U_lattice = 0.35 * jnp.sqrt(cs_squared)
        min_U_lattice = 0.08  # Higher minimum for better flow
        self.U_lattice = jnp.clip(self.U_lattice, min_U_lattice, max_U_lattice)
        
        # Use grid characteristic length as L_lattice
        L_lattice = min(self.grid.nx, self.grid.ny)
        
        # Calculate required lattice viscosity
        nu_lattice = self.U_lattice * L_lattice / target_Re_lattice
        
        # Convert to tau
        tau_target = 0.5 + nu_lattice / cs_squared
        
        # Ensure stability bounds
        tau_min = 0.51  # Just above stability limit
        tau_max = 2.0   # Upper bound for stability
        self.lbm_params.tau = jnp.clip(tau_target, tau_min, tau_max)
        self.lbm_params.omega = 1.0 / self.lbm_params.tau
        
        print(f"LBM: Updated parameters - Re={self.flow.Re:.1f}, tau={self.lbm_params.tau:.4f}, "
              f"U_lattice={self.U_lattice:.4f}, omega={self.lbm_params.omega:.4f}")
        
        # Recompile JIT functions with new parameters
        self.recompile_jit()

    def recompile_jit(self):
        """Recompile JIT functions (called when parameters change)"""
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        print("LBM: JIT cache cleared and recompiled")
