"""
Base solver class with core initialization.
The BaselineSolver class is defined here with its __init__ method.
Other methods are imported from submodules and attached.
"""

import jax
import jax.numpy as jnp
from typing import Optional

# Import from local modules
from ..params import (
    GridParams, FlowParams, GeometryParams, SimulationParams, SimState
)
from ..boundary_conditions import (
    apply_cavity_boundary_conditions, create_cavity_mask,
    apply_taylor_green_boundary_conditions, create_taylor_green_mask,
    apply_backward_step_boundary_conditions
)
from ..pure_functions import step_pure, update_dt_pure, apply_corner_smooth_inlet, apply_corner_smooth_pressure_gradient
from ..operators import (
    grad_x, grad_y, grad_x_nonperiodic, grad_y_nonperiodic,
    divergence, divergence_nonperiodic,
    vorticity, vorticity_nonperiodic
)
from ..les_models import dynamic_smagorinsky, constant_smagorinsky

# Constants
rho = 1.0  # Fluid density

# Import from submodules
from .initializers import _initialize_von_karman_flow, _initialize_cavity_flow, _initialize_taylor_green_flow
from .mask_generator import _compute_mask
from .step_handlers import _step, _step_collocated, _step_mac, get_step_jit
from .flow_control import apply_flow_type, set_obstacle_type, update_naca_angle, inject_dye, set_adaptive_dt, set_fixed_dt
from .visualization_step import step_pure_profiled, step_for_visualization

# Import timestep controllers
try:
    from timestepping.cfl_adaptive_dt import CFLAdaptiveController
except ImportError:
    CFLAdaptiveController = None

# Import NACA
try:
    from obstacles.naca_airfoils import NACA_AVAILABLE
except ImportError:
    NACA_AVAILABLE = False

# Import pressure solvers
try:
    from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
    MAC_PRESSURE_AVAILABLE = True
except ImportError:
    MAC_PRESSURE_AVAILABLE = False

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

try:
    from pressure_solvers import poisson_multigrid, poisson_jacobi
except ImportError:
    poisson_multigrid = None
    poisson_jacobi = None

# Import MAC operators
try:
    from advection_schemes.rk3_mac import rk_step_unified_mac
    MAC_ADVECTION_AVAILABLE = True
except ImportError:
    MAC_ADVECTION_AVAILABLE = False

try:
    from solver.operators_mac import (
        divergence_staggered, divergence_nonperiodic_staggered,
        vorticity_staggered, vorticity_nonperiodic_staggered
    )
    MAC_OPERATORS_AVAILABLE = True
except ImportError:
    MAC_OPERATORS_AVAILABLE = False

# Import scalar advection
try:
    from ..operators import scalar_advection_diffusion_periodic, scalar_advection_diffusion_nonperiodic
except ImportError:
    scalar_advection_diffusion_periodic = None
    scalar_advection_diffusion_nonperiodic = None

# Import staggered scalar advection
try:
    from solver.operators_mac import (
        scalar_advection_diffusion_periodic_staggered,
        scalar_advection_diffusion_nonperiodic_staggered
    )
except ImportError:
    scalar_advection_diffusion_periodic_staggered = None
    scalar_advection_diffusion_nonperiodic_staggered = None

# Import metrics
try:
    from ..metrics import (
        find_stagnation_point, find_separation_point, compute_forces_ibm,
        get_airfoil_surface_mask, detect_vortex_shedding_stability,
        compute_time_averaged_coefficients
    )
except ImportError:
    find_stagnation_point = None
    find_separation_point = None
    compute_forces_ibm = None
    get_airfoil_surface_mask = None
    detect_vortex_shedding_stability = None
    compute_time_averaged_coefficients = None

# Import neural network pressure operator
try:
    from neural_operators.nn_pressure_operator_linear import LearnedPressureOperator
    NN_PRESSURE_AVAILABLE = True
except ImportError:
    NN_PRESSURE_AVAILABLE = False

# Import scalar advection
try:
    from ..operators import scalar_advection_diffusion_periodic, scalar_advection_diffusion_nonperiodic
except ImportError:
    scalar_advection_diffusion_periodic = None
    scalar_advection_diffusion_nonperiodic = None

# Import staggered scalar advection
try:
    from solver.operators_mac import (
        scalar_advection_diffusion_periodic_staggered,
        scalar_advection_diffusion_nonperiodic_staggered
    )
except ImportError:
    scalar_advection_diffusion_periodic_staggered = None
    scalar_advection_diffusion_nonperiodic_staggered = None

# Constants
rho = 1.0  # Fluid density


class BaselineSolver:
    """Main Navier-Stokes solver class"""
    
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
        
        # Default grid dimensions for a general flow type
        # Grid dimensions are set in _initialize_general_flow method

        # Apply percentage-based scaling for NACA airfoil parameters
        if sim_params.obstacle_type == 'naca_airfoil':
            # Scale x-position as percentage of domain width (25% of lx)
            x_percentage = 0.25  # 25% from left
            sim_params.naca_x = x_percentage * grid.lx

            # Scale y-position as percentage of domain height (50% of ly)
            y_percentage = 0.5  # Centered in Y
            sim_params.naca_y = y_percentage * grid.ly

            # Scale chord length as percentage of domain width (15% of lx)
            chord_percentage = 0.15  # 15% of domain width
            sim_params.naca_chord = chord_percentage * grid.lx
            print(f"DEBUG Solver __init__: obstacle_type={sim_params.obstacle_type}, grid.lx={grid.lx:.3f}, scaled naca_chord={sim_params.naca_chord:.3f}")
        else:
            print(f"DEBUG Solver __init__: obstacle_type={sim_params.obstacle_type}, skipping NACA scaling")

        # Compute characteristic length based on flow type and geometry
        from solver.params import compute_characteristic_length
        self.flow.L_char = compute_characteristic_length(sim_params.flow_type, geom, sim_params, sim_params.obstacle_type)
        
        # Resolve flow constraints to ensure consistent physics
        self.flow.resolve()
        print(f"Flow parameters resolved: U={self.flow.U_inf:.3f}, nu={self.flow.nu:.6f}, Re={self.flow.Re:.1f}, L={self.flow.L_char:.3f}")
        
        # Get Re-dependent parameters for stability
        from ..params import get_re_parameters
        re_params = get_re_parameters(self.flow.Re, self.grid.nx)
        
        # Apply Re-dependent parameters
        self.sim_params.smagorinsky_constant = re_params['C_s']
        self.sim_params.nu_h = re_params['nu_h']
        self.sim_params.brinkman_eta = re_params['brinkman_eta']
        self.sim_params.dt_max = re_params['dt_max']
        
        # Store nu_hyper_ratio for use in RK3 scheme
        self.nu_hyper_ratio = re_params['nu_hyper_ratio']
        
        # Wall boundary condition (slip vs no-slip)
        self.slip_walls = True  # Default to slip walls
        
        # Initialize SDF (will be set in _compute_mask)
        self.sdf = None
        
        print(f"Re-dependent parameters: C_s={re_params['C_s']:.3f}, nu_hyper_ratio={re_params['nu_hyper_ratio']:.3f}, "
              f"brinkman_eta={re_params['brinkman_eta']:.3f}, dt_max={re_params['dt_max']:.4f}")
        
        # Initialize divergence PID controller
        if sim_params.adaptive_dt and CFLAdaptiveController is not None:
            self.dt_controller = CFLAdaptiveController(
                dt_min=sim_params.dt_min, dt_max=sim_params.dt_max
            )
        else:
            self.dt_controller = None
        
        # Smart dt initialization
        print(f"DEBUG: adaptive_dt = {self.sim_params.adaptive_dt} during initialization")
        if self.sim_params.adaptive_dt:
            # Always use CFL-based dt when adaptive dt is enabled for stability
            # Use lower CFL target for high velocities and high Reynolds numbers
            if self.sim_params.flow_type == 'lid_driven_cavity':
                cfl_target = 0.1  # Conservative for LDC
            elif self.flow.U_inf > 5.0:
                cfl_target = 0.05  # Very conservative for very high velocities
            elif self.flow.U_inf > 3.0:
                cfl_target = 0.1   # Conservative for high velocities
            elif self.flow.U_inf > 1.5:
                cfl_target = 0.15  # Conservative for moderate velocities
            else:
                cfl_target = 0.3   # Normal for low velocities

            # Further reduce CFL target for low viscosities (high Reynolds numbers)
            # Low viscosity means less viscous damping, requiring smaller timesteps for stability
            if self.flow.nu < 1e-4:
                cfl_target = min(cfl_target, 0.1)  # Reduced from 0.02 - less aggressive
            elif self.flow.nu < 1e-3:
                cfl_target = min(cfl_target, 0.15)   # Reduced from 0.05 - less aggressive
            elif self.flow.nu < 2e-3:
                cfl_target = min(cfl_target, 0.2)    # Reduced from 0.1 - less aggressive

            dx = self.grid.dx
            dy = self.grid.dy
            max_velocity = self.flow.U_inf
            dt_cfl = cfl_target * min(dx, dy) / (max_velocity + 1e-8)
            dt_diffusion = 0.25 * min(dx**2, dy**2) / self.flow.nu

            # Reduce dt_max for low viscosity and high velocity cases to prevent instability
            dt_max = self.sim_params.dt_max
            if self.sim_params.flow_type == 'lid_driven_cavity':
                dt_max = min(dt_max, 0.008)  # Relaxed from 0.005
            if self.flow.nu < 1e-4:
                dt_max = min(dt_max, 0.003)  # Relaxed from 0.0015
            elif self.flow.nu < 1e-3:
                dt_max = min(dt_max, 0.005)  # Relaxed from 0.003
            elif self.flow.nu < 2e-3:
                dt_max = min(dt_max, 0.006)  # Relaxed from 0.0045

            # Additional dt_max reduction for high velocities
            if self.flow.U_inf > 5.0:
                dt_max = min(dt_max, 0.003)  # Relaxed from 0.0015
            elif self.flow.U_inf > 3.0:
                dt_max = min(dt_max, 0.005)  # Relaxed from 0.003
            elif self.flow.U_inf > 1.5:
                dt_max = min(dt_max, 0.006)  # Relaxed from 0.0045

            self.dt = min(dt_cfl, dt_diffusion, dt_max)
            self.dt = max(self.dt, self.sim_params.dt_min)
            if dt is not None and dt != self.dt:
                print(f"User-specified dt={dt:.6f} overridden by CFL-based dt={self.dt:.6f} for stability (CFL={cfl_target})")
            else:
                print(f"Using adaptive dt, auto-calculated initial dt = {self.dt:.6f} (CFL={cfl_target})")
        elif dt is not None:
            self.dt = dt
            self.sim_params.fixed_dt = dt
            print(f"Using user-specified fixed dt = {dt:.6f}")
        else:
            self.dt = self.sim_params.fixed_dt
            print(f"Using fixed dt = {self.dt:.6f} from simulation parameters")
        
        # Update Y positions based on actual domain height
        # Only set default center if not already set (preserves user slider position)
        domain_center_y = self.grid.ly / 2
        if not hasattr(self.geom, 'center_y') or self.geom.center_y is None:
            self.geom.center_y = jnp.array(domain_center_y)
        if not hasattr(self.sim_params, 'naca_airfoil') or self.sim_params.naca_airfoil == 'none':
            if not hasattr(self.sim_params, 'naca_y') or self.sim_params.naca_y is None:
                self.sim_params.naca_y = domain_center_y
        
        # Grid-consistent ε
        from ..params import compute_eps_multiplier
        jax.clear_caches()  # Clear JIT cache to ensure new code is compiled
        if self.sim_params.auto_eps_multiplier:
            self.sim_params.eps_multiplier = compute_eps_multiplier(self.flow.Re)
            print(f"Auto-computed eps_multiplier = {self.sim_params.eps_multiplier} from Re = {self.flow.Re:.1f}")
        self.sim_params.eps = self.sim_params.eps_multiplier * self.grid.dx
        print(f"Setting ε = {self.sim_params.eps:.4f} ({self.sim_params.eps_multiplier} * dx)")
        
        # Pre-compute mask
        self.mask = self._compute_mask()
        
        # Initialize based on flow type
        if self.sim_params.flow_type == 'lid_driven_cavity':
            self._initialize_cavity_flow()
        elif self.sim_params.flow_type == 'taylor_green':
            self._initialize_taylor_green_flow()
        else:
            self._initialize_von_karman_flow()
        
        self._jit_cache = {}
        try:
            # Force adaptive_dt=False to prevent dt mismatch
            self.sim_params.adaptive_dt = False
            self._step_jit = self.get_step_jit()
            print(f"Successfully initialized _step_jit with adaptive_dt=False")
        except Exception as e:
            print(f"ERROR: Failed to initialize _step_jit: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Use appropriate divergence/vorticity functions based on grid type and flow type
        if self.sim_params.grid_type == 'mac':
            if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
                self._vorticity = jax.jit(vorticity_nonperiodic_staggered, static_argnums=(2, 3))
                self._divergence = jax.jit(divergence_nonperiodic_staggered, static_argnums=(2, 3))
            else:
                self._vorticity = jax.jit(vorticity_staggered, static_argnums=(2, 3))
                self._divergence = jax.jit(divergence_staggered, static_argnums=(2, 3))
        else:
            # Collocated grid
            from ..operators import vorticity, vorticity_nonperiodic, divergence, divergence_nonperiodic
            if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
                self._vorticity = jax.jit(vorticity_nonperiodic, static_argnums=(2, 3))
                self._divergence = jax.jit(divergence_nonperiodic, static_argnums=(2, 3))
            else:
                self._vorticity = jax.jit(vorticity, static_argnums=(2, 3))
                self._divergence = jax.jit(divergence, static_argnums=(2, 3))
        
        print(f"Initialized with: {self.sim_params.advection_scheme} advection, {self.sim_params.pressure_solver} pressure solver, grid_type={self.sim_params.grid_type}")
        
        self.history = {
            'time': [], 'dt': [], 'drag': [], 'lift': [],
            'l2_change': [], 'rms_change': [], 'l2_change_u': [], 'l2_change_v': [], 'max_change': [], 'change_99p': [], 'rel_change': [],
            'rms_divergence': [], 'l2_divergence': [],
            'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 'Cp_min': [], 'wake_deficit': [], 'strouhal': [], 'time': []}
        }
        self.iteration = 0
        
        self.u_prev = jnp.copy(self.u)
        self.v_prev = jnp.copy(self.v)
        self.c = jnp.zeros((self.grid.nx, self.grid.ny))
        self.current_pressure = jnp.zeros((self.grid.nx, self.grid.ny))
        self.compute_airfoil_metrics = False
        self.metrics_frame_skip = 100  # Compute metrics every N frames (reduced from 1 for better performance)
        
        # Scalar/dye update flag - set to True to match UI dye checkbox default state
        self.enable_scalar_update = True  # Enabled by default to match UI checkbox state
        
        # Profiling flag - set to True to enable timing per timestep
        self.enable_profiling = False  # Disabled to focus on visualization profiling
        self.detailed_profiling = False  # Set to True for detailed breakdown of step_jit internals (slower)
        self.profiling_data = {
            'step_jit': [],
            'adaptive_dt': [],
            'scalar_update': [],
            'diagnostics': [],
            'airfoil_metrics': []
        }
        # Add detailed profiling categories if detailed profiling is enabled
        if self.detailed_profiling:
            self.profiling_data.update({
                'les_compute': [],
                'advection': [],
                'divergence': [],
                'pressure_solve': [],
                'pressure_gradient': [],
                'pressure_correction': [],
                'boundary_conditions': []
            })
        
        self.state = SimState(
            u=self.u, v=self.v, p=self.current_pressure,
            u_prev=self.u_prev, v_prev=self.v_prev, c=self.c,
            dt=self.dt, iteration=self.iteration,
            grid_type=self.sim_params.grid_type,
            integral=self.dt_controller.integral if self.dt_controller else 0.0,
            prev_error=self.dt_controller.prev_error if self.dt_controller else 0.0
        )
        
        # Initialize neural network pressure model if requested
        self.nn_pressure_model = None
        if self.sim_params.pressure_solver == 'nn' and NN_PRESSURE_AVAILABLE:
            key = jax.random.PRNGKey(self.seed)
            self.nn_pressure_model = LearnedPressureOperator(in_channels=2, features=32,
                                                             num_blocks=3, key=key)
            print("Using learned pressure operator (Equinox) - untrained weights")
        elif self.sim_params.pressure_solver == 'nn' and not NN_PRESSURE_AVAILABLE:
            print("Warning: pressure_solver='nn' but NN_PRESSURE_AVAILABLE=False. Falling back to multigrid.")
            self.sim_params.pressure_solver = 'multigrid'

# Attach methods from submodules
BaselineSolver._initialize_von_karman_flow = _initialize_von_karman_flow
BaselineSolver._initialize_cavity_flow = _initialize_cavity_flow
BaselineSolver._initialize_taylor_green_flow = _initialize_taylor_green_flow
BaselineSolver._compute_mask = _compute_mask
BaselineSolver._step = _step
BaselineSolver._step_collocated = _step_collocated
BaselineSolver._step_mac = _step_mac
BaselineSolver.get_step_jit = get_step_jit
BaselineSolver.apply_flow_type = apply_flow_type
BaselineSolver.set_obstacle_type = set_obstacle_type
BaselineSolver.update_naca_angle = update_naca_angle
BaselineSolver.inject_dye = inject_dye
BaselineSolver.set_adaptive_dt = set_adaptive_dt
BaselineSolver.set_fixed_dt = set_fixed_dt
BaselineSolver.step_pure_profiled = step_pure_profiled
BaselineSolver.step_for_visualization = step_for_visualization
