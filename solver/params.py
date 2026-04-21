"""
Parameter dataclasses for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Dict, Tuple


@jax.tree_util.register_dataclass
@dataclass
class SimState:
    """Full simulation state as a PyTree for JAX compatibility
    
    For collocated grid: u, v, p are all (nx, ny)
    For MAC staggered grid: u is (nx+1, ny), v is (nx, ny+1), p is (nx, ny)
    """
    u: jnp.ndarray
    v: jnp.ndarray
    p: jnp.ndarray  # Pressure field (always cell-centered)
    u_prev: jnp.ndarray
    v_prev: jnp.ndarray
    c: jnp.ndarray  # Scalar dye field (always cell-centered)
    dt: float
    iteration: int
    grid_type: str = 'collocated'  # 'collocated' or 'mac'
    # PID controller internal state
    integral: float = 0.0
    prev_error: float = 0.0


@dataclass
class GridParams:
    nx: int
    ny: int
    lx: float
    ly: float
    dx: float = field(init=False)
    dy: float = field(init=False)
    x: jnp.ndarray = field(init=False)
    y: jnp.ndarray = field(init=False)
    X: jnp.ndarray = field(init=False)
    Y: jnp.ndarray = field(init=False)
    # Staggered grid coordinates (for MAC grid)
    x_u: jnp.ndarray = field(init=False)  # x-velocity face locations (nx+1)
    y_v: jnp.ndarray = field(init=False)  # y-velocity face locations (ny+1)
    X_u: jnp.ndarray = field(init=False)  # Meshgrid for u-faces
    Y_v: jnp.ndarray = field(init=False)  # Meshgrid for v-faces
    
    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x = jnp.linspace(0, self.lx, self.nx)
        self.y = jnp.linspace(0, self.ly, self.ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')
        # Staggered coordinates (face centers)
        self.x_u = jnp.linspace(0, self.lx, self.nx + 1)  # x-faces including boundaries
        self.y_v = jnp.linspace(0, self.ly, self.ny + 1)  # y-faces including boundaries
        self.X_u, _ = jnp.meshgrid(self.x_u, self.y, indexing='ij')  # For u-velocity (nx+1, ny)
        _, self.Y_v = jnp.meshgrid(self.x, self.y_v, indexing='ij')  # For v-velocity (nx, ny+1)


def compute_characteristic_length(flow_type: str, geom, sim_params=None, obstacle_type: str = 'cylinder') -> float:
    """
    Compute characteristic length based on flow type and geometry.

    Args:
        flow_type: Type of flow simulation
        geom: GeometryParams object containing geometry information
        sim_params: SimulationParams object containing obstacle parameters
        obstacle_type: Type of obstacle ('cylinder' or 'naca_airfoil')

    Returns:
        Characteristic length L for Reynolds number calculation
    """
    # For NACA airfoils, use chord length
    if obstacle_type == 'naca_airfoil':
        if sim_params is not None and hasattr(sim_params, 'naca_chord'):
            return float(sim_params.naca_chord)
        else:
            return 2.0  # Default chord length

    if flow_type == "von_karman":
        return 2.0 * float(geom.radius)
    elif flow_type == "lid_driven_cavity":
        # Use cavity width (domain width)
        return float(geom.lx) if hasattr(geom, 'lx') else 1.0
    elif flow_type == "channel_flow":
        # Use channel height
        return float(geom.ly) if hasattr(geom, 'ly') else 1.0
    elif flow_type == "backward_step":
        # Use channel height
        return float(geom.ly) if hasattr(geom, 'ly') else 1.0
    elif flow_type == "taylor_green":
        # Use domain size (2π for periodic box)
        return float(2.0 * jnp.pi)
    else:
        # Default fallback
        return 1.0


@dataclass
class FlowConstraints:
    """Constraint locks for flow parameters."""
    lock_U: bool = True
    lock_nu: bool = True
    lock_Re: bool = False
    lock_L: bool = True


@dataclass
class FlowParams:
    U_inf: float = 1.0
    nu: float = None
    Re: float = None
    L_char: float = 1.0
    constraints: FlowConstraints = field(default_factory=FlowConstraints)
    
    def __post_init__(self):
        """Initialize viscosity if not provided using default constraints."""
        if self.nu is None and self.Re is not None:
            # Default to (U, Re, L) -> solve nu
            self.nu = (self.U_inf * self.L_char) / self.Re
    
    def resolve(self):
        """
        Solve missing variables based on locked constraints.
        Exactly TWO of (U, nu, Re) must be locked (L is usually geometry-driven).
        """
        c = self.constraints
        
        # Count locked physical variables (excluding L which is usually geometry-driven)
        locked_count = sum([c.lock_U, c.lock_nu, c.lock_Re])
        
        # CASE 1: Physical mode (U, nu, L known → Re derived)
        if c.lock_U and c.lock_nu and c.lock_L:
            if self.nu == 0:
                raise ValueError("Viscosity cannot be zero")
            self.Re = (self.U_inf * self.L_char) / self.nu
            return
        
        # CASE 2: (U, Re, L) → solve nu
        if c.lock_U and c.lock_Re and c.lock_L:
            if self.Re == 0:
                raise ValueError("Reynolds number cannot be zero")
            self.nu = (self.U_inf * self.L_char) / self.Re
            return
        
        # CASE 3: (nu, Re, L) → solve U
        if c.lock_nu and c.lock_Re and c.lock_L:
            if self.L_char == 0:
                raise ValueError("Characteristic length cannot be zero")
            self.U_inf = (self.Re * self.nu) / self.L_char
            return
        
        raise ValueError(
            f"Invalid constraint setup. Must lock exactly 2 of (U, nu, Re) + L from geometry. "
            f"Current locks: U={c.lock_U}, nu={c.lock_nu}, Re={c.lock_Re}, L={c.lock_L}"
        )
    
    def compute_reynolds(self):
        """Compute Reynolds number from current state."""
        if self.nu == 0:
            raise ValueError("Viscosity cannot be zero")
        return (self.U_inf * self.L_char) / self.nu


@dataclass
class GeometryParams:
    center_x: jnp.ndarray
    center_y: jnp.ndarray
    radius: jnp.ndarray
    
    def to_dict(self) -> Dict:
        return {
            'center_x': self.center_x,
            'center_y': self.center_y,
            'radius': self.radius
        }
    
    def to_params(self) -> jnp.ndarray:
        """Convert to JAX array for optimization"""
        return jnp.concatenate([jnp.ravel(self.center_x), jnp.ravel(self.center_y), jnp.ravel(self.radius)])
    
    def from_params(self, params: jnp.ndarray):
        """Update from optimized parameters"""
        self.center_x = jnp.array(params[0])
        self.center_y = jnp.array(params[1])
        self.radius = jnp.array(params[2])
    
    def get_bounds(self):
        """Get parameter bounds for optimization"""
        return (
            (0.5, 19.5),   # center_x: keep away from boundaries
            (0.5, 4.0),    # center_y: keep away from boundaries  
            (0.05, 0.5)    # radius: reasonable range
        )


@dataclass
class CavityGeometryParams:
    """Parameters for lid-driven cavity flow"""
    lid_velocity: float = 1.0  # Top wall velocity
    cavity_width: float = 1.0  # Cavity width
    cavity_height: float = 1.0  # Cavity height


@dataclass
class ChannelGeometryParams:
    """Parameters for channel flow"""
    inlet_velocity: float = 1.0  # Inlet velocity
    channel_height: float = 1.0  # Channel height
    channel_length: float = 4.0  # Channel length


@dataclass
class BackwardStepGeometryParams:
    """Parameters for backward-facing step flow"""
    inlet_velocity: float = 1.0  # Inlet velocity
    step_height: float = 0.5  # Step height
    channel_height: float = 1.0  # Channel height
    channel_length: float = 10.0  # Channel length


@dataclass
class TaylorGreenGeometryParams:
    """Parameters for Taylor-Green vortex flow"""
    domain_size: float = 2 * jnp.pi  # Domain size (2π for Taylor-Green)
    initial_amplitude: float = 1.0  # Initial vortex amplitude


def compute_eps_multiplier(Re: float) -> float:
    """
    Compute eps_multiplier based on Reynolds number.
    Uses thicker masks for high Re to improve stability.

    Args:
        Re: Reynolds number

    Returns:
        eps_multiplier: Multiplier for grid spacing to compute epsilon
    """
    if Re <= 1000:
        return 0.01
    elif Re <= 2000:
        return 0.5  # Thicker mask for Re=2000
    elif Re <= 5000:
        return 1.0  # Even thicker for higher Re
    else:
        return 1.5  # Maximum thickness for very high Re


@dataclass
class SimulationParams:
    eps_multiplier: float = 0.01  # Grid-consistent ε: eps = eps_multiplier * dx (set by slider)
    auto_eps_multiplier: bool = False  # If True, eps_multiplier is auto-computed from Re (disabled by default)
    eps: float = 0.02  # Deprecated: will be computed as eps = eps_multiplier * grid.dx
    advection_scheme: str = 'rk3'  # 'rk3', 'spectral', 'weno5', 'tvd'
    limiter: str = 'minmod'  # for TVD
    weno_epsilon: float = 1e-6  # for WENO
    max_cfl: float = 0.3  # maximum CFL number
    adaptive_dt: bool = False  # disabled to prevent timestep oscillation instability
    fixed_dt: float = 0.01  # User-specified fixed timestep
    pressure_solver: str = 'multigrid'  # 'cg', 'fft', 'multigrid', 'sor_masked', 'cg_masked'
    sor_omega: float = 1.5  # SOR relaxation parameter
    pressure_max_iter: int = 100  # max iterations for iterative solvers
    pressure_tolerance: float = 1e-6  # tolerance for early stopping
    multigrid_levels: int = 4  # multigrid grid coarsening levels
    multigrid_v_cycles: int = 7  # multigrid V-cycles per solve
    flow_type: str = 'von_karman'  # 'von_karman', 'lid_driven_cavity', 'taylor_green'
    dt_min: float = 1e-6  # Minimum timestep
    dt_max: float = 0.01   # Maximum timestep
    grid_type: str = 'collocated'  # 'collocated' or 'mac' (staggered grid)

    # NACA airfoil parameters
    obstacle_type: str = 'naca_airfoil'  # 'cylinder', 'naca_airfoil', 'cow', or 'three_cylinder_array'
    naca_airfoil: str = 'NACA 0012'  # Airfoil designation
    naca_chord: float = 0.3  # Chord length (reduced from 0.45 to prevent extending near top boundary)
    naca_angle: float = 0.0  # Angle of attack in degrees (set to 0 to prevent extending near top boundary)
    naca_x: float = 2.0  # X position (will be scaled as 25% of lx when grid changes)
    naca_y: float = 1.5  # Y position (lowered from 2.25 to prevent thin fluid channel near top boundary)
    
    # LES/SGS model parameters
    use_les: bool = False  # Enable/disable LES
    les_model: str = 'dynamic_smagorinsky'  # 'dynamic_smagorinsky' or 'smagorinsky'
    smagorinsky_constant: float = 0.17  # Constant C_s for Smagorinsky model
    
    # Scalar/dye parameters
    use_scalar: bool = True  # Enable/disable scalar field (dye)
    scalar_diffusivity: float = 0.01  # Scalar diffusivity (increased to reduce Gibbs oscillations)
    
    # Hyper-viscosity for damping high-frequency oscillations
    nu_h: float = 5e-4  # 4th-order hyper-viscosity coefficient (increased to damp Gibbs oscillations)
    
    # Brinkman penalization for soft IBM (reduces upstream reflections)
    brinkman_eta: float = 1e-4  # Penalization parameter (smaller = stiffer) - appropriate for solid obstacles with implicit treatment


def get_re_parameters(Re: float, grid_nx: int = 512) -> dict:
    """
    Return Re-dependent simulation parameters for stability.
    
    Args:
        Re: Reynolds number
        grid_nx: Grid resolution in x-direction (used to estimate grid quality)
    
    Returns:
        Dictionary of parameters tuned for current Re
    """
    
    # Base parameters for low Re (Re <= 5000)
    params = {
        'C_s': 0.1,                    # Smagorinsky constant
        'nu_hyper_ratio': 0.0,         # Hyper-viscosity ratio (nu_hyper / nu)
        'advection_scheme': 'rk3_simple',  # 'rk3_simple' or 'rk3_high'
        'nu_h': 0.0,                   # High-order hyper-viscosity
        'brinkman_eta': 1e-4,          # Brinkman penalization for solid obstacles with implicit treatment
        'dt_max': 0.01,                # Maximum timestep
        'cfl_target': 0.3,             # Target CFL number
        'use_les': True,               # Enable LES
    }
    
    # Re-dependent scaling
    if Re <= 1000:
        # Low Re: minimal dissipation, accurate
        params.update({
            'C_s': 0.08,
            'nu_hyper_ratio': 0.0,
            'advection_scheme': 'rk3_simple',
            'nu_h': 0.0,
            'brinkman_eta': 1e-4,    # Solid obstacle with implicit penalization
            'dt_max': 0.005,
            'cfl_target': 0.3,
        })

    elif Re <= 10000:
        # Moderate Re: moderate dissipation
        params.update({
            'C_s': 0.1,
            'nu_hyper_ratio': 0.01,    # 1% hyper-viscosity
            'advection_scheme': 'rk3_simple',
            'nu_h': 0.0,
            'brinkman_eta': 1e-5,    # Solid obstacle with implicit penalization
            'dt_max': 0.008,
            'cfl_target': 0.25,
        })

    elif Re <= 100000:
        # High Re: more dissipation
        params.update({
            'C_s': 0.12,
            'nu_hyper_ratio': 0.03,    # 3% hyper-viscosity
            'advection_scheme': 'rk3_simple',
            'nu_h': 0.0,
            'brinkman_eta': 1e-6,    # Solid obstacle with implicit penalization
            'dt_max': 0.005,
            'cfl_target': 0.2,
        })

    elif Re <= 1000000:
        # Very high Re: strong dissipation
        params.update({
            'C_s': 0.15,
            'nu_hyper_ratio': 0.05,    # 5% hyper-viscosity
            'advection_scheme': 'rk3_high',  # Switch to high-order with filtering
            'nu_h': 1e-5,
            'brinkman_eta': 1e-7,    # Solid obstacle with implicit penalization
            'dt_max': 0.003,
            'cfl_target': 0.15,
        })

    elif Re <= 10000000:
        # Extreme Re: very strong dissipation
        params.update({
            'C_s': 0.18,
            'nu_hyper_ratio': 0.08,    # 8% hyper-viscosity
            'advection_scheme': 'rk3_high',
            'nu_h': 2e-5,
            'brinkman_eta': 1e-7,    # Solid obstacle with implicit penalization
            'dt_max': 0.002,
            'cfl_target': 0.1,
        })

    else:
        # Ultra-high Re: maximum dissipation
        params.update({
            'C_s': 0.20,
            'nu_hyper_ratio': 0.1,     # 10% hyper-viscosity
            'advection_scheme': 'rk3_high',
            'nu_h': 3e-5,
            'brinkman_eta': 1e-7,    # Solid obstacle with implicit penalization
            'dt_max': 0.001,
            'cfl_target': 0.1,
        })
    
    # Grid-aware adjustments
    if grid_nx < 512:
        # Coarser grid needs more dissipation
        params['C_s'] *= 1.2
        params['nu_hyper_ratio'] *= 1.5
        params['brinkman_eta'] *= 1.2
        params['cfl_target'] *= 0.8
    
    return params
