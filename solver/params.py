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
    """Full simulation state as a PyTree for JAX compatibility"""
    u: jnp.ndarray
    v: jnp.ndarray
    p: jnp.ndarray  # Pressure field
    u_prev: jnp.ndarray
    v_prev: jnp.ndarray
    c: jnp.ndarray  # Scalar dye field
    dt: float
    iteration: int
    # PID controller internal state
    integral: float
    prev_error: float


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
    
    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x = jnp.linspace(0, self.lx, self.nx)
        self.y = jnp.linspace(0, self.ly, self.ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')


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
    Higher Re requires wider mask transition for stability.

    Args:
        Re: Reynolds number

    Returns:
        eps_multiplier: Multiplier for grid spacing to compute epsilon
    """
    if Re < 1000:
        return 1.0  # Low Re, sharp mask OK
    elif Re < 3000:
        return 1.5  # Moderate Re
    elif Re < 6000:
        return 2.0  # High Re (current default)
    else:
        return 3.0  # Very high Re, need wide transition


@dataclass
class SimulationParams:
    eps_multiplier: float = 2  # Grid-consistent ε: eps = eps_multiplier * dx (can be auto-computed from Re)
    auto_eps_multiplier: bool = True  # If True, eps_multiplier is auto-computed from Re
    eps: float = 0.02  # Deprecated: will be computed as eps = eps_multiplier * grid.dx
    advection_scheme: str = 'rk3'  # 'rk3', 'spectral', 'weno5', 'tvd'
    limiter: str = 'minmod'  # for TVD
    weno_epsilon: float = 1e-6  # for WENO
    max_cfl: float = 0.3  # maximum CFL number
    adaptive_dt: bool = True  # enable adaptive timestepping
    fixed_dt: float = 0.001  # User-specified fixed timestep
    pressure_solver: str = 'multigrid'  # 'cg', 'fft', 'multigrid', 'sor_masked', 'cg_masked'
    sor_omega: float = 1.5  # SOR relaxation parameter
    pressure_max_iter: int = 100  # max iterations for iterative solvers
    pressure_tolerance: float = 1e-6  # tolerance for early stopping
    multigrid_levels: int = 4  # multigrid grid coarsening levels
    multigrid_v_cycles: int = 2  # multigrid V-cycles per solve
    flow_type: str = 'von_karman'  # 'von_karman', 'lid_driven_cavity', 'taylor_green'
    dt_min: float = 1e-6  # Minimum timestep
    dt_max: float = 0.01   # Maximum timestep
    
    # NACA airfoil parameters
    obstacle_type: str = 'cylinder'  # 'cylinder' or 'naca_airfoil'
    naca_airfoil: str = 'NACA 2412'  # Airfoil designation
    naca_chord: float = 3.0  # Chord length
    naca_angle: float = -10.0  # Angle of attack in degrees
    naca_x: float = 2.0  # X position
    naca_y: float = 2.25  # Y position
    
    # LES/SGS model parameters
    use_les: bool = False  # Enable/disable LES
    les_model: str = 'dynamic_smagorinsky'  # 'dynamic_smagorinsky' or 'smagorinsky'
    smagorinsky_constant: float = 0.17  # Constant C_s for Smagorinsky model
    
    # Scalar/dye parameters
    use_scalar: bool = False  # Enable/disable scalar field (dye)
    scalar_diffusivity: float = 0.001  # Scalar diffusivity
