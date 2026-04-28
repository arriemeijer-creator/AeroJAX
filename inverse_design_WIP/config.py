"""
Configuration management for Inverse Design
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GridConfig:
    """Grid configuration"""
    nx: int = 512
    ny: int = 96
    Lx: float = 20.0
    Ly: float = 3.0


@dataclass
class FlowConfig:
    """Flow configuration"""
    u_inlet: float = 1.0
    nu: float = 0.003
    reynolds: float = 1000.0
    flow_type: str = "von_karman"


@dataclass
class AirfoilConfig:
    """Airfoil configuration"""
    designation: str = "0012"
    chord_length: float = 0.3
    angle_of_attack: float = 0.0
    position_x: float = 5.0
    position_y: float = 1.5


@dataclass
class OptimizationGoals:
    """Optimization goals"""
    target_cl: Optional[float] = 0.8  # Default target lift coefficient
    target_cd: Optional[float] = 0.05  # Default target drag coefficient
    target_strouhal: Optional[float] = None
    target_aoa: Optional[float] = None
    cl_weight: float = 1.0
    cd_weight: float = 1.0
    strouhal_weight: float = 1.0
    shape_regularization: float = 0.1
    
    # Target selection flags
    target_cl_enabled: bool = True
    target_cd_enabled: bool = True
    target_strouhal_enabled: bool = False
    target_aoa_enabled: bool = False
    
    # Variable selection flags (which variables to optimize)
    optimize_aoa: bool = True
    optimize_thickness: bool = True
    optimize_camber_position: bool = False
    optimize_camber: bool = False


@dataclass
class InverseDesignConfig:
    """Main configuration for inverse design"""
    grid: GridConfig = field(default_factory=GridConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    airfoil: AirfoilConfig = field(default_factory=AirfoilConfig)
    goals: OptimizationGoals = field(default_factory=OptimizationGoals)
    
    # Optimization settings
    max_iterations: int = 100
    learning_rate: float = 0.01
    convergence_threshold: float = 1e-6
    
    # Visualization settings
    update_interval: int = 10  # Update visualization every N iterations
    show_residuals: bool = True
    show_forces: bool = True
