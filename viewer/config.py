"""
Configuration management for Baseline Navier-Stokes Viewer
Centralizes settings and constants
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    # Performance settings
    target_vis_fps: int = 60
    frame_skip: int = 1
    level_update_interval: int = 10
    
    # Display settings
    show_velocity: bool = True
    show_vorticity: bool = True
    default_velocity_colormap: str = 'viridis'
    default_vorticity_colormap: str = 'coolwarm'
    
    # Plot settings
    plot_background: str = 'w'
    auto_downsample: bool = True
    downsample_method: str = 'average'


@dataclass
class SimulationConfig:
    """Configuration for simulation settings"""
    # Default parameters
    default_reynolds: float = 1000.0  # Good for vortex shedding
    default_nu: float = 0.003  # Kinematic viscosity
    default_dt: float = 0.001
    default_u_inf: float = None  # Will be calculated from Re, nu, L
    
    # Grid settings
    default_nx: int = 512   # Medium resolution for good performance
    default_ny: int = 96   # Medium resolution for good performance
    default_lx: float = 20.0
    default_ly: float = 3.75
    
    # Pressure solver settings
    pressure_max_iter: int = 50
    pressure_tolerance: float = 1e-4
    
    # Performance settings
    adaptive_dt_enabled: bool = True  # Enable Diffrax adaptive controller
    cfl_target: float = 0.5


class GridPresets:
    """Grid size presets for different flow types"""
    
    PRESETS: Dict[str, List[str]] = {
        'von_karman': [
            '256x48 (Coarse)',
            '512x96 (Medium)', 
            '1024x192 (Fine)',
            '2048x384 (Ultra Fine)'
        ],
        'lid_driven_cavity': [
            '64x64 (Coarse)',
            '128x128 (Medium)',
            '256x256 (Fine)',
            '512x512 (Ultra Fine)'
        ],
        'channel_flow': [
            '128x32 (Coarse)',
            '256x64 (Medium)',
            '512x128 (Fine)',
            '1024x256 (Ultra Fine)'
        ],
        'backward_step': [
            '256x64 (Coarse)',
            '512x128 (Medium)',
            '1024x256 (Fine)',
            '2048x512 (Ultra Fine)'
        ],
        'taylor_green': [
            '64x64 (Coarse)',
            '128x128 (Medium)',
            '256x256 (Fine)',
            '512x512 (Ultra Fine)'
        ]
    }
    
    DOMAIN_SIZES: Dict[str, Tuple[float, float]] = {
        'von_karman': (4.0, 2.0),
        'lid_driven_cavity': (1.0, 1.0),
        'taylor_green': (2*np.pi, 2*np.pi)
    }


class ColormapPresets:
    """Available colormap presets"""
    
    SEQUENTIAL_COLORMAPS = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
        # CET colormaps (sequential)
        'CET-C1', 'CET-C2', 'CET-C3', 'CET-C4', 'CET-C5', 'CET-C6', 'CET-C7',
        'CET-D1', 'CET-D2', 'CET-D3', 'CET-D4', 'CET-D6', 'CET-D7', 'CET-D8', 
        'CET-D9', 'CET-D10', 'CET-D11', 'CET-D12', 'CET-D13',
        'CET-L1', 'CET-L2', 'CET-L3', 'CET-L4', 'CET-L5', 'CET-L6', 'CET-L7', 
        'CET-L8', 'CET-L9', 'CET-L10', 'CET-L11', 'CET-L12', 'CET-L13', 
        'CET-L14', 'CET-L15', 'CET-L16', 'CET-L17', 'CET-L18', 'CET-L19',
        # Additional options
        'PAL-relaxed', 'PAL-relaxed_bright'
    ]
    
    DIVERGING_COLORMAPS = [
        'CET-CBC1', 'CET-CBC2', 'CET-CBD1', 'CET-CBL1', 'CET-CBL2', 
        'CET-CBTC1', 'CET-CBTC2', 'CET-CBTD1', 'CET-CBTL1', 'CET-CBTL2',
        'CET-I1', 'CET-I2', 'CET-I3',
        'CET-R1', 'CET-R2', 'CET-R3', 'CET-R4'
    ]
    
    ALL_COLORMAPS = SEQUENTIAL_COLORMAPS + DIVERGING_COLORMAPS


class SolverPresets:
    """Available solver options"""
    
    ADVECTION_SCHEMES = [
        "rk3", "tvd", "weno5", "spectral"
    ]
    
    PRESSURE_SOLVERS = [
        "fft", "cg", "multigrid"
    ]
    
    FLOW_TYPES = [
        "von_karman", "lid_driven_cavity", "taylor_green"
    ]


class PerformanceSettings:
    """Performance-related settings"""
    
    # Window settings
    DEFAULT_WINDOW_WIDTH = 1600
    DEFAULT_WINDOW_HEIGHT = 900
    MIN_PLOT_WIDTH = 800
    MIN_PLOT_HEIGHT = 600
    
    # FPS limits
    MIN_VIS_FPS = 10
    MAX_VIS_FPS = 120
    DEFAULT_VIS_FPS = 60
    
    # Frame skip limits
    MIN_FRAME_SKIP = 1
    MAX_FRAME_SKIP = 100
    DEFAULT_FRAME_SKIP = 1
    
    # Reynolds number limits
    MIN_REYNOLDS = 10
    MAX_REYNOLDS = 1000
    DEFAULT_REYNOLDS = 1000
    
    # Time step limits
    MIN_DT = 0.0001
    MAX_DT = 0.01
    DEFAULT_DT = 0.002
    
    # Jacobi iteration limits
    MIN_JACOBI_ITER = 1
    MAX_JACOBI_ITER = 1000
    DEFAULT_JACOBI_ITER = 50
    
    # NACA airfoil limits
    MIN_CHORD = 0.1
    MAX_CHORD = 2.0
    DEFAULT_CHORD = 0.5
    
    MIN_ANGLE = -20.0
    MAX_ANGLE = 20.0
    DEFAULT_ANGLE = 0.0


class ConfigManager:
    """Manages configuration and provides easy access to settings"""
    
    def __init__(self):
        self.viz_config = VisualizationConfig()
        self.sim_config = SimulationConfig()
        
    def get_grid_preset(self, flow_type: str) -> List[str]:
        """Get grid presets for a specific flow type"""
        return GridPresets.PRESETS.get(flow_type, GridPresets.PRESETS['von_karman'])
    
    def get_domain_size(self, flow_type: str) -> Tuple[float, float]:
        """Get domain size for a specific flow type"""
        return GridPresets.DOMAIN_SIZES.get(flow_type, (20.0, 4.5))
    
    def get_colormaps(self) -> List[str]:
        """Get all available colormaps"""
        return ColormapPresets.ALL_COLORMAPS
    
    def get_advection_schemes(self) -> List[str]:
        """Get available advection schemes"""
        return SolverPresets.ADVECTION_SCHEMES
    
    def get_pressure_solvers(self) -> List[str]:
        """Get available pressure solvers"""
        return SolverPresets.PRESSURE_SOLVERS
    
    def get_flow_types(self) -> List[str]:
        """Get available flow types"""
        return SolverPresets.FLOW_TYPES
    
    def parse_grid_selection(self, selection: str) -> Tuple[int, int]:
        """Parse grid selection string like '512x96 (Medium)' to (512, 96)"""
        import re
        match = re.search(r'(\d+)x(\d+)', selection)
        if match:
            return int(match.group(1)), int(match.group(2))
        raise ValueError(f"Invalid grid selection: {selection}")
    
    def is_sequential_colormap(self, colormap: str) -> bool:
        """Check if colormap is sequential"""
        return colormap in ColormapPresets.SEQUENTIAL_COLORMAPS
    
    def is_diverging_colormap(self, colormap: str) -> bool:
        """Check if colormap is diverging"""
        return colormap in ColormapPresets.DIVERGING_COLORMAPS
