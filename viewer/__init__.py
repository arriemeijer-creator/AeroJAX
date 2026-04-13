"""
Viewer package for Baseline Navier-Stokes Solver
Refactored modular visualization components
"""

from .ui_components import ControlPanel, InfoPanel
from .visualization import FlowVisualization, ObstacleRenderer, SDFVisualization
from .simulation_controller import SimulationController, SimulationWorker, RecordingManager, DataExporter
from .config import ConfigManager, VisualizationConfig, SimulationConfig

__all__ = [
    # UI Components
    'ControlPanel',
    'InfoPanel', 
    
    # Visualization
    'FlowVisualization',
    'ObstacleRenderer',
    'SDFVisualization',
    
    # Simulation Control
    'SimulationController',
    'SimulationWorker',
    'RecordingManager',
    'DataExporter',
    
    # Configuration
    'ConfigManager',
    'VisualizationConfig',
    'SimulationConfig',
]
