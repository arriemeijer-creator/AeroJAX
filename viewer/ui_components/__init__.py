"""
UI Components module - split into smaller maintainable files.
"""

from .top_console import TopConsole
from .obstacle_controls import ObstacleControls
from .time_controls import TimeControls
from .dye_controls import DyeControls
from .visualization_controls import VisualizationControls
from .info_panel import InfoPanel
from .control_panel import ControlPanel
from .floating_control_bar import FloatingControlBar

__all__ = [
    'TopConsole',
    'ObstacleControls',
    'TimeControls',
    'DyeControls',
    'VisualizationControls',
    'InfoPanel',
    'ControlPanel',
    'FloatingControlBar',
]
