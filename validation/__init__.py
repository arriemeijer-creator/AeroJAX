"""
Validation module for AeroJAX.

This module provides benchmark validation capabilities for various flow configurations,
including Lid-Driven Cavity (LDC) validation against Ghia et al. benchmark data
and von Karman vortex tracking for Strouhal frequency analysis.
"""

from .ldc_validator import LDCValidator
from .ldc_overlay import LDCValidationOverlay
from .vk_validator import VKStrouhalTracker
from .vk_overlay import VKVortexOverlay

__all__ = ['LDCValidator', 'LDCValidationOverlay', 'VKStrouhalTracker', 'VKVortexOverlay']
