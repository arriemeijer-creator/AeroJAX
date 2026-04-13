"""
Timestepping Module for Differential CFD-ML Framework
=====================================================

This module provides adaptive timestepping controllers for the CFD solver.

Components:
- DivergencePIDController: Divergence-based PID adaptive timestepping

The adaptive timestepping system automatically adjusts timestep size based on:
- Velocity field divergence
- PID controller for smooth adjustments
- Configurable dt limits
"""

from .adaptivedt import DivergencePIDController

__all__ = [
    'DivergencePIDController'
]
