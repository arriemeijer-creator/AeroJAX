"""
Differential Backpropagation - Inverse Design GUI
A JAX-based differentiable CFD framework for inverse airfoil design
"""

from .config import InverseDesignConfig

# Don't import from optimizer here to avoid module-level JAX import
# Import directly when needed to allow lazy loading
# from .optimizer import InverseDesigner, OptimizationConfig

__all__ = [
    'InverseDesignConfig',
    # 'InverseDesigner',  # Import directly to allow lazy JAX loading
    # 'OptimizationConfig'
]
