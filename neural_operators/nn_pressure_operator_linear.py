# nn_pressure_operator.py
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional

class LearnedPressureOperator(eqx.Module):
    """Simple linear model that predicts pressure from divergence (rhs) and mask."""
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_channels=2, hidden_size=64, *, key):
        # Simple linear mapping: (2) -> (1) for each spatial point
        key1, key2 = jax.random.split(key)
        # Initialize with small weights to prevent explosion
        self.weight = jax.random.normal(key1, (in_channels, 1)) * 0.01
        self.bias = jax.random.normal(key2, (1,)) * 0.01

    def __call__(self, rhs, mask):
        """
        Args:
            rhs: divergence / dt (scaled) – shape (H, W)
            mask: fluid mask – shape (H, W)
        Returns:
            pressure field – shape (H, W)
        """
        # Stack inputs
        x = jnp.stack([rhs, mask], axis=-1)   # (H, W, 2)
        original_shape = x.shape
        x = x.reshape(-1, 2)  # (H*W, 2)

        # Normalize inputs to prevent numerical instability
        x_mean = jnp.abs(x).mean(axis=0, keepdims=True) + 1e-8
        x = x / x_mean
        
        # Apply linear transformation
        x = jnp.dot(x, self.weight) + self.bias  # (H*W, 1)
        
        # Clip to prevent extreme values
        x = jnp.clip(x, -100.0, 100.0)

        # Reshape back to original spatial dimensions
        x = x.reshape(original_shape[:-1])  # (H, W)
        
        # Ensure no NaN or inf
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        return x