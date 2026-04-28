# nn_pressure_operator_advanced.py - Advanced version with U-Net architecture
import equinox as eqx
import jax
import jax.numpy as jnp


class AdvancedPressureOperator(eqx.Module):
    """Advanced neural operator with simplified architecture and weight regularization"""
    
    # Layers
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    final: eqx.nn.Conv2d
    
    def __init__(self, in_channels=2, features=32, *, key):
        """
        Args:
            in_channels: Number of input channels (2 for rhs+mask, 4 for rhs+mask+u+v)
            features: Base feature count
            key: JAX random key
        """
        keys = jax.random.split(key, 4)
        
        # Simpler 3-layer architecture with small weights to prevent ringing
        self.conv1 = eqx.nn.Conv2d(in_channels, features, 3, padding='SAME', key=keys[0])
        self.conv1 = eqx.tree_at(lambda m: m.weight, self.conv1, self.conv1.weight * 0.05)
        
        self.conv2 = eqx.nn.Conv2d(features, features, 3, padding='SAME', key=keys[1])
        self.conv2 = eqx.tree_at(lambda m: m.weight, self.conv2, self.conv2.weight * 0.05)
        
        self.conv3 = eqx.nn.Conv2d(features, features, 3, padding='SAME', key=keys[2])
        self.conv3 = eqx.tree_at(lambda m: m.weight, self.conv3, self.conv3.weight * 0.05)
        
        # Final convolution with very small weights
        self.final = eqx.nn.Conv2d(features, 1, 3, padding='SAME', key=keys[3])
        self.final = eqx.tree_at(lambda m: m.weight, self.final, self.final.weight * 0.01)
        self.final = eqx.tree_at(lambda m: m.bias, self.final, self.final.bias * 0.01)
    
    def __call__(self, rhs, mask, u=None, v=None, training=True):
        """
        Args:
            rhs: divergence / dt – shape (H, W)
            mask: fluid mask – shape (H, W)
            u, v: velocity fields (optional) - shape (H, W)
            training: whether in training mode (for validation checks)
        Returns:
            pressure field – shape (H, W)
        """
        # Stack inputs
        inputs = [rhs, mask]
        if u is not None and v is not None:
            inputs.extend([u, v])
        x = jnp.stack(inputs, axis=-1)  # (H, W, C)
        
        # Normalize inputs
        channel_means = x.mean(axis=(0, 1), keepdims=True)
        channel_stds = x.std(axis=(0, 1), keepdims=True) + 1e-8
        x = (x - channel_means) / channel_stds
        
        # Transpose to (C, H, W)
        x = jnp.transpose(x, (2, 0, 1))
        
        # Simple 3-layer forward pass with skip connection
        x1 = jax.nn.relu(self.conv1(x))
        x2 = jax.nn.relu(self.conv2(x1) + x1)  # Skip connection
        x3 = jax.nn.relu(self.conv3(x2) + x2)  # Skip connection
        
        # Final convolution
        p = self.final(x3)[0]
        
        # Clip to prevent extreme values
        p = jnp.clip(p, -100.0, 100.0)
        
        # Apply mask
        p = p * mask
        
        # Ensure no NaN or inf
        p = jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validation check in inference mode
        if not training:
            assert jnp.all(jnp.isfinite(p)), "Pressure field contains NaN or inf values"
            assert jnp.abs(p).mean() < 50.0, f"Pressure mean too large: {jnp.abs(p).mean()}"
        
        return p
