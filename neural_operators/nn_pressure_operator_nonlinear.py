# nn_pressure_operator.py - Non-linear version
import equinox as eqx
import jax
import jax.numpy as jnp

class NonLinearPressureOperator(eqx.Module):
    """Small CNN with non-linear activations for vortex capture"""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    
    def __init__(self, in_channels=2, features=16, *, key):  # Base channels: rhs, mask (or 4 with u, v)
        keys = jax.random.split(key, 3)
        # Initialize with small weights to prevent explosion
        # Scale based on number of layers: 0.1^(1/3) ≈ 0.46 for 3 layers
        scale = 0.1 ** (1/3)
        self.conv1 = eqx.nn.Conv2d(in_channels, features, 3, padding='SAME', key=keys[0])
        self.conv2 = eqx.nn.Conv2d(features, features, 3, padding='SAME', key=keys[1])
        self.conv3 = eqx.nn.Conv2d(features, 1, 3, padding='SAME', key=keys[2])
        # Scale down the weights
        self.conv1 = eqx.tree_at(lambda m: m.weight, self.conv1, self.conv1.weight * scale)
        self.conv1 = eqx.tree_at(lambda m: m.bias, self.conv1, self.conv1.bias * scale)
        self.conv2 = eqx.tree_at(lambda m: m.weight, self.conv2, self.conv2.weight * scale)
        self.conv2 = eqx.tree_at(lambda m: m.bias, self.conv2, self.conv2.bias * scale)
        self.conv3 = eqx.tree_at(lambda m: m.weight, self.conv3, self.conv3.weight * scale)
        self.conv3 = eqx.tree_at(lambda m: m.bias, self.conv3, self.conv3.bias * scale)
    
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
        # Stack inputs - include velocity fields if available
        inputs = [rhs, mask]
        if u is not None and v is not None:
            inputs.extend([u, v])
        x = jnp.stack(inputs, axis=-1)  # (H, W, C) where C is 2 or 4
        
        # Normalize inputs with per-channel mean/std for better stability
        channel_means = x.mean(axis=(0, 1), keepdims=True)
        channel_stds = x.std(axis=(0, 1), keepdims=True) + 1e-8
        x = (x - channel_means) / channel_stds
        
        # Transpose to (C, H, W) for Conv2d
        x = jnp.transpose(x, (2, 0, 1))
        
        # Non-linear convolutions with skip connections
        x1 = self.conv1(x)
        x1 = jax.nn.relu(x1)
        
        # Middle convolution with residual connection
        x2 = self.conv2(x1)
        x2 = jax.nn.relu(x2 + x1)  # Residual connection for better gradient flow
        
        # Final convolution
        p = self.conv3(x2)[0]
        
        # Apply efficient smoothing using JAX scan
        def smooth_3x3(p_field):
            p_padded = jnp.pad(p_field, 1, mode='reflect')
            return (p_padded[:-2, :-2] + p_padded[:-2, 1:-1] + p_padded[:-2, 2:] +
                    p_padded[1:-1, :-2] + p_padded[1:-1, 1:-1] + p_padded[1:-1, 2:] +
                    p_padded[2:, :-2] + p_padded[2:, 1:-1] + p_padded[2:, 2:]) / 9.0
        
        # Use jax.lax.scan for efficient repeated smoothing
        def smooth_step(carry, _):
            return smooth_3x3(carry), None
        
        p, _ = jax.lax.scan(smooth_step, p, None, length=8)
        
        # Clip to prevent extreme values
        p = jnp.clip(p, -100.0, 100.0)
        
        # Apply mask to ensure zero pressure in solid
        p = p * mask
        
        # Ensure no NaN or inf
        p = jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validation check in inference mode
        if not training:
            assert jnp.all(jnp.isfinite(p)), "Pressure field contains NaN or inf values"
            assert jnp.abs(p).mean() < 50.0, f"Pressure mean too large: {jnp.abs(p).mean()}"
        
        return p