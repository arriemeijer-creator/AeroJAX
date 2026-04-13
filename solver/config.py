"""
JAX Configuration

Configure JAX settings before any JAX imports.
"""

import jax


def configure_jax(platform='cpu', enable_x64=False, debug_nans=False):
    """
    Configure JAX global settings.
    
    Args:
        platform: 'cpu', 'gpu', or 'tpu'
        enable_x64: Enable 64-bit floats
        debug_nans: Enable NaN debugging
    """
    jax.config.update('jax_platform_name', platform)
    jax.config.update('jax_enable_x64', enable_x64)
    jax.config.update('jax_debug_nans', debug_nans)


# Default configuration
configure_jax(platform='cpu', enable_x64=False, debug_nans=False)
