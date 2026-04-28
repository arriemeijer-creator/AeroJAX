"""
Streaming step for LBM
"""

import jax
import jax.numpy as jnp


def stream(f: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray) -> jnp.ndarray:
    """
    Stream distribution functions to neighboring nodes
    
    Args:
        f: Distribution function (9, nx, ny)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
    
    Returns:
        f_streamed: Streamed distribution (9, nx, ny)
    """
    n_velocities = len(cx)
    f_streamed = jnp.zeros_like(f)
    
    # Stream each direction
    for i in range(n_velocities):
        # Shift array by (cx[i], cy[i])
        # jnp.roll handles periodic boundaries by default
        f_streamed = f_streamed.at[i].set(
            jnp.roll(f[i], shift=(cx[i], cy[i]), axis=(0, 1))
        )
    
    return f_streamed


@jax.jit
def streaming_step(f: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled streaming step
    
    Args:
        f: Distribution function (9, nx, ny)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
    
    Returns:
        f_streamed: Streamed distribution (9, nx, ny)
    """
    return stream(f, cx, cy)
