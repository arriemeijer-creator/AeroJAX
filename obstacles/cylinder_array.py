"""
Obstacle module for cylinder arrays.
"""

import jax.numpy as jnp
import jax

@jax.jit
def sdf_three_cylinders(X: jnp.ndarray, Y: jnp.ndarray, 
                       center_x: float = 5.0, center_y: float = 1.875,
                       diameter: float = 0.5, spacing: float = 0.5) -> jnp.ndarray:
    """
    Signed distance function for 3 cylinders stacked behind each other.
    
    The cylinders are arranged in the x-direction (flow direction) with
    configurable spacing between cylinder centers.
    
    Args:
        X, Y: Grid coordinates
        center_x: X-coordinate of the first (front) cylinder center
        center_y: Y-coordinate of all cylinder centers (vertically aligned)
        diameter: Diameter of each cylinder
        spacing: Spacing between cylinder centers
        
    Returns:
        Signed distance field (negative inside cylinders, positive outside)
    """
    radius = diameter / 2.0
    
    # Cylinder 1 (front)
    sdf1 = jnp.sqrt((X - center_x)**2 + (Y - center_y)**2) - radius
    
    # Cylinder 2 (middle)
    sdf2 = jnp.sqrt((X - (center_x + spacing))**2 + (Y - center_y)**2) - radius
    
    # Cylinder 3 (rear)
    sdf3 = jnp.sqrt((X - (center_x + 2 * spacing))**2 + (Y - center_y)**2) - radius
    
    # Combine using minimum (union of all cylinders)
    sdf_combined = jnp.minimum(jnp.minimum(sdf1, sdf2), sdf3)
    
    return sdf_combined


@jax.jit
def create_three_cylinder_mask(X: jnp.ndarray, Y: jnp.ndarray, 
                               center_x: float = 5.0, center_y: float = 1.875,
                               diameter: float = 0.5, spacing: float = 0.5,
                               eps: float = 0.05) -> jnp.ndarray:
    """
    Create obstacle mask for 3 cylinders stacked behind each other.
    
    Args:
        X, Y: Grid coordinates
        center_x: X-coordinate of the first (front) cylinder center
        center_y: Y-coordinate of all cylinder centers (vertically aligned)
        diameter: Diameter of each cylinder
        spacing: Spacing between cylinder centers
        eps: Smoothing parameter for the sigmoid transition
        
    Returns:
        Mask array (1 = fluid, 0 = solid) with smooth transition
    """
    sdf = sdf_three_cylinders(X, Y, center_x, center_y, diameter, spacing)
    mask = jax.nn.sigmoid(sdf / eps)
    return mask
