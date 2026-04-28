"""
Boundary conditions for LBM
"""

import jax
import jax.numpy as jnp


def apply_bounce_back(f: jnp.ndarray, mask: jnp.ndarray, 
                      opposite: jnp.ndarray) -> jnp.ndarray:
    """
    Apply bounce-back boundary condition at obstacle nodes only
    (NOT for top/bottom walls - those use free-slip)
    
    Args:
        f: Distribution function (9, nx, ny)
        mask: Obstacle mask (1 = fluid, 0 = solid, with smooth transitions)
        opposite: Opposite direction indices (9,)
    
    Returns:
        f_bb: Distribution with bounce-back applied
    """
    f_bb = f.copy()
    
    # Vectorized bounce-back: at solid nodes (mask < 0.5), reflect to opposite direction
    # Use threshold for smooth masks
    solid_mask = mask < 0.5
    
    # Apply bounce-back for all directions simultaneously
    # Need to expand solid_mask to match f dimensions: (nx, ny) -> (9, nx, ny)
    solid_mask_expanded = jnp.expand_dims(solid_mask, axis=0)  # (1, nx, ny)
    f_bb = jnp.where(solid_mask_expanded, f[opposite], f)
    
    return f_bb


def apply_inlet_outlet(f: jnp.ndarray, rho_inlet: float, u_inlet: float,
                       cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                       cs_squared: float, nx: int, ny: int, mask: jnp.ndarray = None,
                       opposite: jnp.ndarray = None) -> jnp.ndarray:
    """
    Apply equilibrium boundary conditions at inlet (left) and outlet (right)
    for channel/von Karman flows
    
    Args:
        f: Distribution function (9, nx, ny)
        rho_inlet: Inlet density
        u_inlet: Inlet velocity (x-direction)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
        mask: Obstacle mask (1=fluid, 0=solid) - used to avoid setting inlet on obstacle
        opposite: Opposite direction indices (9,) - required for JIT compatibility
    
    Returns:
        f_bc: Distribution with inlet/outlet conditions
    """
    from .collision import equilibrium
    
    f_bc = f.copy()
    
    # Inlet (left boundary, x=0)
    # Set equilibrium with prescribed density and velocity
    # Create 2D fields for the boundary (ny, 1) to match equilibrium expectations
    u_inlet_field = jnp.full((ny, 1), u_inlet)
    v_inlet_field = jnp.zeros((ny, 1))
    rho_inlet_field = jnp.full((ny, 1), rho_inlet)
    
    f_eq_inlet = equilibrium(rho_inlet_field, u_inlet_field, v_inlet_field, cx, cy, w, cs_squared)
    
    # Apply to left boundary (squeeze the extra dimension)
    # For now, apply to entire left boundary regardless of mask to ensure flow enters
    f_bc = f_bc.at[:, 0, :].set(f_eq_inlet[:, :, 0])
    
    # Outlet (right boundary, x=nx-1)
    # Zero-gradient (copy from nx-2)
    f_bc = f_bc.at[:, -1, :].set(f[:, -2, :])
    
    # Free-slip boundary conditions for top and bottom walls
    # Set normal velocity to zero, preserve tangential velocity
    if opposite is None:
        raise ValueError("opposite array must be provided for JIT compatibility")
    
    # For free-slip, we need to compute macroscopic variables at the wall
    # and set the distribution functions to equilibrium with v=0 (no normal flow)
    from .collision import equilibrium
    
    # Bottom wall (y=0) - free slip
    # Get macroscopic values at the wall
    rho_wall = jnp.sum(f_bc[:, :, 0], axis=0)
    u_wall = (f_bc[1, :, 0] - f_bc[3, :, 0] + f_bc[5, :, 0] - f_bc[6, :, 0] - f_bc[7, :, 0] + f_bc[8, :, 0]) / rho_wall
    v_wall = 0.0  # No normal velocity for free-slip
    
    # Set equilibrium with v=0 at bottom wall
    u_wall_2d = u_wall[None, :]
    v_wall_2d = jnp.zeros_like(u_wall_2d)
    rho_wall_2d = rho_wall[None, :]
    
    f_eq_wall = equilibrium(rho_wall_2d, u_wall_2d, v_wall_2d, cx, cy, w, cs_squared)
    f_bc = f_bc.at[:, :, 0].set(f_eq_wall[:, :, 0])
    
    # Top wall (y=ny-1) - free slip
    rho_wall = jnp.sum(f_bc[:, :, -1], axis=0)
    u_wall = (f_bc[1, :, -1] - f_bc[3, :, -1] + f_bc[5, :, -1] - f_bc[6, :, -1] - f_bc[7, :, -1] + f_bc[8, :, -1]) / rho_wall
    v_wall = 0.0  # No normal velocity for free-slip
    
    u_wall_2d = u_wall[None, :]
    v_wall_2d = jnp.zeros_like(u_wall_2d)
    rho_wall_2d = rho_wall[None, :]
    
    f_eq_wall = equilibrium(rho_wall_2d, u_wall_2d, v_wall_2d, cx, cy, w, cs_squared)
    f_bc = f_bc.at[:, :, -1].set(f_eq_wall[:, :, 0])
    
    return f_bc


def apply_lid_driven_cavity_bc(f: jnp.ndarray, u_lid: float,
                                cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                                cs_squared: float, nx: int, ny: int, opposite: jnp.ndarray) -> jnp.ndarray:
    """
    Apply lid-driven cavity boundary conditions
    - Top wall: moving lid with velocity u_lid
    - Other walls: no-slip (bounce-back)
    
    Args:
        f: Distribution function (9, nx, ny)
        u_lid: Lid velocity
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
        opposite: Opposite direction indices (9,)
    
    Returns:
        f_bc: Distribution with cavity BCs
    """
    from .collision import equilibrium
    
    f_bc = f.copy()
    
    # Top wall (y=ny-1): moving lid
    # Create 2D fields for the boundary (nx, 1) to match equilibrium expectations
    u_lid_field = jnp.full((nx, 1), u_lid)
    v_lid_field = jnp.zeros((nx, 1))
    rho_lid_field = jnp.ones((nx, 1))  # Assume unit density
    
    f_eq_lid = equilibrium(rho_lid_field, u_lid_field, v_lid_field, cx, cy, w, cs_squared)
    
    # Apply to top boundary (squeeze the extra dimension)
    f_bc = f_bc.at[:, :, -1].set(f_eq_lid[:, :, 0])
    
    # Bottom wall (y=0): no-slip (bounce-back)
    f_bc = f_bc.at[:, :, 0].set(f_bc[opposite, :, 0])
    
    # Left wall (x=0): no-slip (bounce-back)
    f_bc = f_bc.at[:, 0, :].set(f_bc[opposite, 0, :])
    
    # Right wall (x=nx-1): no-slip (bounce-back)
    f_bc = f_bc.at[:, -1, :].set(f_bc[opposite, -1, :])
    
    return f_bc


def apply_taylor_green_bc(f: jnp.ndarray) -> jnp.ndarray:
    """
    Apply periodic boundary conditions for Taylor-Green vortex
    (Note: streaming already handles periodic via jnp.roll)
    
    Args:
        f: Distribution function (9, nx, ny)
    
    Returns:
        f_bc: Distribution (unchanged for periodic)
    """
    # Periodic BCs are handled by jnp.roll in streaming step
    return f


def apply_boundary_conditions(f: jnp.ndarray, mask: jnp.ndarray,
                               opposite: jnp.ndarray, flow_type: str = 'von_karman',
                               u_inlet: float = 0.0, nx: int = 0, ny: int = 0,
                               cx: jnp.ndarray = None, cy: jnp.ndarray = None,
                               w: jnp.ndarray = None, cs_squared: float = None) -> jnp.ndarray:
    """
    Apply all boundary conditions based on flow type
    
    Args:
        f: Distribution function (9, nx, ny)
        mask: Obstacle mask
        opposite: Opposite direction indices (9,)
        flow_type: Type of flow ('von_karman', 'lid_driven_cavity', 'taylor_green')
        u_inlet: Inlet velocity for channel flows
        nx: Grid size in x
        ny: Grid size in y
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
    
    Returns:
        f_bc: Distribution with boundary conditions
    """
    # Apply bounce-back for obstacles
    f_bc = apply_bounce_back(f, mask, opposite)
    
    # Apply flow-specific boundary conditions
    if flow_type == 'von_karman':
        f_bc = apply_inlet_outlet(f_bc, 1.0, u_inlet, cx, cy, w, cs_squared, nx, ny, mask, opposite)
    elif flow_type == 'lid_driven_cavity':
        f_bc = apply_lid_driven_cavity_bc(f_bc, u_inlet, cx, cy, w, cs_squared, nx, ny, opposite)
    elif flow_type == 'taylor_green':
        f_bc = apply_taylor_green_bc(f_bc)
    
    return f_bc
