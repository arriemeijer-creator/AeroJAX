"""
Mask generation methods for BaselineSolver.
Computes obstacle masks from signed distance functions (SDFs).
"""

import jax
import jax.numpy as jnp


def _compute_mask(self) -> jnp.ndarray:
    """Compute the obstacle mask based on geometry"""
    # Special case for lid_driven_cavity and taylor_green - all fluid (mask = 1 everywhere)
    # Check this first before obstacle_type to ensure these flow types have no obstacles
    if self.sim_params.flow_type == 'lid_driven_cavity':
        self.sdf = None  # No SDF for all-fluid case
        return jnp.ones_like(self.grid.X)
    if self.sim_params.flow_type == 'taylor_green':
        self.sdf = None  # No SDF for all-fluid case
        return jnp.ones_like(self.grid.X)
    
    def smooth_mask(mask: jnp.ndarray) -> jnp.ndarray:
        """Simple smoothing using averaging to reduce sharp gradients - vectorized"""
        # Simple 3x3 averaging kernel
        kernel = jnp.ones((3, 3)) / 9.0
        # Pad the mask
        mask_padded = jnp.pad(mask, ((1, 1), (1, 1)), mode='edge')
        # Vectorized convolution using JAX
        smoothed = jax.lax.conv_general_dilated(
            mask_padded[None, None, :, :],
            kernel[None, None, :, :],
            window_strides=(1, 1),
            padding='VALID'
        )[0, 0]
        return smoothed
    
    if hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'naca_airfoil':
        from obstacles.naca_airfoils import NACAParams, create_naca_mask, parse_naca_4digit, parse_naca_5digit
        
        # Parse NACA designation
        naca_str = self.sim_params.naca_airfoil.upper().replace('NACA', '').strip()
        if len(naca_str) == 4:
            m, p, t = parse_naca_4digit(naca_str)
            airfoil_type = '4-digit'
        elif len(naca_str) == 5:
            cl, p, m, t = parse_naca_5digit(naca_str)
            airfoil_type = '5-digit'
        else:
            raise ValueError(f"Unsupported NACA designation: {self.sim_params.naca_airfoil}")
        
        naca_params = NACAParams(
            airfoil_type=airfoil_type,
            designation=self.sim_params.naca_airfoil,
            chord_length=self.sim_params.naca_chord,
            angle_of_attack=self.sim_params.naca_angle,
            position_x=self.sim_params.naca_x,
            position_y=self.sim_params.naca_y
        )
        # Unified masking: single sigmoid from SDF to χ (solid fraction)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider
        # Get SDF from NACA function, then apply unified sigmoid
        from obstacles.naca_airfoils import naca_surface_distance
        if airfoil_type == '4-digit':
            sdf = naca_surface_distance(self.grid.X, self.grid.Y, naca_params.chord_length,
                                       naca_params.angle_of_attack, naca_params.position_x,
                                       naca_params.position_y, m, p, t)
        else:  # 5-digit
            sdf = naca_surface_distance(self.grid.X, self.grid.Y, naca_params.chord_length,
                                       naca_params.angle_of_attack, naca_params.position_x,
                                       naca_params.position_y, cl, p, m, t)
        chi = jax.nn.sigmoid(-sdf / epsilon)  # 1 inside solid, 0 outside
        mask = 1.0 - chi  # 1 in fluid, 0 in solid
        self.sdf = sdf  # Store SDF for Brinkman penalization
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'cow':
        from obstacles.cow import sdf_cow_side
        # Compute cow position relative to grid bounds
        # Use cow_x and cow_y from sim_params if available, otherwise use defaults
        cow_x = getattr(self.sim_params, 'cow_x', self.grid.lx * 0.25)  # 25% of domain width default
        cow_y = getattr(self.sim_params, 'cow_y', self.grid.ly * 0.35)  # 35% of domain height default
        # Compute scale factor based on grid dimensions relative to reference (20x3.75)
        ref_lx = 20.0
        ref_ly = 3.75
        scale_x = self.grid.lx / ref_lx
        scale_y = self.grid.ly / ref_ly
        cow_scale = (scale_x + scale_y) / 2.0  # Average of x and y scaling
        # Unified masking: single sigmoid from SDF to χ (solid fraction)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider
        sdf = sdf_cow_side(self.grid.X, self.grid.Y, cow_x, cow_y, cow_scale)
        chi = jax.nn.sigmoid(-sdf / epsilon)  # 1 inside solid, 0 outside
        mask = 1.0 - chi  # 1 in fluid, 0 in solid
        self.sdf = sdf  # Store SDF for Brinkman penalization
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'three_cylinder_array':
        from obstacles.cylinder_array import sdf_three_cylinders
        cylinder_x = getattr(self.sim_params, 'cylinder_x', 5.0)
        cylinder_y = getattr(self.sim_params, 'cylinder_y', self.grid.ly / 2.0)
        cylinder_diameter = getattr(self.sim_params, 'cylinder_diameter', 0.5)
        cylinder_spacing = getattr(self.sim_params, 'cylinder_spacing', 0.5)
        # Unified masking: single sigmoid from SDF to χ (solid fraction)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider
        sdf = sdf_three_cylinders(self.grid.X, self.grid.Y, cylinder_x, cylinder_y, cylinder_diameter, cylinder_spacing)
        chi = jax.nn.sigmoid(-sdf / epsilon)  # 1 inside solid, 0 outside
        mask = 1.0 - chi  # 1 in fluid, 0 in solid
        self.sdf = sdf  # Store SDF for Brinkman penalization
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'custom':
        from obstacles.freeform_drawer import create_freeform_mask_smooth
        custom_mask = getattr(self.sim_params, 'custom_mask', None)
        if custom_mask is not None:
            # Use user's epsilon setting from slider
            epsilon = self.sim_params.eps
            # Get obstacle center position from sliders
            center_x = getattr(self.sim_params, 'custom_x', self.grid.lx * 0.25)
            center_y = getattr(self.sim_params, 'custom_y', self.grid.ly * 0.5)
            # Scale the custom obstacle to fit in the domain while preserving aspect ratio
            # Use the smaller dimension to determine scale, so the drawing fits
            mask_height, mask_width = custom_mask.shape
            
            # Calculate scale to fit in domain (use 60% of the smaller dimension)
            domain_min_dim = min(self.grid.lx, self.grid.ly)
            scale = domain_min_dim * 0.6
            
            # Use same scale for both dimensions to preserve aspect ratio
            scale_x = scale
            scale_y = scale
            
            # Calculate offset to center the obstacle at the specified position
            # offset is the center position
            offset_x = center_x
            offset_y = center_y
            
            mask = create_freeform_mask_smooth(self.grid.X, self.grid.Y, custom_mask, 
                                              scale_x=scale_x, scale_y=scale_y,
                                              offset_x=offset_x, offset_y=offset_y,
                                              smooth_width=epsilon)
            # Custom mask doesn't have SDF, set to None
            self.sdf = None
            return mask
        else:
            # Fallback to cylinder if no custom mask
            X, Y = self.grid.X, self.grid.Y
            phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
            epsilon = self.sim_params.eps
            chi = jax.nn.sigmoid(-phi / epsilon)
            mask = 1.0 - chi
            self.sdf = phi  # Store SDF for Brinkman penalization
            return mask
    else:
        X, Y = self.grid.X, self.grid.Y
        phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
        # Unified masking: single sigmoid from SDF to χ (solid fraction)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider
        chi = jax.nn.sigmoid(-phi / epsilon)  # 1 inside solid, 0 outside
        mask = 1.0 - chi  # 1 in fluid, 0 in solid
        self.sdf = phi  # Store SDF for Brinkman penalization
        return mask
