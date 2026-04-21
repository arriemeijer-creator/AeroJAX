import jax.numpy as jnp
import jax
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class NACAParams:
    """Parameters for NACA airfoil generation"""
    airfoil_type: str  # '4-digit' or '5-digit'
    designation: str   # e.g., '2412' or '23012'
    chord_length: float
    angle_of_attack: float  # degrees
    position_x: float
    position_y: float

def parse_naca_4digit(designation: str) -> Tuple[float, float, float]:
    """Parse NACA 4-digit designation"""
    # Extract just the digits from the designation
    digits = ''.join(filter(str.isdigit, designation))
    if len(digits) != 4:
        raise ValueError(f"Invalid 4-digit NACA designation: {designation}")
    
    m = int(digits[0]) / 100.0  # Maximum camber
    p = int(digits[1]) / 10.0   # Position of maximum camber
    t = int(digits[2:]) / 100.0 # Maximum thickness
    return m, p, t

def parse_naca_5digit(designation: str) -> Tuple[float, float, float, float]:
    """Parse NACA 5-digit designation"""
    # Extract just the digits from the designation
    digits = ''.join(filter(str.isdigit, designation))
    if len(digits) != 5:
        raise ValueError(f"Invalid 5-digit NACA designation: {designation}")
    
    cl = int(digits[:2]) / 20.0  # Design lift coefficient
    p = int(digits[2]) / 20.0    # Position of maximum camber
    m = int(digits[3]) / 10.0    # Maximum camber
    t = int(digits[4:]) / 100.0  # Maximum thickness
    return cl, p, m, t

@jax.jit
def generate_naca_4digit(x: jnp.ndarray, m: float, p: float, t: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate NACA 4-digit airfoil coordinates"""
    # Thickness distribution
    yt = 5 * t * (0.2969 * jnp.sqrt(jnp.abs(x)) - 0.1260 * x - 
                  0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line
    yc = jnp.where(x <= p,
                  m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    
    # Camber line angle
    dyc_dx = jnp.where(x <= p,
                      2 * m / p**2 * (p - x),
                      2 * m / (1 - p)**2 * (p - x))
    theta = jnp.arctan(dyc_dx)
    
    # Upper and lower surface coordinates
    xu = x - yt * jnp.sin(theta)
    yu = yc + yt * jnp.cos(theta)
    xl = x + yt * jnp.sin(theta)
    yl = yc - yt * jnp.cos(theta)
    
    return xu, yu, xl, yl

@jax.jit
def generate_naca_5digit(x: jnp.ndarray, cl: float, p: float, m: float, t: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate NACA 5-digit airfoil coordinates (simplified)"""
    # For simplicity, use modified 4-digit formula with adjusted parameters
    # In practice, 5-digit airfoils have more complex camber lines
    m_adj = m * 0.8  # Adjusted camber
    # Handle p=0 case to prevent division by zero (reflexed/symmetric airfoils)
    p_adj = jnp.where(p > 0, p * 0.8, 0.01)  # Avoid p=0 to prevent division by zero
    
    return generate_naca_4digit(x, m_adj, p_adj, t)

def rotate_airfoil(x: jnp.ndarray, y: jnp.ndarray, angle_deg: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate airfoil coordinates by angle (degrees) - STANDARD CONVENTION: positive = nose UP"""
    angle_rad = jnp.radians(angle_deg)  # NO negative sign!
    x_rot = x * jnp.cos(angle_rad) - y * jnp.sin(angle_rad)
    y_rot = x * jnp.sin(angle_rad) + y * jnp.cos(angle_rad)
    return x_rot, y_rot

def scale_airfoil(x: jnp.ndarray, y: jnp.ndarray, chord_length: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Scale airfoil to desired chord length"""
    return x * chord_length, y * chord_length

def translate_airfoil(x: jnp.ndarray, y: jnp.ndarray, pos_x: float, pos_y: float, chord_length: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Translate airfoil to desired position (pos_x is leading edge)"""
    # pos_x is the leading edge position
    return x + pos_x, y + pos_y

@jax.jit
def point_to_line_segment_distance(X: jnp.ndarray, Y: jnp.ndarray, 
                                  x1: float, y1: float, x2: float, y2: float) -> jnp.ndarray:
    """Vectorized point-to-line segment distance for JAX"""
    dx, dy = x2 - x1, y2 - y1
    line_mag_sq = dx**2 + dy**2
    
    # Project point onto line, clamped between 0 and 1
    u = ((X - x1) * dx + (Y - y1) * dy) / (line_mag_sq + 1e-8)
    u = jnp.clip(u, 0, 1)
    
    # Distance to the closest point on segment
    dist_sq = (x1 + u * dx - X)**2 + (y1 + u * dy - Y)**2
    return jnp.sqrt(dist_sq)

@jax.jit
def naca_surface_distance(X: jnp.ndarray, Y: jnp.ndarray, chord_length: float, angle_of_attack: float,
                         position_x: float, position_y: float, m: float, p: float, t: float) -> jnp.ndarray:
    """Compute distance to NACA 4-digit airfoil surface using analytical formula"""
    # Transform to airfoil-local coordinates
    # Translate
    X_local = X - position_x
    Y_local = Y - position_y
    
    # Rotate (positive angle - standard convention: positive = nose UP)
    angle_rad = jnp.radians(angle_of_attack)
    X_rot = X_local * jnp.cos(angle_rad) - Y_local * jnp.sin(angle_rad)
    Y_rot = X_local * jnp.sin(angle_rad) + Y_local * jnp.cos(angle_rad)
    
    # Scale to normalized coordinates
    x_norm = X_rot / chord_length
    y_norm = Y_rot / chord_length
    
    # Mask for points within chord bounds
    in_chord = (x_norm >= 0) & (x_norm <= 1)
    
    # Compute thickness distribution
    # Ensure x_norm is bounded to prevent NaN in sqrt
    x_norm_safe = jnp.clip(x_norm, 0.0, 1.0)
    yt = 5 * t * (0.2969 * jnp.sqrt(jnp.abs(x_norm_safe)) - 0.1260 * x_norm_safe - 
                  0.3516 * x_norm_safe**2 + 0.2843 * x_norm_safe**3 - 0.1015 * x_norm_safe**4)
    
    # Compute camber line
    yc = jnp.where(x_norm <= p,
                  m / p**2 * (2 * p * x_norm - x_norm**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x_norm - x_norm**2))
    
    # Camber line angle
    dyc_dx = jnp.where(x_norm <= p,
                      2 * m / p**2 * (p - x_norm),
                      2 * m / (1 - p)**2 * (p - x_norm))
    theta = jnp.arctan(dyc_dx)
    
    # Upper and lower surface coordinates
    xu = x_norm - yt * jnp.sin(theta)
    yu = yc + yt * jnp.cos(theta)
    xl = x_norm + yt * jnp.sin(theta)
    yl = yc - yt * jnp.cos(theta)
    
    # Distance to upper surface
    dist_upper = jnp.abs(y_norm - yu) * chord_length
    
    # Distance to lower surface  
    dist_lower = jnp.abs(y_norm - yl) * chord_length
    
    # Take minimum distance to either surface
    dist_surface = jnp.minimum(dist_upper, dist_lower)
    
    # For points outside chord bounds, use distance to leading/trailing edge
    dist_leading = jnp.sqrt((X_rot)**2 + (Y_rot)**2)
    dist_trailing = jnp.sqrt((X_rot - chord_length)**2 + (Y_rot)**2)
    
    # Combine distances
    dist = jnp.where(in_chord, dist_surface, jnp.minimum(dist_leading, dist_trailing))
    
    # Determine inside/outside using point-in-polygon test
    # Simplified: check if point is below upper surface and above lower surface
    above_lower = y_norm >= yl - 0.01  # Small tolerance
    below_upper = y_norm <= yu + 0.01  # Small tolerance
    inside = in_chord & above_lower & below_upper
    
    # Create signed distance
    sdf = jnp.where(inside, -dist, dist)
    
    return sdf

def create_naca_mask(X: jnp.ndarray, Y: jnp.ndarray, params: NACAParams, eps: float = 0.05) -> jnp.ndarray:
    """Create SDF mask for NACA airfoil"""
    # Create specialized functions for each airfoil type to avoid JAX issues
    if params.airfoil_type == '4-digit':
        return _create_naca_4digit_mask(X, Y, params.chord_length, params.angle_of_attack,
                                       params.position_x, params.position_y, eps, params.designation)
    elif params.airfoil_type == '5-digit':
        return _create_naca_5digit_mask(X, Y, params.chord_length, params.angle_of_attack,
                                       params.position_x, params.position_y, eps, params.designation)
    else:
        raise ValueError(f"Unsupported airfoil type: {params.airfoil_type}")

# Create specialized functions for each airfoil type to handle static arguments
def _create_naca_4digit_mask(X: jnp.ndarray, Y: jnp.ndarray, chord_length: float, angle_of_attack: float,
                            position_x: float, position_y: float, eps: float, designation: str) -> jnp.ndarray:
    """Create mask for 4-digit NACA airfoil"""
    # Parse designation outside of JIT
    m, p, t = parse_naca_4digit(designation)
    
    # JIT-compiled function with numeric parameters only
    return _create_naca_4digit_mask_jit(X, Y, chord_length, angle_of_attack, position_x, position_y, eps, m, p, t)

@jax.jit
def _create_naca_4digit_mask_jit(X: jnp.ndarray, Y: jnp.ndarray, chord_length: float, angle_of_attack: float,
                                position_x: float, position_y: float, eps: float, m: float, p: float, t: float) -> jnp.ndarray:
    """JIT-compiled 4-digit mask creation"""
    sdf = naca_surface_distance(X, Y, chord_length, angle_of_attack, position_x, position_y, m, p, t)
    mask = jax.nn.sigmoid(sdf / eps)
    return mask

def _create_naca_5digit_mask(X: jnp.ndarray, Y: jnp.ndarray, chord_length: float, angle_of_attack: float,
                            position_x: float, position_y: float, eps: float, designation: str) -> jnp.ndarray:
    """Create mask for 5-digit NACA airfoil"""
    # Parse designation outside of JIT
    cl, p, m, t = parse_naca_5digit(designation)
    
    # JIT-compiled function with numeric parameters only
    return _create_naca_5digit_mask_jit(X, Y, chord_length, angle_of_attack, position_x, position_y, eps, cl, p, m, t)

def _create_naca_5digit_mask_jit(X: jnp.ndarray, Y: jnp.ndarray, chord_length: float, angle_of_attack: float,
                                position_x: float, position_y: float, eps: float, cl: float, p: float, m: float, t: float) -> jnp.ndarray:
    """JIT-compiled 5-digit mask creation using segment-based approach"""
    # Generate airfoil coordinates
    x_normalized = jnp.linspace(0, 1, 100)
    xu, yu, xl, yl = generate_naca_5digit(x_normalized, cl, p, m, t)
    
    # Scale, rotate, and translate
    xu, yu = scale_airfoil(xu, yu, chord_length)
    xl, yl = scale_airfoil(xl, yl, chord_length)
    
    xu, yu = rotate_airfoil(xu, yu, angle_of_attack)
    xl, yl = rotate_airfoil(xl, yl, angle_of_attack)
    
    xu, yu = translate_airfoil(xu, yu, position_x, position_y, chord_length)
    xl, yl = translate_airfoil(xl, yl, position_x, position_y, chord_length)
    
    # Combine upper and lower surfaces
    airfoil_x = jnp.concatenate([xu, xl[::-1]])
    airfoil_y = jnp.concatenate([yu, yl[::-1]])
    
    # Vectorized distance computation using jax.lax.scan
    num_segments = len(airfoil_x) - 1
    
    def process_segment(min_dist, idx):
        x1, y1 = airfoil_x[idx], airfoil_y[idx]
        x2, y2 = airfoil_x[idx+1], airfoil_y[idx+1]
        dist = point_to_line_segment_distance(X, Y, x1, y1, x2, y2)
        return jnp.minimum(min_dist, dist), None
    
    min_dist, _ = jax.lax.scan(process_segment, jnp.full_like(X, jnp.inf), jnp.arange(num_segments))
    
    # Point-in-polygon test
    inside = Y < jnp.interp(X, airfoil_x, airfoil_y)
    
    # Create signed distance
    sdf = jnp.where(inside, -min_dist, min_dist)
    
    # Smooth the mask
    mask = jax.nn.sigmoid(sdf / eps)
    
    return mask

# Predefined airfoil configurations
NACA_AIRFOILS = {
    'NACA 0012': {'type': '4-digit', 'designation': '0012', 'description': 'Symmetric 12% thickness'},
    'NACA 2412': {'type': '4-digit', 'designation': '2412', 'description': '2% camber, 12% thickness'},
    'NACA 4412': {'type': '4-digit', 'designation': '4412', 'description': '4% camber, 12% thickness'},
    'NACA 23012': {'type': '5-digit', 'designation': '23012', 'description': 'Design lift 0.3, 12% thickness'},
    'NACA 6412': {'type': '4-digit', 'designation': '6412', 'description': '6% camber, 12% thickness'},
    'NACA 0018': {'type': '4-digit', 'designation': '0018', 'description': 'Symmetric 18% thickness'},
    'NACA 2418': {'type': '4-digit', 'designation': '2418', 'description': '2% camber, 18% thickness'},
}

def get_default_naca_params(airfoil_name: str, chord_length: float = 0.5, angle_of_attack: float = 0.0,
                           position_x: float = 2.0, position_y: float = 2.25) -> NACAParams:
    """Get default parameters for a predefined NACA airfoil"""
    if airfoil_name not in NACA_AIRFOILS:
        raise ValueError(f"Unknown airfoil: {airfoil_name}")
    
    config = NACA_AIRFOILS[airfoil_name]
    return NACAParams(
        airfoil_type=config['type'],
        designation=config['designation'],
        chord_length=chord_length,
        angle_of_attack=angle_of_attack,
        position_x=position_x,
        position_y=position_y
    )
