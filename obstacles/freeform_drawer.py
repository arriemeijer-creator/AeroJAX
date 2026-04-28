import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import distance_transform_edt, zoom
from typing import Optional, Tuple
import pygame

def create_freeform_sdf(mask: np.ndarray, X: jnp.ndarray, Y: jnp.ndarray, 
                       scale_x: float = 1.0, scale_y: float = 1.0,
                       offset_x: float = 0.0, offset_y: float = 0.0,
                       preserve_aspect: bool = True) -> jnp.ndarray:
    """
    Create a scalable SDF from a drawn mask.
    
    Args:
        mask: Binary mask from the drawer (255 = obstacle, 0 = fluid)
        X, Y: Grid coordinates from the solver
        scale_x: Horizontal scaling factor (physical width of mask)
        scale_y: Vertical scaling factor (physical height of mask)
        offset_x: Horizontal offset (center x position)
        offset_y: Vertical offset (center y position)
        preserve_aspect: If True, preserve the drawn aspect ratio
        
    Returns:
        JAX SDF array that can be used in the solver
    """
    # Convert mask to SDF
    sdf_np = mask_to_sdf(mask)
    
    # Get mask dimensions
    mask_height, mask_width = sdf_np.shape
    
    # Handle aspect ratio
    if preserve_aspect:
        # Use the larger scale to preserve aspect ratio
        aspect_ratio = mask_width / mask_height
        if scale_x > scale_y:
            effective_scale_y = scale_x / aspect_ratio
            effective_scale_x = scale_x
        else:
            effective_scale_x = scale_y * aspect_ratio
            effective_scale_y = scale_y
    else:
        effective_scale_x = scale_x
        effective_scale_y = scale_y
    
    # Map physical coordinates to mask coordinates
    # First, center the coordinates around the obstacle position
    X_centered = X - offset_x
    Y_centered = Y - offset_y
    
    # Scale to mask coordinate space
    # The physical range [-scale/2, scale/2] maps to mask coordinates [0, width-1]
    X_mask = (X_centered / effective_scale_x + 0.5) * (mask_width - 1)
    Y_mask = (Y_centered / effective_scale_y + 0.5) * (mask_height - 1)
    
    # Note: No clipping needed - we use mode='constant' to handle out-of-bounds points
    # This prevents unwanted connections at mask boundaries
    
    # Use jax.scipy.ndimage.map_coordinates for efficient interpolation
    from jax.scipy.ndimage import map_coordinates
    
    # Stack coordinates for map_coordinates
    # Note: map_coordinates expects (y, x) order
    coordinates = jnp.stack([Y_mask, X_mask])
    
    # Perform interpolation with constant boundary to prevent self-closing artifacts
    # Use a large positive value (outside obstacle) for points outside the mask
    sdf_jax = map_coordinates(sdf_np, coordinates, order=1, mode='constant', cval=100.0)
    
    return sdf_jax


def create_freeform_sdf_vectorized(mask: np.ndarray, X: jnp.ndarray, Y: jnp.ndarray,
                                   scale_x: float = 1.0, scale_y: float = 1.0,
                                   offset_x: float = 0.0, offset_y: float = 0.0) -> jnp.ndarray:
    """
    Vectorized version using jax.scipy.ndimage.map_coordinates.
    This is more efficient and handles interpolation better.
    """
    from jax.scipy.ndimage import map_coordinates
    
    # Convert mask to SDF
    sdf_np = mask_to_sdf(mask)
    
    # Get mask dimensions
    mask_height, mask_width = sdf_np.shape
    
    # Map physical coordinates to mask coordinates
    X_centered = X - offset_x
    Y_centered = Y - offset_y
    
    # Scale to mask coordinates (0 to width-1)
    X_mask = (X_centered / scale_x + 0.5) * (mask_width - 1)
    Y_mask = (Y_centered / scale_y + 0.5) * (mask_height - 1)
    
    # Note: No clipping needed - we use mode='constant' to handle out-of-bounds points
    # This prevents unwanted connections at mask boundaries
    
    # Stack coordinates for map_coordinates
    # Note: map_coordinates expects (y, x) order
    coordinates = jnp.stack([Y_mask, X_mask])
    
    # Perform interpolation with constant boundary to prevent self-closing artifacts
    # Use a large positive value (outside obstacle) for points outside the mask
    sdf_jax = map_coordinates(sdf_np, coordinates, order=1, mode='constant', cval=100.0)
    
    return sdf_jax


def create_freeform_mask_smooth(X: jnp.ndarray, Y: jnp.ndarray, mask: np.ndarray,
                                scale_x: float = 1.0, scale_y: float = 1.0,
                                offset_x: float = 0.0, offset_y: float = 0.0,
                                smooth_width: float = 0.03) -> jnp.ndarray:
    """
    Create a smooth mask with proper boundary smoothing.
    
    Args:
        smooth_width: Width of the smooth transition in physical units
    """
    sdf = create_freeform_sdf_vectorized(mask, X, Y, scale_x, scale_y, offset_x, offset_y)
    
    # Apply smooth transition using sigmoid
    # Negative SDF = inside obstacle (solid), Positive SDF = outside (fluid)
    # We want mask = 0 for solid, 1 for fluid
    # Use a sharper transition (divide by 10) to prevent bridging disconnected bodies
    mask_jax = jax.nn.sigmoid(sdf / (smooth_width * 0.1))
    
    return mask_jax


# Add debugging visualization to see what's happening
def debug_sdf_mapping(mask: np.ndarray, X: jnp.ndarray, Y: jnp.ndarray,
                     scale_x: float, scale_y: float, offset_x: float, offset_y: float):
    """Debug function to check coordinate mapping."""
    import matplotlib.pyplot as plt
    
    # Get mask dimensions
    mask_height, mask_width = mask.shape
    
    # Compute mapped coordinates for a few points
    X_centered = X - offset_x
    Y_centered = Y - offset_y
    
    X_mask = (X_centered / scale_x + 0.5) * (mask_width - 1)
    Y_mask = (Y_centered / scale_y + 0.5) * (mask_height - 1)
    
    print(f"Physical X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Physical Y range: [{Y.min():.2f}, {Y.max():.2f}]")
    print(f"Mask X range: [{X_mask.min():.2f}, {X_mask.max():.2f}]")
    print(f"Mask Y range: [{Y_mask.min():.2f}, {Y_mask.max():.2f}]")
    print(f"Scale X: {scale_x}, Scale Y: {scale_y}")
    print(f"Offset: ({offset_x}, {offset_y})")
    
    # Visualize mapping
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original mask
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title(f"Original Mask ({mask_width}x{mask_height})")
    axes[0].set_xlabel("Mask X")
    axes[0].set_ylabel("Mask Y")
    
    # Show which parts of the mask are being sampled
    sample_mask = ((X_mask >= 0) & (X_mask < mask_width) & 
                   (Y_mask >= 0) & (Y_mask < mask_height)).reshape(X.shape)
    
    axes[1].imshow(sample_mask, cmap='hot', interpolation='nearest')
    axes[1].set_title(f"Sampled Region ({(sample_mask.sum() / sample_mask.size * 100):.1f}% of grid)")
    axes[1].set_xlabel("Grid X index")
    axes[1].set_ylabel("Grid Y index")
    
    plt.tight_layout()
    plt.show()
    
    return X_mask, Y_mask


class FreeformDrawer:
    """Pygame-based freeform drawing interface for custom obstacles."""
    
    def __init__(self, width: int = 512, height: int = 512):
        """
        Initialize the drawing interface.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
        """
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width), dtype=np.uint8)
        self.drawing = False
        self.last_pos = None
        self.brush_size = 5
        self.running = False
        self.saved_mask = None
        
        # Initialize pygame immediately
        self._init_pygame()
        
    def _init_pygame(self):
        """Initialize pygame."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Draw Custom Obstacle - Left click to draw, Right click to erase, 'S' to save, 'C' to clear")
        self.clock = pygame.time.Clock()
        
    def _draw_line(self, start: Tuple[int, int], end: Tuple[int, int], erase: bool = False):
        """
        Draw a line on the canvas.
        
        Args:
            start: Start (x, y) coordinates
            end: End (x, y) coordinates
            erase: If True, erase (set to 0), otherwise draw (set to 255)
        """
        x0, y0 = start
        x1, y1 = end
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                self._draw_circle(x, y, self.brush_size, erase)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
            # Ensure the last point is drawn
            self._draw_circle(x, y, self.brush_size, erase)
        else:
            err = dy / 2.0
            while y != y1:
                self._draw_circle(x, y, self.brush_size, erase)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
            # Ensure the last point is drawn
            self._draw_circle(x, y, self.brush_size, erase)
        
    def _draw_circle(self, cx: int, cy: int, radius: int, erase: bool = False):
        """
        Draw a filled circle at the given position.
        
        Args:
            cx: Center x coordinate
            cy: Center y coordinate
            radius: Circle radius
            erase: If True, erase (set to 0), otherwise draw (set to 255)
        """
        value = 0 if erase else 255
        
        # Create a grid of points
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        
        # Get the region of the canvas
        y_start = max(0, cy - radius)
        y_end = min(self.height, cy + radius + 1)
        x_start = max(0, cx - radius)
        x_end = min(self.width, cx + radius + 1)
        
        # Calculate the mask region
        mask_y_start = max(0, radius - (cy - y_start))
        mask_y_end = min(2*radius+1, mask_y_start + (y_end - y_start))
        mask_x_start = max(0, radius - (cx - x_start))
        mask_x_end = min(2*radius+1, mask_x_start + (x_end - x_start))
        
        if mask_y_end > mask_y_start and mask_x_end > mask_x_start:
            mask_region = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            self.canvas[y_start:y_end, x_start:x_end] = np.where(
                mask_region, value, self.canvas[y_start:y_end, x_start:x_end]
            )
    
    def run(self) -> Optional[np.ndarray]:
        """
        Run the drawing interface.
        
        Returns:
            Binary mask of the drawn obstacle (255 = obstacle, 0 = fluid)
            Returns None if the window is closed without saving.
        """
        self.running = True
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click - draw
                        self.drawing = True
                        self.last_pos = pygame.mouse.get_pos()
                        # Draw at the clicked position immediately
                        self._draw_circle(*self.last_pos, self.brush_size, erase=False)
                    elif event.button == 3:  # Right click - erase
                        self.drawing = True
                        self.last_pos = pygame.mouse.get_pos()
                        # Erase at the clicked position
                        self._draw_circle(*self.last_pos, self.brush_size, erase=True)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in [1, 3]:  # Left or right click released
                        self.drawing = False
                        self.last_pos = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        current_pos = pygame.mouse.get_pos()
                        buttons = pygame.mouse.get_pressed()
                        erase = buttons[2]  # Right mouse button
                        if self.last_pos:
                            self._draw_line(self.last_pos, current_pos, erase)
                        self.last_pos = current_pos
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # Save
                        # Flip canvas vertically to match solver's coordinate system
                        # Pygame has y=0 at top, solver has y=0 at bottom
                        self.saved_mask = np.flipud(self.canvas).copy()
                        pygame.quit()
                        return self.saved_mask
                    elif event.key == pygame.K_c:  # Clear
                        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)
                    elif event.key == pygame.K_ESCAPE:  # Cancel
                        pygame.quit()
                        return None
                    elif event.key == pygame.K_UP:  # Increase brush size
                        self.brush_size = min(50, self.brush_size + 2)
                    elif event.key == pygame.K_DOWN:  # Decrease brush size
                        self.brush_size = max(1, self.brush_size - 2)
            
            # Display the canvas
            # Convert canvas to RGB for display (grayscale)
            # Pygame expects (width, height, 3) but canvas is (height, width)
            # Use transpose to swap axes
            display_surface = pygame.surfarray.make_surface(np.stack([self.canvas]*3, axis=-1).transpose(1, 0, 2))
            self.screen.blit(display_surface, (0, 0))
            
            # Display brush size
            font = pygame.font.Font(None, 36)
            text = font.render(f"Brush: {self.brush_size} | S: Save | C: Clear | ESC: Cancel | ↑/↓: Brush size", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        return None


def mask_to_sdf(mask: np.ndarray, resolution: int = 512) -> np.ndarray:
    """
    Convert a binary mask to a signed distance function.
    
    Args:
        mask: Binary mask (255 = obstacle, 0 = fluid)
        resolution: Target resolution for SDF computation
        
    Returns:
        Signed distance function array (negative inside obstacle, positive outside)
    """
    # Normalize mask to binary (0 or 1)
    binary_mask = (mask > 128).astype(np.uint8)
    
    # Compute distance transform
    # Distance transform gives distance to nearest zero pixel
    # For points inside the obstacle (mask=1), we need negative distances
    # For points outside (mask=0), we need positive distances
    
    # Distance to background (outside obstacle) - gives positive distances for interior points
    dist_inside = distance_transform_edt(binary_mask)
    
    # Distance to foreground (inside obstacle) - gives positive distances for exterior points
    dist_outside = distance_transform_edt(1 - binary_mask)
    
    # Combine: negative inside, positive outside
    sdf = np.where(binary_mask, -dist_inside, dist_outside)
    
    return sdf