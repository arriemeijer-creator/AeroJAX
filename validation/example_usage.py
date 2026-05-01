"""
Example usage of the LDC validation module.

This script demonstrates how to use the LDCValidator to validate
Lid-Driven Cavity simulations against Ghia et al. benchmark data.
"""

import sys
import os
# Add parent directory to path to import validation module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from validation import LDCValidator


def example_basic_usage():
    """Basic usage example with synthetic snapshot data."""
    print("=== Basic LDC Validator Usage ===\n")
    
    # Create validator for Re=1000
    validator = LDCValidator.load(Re=1000)
    
    # Create synthetic snapshot data
    # In real usage, this would come from solver snapshots
    nx, ny = 128, 128
    lx, ly = 1.0, 1.0
    
    # Create synthetic velocity field with a vortex near center
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Vortex center at (0.53, 0.57) - close to Ghia benchmark
    vortex_x, vortex_y = 0.53, 0.57
    
    # Simple vortex velocity field
    dx = X - vortex_x
    dy = Y - vortex_y
    r2 = dx**2 + dy**2 + 0.01
    
    u = -dy / r2
    v = dx / r2
    
    # Add lid-driven flow (top wall moving right)
    u[:, -1] = 1.0
    
    # Create mock snapshot object
    class MockSnapshot:
        def __init__(self, u, v, p, mask, dx, dy, iteration=0, timestamp=0.0):
            self.u = u
            self.v = v
            self.p = p
            self.mask = mask
            self.dx = dx
            self.dy = dy
            self.nx = u.shape[0]
            self.ny = u.shape[1]
            self.iteration = iteration
            self.timestamp = timestamp
    
    snapshot = MockSnapshot(
        u=u,
        v=v,
        p=np.zeros_like(u),
        mask=np.ones_like(u),
        dx=lx / (nx - 1),
        dy=ly / (ny - 1),
        iteration=100,
        timestamp=0.5
    )
    
    # Compute validation
    validator.compute(snapshot)
    
    # Get results
    center = validator.get_vortex_center()
    error = validator.get_error()
    
    print(f"Reference Reynolds number: {validator.Re}")
    print(f"Reference vortex center: {validator.get_reference_center()}")
    print(f"Computed vortex center: ({center.x:.4f}, {center.y:.4f})")
    print(f"Error vector: (dx={error.dx:.4f}, dy={error.dy:.4f})")
    print(f"L2 distance error: {error.l2_distance:.4f}")
    print()


def example_multiple_reynolds():
    """Example validating against multiple Reynolds numbers."""
    print("=== Multiple Reynolds Numbers ===\n")
    
    # Supported Reynolds numbers
    reynolds_numbers = [100, 400, 1000, 3200, 5000, 7500]
    
    print("Supported Reynolds numbers:")
    for Re in reynolds_numbers:
        ref_center = LDCValidator.load(Re).get_reference_center()
        print(f"  Re={Re:4d}: {ref_center}")
    print()


def example_history_tracking():
    """Example tracking validation over multiple timesteps."""
    print("=== History Tracking ===\n")
    
    validator = LDCValidator.load(Re=1000)
    
    # Simulate multiple timesteps
    for i in range(5):
        # Create synthetic snapshot with drifting vortex center
        nx, ny = 64, 64
        lx, ly = 1.0, 1.0
        
        # Vortex center drifts slightly over time
        vortex_x = 0.53 + 0.001 * i
        vortex_y = 0.57 + 0.001 * i
        
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        dx = X - vortex_x
        dy = Y - vortex_y
        r2 = dx**2 + dy**2 + 0.01
        
        u = -dy / r2
        v = dx / r2
        u[:, -1] = 1.0
        
        class MockSnapshot:
            def __init__(self, u, v, dx, dy, iteration, timestamp):
                self.u = u
                self.v = v
                self.p = np.zeros_like(u)
                self.mask = np.ones_like(u)
                self.dx = dx
                self.dy = dy
                self.nx = u.shape[0]
                self.ny = u.shape[1]
                self.iteration = iteration
                self.timestamp = timestamp
        
        snapshot = MockSnapshot(
            u=u, v=v,
            dx=lx/(nx-1), dy=ly/(ny-1),
            iteration=i*10, timestamp=i*0.01
        )
        
        validator.compute(snapshot)
    
    # Get history
    history = validator.get_history()
    
    print(f"Validation history ({len(history.vortex_centers)} timesteps):")
    print("  Iteration | Center (x, y) | L2 Error")
    print("  " + "-" * 45)
    for i, (center, error, iter) in enumerate(zip(
        history.vortex_centers, history.errors, history.iterations
    )):
        print(f"  {iter:9d} | ({center.x:.4f}, {center.y:.4f}) | {error.l2_distance:.4f}")
    print()


def example_gui_integration():
    """Example of GUI overlay integration."""
    print("=== GUI Overlay Integration ===\n")
    
    print("To use GUI overlays in the viewer:")
    print()
    print("1. Create validator:")
    print("   validator = LDCValidator.load(Re=1000)")
    print()
    print("2. Create overlay:")
    print("   from validation.ldc_overlay import LDCValidationOverlay")
    print("   overlay = LDCValidationOverlay(validator, plot_widget)")
    print()
    print("3. Update after each timestep:")
    print("   validator.compute(snapshot)")
    print("   overlay.update()")
    print()
    print("4. Toggle visibility:")
    print("   overlay.set_visible(True)")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_multiple_reynolds()
    example_history_tracking()
    example_gui_integration()
    
    print("=== Extension Hooks ===\n")
    print("The following extension hooks are available (placeholders):")
    print("  - validator.detect_secondary_vortices(snapshot)")
    print("  - validator.compute_centerline_velocity(snapshot)")
    print("  - validator.compare_centerline_profiles(snapshot)")
    print("  - validator.track_multi_vortex_structure(snapshot)")
    print()
