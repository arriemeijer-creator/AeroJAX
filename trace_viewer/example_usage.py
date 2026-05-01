"""
Example usage of the AeroJAX Solver Trace Viewer.

This script demonstrates how to:
1. Create snapshots from simulation data
2. Save snapshots to disk
3. Load snapshots in the trace viewer
"""

import numpy as np
from trace_viewer import (
    Snapshot, SnapshotSeries,
    save_snapshot, save_snapshots_to_directory,
    SnapshotRecorder
)


def example_create_snapshots():
    """Example: Create sample snapshots for testing."""
    print("Creating example snapshots...")
    
    # Grid parameters
    nx, ny = 64, 64
    dx, dy = 0.01, 0.01
    dt = 0.001
    nu = 0.003
    
    # Create sample velocity fields (simple channel flow)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create a few snapshots
    for step in range(10):
        # Simple parabolic profile
        u = 1.0 * (1 - (Y - 0.5)**2 / 0.25) * (0.5 + 0.1 * np.sin(step * 0.5))
        v = 0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        p = np.zeros((nx, ny))
        mask = np.zeros((nx, ny))
        
        # Add obstacle (cylinder)
        cx, cy = 0.3, 0.5
        radius = 0.1
        mask[(X - cx)**2 + (Y - cy)**2 < radius**2] = 1
        
        # Create snapshot
        snapshot = Snapshot(
            u=u,
            v=v,
            p=p,
            mask=mask,
            dt=dt,
            nu=nu,
            dx=dx,
            dy=dy,
            Re=100.0,
            CFL=0.5,
            flow_type="channel_flow",
            iteration=step,
            timestamp=step * dt
        )
        
        # Save snapshot
        filepath = f"example_snapshots/snapshot_{step:06d}.pkl"
        save_snapshot(snapshot, filepath)
        print(f"Saved snapshot {step} to {filepath}")
    
    print(f"Created 10 example snapshots in 'example_snapshots/' directory")


def example_snapshot_recorder():
    """Example: Using SnapshotRecorder in a simulation loop."""
    print("\nExample: Using SnapshotRecorder...")
    
    # Initialize recorder
    recorder = SnapshotRecorder(
        output_dir="recorder_snapshots",
        save_interval=2  # Save every 2 timesteps
    )
    
    # Simulate a simple loop
    for step in range(20):
        # Simulate some field evolution
        nx, ny = 32, 32
        u = np.random.randn(nx, ny) * 0.1
        v = np.random.randn(nx, ny) * 0.1
        p = np.random.randn(nx, ny) * 0.01
        mask = np.zeros((nx, ny))
        
        # Record snapshot
        recorder.record_step(
            u=u,
            v=v,
            p=p,
            mask=mask,
            dt=0.001,
            nu=0.003,
            dx=0.01,
            dy=0.01,
            Re=100.0,
            CFL=0.5,
            flow_type="test",
            timestamp=step * 0.001
        )
    
    print(f"Recorded {recorder.get_snapshot_count()} snapshots")
    print(f"Snapshots saved to 'recorder_snapshots/' directory")


def example_launch_viewer():
    """Example: Launch the trace viewer GUI."""
    print("\nTo launch the trace viewer GUI, run:")
    print("  python -m trace_viewer.viewer_window")
    print("\nOr from Python:")
    print("  from trace_viewer import TraceViewerWindow")
    print("  from PyQt6.QtWidgets import QApplication")
    print("  import sys")
    print("  app = QApplication(sys.argv)")
    print("  window = TraceViewerWindow()")
    print("  window.show()")
    print("  sys.exit(app.exec())")


if __name__ == "__main__":
    print("AeroJAX Solver Trace Viewer - Example Usage")
    print("=" * 50)
    
    # Run examples
    example_create_snapshots()
    example_snapshot_recorder()
    example_launch_viewer()
    
    print("\n" + "=" * 50)
    print("Examples complete!")
    print("You can now launch the trace viewer and load the snapshots.")
