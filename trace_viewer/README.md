# AeroJAX Solver Trace Viewer

A decoupled solver trace viewer for AeroJAX that provides step-by-step numerical inspection of CFD solver operations.

## Overview

The trace viewer is designed with strict separation from the live solver:
- **Solver (backend)**: Generates and stores snapshots during simulation
- **Trace Viewer (this module)**: Reads snapshots only, no runtime coupling

This design enables deterministic replay, numerical inspection, and debugging without affecting solver performance.

## Features

- **Step Navigation**: Navigate through simulation timesteps with buttons, slider, and direct input
- **Solver Trace Pipeline**: Visual decomposition of the 5-step solver algorithm:
  1. Advection-Diffusion
  2. Divergence computation
  3. Pressure solve
  4. Velocity correction
  5. State update
- **LaTeX Equations**: Static rendering of numerical methods
- **Subdomain Viewer**: Inspect field values (u, v, p, divergence) in selected regions
- **Metrics Display**: CFL, Reynolds number, max/min field values

## Architecture

```
trace_viewer/
├── __init__.py              # Module exports
├── snapshot.py              # Snapshot data model
├── trace_builder.py         # Field reconstruction and trace building
├── navigation_panel.py      # Step navigation controls
├── solver_trace_panel.py    # 5-step pipeline display with LaTeX
├── subdomain_viewer.py      # Numerical field inspection
├── snapshot_utils.py        # Loading/saving utilities
├── viewer_window.py         # Main application window
└── README.md                # This file
```

## Usage

### Standalone Viewer

```python
from trace_viewer import TraceViewerWindow
from PyQt6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = TraceViewerWindow()
window.show()
sys.exit(app.exec())
```

### Creating Snapshots from Solver

```python
from trace_viewer.snapshot_utils import create_snapshot_from_sim_state
import numpy as np

# After each solver step, create a snapshot
snapshot = create_snapshot_from_sim_state(
    u=u_field,
    v=v_field,
    p=p_field,
    mask=obstacle_mask,
    dt=dt,
    nu=nu,
    dx=dx,
    dy=dy,
    divergence=divergence_field,  # optional
    Re=Reynolds_number,           # optional
    CFL=cfl_number,               # optional
    flow_type="von_karman",       # optional
    iteration=timestep,
    timestamp=simulation_time
)

# Save snapshot
snapshot.save(f"snapshots/snapshot_{timestep:06d}.pkl")
```

### Loading Snapshots

```python
from trace_viewer.snapshot_utils import load_snapshots_from_directory
from trace_viewer import SnapshotSeries

# Load all snapshots from directory
snapshots = load_snapshots_from_directory("snapshots/")
series = SnapshotSeries(snapshots)

# Or load a pre-saved series
from trace_viewer.snapshot_utils import load_snapshot_series
series = load_snapshot_series("simulation_trace.pkl")
```

## Snapshot Data Model

Required fields:
- `u` (Nx, Ny array): x-velocity field
- `v` (Nx, Ny array): y-velocity field
- `p` (Nx, Ny array): pressure field
- `mask` (Nx, Ny array): obstacle mask
- `dt` (float): timestep size
- `nu` (float): kinematic viscosity
- `dx`, `dy` (float): grid spacing

Optional fields:
- `divergence` (Nx, Ny array): pre-computed divergence
- `Re` (float): Reynolds number
- `CFL` (float): CFL number
- `flow_type` (str): flow configuration type
- `solver_metadata` (dict): solver-specific information
- `iteration` (int): timestep index
- `timestamp` (float): simulation time

## Trace Builder

The `TraceBuilder` class reconstructs intermediate fields and builds the solver trace:

```python
from trace_viewer.trace_builder import TraceBuilder

builder = TraceBuilder(snapshot)
trace = builder.build_trace()

# Access reconstructed fields
u_star = trace.reconstructed_fields['u_star']
divergence = trace.reconstructed_fields['divergence']

# Access metrics
cfl = trace.metrics['CFL']
max_u = trace.metrics['max_u']

# Extract subdomain for inspection
subdomain = builder.extract_subdomain(x_min=10, x_max=20, y_min=10, y_max=20)
```

## UI Components

### Navigation Panel (Left)
- Current timestep display
- Step forward/backward buttons
- Timeline slider
- Direct step input spinbox

### Solver Trace Panel (Center)
- 5-step pipeline visualization
- LaTeX equation rendering
- Input/output field lists
- Method descriptions
- Computed metrics

### Subdomain Viewer (Right)
- Bounding box selection
- Tabbed view for u, v, p, divergence
- Color-coded cell values
- Numerical precision display

## Integration with AeroJAX Solver

To enable snapshot saving in the existing solver, add the following to your simulation loop:

```python
from trace_viewer.snapshot_utils import save_snapshot, create_snapshot_from_sim_state

# In your simulation loop
for step in range(num_steps):
    # ... run solver step ...
    
    # Create and save snapshot
    snapshot = create_snapshot_from_sim_state(
        u=sim_state.u,
        v=sim_state.v,
        p=sim_state.p,
        mask=mask,
        dt=dt,
        nu=nu,
        dx=dx,
        dy=dy,
        divergence=divergence,
        Re=Re,
        CFL=cfl,
        flow_type=flow_type,
        iteration=step,
        timestamp=step * dt
    )
    
    save_snapshot(snapshot, f"snapshots/snapshot_{step:06d}.pkl")
```

## Requirements

- PyQt6
- numpy
- matplotlib (for LaTeX rendering)

## Design Principles

1. **Decoupling**: No runtime coupling with solver
2. **Deterministic**: Same snapshots produce same traces
3. **Efficient**: NumPy slicing for subdomain extraction
4. **Modular**: Clean separation of concerns
5. **Extensible**: Easy to add new trace steps or visualizations

## Future Enhancements

- Add support for MAC staggered grid visualization
- Export traces to PDF/HTML reports
- Compare multiple simulation runs
- Custom trace step definitions
- Field animation playback
