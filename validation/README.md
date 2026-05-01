# LDC Validation Module

Benchmark validation module for Lid-Driven Cavity (LDC) flow simulations in AeroJAX.

## Overview

This module provides validation against the canonical Ghia et al. (1982) benchmark data for Lid-Driven Cavity flow. It is fully decoupled from the solver and only consumes snapshot data, making it suitable for:

- Validating physical correctness of emergent vortex structures
- Comparing solver variants (NS vs LBM vs ML operators)
- Visually grounding simulations against canonical CFD literature

## Features

- **Benchmark Data**: Primary vortex center locations for Re = [100, 400, 1000, 3200, 5000, 7500]
- **Vortex Center Detection**: Automatic detection from velocity fields using vorticity minimization
- **Error Metrics**: L2 distance error, error vector components
- **GUI Overlays**: Visualization markers for benchmark and simulation vortex centers
- **History Tracking**: Track vortex center evolution over time
- **Extension Hooks**: Placeholders for secondary vortex detection and centerline velocity comparison

## Usage

### Basic Validation

```python
from validation import LDCValidator

# Create validator for specific Reynolds number
validator = LDCValidator.load(Re=1000)

# Compute validation from snapshot
validator.compute(snapshot)

# Get results
center = validator.get_vortex_center()
error = validator.get_error()

print(f"Computed center: ({center.x:.4f}, {center.y:.4f})")
print(f"L2 error: {error.l2_distance:.4f}")
```

### GUI Overlay Integration

```python
from validation import LDCValidator
from validation.ldc_overlay import LDCValidationOverlay

# Create validator
validator = LDCValidator.load(Re=1000)

# Create overlay for plot widget
overlay = LDCValidationOverlay(validator, plot_widget)

# Update after each timestep
validator.compute(snapshot)
overlay.update()

# Toggle visibility
overlay.set_visible(True)
```

### History Tracking

```python
validator = LDCValidator.load(Re=1000)

# Process multiple snapshots
for snapshot in snapshots:
    validator.compute(snapshot)

# Get complete history
history = validator.get_history()
for center, error, iteration in zip(
    history.vortex_centers, history.errors, history.iterations
):
    print(f"Iter {iteration}: center=({center.x:.3f}, {center.y:.3f}), error={error.l2_distance:.4f}")
```

## API Reference

### LDCValidator

**Class Methods:**
- `LDCValidator.load(Re: float) -> LDCValidator`: Create validator for specific Reynolds number

**Instance Methods:**
- `compute(snapshot)`: Compute vortex center and validation error from snapshot
- `get_vortex_center() -> Optional[VortexCenter]`: Get most recently computed vortex center
- `get_error() -> Optional[ValidationError]`: Get most recently computed validation error
- `get_history() -> ValidationHistory`: Get complete validation history
- `get_reference_center() -> Tuple[float, float]`: Get benchmark vortex center for current Re
- `clear_history()`: Clear validation history

**Extension Hooks (placeholders):**
- `detect_secondary_vortices(snapshot)`: Detect secondary corner vortices (high Re)
- `compute_centerline_velocity(snapshot)`: Extract centerline velocity profiles
- `compare_centerline_profiles(snapshot)`: Compare profiles against Ghia data
- `track_multi_vortex_structure(snapshot)`: Track multiple vortex structures

### LDCValidationOverlay

**Constructor:**
- `LDCValidationOverlay(validator: LDCValidator, plot_widget)`: Create overlay for plot widget

**Methods:**
- `update()`: Update overlay markers based on current validation results
- `set_visible(visible: bool)`: Show/hide overlay
- `remove()`: Remove overlay items from plot

## Benchmark Data

Primary vortex center locations from Ghia et al. (1982):

| Re  | Reference Center (x, y) |
|-----|------------------------|
| 100 | (0.50, 0.54)           |
| 400 | (0.52, 0.56)           |
| 1000| (0.53, 0.57)           |
| 3200| (0.53, 0.58)           |
| 5000| (0.53, 0.585)          |
| 7500| (0.54, 0.59)           |

## Vortex Center Detection

The validator computes the primary vortex center by:

1. Computing vorticity field: ω = ∂v/∂x - ∂u/∂y
2. Restricting search to central region: [0.2, 0.8] × [0.2, 0.8] (normalized)
3. Finding minimum absolute vorticity location (vortex center)
4. Converting to normalized coordinates

## Architecture Constraints

- **Decoupled**: Does not modify solver code
- **Read-only**: Only consumes snapshot data
- **Deterministic**: Same snapshot input produces same output
- **Lightweight**: No expensive recomputation loops

## Reference

Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

## Example

See `validation/example_usage.py` for complete examples including:
- Basic usage with synthetic data
- Multiple Reynolds number validation
- History tracking over timesteps
- GUI overlay integration
