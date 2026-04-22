# AeroJAX

**A differentiable, structure-preserving framework for real-time incompressible flow simulation, control, and inverse design. Built with JAX.**

> **AeroJAX** is a high-performance, JAX-native CFD engine architected for real-time physics simulation, gradient-based optimization, and seamless neural operator integration. By combining rigorous numerical methods (staggered MAC grids) with automatic differentiation, it enables end-to-end differentiable workflows for shape optimization and physical AI research.


[![JAX](https://img.shields.io/badge/JAX-0.9.2-9cf?logo=jax&logoColor=white)](https://jax.readthedocs.io/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.8.1-41cd52?logo=qt&logoColor=white)](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
[![License](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)

[![GitHub stars](https://img.shields.io/github/stars/arriemeijer-creator/AeroJAX?style=social)](https://github.com/arriemeijer-creator/AeroJAX)
[![GitHub forks](https://img.shields.io/github/forks/arriemeijer-creator/AeroJAX?style=social)](https://github.com/arriemeijer-creator/AeroJAX)



![Demo Animation](NACA_Airfoil.gif)



**The GUI**
![GUI Screenshot](GUI.png)

**Key Dependencies:**
- [JAX](https://jax.readthedocs.io/) - Differentiable numerical computing
- [PyQt6](https://www.riverbankcomputing.com/static/Docs/PyQt6/) - GUI framework
- [pyqtgraph](https://pyqtgraph.readthedocs.io/) - Scientific visualization
- [NumPy](https://numpy.org/) - Numerical computing

---

## Quick Start

```bash
git clone https://github.com/arriemeijer-creator/AeroJAX
cd AeroJAX
pip install -r requirements.txt
python main.py
```

**First run tips:**

- Start with 512×96 grid for real-time performance
- Try von_karman flow with a NACA 2412 airfoil
- Enable "Adaptive dt" to see the PID controller in action

---

## Capabilities at a glance

| Category | Capabilities |
|----------|---------------|
| **Solvers** | Collocated / MAC staggered, RK3/RK2, FFT/CG/Multigrid pressure |
| **Turbulence** | LES (dynamic/smagorinsky), hyper-viscosity |
| **Obstacles** | Cylinder, NACA (4/5-digit), Cow, 3-cylinder, freeform draw |
| **Controls** | Lockable Re/U/ν, adaptive CFL dt, Brinkman penalization |
| **Diagnostics** | CL/CD, stagnation/separation, Cp_min, wake deficit, L2/max/relative errors, CSV export |
| **GUI** | Real-time velocity/vorticity/pressure/dye plots, streamlines, quivers, video export, light/dark themes |
| **Performance** | 297–31 FPS (512×96 → 2084×384), decoupled threads, shared memory |

---

## What Makes This Framework Distinct

AeroJAX is designed to bridge the gap between classical numerical analysis and modern machine learning:

- **Structure-Preserving Numerics**: Utilizes a staggered MAC grid to ensure mass conservation and eliminate pressure-velocity decoupling (checkerboard artifacts).
- **End-to-End Differentiable**: Every solver step is JAX-native, allowing for `jax.grad` and `jax.vmap` across the full simulation for adjoint-based inverse design.
- **Real-Time Inverse Design**: Morph geometries and optimize shapes live via differentiable immersed boundaries (Brinkman penalization) and smoothed Heaviside masks.
- **Neural-Ready Architecture**: High-throughput synthetic data generation and latent-space acceleration hooks for geometric algebra and multiscale neural operators.
- **Control-Driven Stability**: Adaptive divergence-aware timestep control using PID feedback instead of static CFL tuning.

This makes it useful for:

- CFD education and visualization
- Numerical method development and experimentation
- Neural operator and CFD–ML research
- Inverse design prototyping and flow interrogation

---

## Core Simulation Architecture

### 1. Solver Design & Numerics

The framework implements a JAX-accelerated incompressible Navier–Stokes solver with:

- **Spatial Discretization:** Staggered Cartesian grid (MAC) for robust pressure-velocity coupling.
- **Time Integration:** Explicit RK3 advection scheme with projection-based fractional-step coupling.
- **Immersed Boundary:** Brinkman penalization for differentiable fluid-structure interaction.
- **Pressure Projection:** Selectable solvers including FFT-based Poisson, Conjugate Gradient, and Multigrid.

The solver is designed for real-time execution and parameter interactivity rather than batch-only workflows.

### 2. Immersed Boundary System

#### Brinkman Penalization

Solid bodies are embedded using a porous-medium forcing formulation:

- Velocity damping inside solid regions via resistance term
- No body-fitted meshing required
- Smooth fluid–structure interaction

#### Smoothed Interface Representation

Geometry is represented using a continuous mask field:

- Smoothed Heaviside function
- Interface thickness controlled by epsilon (ε)
- Preserves numerical stability and differentiability

#### Supported geometries

- Cylinders
- NACA 4-digit and 5-digit airfoils
- Cow (simplified animal shape)
- Three-cylinder array
- Freeform (custom drawn obstacles)
- Extensible parametric shapes

### 3. Adaptive Time Stepping (Divergence-PID Controller)

Time integration is governed by a closed-loop feedback controller.

#### Control Objective

`max(|∇ · u|) → target divergence`

#### Control System

A PID controller adjusts timestep based on incompressibility error:

- Proportional: instantaneous divergence error
- Integral: accumulated drift correction
- Derivative: instability growth prediction

#### Stability Constraints

Timestep is additionally bounded by:

- Brinkman stiffness limiter (η-dependent)
- Hard limits: dt_min / dt_max

> The solver behaves as a self-regulating numerical dynamical system with feedback-controlled stability.

### 4. Numerical Methods

#### Advection

- Explicit RK3 scheme (collocated or MAC‑compatible)

#### Pressure Projection (selectable)

- FFT-based Poisson solver (periodic cases)
- Conjugate Gradient solver
- Multigrid solver (collocated and MAC versions)

#### Spatial Discretization

- Structured Cartesian grid (collocated or MAC staggered)

#### Typical operating resolutions:

- 512 × 96 — real-time interactive
- 1024 × 192 — high fidelity
- up to ~3000 × 600 for larger experiments

### 5. Turbulence Modeling (LES Mode)

Optional LES support includes:

- Smagorinsky-type subgrid closure
- Constant or dynamic formulations

Enables more stable higher-Re simulations when needed.

---

## Parallel Architecture

### 6. Simulation–GUI Decoupling

The system is split into two independent subsystems:

- Simulation thread (JAX solver)
- GUI thread (PyQt6 visualization)

#### Communication uses:

- Shared memory buffers for zero-copy field transfer
- Lightweight metadata queues

#### Key properties:

- Simulation and rendering FPS are decoupled
- Frame dropping under load prevents blocking
- Stable interactivity under varying computational cost

### 7. Shared Memory Data Pipeline

Transferred fields include:

- Velocity (u, v)
- Vorticity
- Scalar dye transport
- Derived diagnostics (velocity magnitude, etc.)

This enables real-time visualization without repeated JAX → CPU → GUI copy overhead.

---

## Flow Configuration Space

### Geometry Parameters

- Cylinder radius
- NACA airfoil type, chord length, angle of attack
- Cow scale and position
- Three-cylinder array parameters
- Freeform mask
- Spatial placement

### Flow Parameters

- Reynolds number (Re)
- Domain size
- Inflow conditions (U_inf)

### Numerical Parameters

- epsilon (ε): interface smoothing scale
- eta (η): Brinkman damping strength
- dt bounds
- Pressure solver selection
- LES toggles

---

## Diagnostics and Output Quantities

### Field Outputs

- Velocity field (u, v)
- Pressure field (p)
- Vorticity field (ω)
- Scalar dye transport field
- Immersed boundary mask field

### Numerical Diagnostics

- Maximum divergence
- CFL tracking
- L2 / max / relative change metrics
- Component-wise velocity error measures

### Aerodynamic Metrics (Airfoils)

Computed in real time

- Lift coefficient (CL)
- Drag coefficient (CD)
- Stagnation point location
- Separation point estimate
- Minimum pressure coefficient (Cp_min)
- Wake deficit metrics

---

## Interactive GUI System

### Simulation Controls

- Start / pause / reset
- Grid resizing
- Flow regime switching (von Kármán, lid-driven cavity, Taylor–Green)

### Physical Controls

- Reynolds number
- Geometry configuration
- Angle of attack
- Obstacle type selection

### Numerical Controls

- Solver type selection (collocated / MAC)
- LES toggle
- epsilon and eta tuning
- Adaptive timestep settings

### Visualization Controls

- Velocity / vorticity / dye display modes
- Colormap selection
- FPS controls
- Auto-scaling tools
- Frame export / recording

---

## Performance Characteristics

AeroJAX is designed for stable real-time performance across a wide range of resolutions, with simulation throughput largely independent of GUI rendering load due to the decoupled architecture.

### Typical CPU performance

| Configuration | Solver + visualization only | With live metrics + diagnostics |
|--------------|------------------------------|----------------------------------|
| 512 × 96     | ~297 FPS                     | ~170 FPS                         |
| 1024 × 192   | ~131 FPS                     | ~91 FPS                          |
| 2084 × 384   | ~37 FPS                      | ~31 FPS                          |

### Key observations:

- Real-time interactivity is maintained at standard working resolutions
- Diagnostic overhead remains modest even with live force and stability metrics enabled
- Shared-memory rendering prevents visualization from throttling solver throughput
- Higher-resolution runs remain practical for detailed analysis and dataset generation

> Solver performance is governed primarily by numerical workload rather than interface overhead, enabling consistent interactive use across exploratory and higher-fidelity modes.

---

## Validation & Benchmark Cases

### Validated against

- Lid-driven cavity flow (currently removed because of stability issues)
- Taylor–Green vortex decay
- Von Kármán vortex shedding

### Used for

- Solver verification
- Numerical stability testing
- ML dataset generation

---

## System Characterization

### This framework is best understood as

- A real-time incompressible CFD solver
- A Brinkman immersed boundary system
- A feedback-controlled numerical dynamical system
- A live diagnostic physics platform
- A synthetic data engine for ML workflows

### It is not

- A general-purpose industrial CFD suite
- A mesh-based finite-volume framework
- A high-Re DNS turbulence solver
- A fully validated aerodynamic design tool (yet)

---

## Design Philosophy

- Minimal abstraction overhead
- Direct control of numerical and physical parameters
- Tight coupling between simulation and observation
- Real-time feedback loops for stability and control
- Differentiability-first JAX-native design

---

## Project Structure

```
GitHub/
├── LICENSE
├── README.md
├── requirements.txt
├── main.py (1390 lines)
├── Flow.png
├── GUI.png
├── NACA_Airfoil.gif
├── grid_info_20260418_190301.json
│
├── advection_schemes/
│   ├── __init__.py (4 lines)
│   ├── rk3_mac.py (253 lines)
│   ├── rk3_simple_new.py (298 lines)
│   └── utils.py (37 lines)
│
├── inverse_design_WIP/
│   ├── __init__.py (16 lines)
│   ├── config.py (66 lines)
│   ├── main.py (416 lines)
│   ├── optimizer.py (565 lines)
│   ├── ui_components.py (416 lines)
│   └── visualization.py (246 lines)
│
├── obstacles/
│   ├── cow.py (275 lines)
│   ├── cylinder_array.py (66 lines)
│   ├── freeform_drawer.py (388 lines)
│   └── naca_airfoils.py (287 lines)
│
├── pressure_solvers/
│   ├── __init__.py (4 lines)
│   ├── multigrid_solver.py (223 lines)
│   └── multigrid_solver_mac.py (184 lines)
│
├── solver/
│   ├── __init__.py (77 lines)
│   ├── boundary_conditions.py (97 lines)
│   ├── brinkman.py (108 lines)
│   ├── config.py (25 lines)
│   ├── geometry.py (25 lines)
│   ├── les_models.py (135 lines)
│   ├── metrics.py (305 lines)
│   ├── operators.py (115 lines)
│   ├── operators_mac.py (215 lines)
│   ├── params.py (405 lines)
│   └── solver.py (1482 lines)
│
├── timestepping/
│   ├── __init__.py (20 lines)
│   └── adaptivedt.py (53 lines)
│
└── viewer/
    ├── __init__.py (31 lines)
    ├── config.py (229 lines)
    ├── display_manager.py (757 lines)
    ├── flow_manager.py (462 lines)
    ├── modern_stylesheet.py (975 lines)
    ├── naca_handler.py (146 lines)
    ├── parameter_handlers.py (826 lines)
    ├── simulation_controller.py (838 lines)
    ├── state/
    │   ├── __init__.py (63 lines)
    │   └── store.py (404 lines)
    ├── ui_components/
    │   ├── __init__.py (23 lines)
    │   ├── collapsible_groupbox.py (126 lines)
    │   ├── control_panel.py (630 lines)
    │   ├── dye_controls.py (95 lines)
    │   ├── floating_control_bar.py (196 lines)
    │   ├── info_panel.py (482 lines)
    │   ├── obstacle_controls.py (403 lines)
    │   ├── time_controls.py (47 lines)
    │   ├── top_console.py (42 lines)
    │   └── visualization_controls.py (190 lines)
    └── visualization/
        ├── __init__.py (13 lines)
        ├── flow_visualization.py (1766 lines)
        ├── obstacle_renderer.py (562 lines)
        └── sdf_visualization.py (123 lines)
```

**Core solver (differentiable, JAX):** ~3,500 lines  
**Obstacles:** ~1,016 lines  
**GUI & visualization:** ~8,000 lines  
**Advection & pressure solvers:** ~998 lines  
**Inverse design:** ~1,725 lines  
**Total functional code:** ~15,239 lines

---

## License

GNU Lesser General Public License v3.0

---

## Author

Arno Meijer  
Mechanical Engineer | CFD–ML Systems Developer

---

## Citation

```bibtex
@misc{meijer2026aerojax,
  author = {Meijer, Arno},
  title = {AeroJAX: Differentiable, Structure-Preserving CFD Framework},
  year = {2026},
  url = {https://github.com/arriemeijer-creator/AeroJAX}
}
