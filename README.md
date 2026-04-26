# AeroJAX

**A real-time, JAX-native CFD framework for interactive flow research, control, and inverse design.**

AeroJAX evolved from a need for high-throughput synthetic data for Neural Operators (FNOs, CNNs) into a full-scale research platform. It bridges the gap between classical numerical analysis and modern ML workflows, providing a fast environment where physical parameters and solver architectures can be interrogated in real-time.

[![JAX](https://img.shields.io/badge/JAX-0.9.2-9cf?logo=jax&logoColor=white)](https://jax.readthedocs.io/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.8.1-41cd52?logo=qt&logoColor=white)](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
[![License](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

![NACA Airfoil Simulation](NACA_Airfoil.gif)

## Why AeroJAX is Distinct

Most CFD suites are built for batch processing on massive clusters. AeroJAX is built for **live interrogation**:

- **Optimization-Ready:** Every solver step is JAX-native and end-to-end differentiable. This enables adjoint-based inverse design and shape optimization loops directly within the framework.
- **Interactive Physics Sandbox:** Unlike traditional solvers, you can hand-draw obstacles, toggle LES models, or swap pressure solvers mid-simulation to observe numerical stability and flow transitions instantly.
- **CPU-First Engineering:** Architected to deliver high-fidelity results on standard workstations. By decoupling the JAX simulation core from the rendering thread, the solver maintains high throughput even at resolutions up to ~3000x600 without a GPU.
- **ML Data Engine:** While it functions as a standalone research tool, its speed and parametric control make it a primary engine for generating physics-consistent datasets for Neural Operator research.

---

## The GUI
![GUI Screenshot](GUI.png)

## Quick Start

```bash
git clone https://github.com/arriemeijer-creator/AeroJAX
cd AeroJAX
pip install -r requirements.txt
python main.py
```

### First run tips

- Start with 512×96 grid for real-time performance
- Try von_karman flow with a NACA 2412 airfoil
- Enable "Adaptive dt" to see the PID controller in action

## Technical Capabilities

AeroJAX is a physics sandbox designed to be visualization-agnostic—if the flow doesn't look right, the physics aren't right. It allows for hot-swapping numerical and physical parameters on the fly.

### Solver & Numerics

- **Discretization**: Toggle between Collocated and MAC staggered grids
- **Integrators**: Explicit RK3 and RK2 schemes
- **Pressure Projection**: Selectable FFT, Conjugate Gradient, and Multigrid Poisson solvers
- **Precision**: Global float32 / float64 toggles for performance vs. stability testing
- **Stability**: Closed-loop Adaptive CFL control using a PID controller that modulates dt based on real-time divergence error

### Obstacle System & Interactivity

- **Parametric Geometries**: Instant generation of Cylinders, NACA 4/5-digit airfoils, and 3-cylinder arrays
- **Freeform Draw**: Draw custom masks via Pygame; the engine instantly injects the geometry into the domain using Brinkman penalization
- **Flow Control**: Lockable Reynolds number (Re), Viscosity, or U_inf inputs (the engine auto-calculates the third)
- **Boundary Conditions**: Toggleable slip / no-slip conditions

### Neural Operator Integration - AVAILABLE ON NEXT RELEASE

AeroJAX v2.0 will include a built-in training pipeline for neural pressure solvers (CNNs, Linear, and Latent-space architectures), allowing users to generate parameterized training data on-the-fly, train custom operators, and hot-swap them into the simulation loop to replace traditional Poisson solvers.

### Turbulence & Advanced Physics

- **LES Modeling**: Support for Dynamic/Smagorinsky Large Eddy Simulation and hyper-viscosity
- **Transport**: Scalar dye injection and particle injection for flow visualization
- **Domain Presets**: Quick-load for Lid-Driven Cavity (LDC) and Taylor-Green Vortex (TGV) cases

### Diagnostics & Visualization

- **Live Aerodynamics**: Real-time calculation of C<sub>L</sub>, C<sub>D</sub>, C<sub>p_min</sub>, and wake deficit
  - *Numerical Note*: Due to the nature of Brinkman penalization and smoothed Heaviside masks, conventional pressure-integration methods for force coefficients are unreliable. AeroJAX utilizes a Circulation-based C<sub>L</sub> (via Kutta-Joukowski theorem) and Momentum Deficit-based C<sub>D</sub> to ensure robust aerodynamic metrics.
- **Flow Physics**: Automatic tracking of stagnation and separation lines
- **Interactivity**: Info-on-hover diagnostics, sliders for obstacle positioning, and log-based color scale toggles
- **Export**: Video export functionality and CSV diagnostic logging

## Performance & Architecture

Since this was developed on a CPU-only stack, the architecture is strictly optimized to maximize throughput:

- **Decoupled Pipeline**: The JAX simulation core is isolated from the rendering thread to prevent visualization from throttling solver FPS
- **State Management**: Redux-style state management ensures predictable GUI behavior under heavy computational load
- **Data Pipeline**: Uses shared-memory buffers for zero-copy field transfers between the solver and the PyQt6/pyqtgraph visualizer

### Benchmarks (Tested on CPU)

| Grid Resolution | Solver + Viz | With Full Diagnostics |
|-----------------|--------------|----------------------|
| 512 × 96        | ~297 FPS     | ~170 FPS              |
| 1024 × 192      | ~131 FPS     | ~91 FPS               |
| 2084 × 384      | ~37 FPS      | ~31 FPS               |

## Known Limitations

- **2D only**: AeroJAX is designed for 2D flow research and rapid prototyping
- **Force coefficients are qualitative**: Due to Brinkman penalization, C<sub>L</sub>/C<sub>D</sub> values are useful for relative comparison and optimization trends, not absolute prediction
- **Moderate Reynolds numbers**: Best suited for Re < 10,000 (laminar to early turbulent regimes)
- **CPU-focused**: While JAX supports GPUs, the architecture is specifically optimized for workstation CPUs

## Project Structure

- `solver/`: Core JAX-accelerated Navier-Stokes kernels and Brinkman penalization
- `pressure_solvers/`: Multigrid, FFT, and CG implementations
- `viewer/`: PyQt6 GUI components and Redux state logic
- `obstacles/`: Parametric NACA generators and freeform drawing logic
- `inverse_design_WIP/`: Experimental module for adjoint-based shape optimization

## License

GNU Lesser General Public License v3.0

## Author

Arno Meijer - Mechanical Engineer | CFD–ML Systems Developer
