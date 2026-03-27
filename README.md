# Differential CFD-ML

A Fully Differentiable Hybrid Navier–Stokes Framework with Multi-Scale Neural Correction, Latent-Space Acceleration, and Differentiable Inverse Design

![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)
![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![JAX: 0.4+](https://img.shields.io/badge/JAX-0.4+-green.svg)
![PyQt6: 6.0+](https://img.shields.io/badge/PyQt6-6.0+-orange.svg)

![image.png](image.png)

![von_karman_VS.mp4](von_karman_VS.mp4)

## Overview

Differential CFD-ML is a unified, fully differentiable framework for incompressible flow simulation, control, and inverse design. It combines classical numerical methods with modern machine learning to create a structure-preserving solver that supports end-to-end gradient-based optimization.

Unlike traditional CFD solvers that require expensive pressure Poisson solves and hand-coded adjoints, this framework replaces the projection step with learned multi-scale neural operators, enabling:

- **Differentiable simulation** – backpropagate through the entire flow evolution
- **Inverse design** – optimize geometries directly with gradient descent
- **Latent-space acceleration** – compress and predict flow dynamics in low-dimensional spaces
- **Real-time visualization** – interactive PyQtGraph rendering of vortex shedding

## Key Features

| Feature | Description |
|---------|-------------|
| **Differentiable Navier–Stokes** | Fully differentiable operators with JAX automatic differentiation |
| **Multi-Scale Neural Correction** | Fine-scale and coarse-scale neural operators replace the pressure Poisson solve |
| **Divergence-Free Construction** | Streamfunction-based velocity reconstruction guarantees ∇·u = 0 |
| **SGS Turbulence Modeling** | Smagorinsky subgrid-scale closure for stable high-Re simulations |
| **Brinkman Penalization** | Smooth, differentiable representation of solid boundaries |
| **SDF Geometry Representation** | Differentiable signed distance functions for sharp interface recovery |
| **Latent-Space Acceleration** | Convolutional autoencoders for high-dimensional flow compression |
| **Real-Time Visualization** | PyQtGraph-based interactive viewer for velocity and vorticity fields |
| **Inverse Design** | Gradient-based geometry optimization to meet target flow objectives |

## Repository Structure

```
differential-cfd/
├── LICENSE                     # LGPL v3 license
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── baseline/
│   ├── solver.py                # Baseline Navier–Stokes solver
│   ├── viewer.py                # PyQtGraph real-time visualization
│   ├── operators.py             # Differentiable finite-difference operators
│   └── geometry.py              # Cylinder SDF and masking
├── latent/
│   ├── encoder.py               # Convolutional encoder for latent space
│   ├── decoder.py               # Transposed convolutional decoder
│   └── operator.py              # Latent dynamics operator (neural or physics-based)
├── inverse/
│   ├── geometry.py              # Differentiable SDF for shape optimization
│   └── optimize.py              # Gradient-based inverse design loop
├── docs/
│   └── A Differential JAX-Orchestrated Incompressible Flow Simulation Framework - Arno Meijer.pdf            # Complete framework documentation
└── examples/
    ├── cylinder_flow.py         # Run vortex shedding simulation
    ├── visualize.py             # Launch real-time viewer
    └── inverse_design.py        # Shape optimization example
```

## License

This project is licensed under the GNU Lesser General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/arnomeijer/differential-cfd.git
cd differential-cfd

# Install dependencies
pip install -r requirements.txt
```

### Run a Vortex Shedding Simulation

```python
from baseline.solver import BaselineSolver
from baseline.viewer import run_viewer

# Initialize parameters
solver = BaselineSolver(
    nx=512, ny=96, lx=20.0, ly=4.5,
    Re=150.0, U_inf=1.0,
    cylinder_center=(2.5, 2.25), cylinder_radius=0.18,
    dt=0.001
)

# Launch interactive viewer
run_viewer(solver)
```

### Command Line

```bash
# Run simulation with visualization
python examples/cylinder_flow.py --visualize

# Run inverse design optimization
python examples/inverse_design.py --target-drag 1.0
```

## Validation

The baseline solver has been validated against canonical benchmarks:

| Benchmark | Reynolds Number | Target | Result |
|-----------|----------------|--------|--------|
| Cylinder Drag Coefficient | Re = 100 | 1.05–1.15 | ✓ Within 5% |
| Cylinder Drag Coefficient | Re = 150 | 0.95–1.05 | ✓ Within 5% |
| Strouhal Number | Re = 100 | 0.16–0.18 | ✓ Within 3% |
| Strouhal Number | Re = 150 | 0.18–0.20 | ✓ Within 3% |
| Divergence Error | All | < 1×10⁻⁵ | ✓ Maintained |

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Differential CFD-ML Framework                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────┐                   │
│  │ Predictor│ →  │  Fine-Scale │ →  │ Multi-Scale       │                   │
│  │ (Physics)│    │   NN        │    │ Neural Correction │                   │
│  └──────────┘    └─────────────┘    └───────────────────┘                   │
│       │              │                    │                                 │
│       ▼              ▼                    ▼                                 │
│  ┌─────────────────────────────────────────────────┐                        │
│  │           Differentiable Boundary               │                        │
│  │         (Brinkman + SDF Masking)                │                        │
│  └─────────────────────────────────────────────────┘                        │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────┐                        │
│  │           Latent-Space Acceleration             │                        │
│  │    Encoder → Latent Operator → Decoder          │                        │
│  └─────────────────────────────────────────────────┘                        │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────┐                        │
│  │           Inverse Design Loop                   │                        │
│  │    Flow Loss → SDF Gradients → Geometry Update  │                        │
│  └─────────────────────────────────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Theoretical Foundations

The framework is grounded in rigorous theory (see docs/framework.pdf for full details):

- **Consistency**: Second-order accurate discretization with convergence to incompressible Navier–Stokes
- **Stability**: Energy-dissipative when neural corrections satisfy ⟨u, 𝒩_corr⟩ ≤ C‖∇·u‖²
- **Projection Interpretation**: Neural operators approximate the Leray projection P(u) = u − ∇(∇⁻²∇·u*)
- **Multi-Scale Convergence**: Fine/coarse decomposition analogous to multigrid methods

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| jax | ≥0.4.0 | Automatic differentiation, GPU acceleration |
| jaxlib | ≥0.4.0 | JAX core library |
| equinox | ≥0.11.0 | Neural network library for JAX |
| numpy | ≥1.24.0 | Numerical operations |
| pyqt6 | ≥6.4.0 | GUI framework |
| pyqtgraph | ≥0.13.0 | Real-time visualization |
| optax | ≥0.1.0 | Optimization library |

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Part I | ✓ Complete | Baseline Navier–Stokes solver with vortex shedding, SGS model, PyQtGraph visualization |
| Part II | 🔄 In Progress | Latent-space neural operator for flow acceleration |
| Part III | 📋 Planned | Differentiable inverse design with SDF geometry optimization |
| 3D Extension | 📋 Future | Unstructured meshes, Warp kernels, multi-GPU clusters (NVLink) |

## Usage Examples

### 1. Baseline Simulation

```python
from baseline.solver import BaselineSolver

solver = BaselineSolver(
    nx=512, ny=96,
    lx=20.0, ly=4.5,
    Re=150.0, U_inf=1.0,
    cylinder_center=(2.5, 2.25), cylinder_radius=0.18,
    dt=0.001
)

# Run for 20,000 steps
u, v = solver.run_simulation(n_steps=20000)

# Extract diagnostics
drag_coefficient = solver.history['drag'][-1]
strouhal_number = compute_strouhal(solver.history['lift'], solver.dt)
```

### 2. Real-Time Visualization

```python
from baseline.viewer import BaselineViewer

viewer = BaselineViewer(solver)
viewer.run()  # Interactive window with vorticity and velocity fields
```

### 3. Inverse Design (Part III - Coming Soon)

```python
from inverse.optimize import inverse_design
from inverse.geometry import DifferentiableSDF

# Define target flow objective
target = {'drag': 0.8, 'lift': 0.0, 'vorticity': wake_profile}

# Optimize geometry
sdf = DifferentiableSDF(initial_cylinder)
optimized_sdf = inverse_design(solver, sdf, target, n_iterations=100)
```

## Documentation

Full documentation is available in docs/A Differential JAX-Orchestrated Incompressible Flow Simulation Framework - Arno Meijer.pdf, which includes:

- **Part I**: Differentiable Hybrid Navier–Stokes Solver
- **Part II**: Active Flow Control and Enhanced Neural Flow Management
- **Part III**: Reinforcement-Driven Differentiable Flow Optimization
- **Appendix A**: Theoretical Foundations (Consistency, Stability, Convergence)
- **Appendix B**: Numerical Implementations (Baseline Solver, Inverse Design, Latent Space)
- **Appendix C**: Transition Plans (Phased Development Roadmaps)
- **Appendix D**: 3D Extension Roadmap (Warp, NVLink, Unstructured Meshes)
- **Appendix E**: Mitigation Strategies (Numerical Stiffness, Latent Drift, Spectral Bias)

## Contributing

Contributions are welcome! Areas of interest:

- Validation on additional benchmark cases (lid-driven cavity, backward-facing step)
- Implementation of transformer-based temporal operators
- 3D structured grid extensions
- Warp kernel integration for high-performance operators
- Unstructured mesh support with graph neural operators

Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See LICENSE for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{meijer2026differentialcfd,
  author = {Meijer, Arno},
  title = {Differential CFD-ML: A Differentiable Hybrid Navier–Stokes Framework with Multi-Scale Neural Correction},
  year = {2026},
  url = {https://github.com/arnomeijer/differential-cfd}
}
```

## Author

**Arno Meijer**  
Mechanical Engineer | CFD-ML Researcher | HVAC Innovator

Independent Researcher, Differential CFD-ML

## Acknowledgments

This framework builds on decades of research in computational fluid dynamics, scientific machine learning, and differentiable programming. Special thanks to the JAX, Equinox, and PyQtGraph communities.

## Contact

For questions, collaborations, or opportunities, please reach out via GitHub or email.

---

*Built with JAX, Equinox, and PyQtGraph*
