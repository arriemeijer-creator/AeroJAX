# Differential CFD-ML

**A Differentiable Hybrid Navier–Stokes Framework with Multi-Scale Neural Correction, Latent-Space Acceleration, and Differentiable Inverse Design**

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-orange.svg)](https://www.riverbankcomputing.com/software/pyqt/)

![GUI.png](GUI.png)

![flow_types.png](flow_types.png)

---

## What This Framework Is

Differential CFD-ML is a differentiable test bench for developing neural operators in computational fluid dynamics. Built on JAX, it provides a modular framework for training and testing machine learning-enhanced CFD solvers.

The framework serves two purposes:

1. **Generate ground truth data** from validated numerical solvers across 5 flow types
2. **Provide a modular architecture** for plugging in neural components (pressure solvers, turbulence models, latent dynamics)

If you're working on neural operators for fluid dynamics, this gives you a ready-made environment to train and test them against traditional solvers.

---

## What You Can Build With This

- **Neural pressure solvers** – Replace iterative Poisson solvers with learned mappings
- **Learned turbulence models** – Augment or replace Smagorinsky SGS closures
- **Latent dynamics** – Compress and predict flow evolution in low-dimensional spaces
- **Inverse design** – Optimize geometries with gradient descent through flow physics

---

## Current Status (March 2026)

The baseline solver is complete and working. The neural components are under development.

**Working:**
- 5 flow types with differentiable boundary conditions
- 9 advection schemes (upwind to WENO5 to spectral)
- 7 pressure solvers (Jacobi to multigrid)
- Real-time GUI with 6 plots and video recording
- Data export for training set generation
- Test framework covering 1,680 configurations

**In progress:**
- Neural pressure solver (training pipeline exists, tuning the loss function)

**Planned:**
- Fine-scale and coarse-scale neural correction
- Latent space modeling
- Differentiable inverse design
- Reinforcement learning integration

For a detailed roadmap and theoretical background, see `docs/framework.pdf`.

---

## Core Features

| Category | Features |
|----------|----------|
| **Flow Types** | Von Kármán vortex shedding, lid-driven cavity, channel flow, backward-facing step, Taylor-Green vortex |
| **Advection Schemes** | Upwind, MacCormack, Jos Stam, QUICK, WENO5, TVD, RK3, Spectral |
| **Pressure Solvers** | Jacobi, FFT, ADI, SOR, Gauss-Seidel RB, Conjugate Gradient, Multigrid |
| **Grid Resolutions** | 4 per flow type (Coarse, Medium, Fine, Ultra Fine) |
| **Differentiable Operators** | All finite-difference operators are JIT-compiled and differentiable |
| **Adaptive Timestepping** | CFL-based with flow-type specific safety limits |
| **Testing** | 1,680 configuration test suite with automated validation |
| **Visualization** | PyQtGraph GUI with 6 simultaneous plots |
| **Data Export** | Full fields (velocity, vorticity, pressure) and time history |

---

## Repository Structure
```
differential-cfd/
├── LICENSE
├── README.md
├── requirements.txt
│
├── baseline_clean.py              # Main solver
├── baseline_viewer.py             # GUI entry point
├── test_framework.py             # Test suite
├── collect_code.py               # Code collection utility
├── resize_images.py              # Image resizing utility
│
├── advection_schemes/            # 9 advection schemes
│ ├── __init__.py
│ ├── jos_stam_scheme.py
│ ├── maccormack_scheme.py
│ ├── quick_scheme.py
│ ├── rk3_scheme.py
│ ├── spectral_scheme.py
│ ├── tvd_scheme.py
│ ├── upwind_scheme.py
│ ├── utils.py
│ └── weno5_scheme.py
│
├── pressure_solvers/              # 7 pressure solvers
│ ├── __init__.py
│ ├── jacobi_solver.py
│ ├── fft_solver.py
│ ├── adi_solver.py
│ ├── sor_solver.py
│ ├── gauss_seidel_rb_solver.py
│ ├── cg_solver.py
│ └── multigrid_solver.py
│
├── timestepping/                # Adaptive timestepping
│ ├── __init__.py
│ └── adaptive_dt.py
│
├── baseline/                     # Legacy baseline code
│ ├── geometry.py
│ ├── operators.py
│ ├── solver.py
│ └── viewer.py
│
├── inverse/                     # Inverse design components
│ ├── geometry.py
│ └── optimize.py
│
├── latent/                      # Latent space modeling
│ ├── decoder.py
│ ├── encoder.py
│ └── operator.py
│
├── docs/                        # Documentation
│ ├── roadmap.md
│ └── A Differential JAX-Orchestrated Incompressible Flow Simulation Framework - Arno Meijer.pdf
│
└── examples/                     # Example scripts
    ├── cylinder_flow.py
    ├── benchmark_all.py
    └── inverse_design.py
```


---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/arnomeijer/differential-cfd.git
cd differential-cfd

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for visualization
pip install pyqt6 pyqtgraph pillow
```

### Launch the GUI

```bash
python baseline_viewer.py
```

This launches the comprehensive interactive interface where you can control:

- **Flow Type** - Von Kármán, Cavity, Channel, Backward Step, Taylor-Green
- **Grid Resolution** - Coarse, Medium, Fine, Ultra Fine (per flow type)
- **Advection Scheme** - Upwind, MacCormack, Jos Stam, QUICK, WENO5, TVD, RK3, Spectral
- **Pressure Solver** - Jacobi, FFT, ADI, SOR, Gauss-Seidel RB, CG, Multigrid
- **Reynolds Number** - 10–1000
- **dt Mode** - Fixed or Adaptive (CFL-based)
- **Visualization** - 30+ colormaps, 6 simultaneous plots

### Batch Simulation

For automated testing or data collection:

```python
from baseline_clean import GridParams, FlowParams, GeometryParams, SimulationParams, BaselineSolver

# Set up grid (Medium resolution)
grid = GridParams(nx=512, ny=96, lx=20.0, ly=4.5)

# Set up flow
flow = FlowParams(Re=150.0, U_inf=1.0)

# Set up geometry (cylinder)
geom = GeometryParams(center_x=2.5, center_y=2.25, radius=0.18)

sim_params = SimulationParams()
solver = BaselineSolver(grid, flow, geom, sim_params)

# Run for 20,000 steps
u, v = solver.run_simulation(n_steps=20000)

# Extract diagnostics
drag_coefficient = solver.history['drag'][-1]
```

### Run Tests

```bash
python test_framework.py
```

Select test mode:
1. **Quick test** – 20 configs, 50 frames each (~10-20 minutes)
2. **Medium test** – 50 configs, 100 frames each (~1-2 hours)
3. **Full test** – All 1,680 configs (~1-2 days)
4. **Custom test** – User-defined

### Test Framework Features

- Automated validation of all configurations
- CFL monitoring with color-coded warnings
- Numerical stability detection (NaN/Inf, velocity explosion)
- Performance metrics (steps/sec, average dt)
- Error categorization for debugging
- Results export to CSV

### Field Plots (4)
- **Velocity Magnitude** – Sequential colormap (plasma)
- **Vorticity** – Diverging colormap (RdBu)
- **Streamlines** – Sequential colormap (viridis)
- **Pressure** – Sequential colormap (inferno)

### Live History Plots (2)
- **Drag & Lift** – Real-time force coefficients
- **Kinetic Energy & Enstrophy** – Energy diagnostics

### Controls
- **Real-time** flow type switching with automatic grid update
- **Grid resolution** adjustment on the fly
- **Colormap** selection from 30+ options
- **Video recording** (GIF export)
- **Data export** (CSV files for all fields and history)

## Data Export

The "Export Data" button in the GUI generates:

- `velocity_u_*.csv`, `velocity_v_*.csv`
- `vorticity_*.csv`, `pressure_*.csv`
- `history_*.csv` (time, KE, enstrophy, drag, lift)
- `grid_info_*.json` with all parameters

This is what generates training data for neural operators.

## Neural Integration Points

The framework is designed so you can plug in neural components without touching the core solver.

### Replace Pressure Solver

```python
class NeuralPressureSolver:
    def __init__(self):
        self.model = load_model('pressure_network.pt')
    
    def predict(self, u_star, v_star):
        return self.model(jnp.stack([u_star, v_star]))
```

### Augment Turbulence Model

```python
class NeuralTurbulenceModel:
    def __init__(self):
        self.model = load_model('sgs_network.pt')
    
    def predict(self, u, v):
        return self.model(compute_gradients(u, v))
```

### Latent Space Dynamics

```python
class LatentOperator:
    def __init__(self):
        self.encoder = load_encoder()
        self.decoder = load_decoder()
        self.latent_dynamics = load_dynamics()
    
    def predict(self, u, v):
        z = self.encoder(u, v)
        z_next = self.latent_dynamics(z)
        return self.decoder(z_next)
```
## Flow Types

| Flow Type | Domain | Physics |
|-----------|--------|---------|
| **Von Kármán** | 20×4.5 (channel) | Vortex shedding behind cylinder |
| **Lid-Driven Cavity** | 1×1 (square) | Recirculation with moving lid |
| **Channel Flow** | 4×1 (rectangular) | Poiseuille flow |
| **Backward Step** | 10×1 (step expansion) | Separation bubble | (UNSTABLE - WORKING ON IT)
| **Taylor-Green** | 2π×2π (periodic) | Decaying turbulence |

## Numerical Methods

| Category | Methods | Order | Best For |
|----------|---------|-------|----------|
| **Advection** | Upwind, MacCormack, Jos Stam, QUICK, WENO5, TVD, RK3, Spectral | 1st–5th | From diffusive to high-resolution |
| **Pressure** | Jacobi, FFT, ADI, SOR, Gauss-Seidel RB, CG, Multigrid | Iterative–Spectral | From simple to optimal |

All operators are implemented in JAX and fully differentiable.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **jax** | ≥0.4.0 | Automatic differentiation, GPU acceleration |
| **jaxlib** | ≥0.4.0 | JAX core library |
| **numpy** | ≥1.24.0 | Numerical operations |
| **pyqt6** | ≥6.4.0 | GUI framework |
| **pyqtgraph** | ≥0.13.0 | Real-time visualization |
| **pillow** | ≥9.0.0 | Video recording |

Documentation
Full design documentation is in docs/framework.pdf (70+ pages). It covers:

Part I: Differentiable hybrid Navier-Stokes solver

Part II: Active flow control and neural flow management

Part III: Reinforcement-driven optimization

Appendices on theory, implementation, transition plans, and mitigation strategies

The PDF describes the complete vision. This README describes what actually works right now.

Contributing
Contributions welcome, especially in these areas:

Neural operators (pressure, turbulence, dynamics)

New flow types (airfoil, Rayleigh-Bénard, etc.)

3D extension

Optimization loops

Open an issue or submit a PR.

License
LGPL v3. See LICENSE file.

Author
Arno Meijer

Citation
bibtex
@misc{meijer2026differentialcfd,
  author = {Meijer, Arno},
  title = {Differential CFD-ML},
  year = {2026},
  url = {https://github.com/arnomeijer/differential-cfd}
}
