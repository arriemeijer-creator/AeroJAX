# AeroJAX

*A real-time, JAX-native CFD framework for interactive flow research, control, and inverse design.*

<div align="center">

![NACA Airfoil Simulation](NACA_Airfoil.gif)

[![JAX](https://img.shields.io/badge/JAX-0.9.2-9cf)]
[![PyQt6](https://img.shields.io/badge/PyQt6-6.8.1-41cd52)]
[![License](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)]

</div>


## What makes AeroJAX different?
Most CFD software is batch‑oriented: simulations are configured, run to completion, and results are analysed afterwards. If you want to change the mesh or try a different turbulence model - you start over.

AeroJAX is an interactive CFD framework in which solver parameters, boundary conditions, and immersed geometries can be modified during runtime without restarting the simulation.  Most parameters can be modified during time integration with immediate effect. You can:

- Draw an obstacle with your mouse - the solver injects it on the next timestep using Brinkman penalisation.
- Drag a NACA airfoil across the domain with a slider and watch the wake evolve in real time.
- Pressure solvers (FFT, Conjugate Gradient, Multigrid, or Neural Operator) can be swapped without restarting the simulation.
- Toggle LES models (Smagorinsky / dynamic Smagorinsky) on the fly.
- Inject dye at any point and watch it advect.
- Record video or export frames with one click.

AeroJAX is built on JAX, making each solver step end-to-end differentiable. You can run gradient‑based inverse design (optimise an airfoil shape to minimise drag) without writing a separate adjoint solver.

The framework is CPU‑optimised for real-time performance. Typical performance reaches ~297 FPS at 512×96 on a laptop CPU. No GPU required for smaller grids.

## Quick start

```bash
git clone https://github.com/arriemeijer-creator/AeroJAX
cd AeroJAX
pip install -r requirements.txt
python main.py
```

## The GUI

<div align="center">

![GUI Screenshot](GUI.png)

</div>

### First run tips

- Start with 512×96 grid for real‑time performance.
- Try the default initialized von Kármán with NACA 0012 airfoil at 10° AoA.
- Enable "Adaptive dt" to see the PID controller in action.

## What you can control

The UI is organised into collapsible panels. Everything below can be changed mid‑simulation unless noted.

### Floating Quick-Control Bar
A semi-transparent overlay provides instant access to the most frequent interactions without needing to navigate the sidebar:

- **Playback:** Instant Start (▶), Pause (⏸), and Reset (↺).
- **Layer Toggles:** Hot-swap visibility for Velocity, Divergence, Vorticity, Pressure, and Dye layers.
- **Diagnostics Toggles:** One-click activation of Error Metrics, Airfoil Metrics, and Profiling Overlays.
- **Live Colormaps:** Change colormaps for V, ω, and P on the fly to highlight different physical features.
- **Precision Dye Injection:** Dual-slider (X, Y) control for real-time scalar tracer injection.

### Simulation control (top bar)

- Start / Pause / Reset
- Inverse Design – experimental module for adjoint‑based shape optimisation
- Thermal – placeholder for future heat transfer - Boussinesq-approximation buoyancy flow is already conceptually implemented.
- Theme toggle – light / dark mode

### Grid & solver

| Panel | What you can change |
|-------|---------------------|
| Grid Size | Nx, Ny (64‑4096 / 32‑2048) – requires sim restart |
| Grid Type | Collocated or MAC (staggered) – sim restart |
| Solver Type | Navier‑Stokes or Lattice Boltzmann (D2Q9/D2Q7) - enables direct solver-to-solver comparison under identical flow conditions – sim restart required |
| Precision | float32 / float64 – GUI reload |

### Flow parameters

| Panel | What you can change |
|-------|---------------------|
| Reynolds Number | Lock any two of U_inf, ν, Re – the third auto‑updates. Apply live. |
| Flow Type | von Kármán, Lid-Driven Cavity Flow, Taylor‑Green Vortex – sim restart. |

### Solver parameters

| Control | What it does |
|---------|--------------|
| Multigrid V‑cycles | Number of multigrid cycles (1‑10). Apply live. |
| Hyper ν | Hyperviscosity (0‑0.05) – improves stability for under‑resolved turbulence. |
| Fast Mode (RK2) | Switches from RK3 to RK2 – faster but less accurate. |
| LES | Enable, choose Smagorinsky or dynamic Smagorinsky. Apply live. |
| Pressure Solver | Multigrid, CG (iteration-dependent; slower for poorly conditioned systems), FFT (for periodic BCs like LDC), Jacobi – requires sim restart.|

### Boundary conditions

| Control | What it does |
|---------|--------------|
| Slip Walls | Toggle between no‑slip and free‑slip on domain walls. |
| Mask ε | Brinkman penalisation sharpness (0.01‑1.0). Higher = sharper but stiffer. |

### Obstacles (all live)

- Type: Cylinder (single), NACA airfoil (4‑/5‑digit), Cow (arbitrary demonstration geometry), Three‑cylinder array.
- NACA: Choose from selected 4‑digit and 5‑digit series. Set chord length, angle of attack (AoA) with slider/spinbox.
- Cylinder: Radius (live preview as you input).
- Cylinder array: Diameter, spacing between centres.
- Position: X and Y sliders – drag the obstacle across the domain live.
- Draw custom obstacle: Click and draw a shape with your mouse in the PyGame drawing window. The SDF generator injects it immediately.

### Time stepping

- dt (fixed) – apply live.
- Adaptive dt – PID controller based on divergence error with CFL monitoring (work in progress).

### Visualisation

| Control | What it does |
|---------|--------------|
| Frame skip | Render every N‑th solver frame – improves UI responsiveness. |
| Target FPS | Limits visualisation framerate to save CPU. |
| Show ... | Toggle velocity, vorticity, pressure, dye, particle mode (computationally expensive due to particle advection), SDF mask, streamlines, quivers – all live. |
| Log / Spatial / Adaptive | Colour scale modes. Adaptive auto‑adjusts range to current data. |
| Smooth | Upscales low‑res fields for cleaner display 1x (default) to 10x. Note: this does NOT enhance physics - it uses bilinear interpolation to increase visual fidelity. |
| Colormaps | Separate dropdowns for velocity, vorticity, pressure. Many CET and PAL options. |
| Auto‑scale | One‑click rescaling for each field or all at once. |

### Dye injection

- X / Y position (spinbox or sliders)
- Amount (0‑100%)
- Inject dye – adds a scalar tracer that advects with the flow.

### Diagnostics & metrics

| Panel | What you get |
|-------|--------------|
| Simulation Info | Solver status, simulation time, dt, RMS divergence, Sim FPS, Vis FPS. |
| Error Metrics | L2 change, RMS change, max change, 99th percentile change, relative change, component L2 changes. Enable/disable to save performance. Save all history to CSV. |
| Airfoil Metrics | CL, CD, Strouhal number, stagnation point (in chord fractions), separation point, Cp_min, wake deficit. Toggle on/off. Markers overlay on visualisation. Copy all metrics to clipboard. |

### Neural Operator training (experimental)

- Generate dataset: Run simulation for N steps, save chosen fields (u, v, p, mask, divergence) to .npz.
- Select operator: Choose from any .py file in neural_operators/.
- Architecture: Linear, NonLinear, Advanced (FNO‑like).
- Train: Set epochs, learning rate, batch size, cancel anytime. Progress bar.
- Load trained model: Replace the pressure solver with a neural operator within the simulation loop.

### Export & recording

- Export Frame – save current visualisation as PNG.
- Record – toggle video recording (saves to disk).
- Save State – not yet implemented.

## Numerical methods & Solver details

- Navier‑Stokes (incompressible) finite‑difference solver on collocated or MAC grid.
- Lattice Boltzmann (D2Q9 / D2Q7) with BGK or MRT collision (low-Mach regime).
- Advection: RK3, RK2, multiple specialised variants.
- Pressure projection: FFT (periodic), Conjugate Gradient, Multigrid (V‑cycle).
- Turbulence: Smagorinsky & dynamic Smagorinsky LES.
- Immersed boundaries: Brinkman penalisation with smooth Heaviside mask.
- Differentiability: Every step is JAX‑native. Use jax.grad to optimise shapes / controls.  This enables gradietn-based inverse design and future differentiable control workflows.
- Performance: Decoupled solver / render / metrics threads with shared‑memory zero‑copy buffers. CPU‑optimised – runs at 297 FPS (512×96) on a laptop CPU.
- Live parameter updates: Redux‑style state management. The solver checks for changes each timestep and re‑compiles only when necessary.

## Performance benchmarks (CPU)

*Benchmarks include full solver stepping and rendering.*

| Grid resolution | Solver + rendering | With full diagnostics |
|-----------------|-------------------|----------------------|
| 512 × 96 | ~297 FPS | ~170 FPS |
| 1024 × 192 | ~131 FPS | ~91 FPS |
| 2048 × 384 | ~37 FPS | ~31 FPS |

## Limitations

- 2D only – designed for rapid prototyping and neural operator research.
- Force coefficients (CL, CD) are qualitative – Brinkman penalisation prevents accurate pressure integration. Use for optimisation trends, not absolute values.
- Moderate Reynolds numbers – best for Re < 10,000 (laminar to early turbulent).
- CPU‑focused – GPU execution is supported, but the current architecture is tuned for workstations without dedicated GPUs.
- LBM operates in the low-Mach regime; accuracy degrades if this constraint is violated.

## License

LGPL v3.0 – you can use it in proprietary software as long as you release modifications to the library itself.

## Author

Arno Meijer - Mechanical Engineer | CFD–ML Systems Developer