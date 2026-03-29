# Differential CFD-ML: Project Status & Roadmap

**Last updated: March 2026**

---

## Where Things Actually Stand

Let me be upfront about what's working, what's not, and what's just ideas right now.

---

## ✅ Done & Working

The baseline solver is complete and validated. This means:

### Flow Types (5)
- Von Kármán vortex shedding (cylinder in channel)
- Lid-driven cavity
- Channel flow (Poiseuille)
- Backward-facing step
- Taylor-Green vortex

Switch between them at runtime in the GUI. Each has appropriate boundary conditions, domain size, and initialization.

### Advection Schemes (9)
- **Upwind** (1st order, stable but diffusive)
- **MacCormack** (predictor-corrector)
- **Jos Stam** (semi-Lagrangian, unconditionally stable)
- **QUICK** (3rd order upwind-biased)
- **WENO5** (5th order, shock-capturing)
- **TVD with limiters** (minmod, superbee, van Leer)
- **RK3** (3rd order Runge-Kutta)
- **Spectral** (FFT-based, periodic domains)
- **Plus utility functions** for CFL and dealiasing

All JIT-compiled, all differentiable.

### Pressure Solvers (7)
- **Jacobi** (simple, slow, but works)
- **FFT** (fast, periodic only)
- **ADI** (alternating direction implicit)
- **SOR with red-black ordering**
- **Gauss-Seidel red-black**
- **Conjugate gradient**
- **Multigrid** (geometric, falls back gracefully when grid isn't divisible)

Again, all differentiable. The multigrid solver was a pain to get right with JAX's while_loop restrictions but it works now.

### Visualization (PyQtGraph)
- **6 simultaneous plots**: velocity, vorticity, streamlines, pressure, drag/lift, KE/enstrophy
- **Real-time switching** of everything (flow type, grid resolution, schemes, solvers)
- **30+ colormaps** from CET collection
- **Video recording** to GIF
- **Data export** to CSV

The streamlines are cached (100x speedup). Plot updates alternate between frames (50% reduction in work). It runs at 30 FPS even on 1024×192 grids.

### Testing
The test framework (`test_framework.py`) runs through combinations of:
- 5 flow types × 8 advection schemes × 7 pressure solvers × 2 dt modes × 3 Reynolds numbers

That's **1,680 configurations**. It catches numerical blow-ups, tracks CFL, and logs performance. Quick mode runs approx. 20 configs in 10 minutes. Full mode takes a day or two.

### Data Export
When you hit "Export Data" in the GUI, you get:
- `velocity_u_*.csv`, `velocity_v_*.csv`
- `vorticity_*.csv`, `pressure_*.csv`
- `history_*.csv` (time, KE, enstrophy, drag, lift)
- `grid_info_*.json` with all parameters

This is what generates training data for neural parts.  I am still working on it to generate proper training dataset format files.

---

## In Progress

### Neural Pressure Solver
The idea: replace iterative Poisson solvers with a learned mapping from (u_star, v_star) to pressure.

Current state:
- Data generation pipeline works (exporting fields from baseline solver)
- Architecture designed (CNN with skip connections, similar to U-Net)
- Training script exists but needs tuning

What's blocking: Getting the loss function right. The pressure field is only defined up to a constant, and the network keeps drifting. Need to add a mean-zero constraint.

### Fine-Scale Neural Correction
This is Part I from the design doc. The plan:
1. Extract features (divergence, vorticity, velocity magnitude, mask)
2. Run through a small CNN
3. Add correction to u_star before pressure solve

Not started yet. Waiting on the pressure solver to work first.

---

## 📋 Planned (But Not Started)

### Coarse-Scale / Temporal Networks
Long-range corrections using transformers or convLSTM. The design doc talks about this. Realistically, this is months away.

### Latent Space Modeling
Autoencoder + latent dynamics. I have the architecture sketched but no implementation. The challenge is keeping divergence-free constraints in latent space.

### Differentiable Inverse Design
SDF-based geometry optimization. The math is in the doc. The code is not.

### Reinforcement Learning Integration
This is the furthest out. The doc mentions it, but I haven't touched RL yet. Probably 2027.

---

## What The 70-Page Document Is

That document (`docs/framework.pdf`) is my design specification. It describes the complete vision — what the framework *could* become. It's not a lie or overclaiming; it's a roadmap.

Think of it as:
- **Part I (baseline)** → Mostly implemented
- **Part II (neural enhancement)** → Partially implemented (pressure solver only)
- **Part III (control & RL)** → Design only

The README on GitHub reflects reality. The PDF shows the destination.

---

## Known Issues & Annoyances

### JAX While Loops
The multigrid and CG solvers use `jax.lax.while_loop`, which is fine until you try to jit them inside another loop. Sometimes it explodes. The fallbacks work but they're slower.

### Pressure Solver Stability
FFT solver is fast but requires periodic boundaries. The cylinder flow isn't periodic. For von Kármán, I use Jacobi or CG, which are slow. The neural pressure solver should fix this.

### Adaptive Timestepping
The CFL-based dt adaptation works but oscillates sometimes. I added damping (max change of 2x per step) and it helped. Still not perfect for high Re flows.

### Streamline Computation
The vectorized streamline integration is fast but not perfectly accurate. For publication-quality plots, you'd want proper line integration. For real-time visualization, it's fine.

### Memory Usage
Exporting full fields at high resolution (2048×384) produces ~200MB CSV files. Fine for research, annoying for quick tests. I should add an option to downsample on export.

---

## Next Milestones (Next 3 Months)

1. **Get the neural pressure solver working**  
   Train on cylinder flow data (Re=100, 150, 200). Target: 10x faster than Jacobi, within 5% accuracy.

2. **Integrate it into the solver**  
   Add a "neural" option to the pressure solver dropdown. Compare side-by-side with classical solvers.

3. **Generate the training dataset**  
   Run parameter sweeps (Re, cylinder position, grid resolution) and export everything to HDF5. Make it available for anyone who wants to train their own models.

4. **Document the neural integration API**  
   So others can plug in their own models without touching the core solver.

---

## How To Use This Right Now

### For CFD
Run `python baseline_viewer.py`, pick a flow type, watch it go. Change schemes mid-simulation. Export data.

### For Training Data
Set up a batch script:
```python
for Re in [50, 100, 150, 200, 250]:
    solver = BaselineSolver(grid, FlowParams(Re=Re, U_inf=1.0), ...)
    solver.run_simulation(n_steps=20000)
    # export automatically or call export_data()
For Neural Operator Research
Use the exported CSV files as ground truth. The grid is uniform Cartesian, so you can treat it as image data. Each timestep gives you (u, v, p, ω, mask).

Contributing (If Anyone Wants To)
I'm not actively seeking contributors, but if you're interested in:

Adding a flow type (airfoil? Rayleigh-Bénard?)

Implementing a neural pressure solver in JAX

Porting the solvers to GPU (they already run on GPU via JAX, but more testing needed)

Open an issue or send a PR. The code is LGPL v3, so use it freely.

Questions I Ask Myself
Should I have started with a simpler ML problem? Probably. But where's the fun in that?

Is this overkill for a personal project? Absolutely. That's the point.

Will I ever finish Part III? Maybe. The RL stuff is intimidating. But the baseline works, and that's already useful.

Bottom Line
The baseline solver is production-quality. Use it. The neural parts are coming. The design doc is the vision.

If you just want to simulate flow past a cylinder, clone the repo and run the viewer. If you want to build a differentiable neural CFD framework, the foundation is here.