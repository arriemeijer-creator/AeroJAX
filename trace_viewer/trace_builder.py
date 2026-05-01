"""
Trace Builder for reconstructing intermediate solver fields.

This module implements logic to reconstruct intermediate fields from snapshots
and compute derived metrics for step-by-step solver trace visualization.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from .snapshot import Snapshot


@dataclass
class TraceStep:
    """Represents a single step in the solver pipeline."""
    step_number: int
    step_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    method: str
    equation_latex: str
    description: str


@dataclass
class SolverTrace:
    """Complete trace of solver operations for a single timestep."""
    snapshot: Snapshot
    steps: list[TraceStep]
    reconstructed_fields: Dict[str, np.ndarray]
    metrics: Dict[str, float]


class TraceBuilder:
    """Reconstructs intermediate fields and builds solver trace from snapshots."""
    
    def __init__(self, snapshot: Snapshot):
        """Initialize trace builder with a snapshot."""
        self.snapshot = snapshot
        self.nx, self.ny = snapshot.shape
        self.dx = snapshot.dx
        self.dy = snapshot.dy
        self.dt = snapshot.dt
        self.nu = snapshot.nu
        self.mask = snapshot.mask
        
    def compute_divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute divergence field using finite differences.
        
        D = du/dx + dv/dy
        """
        # Central difference for interior points
        div = np.zeros_like(u)
        
        # du/dx using central difference
        div[1:-1, :] += (u[2:, :] - u[:-2, :]) / (2 * self.dx)
        
        # dv/dy using central difference
        div[:, 1:-1] += (v[:, 2:] - v[:, :-2]) / (2 * self.dy)
        
        # Apply mask
        div = div * (1 - self.mask)
        
        return div
    
    def compute_pressure_gradient(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pressure gradient.
        
        Returns (dp/dx, dp/dy)
        """
        grad_px = np.zeros_like(p)
        grad_py = np.zeros_like(p)
        
        # Central difference for interior points
        grad_px[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * self.dx)
        grad_py[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * self.dy)
        
        return grad_px, grad_py
    
    def reconstruct_advection_diffusion(self) -> Dict[str, np.ndarray]:
        """Reconstruct u_star, v_star after advection-diffusion step.
        
        Since we only have the final state, we approximate this step
        by showing the input state and method description.
        """
        # In a real implementation, if intermediate states were saved,
        # we would load them. Here we provide the input state.
        return {
            'u': self.snapshot.u,
            'v': self.snapshot.v,
            'u_star': self.snapshot.u,  # Placeholder - would be actual intermediate
            'v_star': self.snapshot.v   # Placeholder - would be actual intermediate
        }
    
    def compute_cfl(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute CFL number."""
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        cfl = self.dt * (u_max / self.dx + v_max / self.dy)
        return float(cfl)
    
    def compute_reynolds(self, u_char: float = 1.0, L_char: float = 1.0) -> float:
        """Compute Reynolds number."""
        return u_char * L_char / self.nu
    
    def build_trace(self) -> SolverTrace:
        """Build complete solver trace for the snapshot."""
        steps = []
        reconstructed = {}
        
        # Step 1: Advection-Diffusion
        adv_diff_fields = self.reconstruct_advection_diffusion()
        reconstructed.update(adv_diff_fields)
        
        steps.append(TraceStep(
            step_number=1,
            step_name="Advection-Diffusion",
            inputs={
                'u': 'velocity field',
                'v': 'velocity field',
                'dt': f'{self.dt:.4f}',
                'nu': f'{self.nu:.4f}',
                'dx': f'{self.dx:.4f}',
                'dy': f'{self.dy:.4f}'
            },
            outputs={
                'u_star': 'intermediate velocity',
                'v_star': 'intermediate velocity'
            },
            method="RK3 + Convection + Diffusion",
            equation_latex=r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = \nu \nabla^2 \mathbf{u}",
            description="Advects velocity field and applies viscous diffusion using Runge-Kutta 3 scheme."
        ))
        
        # Step 2: Divergence
        u_star = adv_diff_fields['u_star']
        v_star = adv_diff_fields['v_star']
        divergence = self.compute_divergence(u_star, v_star)
        reconstructed['divergence'] = divergence
        
        steps.append(TraceStep(
            step_number=2,
            step_name="Divergence",
            inputs={
                'u_star': 'intermediate velocity',
                'v_star': 'intermediate velocity'
            },
            outputs={
                'D': 'divergence field'
            },
            method="Finite Difference Divergence Operator",
            equation_latex=r"D = \nabla \cdot \mathbf{u}^* = \frac{\partial u^*}{\partial x} + \frac{\partial v^*}{\partial y}",
            description="Computes divergence of intermediate velocity field to enforce incompressibility."
        ))
        
        # Step 3: Pressure Solve
        steps.append(TraceStep(
            step_number=3,
            step_name="Pressure Solve",
            inputs={
                'D': 'divergence field',
                'dt': f'{self.dt:.4f}',
                'BC': 'boundary conditions'
            },
            outputs={
                'p': 'pressure field'
            },
            method="Conjugate Gradient / Multigrid / FFT",
            equation_latex=r"\nabla^2 p = \frac{D}{\Delta t}",
            description="Solves Poisson equation for pressure to enforce divergence-free velocity field."
        ))
        
        # Step 4: Velocity Correction
        grad_px, grad_py = self.compute_pressure_gradient(self.snapshot.p)
        reconstructed['grad_px'] = grad_px
        reconstructed['grad_py'] = grad_py
        
        steps.append(TraceStep(
            step_number=4,
            step_name="Velocity Correction",
            inputs={
                'u_star': 'intermediate velocity',
                'v_star': 'intermediate velocity',
                'p': 'pressure field',
                'dt': f'{self.dt:.4f}'
            },
            outputs={
                'u_new': 'corrected velocity',
                'v_new': 'corrected velocity'
            },
            method="Pressure Gradient Subtraction",
            equation_latex=r"\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t \nabla p",
            description="Corrects velocity field by subtracting pressure gradient to ensure incompressibility."
        ))
        
        # Step 5: State Update
        steps.append(TraceStep(
            step_number=5,
            step_name="State Update",
            inputs={
                'u_new': 'corrected velocity',
                'v_new': 'corrected velocity',
                'mask': 'obstacle mask'
            },
            outputs={
                'next_state': 'timestep n+1'
            },
            method="Overwrite / Advance Snapshot",
            equation_latex=r"\mathbf{u}^{n+1} \leftarrow \text{apply BCs}(\mathbf{u}^{n+1})",
            description="Applies boundary conditions and advances to next timestep."
        ))
        
        # Compute metrics
        cfl = self.compute_cfl(self.snapshot.u, self.snapshot.v)
        metrics = {
            'CFL': cfl,
            'max_u': float(np.max(np.abs(self.snapshot.u))),
            'max_v': float(np.max(np.abs(self.snapshot.v))),
            'max_p': float(np.max(self.snapshot.p)),
            'min_p': float(np.min(self.snapshot.p)),
            'max_div': float(np.max(np.abs(divergence)))
        }
        
        if self.snapshot.Re is not None:
            metrics['Re'] = self.snapshot.Re
        
        return SolverTrace(
            snapshot=self.snapshot,
            steps=steps,
            reconstructed_fields=reconstructed,
            metrics=metrics
        )
    
    def extract_subdomain(self, x_min: int, x_max: int, y_min: int, y_max: int) -> Dict[str, np.ndarray]:
        """Extract subdomain patch for numerical viewer.
        
        Args:
            x_min, x_max: x-index bounds (inclusive)
            y_min, y_max: y-index bounds (inclusive)
            
        Returns:
            Dictionary with u, v, p, divergence arrays for the subdomain
        """
        # Validate bounds
        x_min = max(0, x_min)
        x_max = min(self.nx - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(self.ny - 1, y_max)
        
        trace = self.build_trace()
        divergence = trace.reconstructed_fields.get('divergence')
        
        return {
            'u': self.snapshot.u[x_min:x_max+1, y_min:y_max+1],
            'v': self.snapshot.v[x_min:x_max+1, y_min:y_max+1],
            'p': self.snapshot.p[x_min:x_max+1, y_min:y_max+1],
            'divergence': divergence[x_min:x_max+1, y_min:y_max+1] if divergence is not None else None,
            'mask': self.snapshot.mask[x_min:x_max+1, y_min:y_max+1]
        }
