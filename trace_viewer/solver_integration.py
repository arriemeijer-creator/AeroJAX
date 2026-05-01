"""
Integration helper for saving snapshots from the AeroJAX solver.

This module provides utilities to hook into the solver and save snapshots
for later analysis in the trace viewer. This maintains decoupling - the solver
only saves data, and the trace viewer only reads data.
"""

import os
import numpy as np
from typing import Optional, Dict, Any
import jax.numpy as jnp

from .snapshot import Snapshot
from .snapshot_utils import save_snapshot, save_snapshots_to_directory


class SnapshotRecorder:
    """Records simulation snapshots during solver execution."""
    
    def __init__(self, output_dir: str = "snapshots", save_interval: int = 1):
        """Initialize the snapshot recorder.
        
        Args:
            output_dir: Directory to save snapshots
            save_interval: Save every N timesteps (default: 1 for every step)
        """
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.snapshots = []
        self.step_count = 0
        self.enabled = True
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def record_step(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        mask: np.ndarray,
        dt: float,
        nu: float,
        dx: float,
        dy: float,
        divergence: Optional[np.ndarray] = None,
        Re: Optional[float] = None,
        CFL: Optional[float] = None,
        flow_type: Optional[str] = None,
        solver_metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> Optional[Snapshot]:
        """Record a single timestep as a snapshot.
        
        Args:
            u: x-velocity field (can be JAX array or numpy array)
            v: y-velocity field (can be JAX array or numpy array)
            p: pressure field (can be JAX array or numpy array)
            mask: obstacle mask (can be JAX array or numpy array)
            dt: timestep size
            nu: kinematic viscosity
            dx: grid spacing in x
            dy: grid spacing in y
            divergence: optional divergence field
            Re: optional Reynolds number
            CFL: optional CFL number
            flow_type: optional flow configuration type
            solver_metadata: optional solver-specific metadata
            timestamp: optional simulation time
            
        Returns:
            The created snapshot if saved, None otherwise
        """
        if not self.enabled:
            return None
            
        # Only save at specified interval
        if self.step_count % self.save_interval != 0:
            self.step_count += 1
            return None
            
        # Convert JAX arrays to numpy if needed
        u_np = np.array(u) if isinstance(u, jnp.ndarray) else u
        v_np = np.array(v) if isinstance(v, jnp.ndarray) else v
        p_np = np.array(p) if isinstance(p, jnp.ndarray) else p
        mask_np = np.array(mask) if isinstance(mask, jnp.ndarray) else mask
        div_np = np.array(divergence) if divergence is not None and isinstance(divergence, jnp.ndarray) else divergence
        
        # Create snapshot
        snapshot = Snapshot(
            u=u_np,
            v=v_np,
            p=p_np,
            mask=mask_np,
            dt=dt,
            nu=nu,
            dx=dx,
            dy=dy,
            divergence=div_np,
            Re=Re,
            CFL=CFL,
            flow_type=flow_type,
            solver_metadata=solver_metadata,
            iteration=self.step_count,
            timestamp=timestamp
        )
        
        self.snapshots.append(snapshot)
        
        # Save immediately to disk
        filepath = os.path.join(self.output_dir, f"snapshot_{self.step_count:06d}.pkl")
        save_snapshot(snapshot, filepath)
        
        self.step_count += 1
        return snapshot
        
    def record_from_sim_state(
        self,
        sim_state,
        mask: np.ndarray,
        dt: float,
        nu: float,
        dx: float,
        dy: float,
        divergence: Optional[np.ndarray] = None,
        Re: Optional[float] = None,
        CFL: Optional[float] = None,
        flow_type: Optional[str] = None,
        solver_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Snapshot]:
        """Record a snapshot from a SimState object.
        
        This is a convenience method for use with the AeroJAX solver's SimState.
        
        Args:
            sim_state: SimState object containing u, v, p fields
            mask: obstacle mask
            dt: timestep size
            nu: kinematic viscosity
            dx: grid spacing in x
            dy: grid spacing in y
            divergence: optional divergence field
            Re: optional Reynolds number
            CFL: optional CFL number
            flow_type: optional flow configuration type
            solver_metadata: optional solver-specific metadata
            
        Returns:
            The created snapshot if saved, None otherwise
        """
        timestamp = sim_state.iteration * dt if hasattr(sim_state, 'iteration') else None
        return self.record_step(
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
            CFL=CFL,
            flow_type=flow_type,
            solver_metadata=solver_metadata,
            timestamp=timestamp
        )
        
    def finalize(self) -> None:
        """Finalize recording (save any remaining snapshots)."""
        # Snapshots are saved immediately in record_step, so this is mainly for cleanup
        pass
        
    def get_snapshot_count(self) -> int:
        """Return the number of snapshots recorded."""
        return len(self.snapshots)
        
    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self.snapshots = []
        self.step_count = 0


def create_snapshot_from_sim_state(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    mask: np.ndarray,
    dt: float,
    nu: float,
    dx: float,
    dy: float,
    divergence: Optional[np.ndarray] = None,
    Re: Optional[float] = None,
    CFL: Optional[float] = None,
    flow_type: Optional[str] = None,
    solver_metadata: Optional[Dict[str, Any]] = None,
    iteration: int = 0,
    timestamp: Optional[float] = None
) -> Snapshot:
    """Create a snapshot from simulation state arrays.
    
    This is a convenience function for the solver to create snapshots.
    Re-exported from snapshot_utils for convenience.
    
    Args:
        u: x-velocity field
        v: y-velocity field
        p: pressure field
        mask: obstacle mask
        dt: timestep size
        nu: kinematic viscosity
        dx: grid spacing in x
        dy: grid spacing in y
        divergence: optional divergence field
        Re: optional Reynolds number
        CFL: optional CFL number
        flow_type: optional flow configuration type
        solver_metadata: optional solver-specific metadata
        iteration: timestep index
        timestamp: simulation time
        
    Returns:
        A Snapshot object
    """
    return Snapshot(
        u=u,
        v=v,
        p=p,
        mask=mask,
        dt=dt,
        nu=nu,
        dx=dx,
        dy=dy,
        divergence=divergence,
        Re=Re,
        CFL=CFL,
        flow_type=flow_type,
        solver_metadata=solver_metadata,
        iteration=iteration,
        timestamp=timestamp
    )


# Example usage with AeroJAX solver
"""
# In your simulation loop:

from trace_viewer.solver_integration import SnapshotRecorder

# Initialize recorder
recorder = SnapshotRecorder(
    output_dir="simulation_snapshots",
    save_interval=5  # Save every 5 timesteps
)

# In your timestep loop:
for step in range(num_steps):
    # ... run solver step ...
    sim_state = solver.step(sim_state, mask, params)
    
    # Record snapshot
    recorder.record_from_sim_state(
        sim_state=sim_state,
        mask=mask,
        dt=dt,
        nu=nu,
        dx=dx,
        dy=dy,
        divergence=divergence_field,  # if computed
        Re=Reynolds_number,
        CFL=cfl_number,
        flow_type="von_karman",
        solver_metadata={"pressure_solver": "multigrid"}
    )

print(f"Recorded {recorder.get_snapshot_count()} snapshots")
"""
