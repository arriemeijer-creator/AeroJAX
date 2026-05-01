"""
Snapshot loading and saving utilities.

Provides functions for managing snapshot files and series.
"""

import os
import pickle
from typing import List, Optional
from pathlib import Path
import numpy as np

from .snapshot import Snapshot, SnapshotSeries


def save_snapshot(snapshot: Snapshot, filepath: str) -> None:
    """Save a single snapshot to file.
    
    Args:
        snapshot: The snapshot to save
        filepath: Path to save the snapshot (will be created if needed)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    snapshot.save(filepath)


def load_snapshot(filepath: str) -> Snapshot:
    """Load a single snapshot from file.
    
    Args:
        filepath: Path to the snapshot file
        
    Returns:
        The loaded snapshot
    """
    return Snapshot.load(filepath)


def save_snapshot_series(series: SnapshotSeries, filepath: str) -> None:
    """Save a snapshot series to file.
    
    Args:
        series: The snapshot series to save
        filepath: Path to save the series (will be created if needed)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    series.save(filepath)


def load_snapshot_series(filepath: str) -> SnapshotSeries:
    """Load a snapshot series from file.
    
    Args:
        filepath: Path to the snapshot series file
        
    Returns:
        The loaded snapshot series
    """
    return SnapshotSeries.load(filepath)


def save_snapshots_to_directory(snapshots: List[Snapshot], directory: str) -> None:
    """Save multiple snapshots to a directory, one file per snapshot.
    
    Args:
        snapshots: List of snapshots to save
        directory: Directory to save snapshots in (will be created if needed)
    """
    os.makedirs(directory, exist_ok=True)
    for i, snapshot in enumerate(snapshots):
        filepath = os.path.join(directory, f"snapshot_{i:06d}.pkl")
        save_snapshot(snapshot, filepath)


def load_snapshots_from_directory(directory: str) -> List[Snapshot]:
    """Load all snapshots from a directory.
    
    Args:
        directory: Directory containing snapshot files
        
    Returns:
        List of loaded snapshots, sorted by filename
    """
    snapshots = []
    snapshot_files = sorted(Path(directory).glob("snapshot_*.pkl"))
    
    for filepath in snapshot_files:
        snapshot = load_snapshot(str(filepath))
        snapshots.append(snapshot)
        
    return snapshots


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
    solver_metadata: Optional[dict] = None,
    iteration: int = 0,
    timestamp: Optional[float] = None
) -> Snapshot:
    """Create a snapshot from simulation state arrays.
    
    This is a convenience function for the solver to create snapshots
    without needing to directly instantiate the Snapshot dataclass.
    
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


def merge_snapshot_series(series_list: List[SnapshotSeries]) -> SnapshotSeries:
    """Merge multiple snapshot series into one.
    
    Args:
        series_list: List of snapshot series to merge
        
    Returns:
        A single snapshot series containing all snapshots
    """
    merged = SnapshotSeries()
    for series in series_list:
        for snapshot in series.snapshots:
            merged.add_snapshot(snapshot)
    return merged


def validate_snapshot_directory(directory: str) -> bool:
    """Check if a directory contains valid snapshot files.
    
    Args:
        directory: Directory to check
        
    Returns:
        True if directory contains valid snapshot files, False otherwise
    """
    if not os.path.isdir(directory):
        return False
    
    snapshot_files = list(Path(directory).glob("snapshot_*.pkl"))
    return len(snapshot_files) > 0
