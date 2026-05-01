"""
AeroJAX Solver Trace Viewer - Decoupled Architecture

This module provides a decoupled trace viewer for inspecting solver snapshots.
It reads saved simulation snapshots and reconstructs step-by-step numerical explanations.

Core Design: Strict separation from live solver runtime.
"""

from .snapshot import Snapshot, SnapshotSeries
from .trace_builder import TraceBuilder, SolverTrace, TraceStep
from .viewer_window import TraceViewerWindow
from .snapshot_utils import (
    save_snapshot, load_snapshot, 
    save_snapshot_series, load_snapshot_series,
    save_snapshots_to_directory, load_snapshots_from_directory,
    create_snapshot_from_sim_state
)
from .solver_integration import SnapshotRecorder

__all__ = [
    'Snapshot', 'SnapshotSeries',
    'TraceBuilder', 'SolverTrace', 'TraceStep',
    'TraceViewerWindow',
    'save_snapshot', 'load_snapshot',
    'save_snapshot_series', 'load_snapshot_series',
    'save_snapshots_to_directory', 'load_snapshots_from_directory',
    'create_snapshot_from_sim_state',
    'SnapshotRecorder'
]
