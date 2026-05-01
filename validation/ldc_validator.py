"""
Lid-Driven Cavity (LDC) benchmark validation module.

This module provides validation against Ghia et al. (1982) benchmark data for
Lid-Driven Cavity flow across multiple Reynolds numbers. It is fully decoupled
from the solver and only consumes snapshot data.

Reference:
Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible
flow using the Navier-Stokes equations and a multigrid method. Journal of
Computational Physics, 48(3), 387-411.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class VortexCenter:
    """Vortex center location and metadata."""
    x: float  # x-coordinate (normalized [0, 1])
    y: float  # y-coordinate (normalized [0, 1])
    method: str  # Method used to compute center
    timestamp: Optional[float] = None  # Simulation time if available


@dataclass
class ValidationError:
    """Error metrics for vortex center comparison."""
    dx: float  # x-error (sim - ref)
    dy: float  # y-error (sim - ref)
    l2_distance: float  # L2 distance error
    reference_center: Tuple[float, float]  # (x_ref, y_ref)
    simulation_center: Tuple[float, float]  # (x_sim, y_sim)


@dataclass
class ValidationHistory:
    """History of validation results over time."""
    vortex_centers: List[VortexCenter] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


class LDCValidator:
    """
    Lid-Driven Cavity benchmark validator.
    
    Compares solver output against Ghia et al. benchmark data for primary
    vortex center locations across multiple Reynolds numbers.
    
    Usage:
        validator = LDCValidator.load(Re=1000)
        validator.compute(snapshot)
        center = validator.get_vortex_center()
        error = validator.get_error()
    """
    
    # Benchmark data: Primary vortex center locations from Ghia et al. (1982)
    # Values are normalized coordinates (x, y) in [0, 1] domain
    BENCHMARK_DATA = {
        100: (0.50, 0.54),
        400: (0.52, 0.56),
        1000: (0.53, 0.57),
        3200: (0.53, 0.58),
        5000: (0.53, 0.585),
        7500: (0.54, 0.59),
    }
    
    # Search domain for vortex center (normalized coordinates)
    SEARCH_DOMAIN = {
        'x_min': 0.2,
        'x_max': 0.8,
        'y_min': 0.2,
        'y_max': 0.8,
    }
    
    def __init__(self, Re: float):
        """
        Initialize validator for a specific Reynolds number.
        
        Args:
            Re: Reynolds number for validation
            
        Raises:
            ValueError: If Re is not in supported benchmark dataset
        """
        if Re not in self.BENCHMARK_DATA:
            supported = list(self.BENCHMARK_DATA.keys())
            raise ValueError(
                f"Re={Re} not supported. Supported Reynolds numbers: {supported}"
            )
        
        self.Re = Re
        self.reference_center = self.BENCHMARK_DATA[Re]
        
        # State
        self.current_vortex_center: Optional[VortexCenter] = None
        self.current_error: Optional[ValidationError] = None
        self.history = ValidationHistory()
        
        # Grid parameters (extracted from snapshot)
        self.dx: Optional[float] = None
        self.dy: Optional[float] = None
        self.nx: Optional[int] = None
        self.ny: Optional[int] = None
        self.lx: Optional[float] = None
        self.ly: Optional[float] = None
    
    @classmethod
    def load(cls, Re: float) -> 'LDCValidator':
        """
        Load validator for a specific Reynolds number.
        
        Args:
            Re: Reynolds number for validation
            
        Returns:
            Configured LDCValidator instance
        """
        return cls(Re)
    
    def compute(self, snapshot) -> None:
        """
        Compute vortex center from snapshot and validate against benchmark.
        
        Args:
            snapshot: Snapshot object containing u, v, p, mask, dx, dy, etc.
                     Expected to have attributes: u, v, dx, dy, iteration, timestamp
        """
        # Extract grid parameters
        self._extract_grid_params(snapshot)
        
        # Compute vortex center
        self.current_vortex_center = self._compute_vortex_center(snapshot)
        
        # Compute error against benchmark
        self.current_error = self._compute_error(self.current_vortex_center)
        
        # Update history
        self.history.vortex_centers.append(self.current_vortex_center)
        self.history.errors.append(self.current_error)
        self.history.iterations.append(getattr(snapshot, 'iteration', 0))
        self.history.timestamps.append(getattr(snapshot, 'timestamp', 0.0))
    
    def _extract_grid_params(self, snapshot) -> None:
        """Extract grid parameters from snapshot."""
        self.dx = snapshot.dx
        self.dy = snapshot.dy
        self.nx = snapshot.nx
        self.ny = snapshot.ny
        
        # Infer domain dimensions from grid spacing and resolution
        self.lx = self.dx * (self.nx - 1)
        self.ly = self.dy * (self.ny - 1)
    
    def _compute_vortex_center(self, snapshot) -> VortexCenter:
        """
        Compute primary vortex center using streamfunction method.
        
        For LDC, the primary vortex center is the point with maximum
        absolute streamfunction (local extremum) within the core region.
        This corresponds to recirculation cell maximum.
        """
        u = np.asarray(snapshot.u)
        v = np.asarray(snapshot.v)
        
        # Method 1: Streamfunction (most robust for LDC)
        psi = self._compute_streamfunction(u, v)
        
        # Search region (avoid boundaries and corner vortices)
        i_min = int(self.SEARCH_DOMAIN['x_min'] * (self.nx - 1))
        i_max = int(self.SEARCH_DOMAIN['x_max'] * (self.nx - 1))
        j_min = int(self.SEARCH_DOMAIN['y_min'] * (self.ny - 1))
        j_max = int(self.SEARCH_DOMAIN['y_max'] * (self.ny - 1))
        
        # Find maximum absolute streamfunction (primary vortex core)
        psi_region = psi[i_min:i_max, j_min:j_max]
        abs_psi = np.abs(psi_region)
        max_idx = np.unravel_index(np.argmax(abs_psi), abs_psi.shape)
        
        # Convert to global indices
        i_center = i_min + max_idx[0]
        j_center = j_min + max_idx[1]
        
        # Refine with sub-grid interpolation (optional, improves accuracy)
        x_norm, y_norm = self._refine_center_subgrid(psi, i_center, j_center)
        
        return VortexCenter(
            x=x_norm,
            y=y_norm,
            method='streamfunction_max',
            timestamp=getattr(snapshot, 'timestamp', None)
        )
    
    def _compute_streamfunction(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute streamfunction psi where u = dpsi/dy, v = -dpsi/dx.
        
        Solves: psi(x,y) = psi(0,0) + ∫(0 to y) u(x, y') dy' (starting from bottom wall)
        """
        psi = np.zeros((self.nx, self.ny))
        
        # Integrate from bottom wall (y=0) upward
        # u velocity component gives vertical variation of psi
        for i in range(self.nx):
            psi[i, 0] = 0.0  # psi = constant along bottom wall (no-penetration)
            for j in range(1, self.ny):
                # Trapezoidal integration: psi = ψ + ∫ u dy
                # Average u between current and previous grid line
                psi[i, j] = psi[i, j-1] + 0.5 * self.dy * (u[i, j] + u[i, j-1])
        
        # Alternative: integrate from left wall using v (cross-check)
        # For LDC, bottom integration is sufficient
        
        # Subtract mean to make psi zero at boundaries (optional)
        # psi -= np.mean(psi)
        
        return psi
    
    def _refine_center_subgrid(self, psi: np.ndarray, i0: int, j0: int, 
                              search_radius: int = 2) -> Tuple[float, float]:
        """
        Refine vortex center using quadratic interpolation around peak.
        
        Finds sub-grid accurate maximum of |psi| within local neighborhood.
        """
        # Extract local neighborhood around initial guess
        i_min = max(1, i0 - search_radius)
        i_max = min(self.nx - 2, i0 + search_radius)
        j_min = max(1, j0 - search_radius)
        j_max = min(self.ny - 2, j0 + search_radius)
        
        x_local = np.arange(i_min, i_max + 1) * self.dx
        y_local = np.arange(j_min, j_max + 1) * self.dy
        psi_local = np.abs(psi[i_min:i_max+1, j_min:j_max+1])
        
        # Find index of maximum in local region
        max_idx = np.unravel_index(np.argmax(psi_local), psi_local.shape)
        
        # If no refinement needed (peak at interior), return default
        if max_idx[0] == 0 or max_idx[0] == psi_local.shape[0]-1 or \
           max_idx[1] == 0 or max_idx[1] == psi_local.shape[1]-1:
            return x_local[max_idx[0]] / self.lx, y_local[max_idx[1]] / self.ly
        
        # Quadratic interpolation in x-direction (using neighboring points)
        # For 1D: f(x) = a x² + b x + c, fit through three points
        i_peak = i_min + max_idx[0]
        j_peak = j_min + max_idx[1]
        
        # Interpolate in x
        x_vals = np.array([i_peak-1, i_peak, i_peak+1]) * self.dx
        psi_vals = np.abs(psi[i_peak-1:i_peak+2, j_peak])
        
        # Solve for peak position using quadratic fit
        # a = (psi_vals[2] - 2*psi_vals[1] + psi_vals[0]) / (2*dx²)
        # x_peak = x_center - b/(2a)
        if psi_vals[2] - 2*psi_vals[1] + psi_vals[0] != 0:
            a = (psi_vals[2] - 2*psi_vals[1] + psi_vals[0]) / (2 * self.dx**2)
            b = (psi_vals[2] - psi_vals[0]) / (2 * self.dx)
            x_peak_subgrid = i_peak * self.dx - b / (2 * a)
        else:
            x_peak_subgrid = i_peak * self.dx
        
        # Interpolate in y similarly
        psi_vals_y = np.abs(psi[i_peak, j_peak-1:j_peak+2])
        if psi_vals_y[2] - 2*psi_vals_y[1] + psi_vals_y[0] != 0:
            a_y = (psi_vals_y[2] - 2*psi_vals_y[1] + psi_vals_y[0]) / (2 * self.dy**2)
            b_y = (psi_vals_y[2] - psi_vals_y[0]) / (2 * self.dy)
            y_peak_subgrid = j_peak * self.dy - b_y / (2 * a_y)
        else:
            y_peak_subgrid = j_peak * self.dy
        
        # Normalize to [0,1] domain
        x_norm = x_peak_subgrid / self.lx
        y_norm = y_peak_subgrid / self.ly
        
        return x_norm, y_norm
    
    def _compute_error(self, vortex_center: VortexCenter) -> ValidationError:
        """
        Compute error metrics against benchmark data.
        
        Args:
            vortex_center: Computed vortex center
            
        Returns:
            ValidationError with error metrics
        """
        x_ref, y_ref = self.reference_center
        x_sim, y_sim = vortex_center.x, vortex_center.y
        
        dx = x_sim - x_ref
        dy = y_sim - y_ref
        l2_distance = np.sqrt(dx**2 + dy**2)
        
        return ValidationError(
            dx=dx,
            dy=dy,
            l2_distance=l2_distance,
            reference_center=(x_ref, y_ref),
            simulation_center=(x_sim, y_sim)
        )
    
    def get_vortex_center(self) -> Optional[VortexCenter]:
        """
        Get the most recently computed vortex center.
        
        Returns:
            VortexCenter object or None if not computed yet
        """
        return self.current_vortex_center
    
    def get_error(self) -> Optional[ValidationError]:
        """
        Get the most recently computed validation error.
        
        Returns:
            ValidationError object or None if not computed yet
        """
        return self.current_error
    
    def get_history(self) -> ValidationHistory:
        """
        Get complete validation history.
        
        Returns:
            ValidationHistory object with all computed results
        """
        return self.history
    
    def get_reference_center(self) -> Tuple[float, float]:
        """
        Get benchmark vortex center for current Reynolds number.
        
        Returns:
            Tuple (x_ref, y_ref) of normalized coordinates
        """
        return self.reference_center
    
    def clear_history(self) -> None:
        """Clear validation history."""
        self.history = ValidationHistory()
    
    # ========================================================================
    # Extension hooks (placeholders for future implementation)
    # ========================================================================
    
    def detect_secondary_vortices(self, snapshot) -> List[VortexCenter]:
        """
        Detect secondary vortices (corner vortices) at high Reynolds numbers.
        
        Placeholder for future implementation.
        
        Args:
            snapshot: Snapshot object
            
        Returns:
            List of secondary vortex centers
        """
        # TODO: Implement secondary vortex detection
        # This would search for vorticity extrema in corner regions
        # [0, 0.2] x [0, 0.2], [0.8, 1] x [0, 0.2], etc.
        return []
    
    def compute_centerline_velocity(self, snapshot) -> Dict[str, np.ndarray]:
        """
        Extract velocity profiles along centerlines for comparison with Ghia.
        
        Placeholder for future implementation.
        
        Args:
            snapshot: Snapshot object
            
        Returns:
            Dictionary with 'u_vertical' and 'v_horizontal' profiles
        """
        # TODO: Implement centerline velocity extraction
        # Extract u along vertical centerline (x = 0.5)
        # Extract v along horizontal centerline (y = 0.5)
        return {}
    
    def compare_centerline_profiles(self, snapshot) -> Dict[str, float]:
        """
        Compare computed centerline profiles against Ghia benchmark data.
        
        Placeholder for future implementation.
        
        Args:
            snapshot: Snapshot object
            
        Returns:
            Dictionary with profile comparison metrics (L2 errors, etc.)
        """
        # TODO: Implement centerline profile comparison
        # Would need to embed Ghia's tabulated velocity profile data
        return {}
    
    def track_multi_vortex_structure(self, snapshot) -> Dict[str, List[VortexCenter]]:
        """
        Track multiple vortex structures (primary, secondary, tertiary).
        
        Placeholder for future implementation.
        
        Args:
            snapshot: Snapshot object
            
        Returns:
            Dictionary mapping vortex names to their centers
        """
        # TODO: Implement multi-vortex tracking
        # Useful for high Re regimes where multiple vortices appear
        return {}
