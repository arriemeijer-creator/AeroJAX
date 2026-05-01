"""
Von Karman Strouhal tracker - Simplified & Robust.

Core insight: For Strouhal number, you don't need to track individual vortices.
Just track the wake's vertical oscillation using cross-correlation or
centerline velocity.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from scipy.fft import fft, fftfreq
from scipy.signal import correlate


@dataclass
class WakeOscillation:
    """Wake oscillation tracking data."""
    time: float
    wake_center_y: float  # Vertical position of wake centerline
    amplitude: float      # Oscillation amplitude
    phase: float          # Phase angle


@dataclass
class TrackedVortex:
    """Persistent vortex tracking data."""
    id: int
    x_norm: float
    y_norm: float
    sign: int  # 1 for positive (clockwise), -1 for negative (counter-clockwise)
    last_seen_time: float
    creation_time: float


class VKStrouhalTracker:
    """
    Von Karman Strouhal number tracker using wake oscillation.
    
    Tracks the vertical oscillation of the wake instead of individual vortices.
    This is far more robust and directly gives the shedding frequency.
    """
    
    def __init__(self,
                 tracking_x: float = 0.5,  # X-position to track wake (downstream)
                 y_range_min: float = 0.1,  # Min y for wake search
                 y_range_max: float = 0.9,  # Max y for wake search
                 min_amplitude: float = 0.005,  # Min oscillation amplitude to detect shedding (lowered)
                 history_length: int = 1000,  # Length of oscillation history
                 wake_x_min: float = 0.5,  # Min x for vortex detection (normalized)
                 wake_x_max: float = 1.0,  # Max x for vortex detection (normalized)
                 solver=None):  # Solver reference for parameter extraction
        """
        Args:
            tracking_x: X-coordinate (normalized) where to track wake oscillation
            y_range_min: Bottom of y-range for wake search
            y_range_max: Top of y-range for wake search
            min_amplitude: Minimum amplitude to consider as valid shedding
            history_length: Number of timesteps to keep in history
            wake_x_min: Minimum x for vortex detection (normalized)
            wake_x_max: Maximum x for vortex detection (normalized)
            solver: Solver reference for extracting characteristic length and velocity
        """
        self.tracking_x = tracking_x
        self.y_range_min = y_range_min
        self.y_range_max = y_range_max
        self.min_amplitude = min_amplitude
        self.history_length = history_length
        self.wake_x_min = wake_x_min
        self.wake_x_max = wake_x_max
        self.solver = solver  # Store solver reference

        # Persistent vortex tracking
        self.tracked_vortices: List[TrackedVortex] = []
        self.next_vortex_id = 0
        self.max_vortex_age = 0.1  # Remove vortices not seen for 0.1 seconds
        self.matching_distance = 0.1  # Max normalized distance to match vortices
        self.peak_distance = 20  # Minimum distance between peaks to avoid duplicates
        self.max_vortices_to_track = 2  # Maximum number of vortices to track (default matches slider)
        
        # Grid parameters
        self.dx = None
        self.dy = None
        self.nx = None
        self.ny = None
        
        # Oscillation history
        self.oscillation_history: List[WakeOscillation] = []
        
        # Lift history (alternative method)
        self.lift_history: List[float] = []
        self.time_history: List[float] = []
        
        # Current state
        self.current_wake_center = 0.5
        self.current_amplitude = 0.0
        self.current_phase = 0.0
        
        # Solver time step (will be updated from actual solver)
        self.solver_dt = 0.005  # Default fallback
    
    @classmethod
    def load(cls, solver=None, **kwargs) -> 'VKStrouhalTracker':
        """Load tracker with default or custom parameters."""
        return cls(solver=solver, **kwargs)
    
    def compute(self, snapshot, lift_coefficient: Optional[float] = None) -> Dict:
        """
        Legacy method for backward compatibility.
        
        Calls update() and returns the result.
        """
        return self.update(snapshot, lift_coefficient)
        
    def update(self, snapshot, lift_coefficient: Optional[float] = None) -> Dict:
        """
        Update wake tracking with new snapshot.
        
        Args:
            snapshot: Snapshot with u, v fields
            lift_coefficient: Optional lift coefficient (for cross-validation)
            
        Returns:
            Dictionary with current wake state and Strouhal estimate
        """
        # Extract grid parameters
        self._extract_grid(snapshot)
        
        # Extract actual solver dt for accurate frequency calculation
        if hasattr(snapshot, 'dt'):
            self.solver_dt = snapshot.dt
        elif hasattr(self, 'solver') and hasattr(self.solver, 'dt'):
            self.solver_dt = self.solver.dt
        else:
            self.solver_dt = 0.005  # Fallback to common dt
        
        # Get velocity fields with safety checks
        try:
            u = np.asarray(snapshot.u)
            v = np.asarray(snapshot.v)
        except:
            # If velocity fields can't be extracted, return previous state
            return {
                'wake_center_y': getattr(self, 'current_wake_center', 0.5),
                'amplitude': getattr(self, 'current_amplitude', 0.0),
                'phase': getattr(self, 'current_phase', 0.0),
                'strouhal_number': self.compute_strouhal(solver=self.solver),
                'is_shedding': False
            }
        
        # Method 1: Track wake using vertical velocity component
        wake_center = self._track_wake_centerline(v)
        
        # Method 2: Track wake using vorticity (cross-validation)
        wake_center_vorticity = self._track_wake_vorticity(u, v)
        
        # Average both methods for robustness
        if wake_center_vorticity is not None:
            wake_center = (wake_center + wake_center_vorticity) / 2
        
        # Compute oscillation amplitude and phase
        amplitude, phase = self._compute_oscillation_metrics(wake_center)
        
        # Store history
        current_time = getattr(snapshot, 'timestamp', len(self.oscillation_history) * 0.01)
        self.oscillation_history.append(WakeOscillation(
            time=current_time,
            wake_center_y=wake_center,
            amplitude=amplitude,
            phase=phase
        ))
        
        # Trim history
        if len(self.oscillation_history) > self.history_length:
            self.oscillation_history = self.oscillation_history[-self.history_length:]
        
        # Store lift history if provided
        if lift_coefficient is not None:
            self.lift_history.append(lift_coefficient)
            self.time_history.append(current_time)
            if len(self.lift_history) > self.history_length:
                self.lift_history = self.lift_history[-self.history_length:]
                self.time_history = self.time_history[-self.history_length:]
        
        # Update current state
        self.current_wake_center = wake_center
        self.current_amplitude = amplitude
        self.current_phase = phase
        
        # Compute Strouhal number
        strouhal = self.compute_strouhal(solver=self.solver)
        
        return {
            'wake_center_y': wake_center,
            'amplitude': amplitude,
            'phase': phase,
            'strouhal_number': strouhal,
            'is_shedding': amplitude > self.min_amplitude
        }
    
    def _extract_grid(self, snapshot) -> None:
        """Extract grid parameters."""
        self.dx = snapshot.dx
        self.dy = snapshot.dy
        self.nx = snapshot.nx
        self.ny = snapshot.ny
    
    def _track_wake_centerline(self, v: np.ndarray) -> float:
        """
        Track wake centerline using vertical velocity component.
        
        In a von Kármán street, the wake oscillates vertically. The centerline
        is where v changes sign.
        """
        # Handle MAC grid shape mismatches
        v_shape = v.shape
        if len(v_shape) != 2:
            return getattr(self, 'current_wake_center', 0.5)
        
        v_nx, v_ny = v_shape
        
        # Adjust tracking indices for actual v field dimensions
        ix = int(self.tracking_x * (v_nx - 1))
        ix = min(max(ix, 0), v_nx - 1)
        
        # Extract vertical velocity at this x-position
        try:
            v_profile = v[ix, :]
        except IndexError:
            # If indexing fails, return previous center
            return getattr(self, 'current_wake_center', 0.5)
        
        # Define y-range to search (adjust for actual v field dimensions)
        j_min = int(self.y_range_min * (v_ny - 1))
        j_max = int(self.y_range_max * (v_ny - 1))
        j_min = max(0, min(j_min, v_ny - 1))
        j_max = max(j_min + 1, min(j_max, v_ny - 1))
        
        v_region = v_profile[j_min:j_max]
        y_indices = np.arange(j_min, j_max)
        
        # Find zero-crossing of v (wake centerline)
        # Use weighted average around sign change
        zero_crossings = []
        for i in range(len(v_region) - 1):
            if v_region[i] * v_region[i+1] < 0:
                # Linear interpolation for exact zero
                t = -v_region[i] / (v_region[i+1] - v_region[i])
                y_zero = y_indices[i] + t * (y_indices[i+1] - y_indices[i])
                zero_crossings.append(y_zero)
        
        if not zero_crossings:
            # No zero crossing found - return previous center or middle
            return getattr(self, 'current_wake_center', 0.5)
        
        # For von Kármán, there should be one primary zero crossing
        # Take the one closest to center if multiple
        y_center_norm = np.mean([0.5])  # Middle of domain
        best_crossing = min(zero_crossings, 
                           key=lambda y: abs(y / (self.ny - 1) - 0.5))
        
        return best_crossing / (self.ny - 1)
    
    def _track_wake_vorticity(self, u: np.ndarray, v: np.ndarray) -> Optional[float]:
        """
        Track wake using vorticity centroid.
        
        Finds the vertical position where positive and negative vorticity
        are balanced in the wake.
        """
        # Handle MAC grid shape mismatches
        u_shape = u.shape
        v_shape = v.shape
        
        if len(u_shape) != 2 or len(v_shape) != 2:
            return None
        
        # For MAC grids, u and v may have different dimensions
        # Use the smaller dimensions for safe operations
        min_nx = min(u_shape[0], v_shape[0])
        min_ny = min(u_shape[1], v_shape[1])
        
        # Trim arrays to common dimensions if needed
        u_safe = u[:min_nx, :min_ny]
        v_safe = v[:min_nx, :min_ny]
        
        try:
            # Compute vorticity with safe arrays
            du_dy = np.gradient(u_safe, axis=1) / self.dy
            dv_dx = np.gradient(v_safe, axis=0) / self.dx
            vorticity = dv_dx - du_dy
        except:
            # If vorticity computation fails, return None
            return None
        
        # X-index for tracking (use actual vorticity dimensions)
        vort_nx, vort_ny = vorticity.shape
        ix = int(self.tracking_x * (vort_nx - 1))
        ix = min(max(ix, 0), vort_nx - 1)
        
        # Extract vorticity profile
        try:
            vort_profile = vorticity[ix, :]
        except IndexError:
            return None
        
        # Y-range (adjust for actual vorticity dimensions)
        j_min = int(self.y_range_min * (vort_ny - 1))
        j_max = int(self.y_range_max * (vort_ny - 1))
        
        # Compute centroid of absolute vorticity
        y_indices = np.arange(j_min, j_max)
        abs_vort = np.abs(vort_profile[j_min:j_max])
        
        if np.sum(abs_vort) > 0:
            centroid = np.sum(y_indices * abs_vort) / np.sum(abs_vort)
            return centroid / (self.ny - 1)
        
        return None
    
    def _compute_oscillation_metrics(self, current_center: float) -> Tuple[float, float]:
        """
        Compute oscillation amplitude and phase from recent history.
        """
        if len(self.oscillation_history) < 10:
            return 0.0, 0.0
        
        # Extract recent y positions
        recent_centers = [osc.wake_center_y for osc in self.oscillation_history[-50:]]
        
        if len(recent_centers) < 5:
            return 0.0, 0.0
        
        # Compute amplitude as RMS of deviation from mean
        mean_center = np.mean(recent_centers)
        deviations = np.array(recent_centers) - mean_center
        amplitude = np.std(deviations)
        
        # Compute phase using Hilbert transform (simplified - use last few points)
        if len(deviations) > 10:
            # Find last zero crossing to estimate phase
            last_crossing = None
            for i in range(len(deviations) - 1, 0, -1):
                if deviations[i] * deviations[i-1] < 0:
                    # Interpolate zero crossing
                    t = -deviations[i-1] / (deviations[i] - deviations[i-1])
                    last_crossing = (len(deviations) - i) + t
                    break
            
            if last_crossing is not None:
                phase = 2 * np.pi * (last_crossing % 1.0)
            else:
                phase = 0.0
        else:
            phase = 0.0
        
        return amplitude, phase
    
    def compute_strouhal(self,
                        freestream_velocity: Optional[float] = None,
                        characteristic_length: Optional[float] = None,
                        solver=None) -> Optional[float]:
        """
        Compute Strouhal number from wake oscillation or lift coefficient.
        
        Prioritizes lift coefficient if available (more accurate),
        otherwise uses wake oscillation.
        
        Args:
            freestream_velocity: Override freestream velocity (extracted from solver if None)
            characteristic_length: Override characteristic length (extracted from solver if None)
            solver: Solver reference to extract parameters from
        """
        # Initialize variables
        if freestream_velocity is None:
            freestream_velocity = None
        if characteristic_length is None:
            characteristic_length = None
        
        # Extract actual parameters from solver if not provided
        if solver is not None:
            if freestream_velocity is None:
                # Check if LBM solver and use effective velocity
                if hasattr(solver, 'U_lattice') and hasattr(solver, 'grid'):
                    # LBM uses lattice units - convert lattice velocity to effective physical velocity
                    # U_physical = U_lattice * dx / dt
                    dx = solver.grid.lx / solver.grid.nx
                    dt = solver.dt
                    u_effective = float(solver.U_lattice) * dx / dt
                    freestream_velocity = u_effective
                else:
                    freestream_velocity = solver.flow.U_inf
                
            if characteristic_length is None:
                # Determine characteristic length based on obstacle type
                # Check obstacle type first to use correct characteristic length
                if hasattr(solver, 'sim_params') and hasattr(solver.sim_params, 'obstacle_type'):
                    obstacle_type = solver.sim_params.obstacle_type
                    
                    if obstacle_type == 'cylinder':
                        if hasattr(solver, 'geom') and hasattr(solver.geom, 'radius'):
                            radius = float(solver.geom.radius.item()) if hasattr(solver.geom.radius, 'item') else float(solver.geom.radius)
                            characteristic_length = 2.0 * radius  # Diameter
                        else:
                            characteristic_length = 1.0  # Fallback
                    elif obstacle_type in ['naca_airfoil', 'airfoil']:
                        if hasattr(solver, 'sim_params') and hasattr(solver.sim_params, 'naca_chord'):
                            characteristic_length = solver.sim_params.naca_chord
                        else:
                            characteristic_length = 1.0  # Fallback
                    else:
                        characteristic_length = 1.0  # Fallback for unknown types
                else:
                    # Fallback: try NACA first, then cylinder
                    if hasattr(solver, 'sim_params') and hasattr(solver.sim_params, 'naca_chord'):
                        characteristic_length = solver.sim_params.naca_chord
                    elif hasattr(solver, 'geom') and hasattr(solver.geom, 'radius'):
                        radius = float(solver.geom.radius.item()) if hasattr(solver.geom.radius, 'item') else float(solver.geom.radius)
                        characteristic_length = 2.0 * radius  # Diameter
                    else:
                        characteristic_length = 1.0  # Fallback
        
        # Use defaults if still None
        if freestream_velocity is None:
            freestream_velocity = 1.0
        if characteristic_length is None:
            characteristic_length = 1.0
        
        # Method 1: Use lift coefficient (most accurate)
        if len(self.lift_history) >= 64:
            strouhal = self._compute_strouhal_from_signal(
                self.lift_history, 
                self.time_history,
                freestream_velocity,
                characteristic_length
            )
            if strouhal is not None:
                return strouhal
        
        # Method 2: Use wake oscillation
        if len(self.oscillation_history) >= 64:
            # Extract wake center time series
            wake_centers = [osc.wake_center_y for osc in self.oscillation_history]
            times = [osc.time for osc in self.oscillation_history]
            
            strouhal = self._compute_strouhal_from_signal(
                wake_centers,
                times,
                freestream_velocity,
                characteristic_length
            )
            if strouhal is not None:
                return strouhal
        
        return None
    
    def _compute_strouhal_from_signal(self, 
                                      signal: List[float],
                                      times: List[float],
                                      freestream_velocity: float,
                                      characteristic_length: float) -> Optional[float]:
        """
        Compute Strouhal number from arbitrary signal using FFT.
        """
        if len(signal) < 64:
            return None
        
        # Convert to numpy array
        signal = np.array(signal[-1024:])
        
        # Detrend
        signal = signal - np.mean(signal)
        
        # Compute time step - use actual solver dt for accurate frequency calculation
        times = np.array(times[-len(signal):])
        if len(times) < 2:
            # Use actual solver dt instead of fixed 0.01
            dt = getattr(self, 'solver_dt', 0.005)
        else:
            # Still compute from times for robustness, but ensure it's reasonable
            computed_dt = np.mean(np.diff(times))
            # Use computed dt if it's reasonable, otherwise use solver dt
            if 0.0001 < computed_dt < 1.0:  # Reasonable dt range
                dt = computed_dt
            else:
                dt = getattr(self, 'solver_dt', 0.005)
        
        # FFT
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, dt)
        
        # Find dominant frequency (positive only)
        positive_mask = xf > 0
        xf_pos = xf[positive_mask]
        yf_pos = np.abs(yf[positive_mask])
        
        if len(xf_pos) == 0:
            return None
        
        # Find peak, excluding very low frequencies (drift)
        min_freq = 0.05
        peak_mask = xf_pos > min_freq
        if not np.any(peak_mask):
            return None
        
        dominant_idx = np.argmax(yf_pos[peak_mask])
        dominant_freq = xf_pos[peak_mask][dominant_idx]
        
        # Strouhal number
        strouhal = dominant_freq * characteristic_length / freestream_velocity
        
        return float(strouhal)
    
    def get_strouhal_frequency(self) -> Optional[float]:
        """Legacy method for backward compatibility."""
        return self.compute_strouhal()
     
    def get_primary_vortex_positions(self, snapshot) -> List[Tuple[float, float, int, int]]:
        """
        Get all tracked vortices in the wake region with persistent IDs.
        Automatically determines the number of primary vortices to track.

        Returns:
            List of (x_norm, y_norm, sign, id) tuples for all tracked vortices
        """
        u = np.asarray(snapshot.u)
        v = np.asarray(snapshot.v)

        # Compute vorticity
        du_dy = np.gradient(u, axis=1) / self.dy
        dv_dx = np.gradient(v, axis=0) / self.dx
        vorticity = dv_dx - du_dy

        # Mask out obstacle region using configurable x_min and x_max
        wake_mask = np.zeros_like(vorticity, dtype=bool)
        wake_start_idx = int(self.wake_x_min * (self.nx - 1))
        wake_end_idx = int(self.wake_x_max * (self.nx - 1))
        wake_mask[wake_start_idx:wake_end_idx, :] = True

        # Mask out top and bottom walls (y < 0.15 or y > 0.85)
        y_min_idx = int(0.15 * (self.ny - 1))
        y_max_idx = int(0.85 * (self.ny - 1))
        wake_mask[:, :y_min_idx] = False
        wake_mask[:, y_max_idx:] = False

        # Apply mask to vorticity
        vorticity_wake = np.where(wake_mask, vorticity, 0)

        # Debug: print vorticity range
        pos_vort_max = np.max(vorticity_wake)
        neg_vort_min = np.min(vorticity_wake)
        print(f"DEBUG: Vorticity range (wake only): [{neg_vort_min:.4f}, {pos_vort_max:.4f}]")

        # Find all valid peaks (positive) and troughs (negative) in 2D
        min_peak_distance_norm = 0.05   # minimum spacing between two distinct vortices
        threshold = 0.01                # vorticity strength threshold

        positive_peaks = self._find_peaks_2d(vorticity_wake, threshold, min_peak_distance_norm, sign='positive')
        negative_peaks = self._find_peaks_2d(vorticity_wake, threshold, min_peak_distance_norm, sign='negative')

        # Combine all detections: (x_norm, y_norm, sign, strength)
        all_detections = []
        for (x_idx, y_idx, val) in positive_peaks:
            x_norm = x_idx / (self.nx - 1)
            y_norm = y_idx / (self.ny - 1)
            all_detections.append((x_norm, y_norm, 1, val))
        for (x_idx, y_idx, val) in negative_peaks:
            x_norm = x_idx / (self.nx - 1)
            y_norm = y_idx / (self.ny - 1)
            all_detections.append((x_norm, y_norm, -1, -val))

        # AUTO-DETECT: Determine how many vortices to track
        optimal_vortex_count = self._determine_optimal_vortex_count(all_detections)
        
        # Update max_vortices_to_track dynamically (but cap at reasonable limits)
        self.max_vortices_to_track = min(optimal_vortex_count, 10)  # Cap at 10 vortices
        
        # Keep only the strongest vortices based on auto-detected count
        all_detections.sort(key=lambda d: d[3], reverse=True)
        all_detections = all_detections[:self.max_vortices_to_track]

        # Current time for tracking
        current_time = getattr(snapshot, 'timestamp', 0.0)

        # Match detections to existing tracked vortices
        self._update_tracked_vortices(all_detections, current_time)

        # Final safety pruning
        if len(self.tracked_vortices) > self.max_vortices_to_track:
            self.tracked_vortices.sort(key=lambda v: v.creation_time)
            self.tracked_vortices = self.tracked_vortices[-self.max_vortices_to_track:]

        # Return currently tracked vortices
        return [(v.x_norm, v.y_norm, v.sign, v.id) for v in self.tracked_vortices]

    def _determine_optimal_vortex_count(self, detections: List[Tuple[float, float, int, float]]) -> int:
        """
        Automatically determine how many primary vortices exist in the wake.
        
        Uses multiple heuristics:
        1. Strength-based clustering: Count significant vortices above adaptive threshold
        2. Pair consistency: Vortices should come in alternating sign pairs
        3. Temporal stability: Track count over time to avoid fluctuations
        
        Returns:
            Optimal number of vortices to track
        """
        if not detections:
            return 2  # Default to classic von Kármán pair
        
        # Sort by strength
        sorted_detections = sorted(detections, key=lambda d: d[3], reverse=True)
        
        # Method 1: Find natural break in vortex strengths (elbow method)
        strengths = [d[3] for d in sorted_detections]
        
        if len(strengths) <= 2:
            return len(strengths)
        
        # Look for significant drop in vortex strength
        strength_ratios = []
        for i in range(1, len(strengths)):
            if strengths[i-1] > 0:
                ratio = strengths[i] / strengths[i-1]
                strength_ratios.append(ratio)
        
        # Find first large drop (> 50% reduction)
        significant_drop_idx = None
        for i, ratio in enumerate(strength_ratios):
            if ratio < 0.5:  # 50% strength reduction
                significant_drop_idx = i + 1  # +1 because ratio i corresponds to vortex i+1
                break
        
        if significant_drop_idx is not None:
            candidate_count = significant_drop_idx
        else:
            # No clear drop, use all detections but cap reasonably
            candidate_count = min(len(detections), 6)
        
        # Method 2: Check alternating sign pattern (essential for von Kármán)
        # Group by x-position to see alternating pattern
        if len(sorted_detections) >= 2:
            # Take top candidates and check sign alternation
            top_vortices = sorted_detections[:candidate_count]
            
            # Sort by x-position to see downstream pattern
            top_vortices_sorted = sorted(top_vortices, key=lambda d: d[0])
            
            # Check if signs alternate properly (+, -, +, - or -, +, -, +)
            signs = [v[2] for v in top_vortices_sorted]
            alternates = all(signs[i] != signs[i+1] for i in range(len(signs)-1))
            
            if not alternates and len(signs) > 2:
                # If signs don't alternate, we might be counting too many
                # Reduce until we get alternation
                for reduced_count in range(candidate_count - 1, 1, -1):
                    reduced_signs = signs[:reduced_count]
                    if all(reduced_signs[i] != reduced_signs[i+1] for i in range(len(reduced_signs)-1)):
                        candidate_count = reduced_count
                        break
        
        # Method 3: Track count over time (temporal consistency)
        if not hasattr(self, '_vortex_count_history'):
            self._vortex_count_history = []
        
        # Add current candidate to history (keep last 50 timesteps)
        self._vortex_count_history.append(candidate_count)
        if len(self._vortex_count_history) > 50:
            self._vortex_count_history.pop(0)
        
        # Use median/mean of recent history for stability
        if len(self._vortex_count_history) > 10:
            # Take median of recent counts to avoid rapid fluctuations
            stable_count = int(np.median(self._vortex_count_history[-20:]))
            # But don't let it drop too quickly (hysteresis)
            current_max = getattr(self, 'max_vortices_to_track', 2)
            if stable_count < current_max and stable_count < candidate_count:
                # Only decrease slowly (prevents flickering)
                if len(self._vortex_count_history) > 30 and all(c < current_max for c in self._vortex_count_history[-10:]):
                    return stable_count
                else:
                    return current_max
            else:
                return stable_count
        
        # Fallback to candidate count (ensure at least 2 for von Kármán street)
        return max(2, candidate_count)

    def _find_peaks_2d(self, field: np.ndarray, threshold: float,
                       min_distance_norm: float, sign: str) -> List[Tuple[int, int, float]]:
        """
        Find local maxima (sign='positive') or minima (sign='negative') in a 2D array,
        ensuring peaks are separated by at least `min_distance_norm` in normalized coordinates.

        Returns:
            List of (row_index, col_index, value) for each peak.
        """
        nx, ny = field.shape
        dx_norm = 1.0 / (nx - 1)
        dy_norm = 1.0 / (ny - 1)

        # For minima, we work on the negative field
        if sign == 'negative':
            search_field = -field
        else:
            search_field = field

        # Step 1: find all points that are local maxima (strictly greater than 8 neighbours)
        candidates = []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if search_field[i, j] <= threshold:
                    continue
                # Check 8 neighbours
                if (search_field[i, j] > search_field[i-1, j-1] and
                    search_field[i, j] > search_field[i-1, j] and
                    search_field[i, j] > search_field[i-1, j+1] and
                    search_field[i, j] > search_field[i, j-1] and
                    search_field[i, j] > search_field[i, j+1] and
                    search_field[i, j] > search_field[i+1, j-1] and
                    search_field[i, j] > search_field[i+1, j] and
                    search_field[i, j] > search_field[i+1, j+1]):
                    candidates.append((i, j, search_field[i, j]))

        if not candidates:
            return []

        # Step 2: sort by strength (descending)
        candidates.sort(key=lambda c: c[2], reverse=True)

        # Step 3: suppress peaks that are too close to a stronger one
        kept = []
        for cand in candidates:
            i_c, j_c, val_c = cand
            keep = True
            for (i_k, j_k, val_k) in kept:
                # Euclidean distance in normalized coordinates
                dist = np.sqrt(((i_c - i_k) * dx_norm) ** 2 + ((j_c - j_k) * dy_norm) ** 2)
                if dist < min_distance_norm:
                    keep = False
                    break
            if keep:
                kept.append((i_c, j_c, val_c))

        # Convert coordinates back to original sign if necessary
        if sign == 'negative':
            return [(i, j, -val) for (i, j, val) in kept]
        else:
            return kept

    def _update_tracked_vortices(self, detections: List[Tuple[float, float, int, float]], current_time: float):
        """
        Update tracked vortices with new detections using proximity matching.

        Args:
            detections: List of (x, y, sign, strength) tuples for current detections
            current_time: Current simulation time
        """
        # Mark all tracked vortices as not seen
        for vortex in self.tracked_vortices:
            vortex.last_seen_time = current_time  # Will be updated if matched

        # Match detections to tracked vortices
        matched_indices = set()
        for det_x, det_y, det_sign, det_strength in detections:
            best_match = None
            best_distance = self.matching_distance

            for i, vortex in enumerate(self.tracked_vortices):
                if vortex.sign == det_sign:  # Only match same sign
                    distance = np.sqrt((vortex.x_norm - det_x)**2 + (vortex.y_norm - det_y)**2)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = i

            if best_match is not None:
                # Update existing vortex
                self.tracked_vortices[best_match].x_norm = det_x
                self.tracked_vortices[best_match].y_norm = det_y
                self.tracked_vortices[best_match].last_seen_time = current_time
                matched_indices.add(best_match)
            else:
                # Create new vortex
                new_vortex = TrackedVortex(
                    id=self.next_vortex_id,
                    x_norm=det_x,
                    y_norm=det_y,
                    sign=det_sign,
                    last_seen_time=current_time,
                    creation_time=current_time
                )
                self.tracked_vortices.append(new_vortex)
                self.next_vortex_id += 1

        # Remove vortices that are out of bounds or too old
        self.tracked_vortices = [
            v for v in self.tracked_vortices
            if (self.wake_x_min <= v.x_norm <= self.wake_x_max and
                current_time - v.last_seen_time < self.max_vortex_age)
        ]

        # Limit to max_vortices_to_track (keep strongest by creation time)
        if len(self.tracked_vortices) > self.max_vortices_to_track:
            # Sort by creation time (older vortices first) and keep the newest ones
            self.tracked_vortices.sort(key=lambda v: v.creation_time)
            self.tracked_vortices = self.tracked_vortices[-self.max_vortices_to_track:]

    def prune_tracked_vortices(self):
        """Immediately prune tracked vortices to current max_vortices_to_track."""
        if len(self.tracked_vortices) > self.max_vortices_to_track:
            # Sort by creation time and keep newest
            self.tracked_vortices.sort(key=lambda v: v.creation_time)
            self.tracked_vortices = self.tracked_vortices[-self.max_vortices_to_track:]
    
    def get_shedding_statistics(self) -> Dict:
        """
        Get comprehensive shedding statistics.
        """
        strouhal = self.compute_strouhal(solver=self.solver)
        
        # Compute oscillation characteristics
        if len(self.oscillation_history) > 0:
            recent_amplitudes = [osc.amplitude for osc in self.oscillation_history[-100:]]
            mean_amplitude = np.mean(recent_amplitudes) if recent_amplitudes else 0.0
            max_amplitude = np.max(recent_amplitudes) if recent_amplitudes else 0.0
        else:
            mean_amplitude = 0.0
            max_amplitude = 0.0
        
        return {
            'strouhal_number': strouhal,
            'is_shedding': self.current_amplitude > self.min_amplitude,
            'oscillation_amplitude': self.current_amplitude,
            'mean_amplitude': mean_amplitude,
            'max_amplitude': max_amplitude,
            'wake_center_y': self.current_wake_center,
            'tracking_x': self.tracking_x
        }
    
    def get_visualization_data(self, snapshot) -> Dict:
        """
        Get data for GUI visualization.

        Args:
            snapshot: Current simulation snapshot

        Returns:
            Dictionary with visualization data for overlay
        """
        # Get primary vortex positions (auto-detects count)
        primary_vortices = self.get_primary_vortex_positions(snapshot)

        return {
            'primary_vortices': primary_vortices,
            'wake_center': self.current_wake_center,
            'tracking_x': self.tracking_x,
            'strouhal': self.compute_strouhal(solver=self.solver),
            'is_shedding': self.current_amplitude > self.min_amplitude,
            'vortex_count': len(primary_vortices)  # Add this for UI feedback
        }

    def get_vortex_statistics(self) -> Dict:
        """Get statistics about auto-detected vortices for debugging."""
        return {
            'auto_detected_count': getattr(self, 'max_vortices_to_track', 2),
            'tracked_vortices': len(self.tracked_vortices),
            'vortex_count_history': getattr(self, '_vortex_count_history', [])[-10:],
            'detection_threshold': 0.01,
            'peak_distance': 0.05
        }
