"""
Force computation and airfoil metrics for the Navier-Stokes solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from scipy import signal
from .operators import grad_x, grad_y, vorticity, grad_x_nonperiodic, grad_y_nonperiodic


@jax.jit
def interpolate_to_cell_center(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Interpolate staggered velocities to cell centers.
    u: (nx+1, ny) at x-faces -> u_center: (nx, ny)
    v: (nx, ny+1) at y-faces -> v_center: (nx, ny)
    """
    u_center = 0.5 * (u[1:, :] + u[:-1, :])
    v_center = 0.5 * (v[:, 1:] + v[:, :-1])
    return u_center, v_center


def get_airfoil_surface_mask(mask: jnp.ndarray, dx: float, threshold: float = 0.1) -> jnp.ndarray:
    """Find cells where mask gradient is large (the interface)"""
    dm_dx = grad_x_nonperiodic(mask, dx)
    dm_dy = grad_y_nonperiodic(mask, dx)
    grad_mag = jnp.sqrt(dm_dx**2 + dm_dy**2)
    return grad_mag > threshold


def find_stagnation_point(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray, p: jnp.ndarray,
                          grid_X: jnp.ndarray, dx: float, threshold: float = 0.1) -> float:
    """Find stagnation point on airfoil surface using pressure (highest pressure), returned in absolute domain coordinates"""
    surface = get_airfoil_surface_mask(mask, dx, threshold)
    # Use pressure instead of velocity - stagnation point has highest pressure
    surface_pressure = jnp.where(surface, p, -jnp.inf)
    max_idx = jnp.argmax(surface_pressure)
    stag_x = float(grid_X.flatten()[max_idx])
    return stag_x


def find_separation_point(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                          grid_X: jnp.ndarray, dx: float, dy: float, threshold: float = 0.1, grid_type: str = 'collocated') -> float:
    """Find separation by wall shear stress sign change on surface, returned in absolute domain coordinates"""
    surface = get_airfoil_surface_mask(mask, dx, threshold)
    if grid_type == 'mac':
        from solver.operators_mac import vorticity_nonperiodic_staggered
        vort = vorticity_nonperiodic_staggered(u, v, dx, dy)
    else:
        vort = vorticity(u, v, dx, dy)
    # Only consider surface cells
    surface_vort = jnp.where(surface, vort, 0.0)

    # Get x-coordinates of surface cells with positive vs negative vorticity
    surface_x = jnp.where(surface, grid_X, jnp.inf)
    pos_vort_x = jnp.where(surface_vort > 0, surface_x, jnp.inf)
    neg_vort_x = jnp.where(surface_vort < 0, surface_x, -jnp.inf)

    # Separation is where vorticity changes sign
    min_pos = jnp.min(pos_vort_x)
    max_neg = jnp.max(neg_vort_x)

    if min_pos < jnp.inf and max_neg > -jnp.inf:
        return float((min_pos + max_neg) / 2)  # Midpoint of sign change
    return 0.0


def detect_strouhal_stability(cl_history: np.ndarray, times: np.ndarray,
                             U_inf: float, chord_length: float,
                             min_oscillations: int = 3,
                             frequency_tolerance: float = 0.3) -> Tuple[bool, float, Optional[float]]:
    """
    Detect if vortex shedding has reached stable Strouhal number.

    Args:
        cl_history: Array of lift coefficient values over time
        times: Array of corresponding time values
        U_inf: Freestream velocity
        chord_length: Airfoil chord length
        min_oscillations: Minimum number of oscillations to consider
        frequency_tolerance: Tolerance for Strouhal number stability (±30%)

    Returns:
        (is_stable, strouhal_number, dominant_frequency)
    """
    if len(cl_history) < min_oscillations * 5:  # Need enough data points (reduced from 10 to 5)
        return False, 0.0, None

    # Remove mean
    cl_detrended = cl_history - np.mean(cl_history)

    # Lomb-Scargle periodogram (works with uneven sampling)
    # Define frequency grid to search
    total_time = times[-1] - times[0]
    f_min = 0.01  # Min frequency (Hz)
    f_max = min(10.0, len(cl_history) / total_time)  # Max frequency (Hz), limited by Nyquist
    freqs = np.linspace(f_min, f_max, 1000)

    # Compute periodogram
    periodogram = signal.lombscargle(times, cl_detrended, freqs, normalize=True)

    # Find dominant frequency
    dominant_freq_idx = np.argmax(periodogram)
    dominant_freq = freqs[dominant_freq_idx]

    # Calculate Strouhal number: St = f * L / U
    strouhal = dominant_freq * chord_length / U_inf

    # Debug output - commented out for performance
    # print(f"  Freq analysis (Lomb-Scargle): total_time={total_time:.2f}s, dominant_freq={dominant_freq:.2f}Hz, "
    #       f"expected_osc={dominant_freq*total_time:.1f}, strouhal={strouhal:.3f}")

    # Expected Strouhal range for cylinder/airfoil vortex shedding (0.05-0.4) - widened range
    expected_strouhal_range = (0.05, 0.4)

    # Check if Strouhal is in expected range
    in_expected_range = expected_strouhal_range[0] <= strouhal <= expected_strouhal_range[1]

    # Check if we have enough oscillations at this frequency
    num_oscillations = dominant_freq * total_time
    enough_oscillations = num_oscillations >= min_oscillations

    # Lomb-Scargle doesn't provide power ratio check like Welch's method
    # Just check Strouhal range and oscillations
    is_stable = in_expected_range and enough_oscillations

    return is_stable, strouhal, dominant_freq


def detect_vortex_amplitude_stability(cl_history: np.ndarray,
                                     window_size: int = 15,
                                     amplitude_tolerance: float = 0.2) -> Tuple[bool, float]:
    """
    Detect if vortex amplitude has stabilized (oscillations are consistent).

    Args:
        cl_history: Array of lift coefficient values over time
        window_size: Size of rolling window for amplitude calculation
        amplitude_tolerance: Relative tolerance for amplitude stability

    Returns:
        (is_stable, current_amplitude)
    """
    if len(cl_history) < window_size * 2:
        return False, 0.0

    # Calculate rolling amplitude (max - min in window)
    amplitudes = []
    for i in range(len(cl_history) - window_size + 1):
        window = cl_history[i:i + window_size]
        amplitude = np.max(window) - np.min(window)
        amplitudes.append(amplitude)

    amplitudes = np.array(amplitudes)

    # Check if amplitude has stabilized (coefficient of variation low)
    if len(amplitudes) < window_size:
        return False, 0.0

    recent_amplitudes = amplitudes[-window_size:]
    amplitude_mean = np.mean(recent_amplitudes)
    amplitude_std = np.std(recent_amplitudes)

    # Coefficient of variation should be small (< tolerance) - increased from 0.1 to 0.2
    cv = amplitude_std / (amplitude_mean + 1e-10)
    is_stable = cv < amplitude_tolerance

    return is_stable, amplitude_mean


def detect_vortex_shedding_stability(cl_history: np.ndarray, times: np.ndarray,
                                     U_inf: float, chord_length: float,
                                     min_stable_periods: int = 3) -> Tuple[bool, int, float]:
    """
    Combined stability detection using both Strouhal and amplitude checks.

    Args:
        cl_history: Array of lift coefficient values over time
        times: Array of corresponding time values
        U_inf: Freestream velocity
        chord_length: Airfoil chord length
        min_stable_periods: Minimum number of stable periods to confirm stability

    Returns:
        (is_stable, stable_start_index, strouhal_number)
    """
    if len(cl_history) < 50:  # Need minimum data
        return False, 0, 0.0

    # Check Strouhal stability
    strouhal_stable, strouhal, dominant_freq = detect_strouhal_stability(
        cl_history, times, U_inf, chord_length
    )

    # Check amplitude stability
    amp_stable, current_amp = detect_vortex_amplitude_stability(cl_history)

    # Combined stability check
    is_stable = strouhal_stable and amp_stable

    if is_stable:
        # Find the point where stability was achieved
        # Look for when both conditions were met for min_stable_periods
        stable_indices = []
        for i in range(len(cl_history) - min_stable_periods * 10):
            window_cl = cl_history[i:]
            window_times = times[i:]

            s_stable, _, _ = detect_strouhal_stability(window_cl, window_times, U_inf, chord_length)
            a_stable, _ = detect_vortex_amplitude_stability(window_cl)

            if s_stable and a_stable:
                stable_indices.append(i)

        if len(stable_indices) > 0:
            # Use the earliest stable point
            stable_start = min(stable_indices)
            return True, stable_start, strouhal

    return False, 0, strouhal


def compute_time_averaged_coefficients(cl_history: np.ndarray, cd_history: np.ndarray,
                                      stable_start_index: int) -> Tuple[float, float, int]:
    """
    Compute time-averaged CL and CD from stable period onwards.

    Args:
        cl_history: Array of lift coefficient values
        cd_history: Array of drag coefficient values
        stable_start_index: Index from which to start averaging

    Returns:
        (avg_cl, avg_cd, num_samples)
    """
    if stable_start_index >= len(cl_history):
        return 0.0, 0.0, 0

    cl_stable = cl_history[stable_start_index:]
    cd_stable = cd_history[stable_start_index:]

    avg_cl = np.mean(cl_stable)
    avg_cd = np.mean(cd_stable)
    num_samples = len(cl_stable)

    return avg_cl, avg_cd, num_samples


@jax.jit
def compute_CL_circulation(vorticity: jnp.ndarray, mask: jnp.ndarray, dx: float, dy: float,
                          U_inf: float, chord: float, fluid_threshold: float = 0.95) -> jnp.ndarray:
    """
    Compute lift coefficient from circulation (Kutta-Joukowski theorem).

    Args:
        vorticity: Vorticity field
        mask: Solid/fluid mask (1.0 = fluid, 0.0 = solid)
        dx, dy: Grid spacing
        U_inf: Freestream velocity
        chord: Airfoil chord length
        fluid_threshold: Threshold to exclude transition zone (0.95 = exclude cells with mask < 0.95)

    Returns:
        CL: Lift coefficient from circulation
    """
    # Create clean fluid mask - exclude transition zone
    fluid_mask = jnp.where(mask >= fluid_threshold, 1.0, 0.0)

    # Compute circulation by integrating vorticity over fluid region
    gamma = jnp.sum(vorticity * fluid_mask) * dx * dy

    # Kutta-Joukowski: L = rho * U * gamma, CL = L / (0.5 * rho * U^2 * chord) = 2 * gamma / (U * chord)
    CL = 2.0 * gamma / (U_inf * chord)

    return CL


def compute_circulation_contour(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float,
                                x_min: float, x_max: float, y_min: float, y_max: float) -> jnp.ndarray:
    """
    Compute circulation around a rectangular contour using line integral.

    Args:
        u: x-velocity field (cell-centered or interpolated to cell centers)
        v: y-velocity field (cell-centered or interpolated to cell centers)
        dx, dy: Grid spacing
        x_min, x_max: Contour bounds in x-direction
        y_min, y_max: Contour bounds in y-direction

    Returns:
        gamma: Circulation ∮ u·dl
    """
    # Find indices (not JIT-compiled)
    i_min = int(x_min / dx)
    i_max = int(x_max / dx)
    j_min = int(y_min / dy)
    j_max = int(y_max / dy)

    # Clamp indices to grid bounds
    nx, ny = u.shape
    i_min = max(0, min(i_min, nx - 1))
    i_max = max(0, min(i_max, nx - 1))
    j_min = max(0, min(j_min, ny - 1))
    j_max = max(0, min(j_max, ny - 1))

    # Ensure i_max > i_min and j_max > j_min
    if i_max <= i_min:
        i_max = i_min + 1
    if j_max <= j_min:
        j_max = j_min + 1

    # Circulation: ∮ u·dl = ∫(u dx + v dy) along contour
    # Bottom edge: (i_min→i_max, j_min) - v component
    gamma = jnp.sum(v[i_min:i_max, j_min]) * dx

    # Right edge: (i_max, j_min→j_max) - u component
    gamma += jnp.sum(u[i_max, j_min:j_max]) * dy

    # Top edge: (i_max→i_min, j_max) - v component (negative direction)
    gamma -= jnp.sum(v[i_min:i_max, j_max]) * dx

    # Left edge: (i_min, j_max→j_min) - u component (negative direction)
    gamma -= jnp.sum(u[i_min, j_min:j_max]) * dy

    return gamma


def compute_lift_circulation_contour(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                                     dx: float, dy: float,
                                     U_inf: float, chord: float,
                                     airfoil_x: float, airfoil_y: float,
                                     contour_margin: float = 1.0) -> jnp.ndarray:
    """
    Compute lift coefficient using circulation around rectangular contour (Kutta-Joukowski).
    Only integrates in fluid regions (mask > 0.5).

    Args:
        u: x-velocity field (cell-centered)
        v: y-velocity field (cell-centered)
        mask: Solid/fluid mask (1.0 = fluid, 0.0 = solid)
        dx, dy: Grid spacing
        U_inf: Freestream velocity
        chord: Airfoil chord length
        airfoil_x: Airfoil center x-position
        airfoil_y: Airfoil center y-position
        contour_margin: Margin around airfoil for contour (default 2.0)

    Returns:
        CL: Lift coefficient
    """
    # Define rectangular contour around airfoil
    x_min = airfoil_x - chord/2 - contour_margin
    x_max = airfoil_x + chord/2 + contour_margin
    y_min = airfoil_y - chord/2 - contour_margin
    y_max = airfoil_y + chord/2 + contour_margin

    # Get indices
    i_min = int(x_min / dx)
    i_max = int(x_max / dx)
    j_min = int(y_min / dy)
    j_max = int(y_max / dy)

    # Clamp to grid
    nx, ny = u.shape
    i_min = max(0, i_min)
    i_max = min(nx - 1, i_max)
    j_min = max(0, j_min)
    j_max = min(ny - 1, j_max)

    # Ensure i_max > i_min and j_max > j_min
    if i_max <= i_min:
        i_max = i_min + 1
    if j_max <= j_min:
        j_max = j_min + 1

    # Diagnostics: print contour bounds and circulation
    print(f"CIRCULATION CONTOUR DIAGNOSTICS:")
    print(f"  Contour bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"  Grid indices: i=[{i_min}, {i_max}], j=[{j_min}, {j_max}]")
    print(f"  Domain bounds: x=[0, {nx*dx:.2f}], y=[0, {ny*dy:.2f}]")
    print(f"  Contour clipped: {x_min < 0 or x_max > nx*dx or y_min < 0 or y_max > ny*dy}")

    # Mask the velocity fields (set velocity to zero inside solid)
    # Zero out velocities inside solid (mask < 0.5)
    u_masked = u * (mask > 0.5).astype(float)
    v_masked = v * (mask > 0.5).astype(float)

    # Compute circulation on masked velocities (reverse sign for CCW contour)
    # Bottom edge
    gamma = jnp.sum(v_masked[i_min:i_max, j_min]) * dx
    # Right edge
    gamma += jnp.sum(u_masked[i_max, j_min:j_max]) * dy
    # Top edge (negative direction)
    gamma -= jnp.sum(v_masked[i_min:i_max, j_max]) * dx
    # Left edge (negative direction)
    gamma -= jnp.sum(u_masked[i_min, j_min:j_max]) * dy

    # Reverse sign for CCW contour
    gamma = -gamma

    # Kutta-Joukowski: CL = 2 * gamma / (U_inf * chord)
    CL = 2.0 * gamma / (U_inf * chord)

    print(f"  Circulation gamma: {gamma:.6f}")
    print(f"  Computed CL: {CL:.6f}")

    return CL


def compute_drag_momentum_deficit(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                                   dx: float, dy: float,
                                   U_inf: float, chord: float,
                                   airfoil_x: float, chord_length: float,
                                   lx: float,
                                   wake_location: float = 8.0) -> jnp.ndarray:
    """
    Compute drag coefficient from momentum deficit in the wake.

    Args:
        u: x-velocity field (cell-centered)
        v: y-velocity field (cell-centered)
        mask: Solid/fluid mask
        dx, dy: Grid spacing
        U_inf: Freestream velocity
        chord: Airfoil chord length
        airfoil_x: Airfoil center x-position
        chord_length: Airfoil chord length
        lx: Domain length
        wake_location: X-position for wake measurement (default 8.0, deprecated)

    Returns:
        CD: Drag coefficient
    """
    # Find wake plane index (downstream of airfoil)
    # Use 60% of domain length to avoid outlet boundary effects
    wake_x = 0.6 * lx
    i_wake = int(wake_x / dx)

    nx = u.shape[0]
    if i_wake >= nx:
        return 0.0  # Not enough domain

    # Velocity profile at wake plane
    u_wake = u[i_wake, :]

    # Only integrate in fluid region (not inside airfoil)
    # Use mask interpolated to this plane
    mask_wake = mask[i_wake, :] if mask.shape[0] > i_wake else mask

    # Momentum deficit: ∫ (U_inf - u) * u dy (correct formula)
    deficit = (U_inf - u_wake) * u_wake

    # Only integrate where mask > 0.99 (pure fluid, avoid Brinkman transition zone)
    fluid = (mask_wake > 0.99).astype(float)

    # Only integrate where u < U_inf (true wake deficit, not acceleration regions)
    wake_region = (u_wake < U_inf).astype(float)
    integration_mask = fluid * wake_region

    # Diagnostics: check for regions where u > U_inf (causes negative deficit)
    u_gt_uinf = (u_wake > U_inf).astype(float)
    n_gt_uinf = jnp.sum(u_gt_uinf * fluid)
    n_wake = jnp.sum(wake_region * fluid)
    if n_gt_uinf > 0:
        print(f"WAKE PLANE DIAGNOSTICS (x={wake_x:.2f}):")
        print(f"  Fluid cells: {jnp.sum(fluid)}, Wake cells (u<U_inf): {n_wake}, Acceleration cells (u>U_inf): {n_gt_uinf}")
        print(f"  u_wake min: {jnp.min(u_wake * fluid):.4f}, max: {jnp.max(u_wake * fluid):.4f}, mean: {jnp.mean(u_wake * fluid):.4f}")
        print(f"  U_inf: {U_inf:.4f}")

    # Integrate only in wake region (where u < U_inf)
    momentum_deficit = jnp.sum(deficit * integration_mask) * dy

    # Drag coefficient: CD = (2 / (U_inf * chord)) * momentum_deficit
    CD = 2.0 * momentum_deficit / (U_inf * chord)

    if CD < 0:
        print(f"WARNING: Negative CD = {CD:.4f}, momentum_deficit = {momentum_deficit:.4f}")

    return CD


def compute_forces_ibm(u: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray,
                       x: jnp.ndarray, y: jnp.ndarray,
                       mask: jnp.ndarray, dx: float, dy: float,
                       U_inf: float, chord: float,
                       airfoil_x: float, airfoil_y: float,
                       lx: float,
                       grid_type: str = 'collocated') -> Tuple[float, float]:
    """
    Compute forces using IBM-appropriate methods (circulation-based).

    Args:
        u: x-velocity field
        v: y-velocity field
        w: Vorticity field (not used in momentum deficit method)
        x: X-coordinate grid
        y: Y-coordinate grid
        mask: Solid/fluid mask
        dx, dy: Grid spacing
        U_inf: Freestream velocity
        chord: Airfoil chord length
        airfoil_x: Airfoil center x-position
        airfoil_y: Airfoil center y-position
        lx: Domain length
        grid_type: 'collocated' or 'mac'

    Returns:
        (CL, CD): Lift and drag coefficients
    """
    # For MAC grid, interpolate velocities to cell centers first
    if grid_type == 'mac':
        u_center, v_center = interpolate_to_cell_center(u, v)
    else:
        u_center, v_center = u, v

    # Compute lift using circulation contour method with masking
    CL = compute_lift_circulation_contour(u_center, v_center, mask, dx, dy,
                                          U_inf, chord, airfoil_x, airfoil_y)

    # Compute drag using momentum deficit method (IBM-appropriate)
    CD = compute_drag_momentum_deficit(u_center, v_center, mask, dx, dy,
                                      U_inf, chord, airfoil_x, chord, lx)

    return float(CL), float(CD)
