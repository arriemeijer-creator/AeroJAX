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
def compute_forces(u: jnp.ndarray, v: jnp.ndarray, p: jnp.ndarray, mask: jnp.ndarray,
                   dx: float, dy: float, nu: float, chord_length: float = 3.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute drag and lift forces on immersed boundaries by integrating over actual surface"""
    dp_dx = grad_x_nonperiodic(p, dx)
    dp_dy = grad_y_nonperiodic(p, dy)
    du_dx = grad_x_nonperiodic(u, dx)
    du_dy = grad_y_nonperiodic(u, dy)
    dv_dx = grad_x_nonperiodic(v, dx)
    dv_dy = grad_y_nonperiodic(v, dy)
    sigma_xx = p + 2.0 * nu * du_dx  # Fixed sign: +p instead of -p (pressure correction already accounts for sign)
    sigma_yy = p + 2.0 * nu * dv_dy  # Fixed sign: +p instead of -p
    sigma_xy = nu * (du_dy + dv_dx)
    dm_dx = grad_x_nonperiodic(mask, dx)
    dm_dy = grad_y_nonperiodic(mask, dy)

    # Normalize the gradient to get proper normal vector
    mag_grad = jnp.sqrt(dm_dx**2 + dm_dy**2)
    # Only calculate normals where mask gradient is significant to avoid noise amplification
    safe_mag = jnp.where(mag_grad > 1e-5, mag_grad, 1.0)
    nx = jnp.where(mag_grad > 1e-5, dm_dx / safe_mag, 0.0)
    ny = jnp.where(mag_grad > 1e-5, dm_dy / safe_mag, 0.0)

    # Identify actual surface cells (where mask gradient is large)
    # Use threshold to separate actual surface from diffuse layer
    surface_threshold = 0.5 * jnp.max(mag_grad)
    is_surface = mag_grad > surface_threshold

    # Force exerted by fluid on body = - (Force exerted by body on fluid)
    # Compute forces ONLY on actual surface cells, not diffuse layer
    fx_contrib = -(sigma_xx * nx + sigma_xy * ny) * is_surface
    fy_contrib = -(sigma_xy * nx + sigma_yy * ny) * is_surface

    drag = jnp.sum(fx_contrib) * dx * dy
    lift = jnp.sum(fy_contrib) * dx * dy

    # Return additional diagnostics for debugging
    max_grad = jnp.max(mag_grad)
    num_surface_cells = jnp.sum(is_surface)
    surface_length_estimate = num_surface_cells * dx

    return drag, lift, max_grad, surface_length_estimate


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
                          grid_X: jnp.ndarray, dx: float, dy: float, threshold: float = 0.1) -> float:
    """Find separation by wall shear stress sign change on surface, returned in absolute domain coordinates"""
    surface = get_airfoil_surface_mask(mask, dx, threshold)
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
