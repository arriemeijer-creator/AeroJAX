"""
Divergence-based PID Controller for Adaptive Timestepping
"""
import jax.numpy as jnp


class DivergencePIDController:
    def __init__(self, target_div=1e-4, Kp=0.5, Ki=0.05, Kd=0.1, dt_min=1e-5, dt_max=0.01, integral_limit=1.0):
        self.target = target_div
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update(self, current_dt, div_max, eta_max=None, dt_max=None):
        # Safeguard against invalid divergence values
        if not jnp.isfinite(div_max) or div_max < 0:
            # If divergence is invalid or negative, reduce timestep conservatively
            new_dt = current_dt * 0.5
            if dt_max is None:
                dt_max = self.dt_max
            new_dt = max(self.dt_min, min(dt_max, new_dt))
            return new_dt

        error = div_max - self.target

        # Use provided dt_max or fall back to self.dt_max
        if dt_max is None:
            dt_max = self.dt_max

        # Anti-windup: clamp integral term
        self.integral += error * current_dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        derivative = (error - self.prev_error) / (current_dt + 1e-8)
        correction = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        factor = max(0.5, min(2.0, 1.0 + correction))
        new_dt = current_dt * factor

        # Step 5: Brinkman stiffness limiter (ChatGPT's critical addition)
        if eta_max is not None and eta_max > 0:
            rho = 1.0
            C_safety = 0.5  # Conservative safety factor
            dt_brinkman_limit = C_safety * (rho / (eta_max + 1e-8))
            new_dt = min(new_dt, dt_brinkman_limit)

        new_dt = max(self.dt_min, min(dt_max, new_dt))
        self.prev_error = error
        return new_dt
