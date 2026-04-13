"""
Example usage of Diffrax adaptive controller integration
"""

# Example 1: Using Diffrax adaptive controller
from timestepping.adaptive_dt import AdaptiveDtController

# Create controller with Diffrax (default)
controller = AdaptiveDtController(flow_type='von_karman', use_diffrax=True)
print(f"Using Diffrax: {controller.use_diffrax}")

# Get initial timestep
dt = controller.get_initial_dt(U_inf=1.0, dx=0.04, dy=0.04)
print(f"Initial dt: {dt:.6f}")

# Example 2: Using basic adaptive controller (fallback)
basic_controller = AdaptiveDtController(flow_type='von_karman', use_diffrax=False)
print(f"Using basic controller: {not basic_controller.use_diffrax}")

# Example 3: Integration with solver (pseudo-code)
"""
# In your solver initialization:
from timestepping.adaptive_dt import set_adaptive_dt

# Enable Diffrax adaptive mode
set_adaptive_dt(solver_instance, 
                rtol=1e-4, 
                atol=1e-6, 
                dt_min=1e-6, 
                dt_max=0.002,
                use_diffrax=True)

# Or use basic adaptive mode
set_adaptive_dt(solver_instance, 
                max_cfl=0.3,
                dt_min=1e-6, 
                dt_max=0.002,
                use_diffrax=False)
"""

# Example 4: Flow-specific configurations
print("\nFlow-specific configurations:")
for flow_type in ['von_karman', 'taylor_green', 'lid_driven_cavity']:
    controller = AdaptiveDtController(flow_type=flow_type, use_diffrax=True)
    print(f"{flow_type}: rtol={controller.diffrax_controller.step_controller.rtol}, "
          f"dt_max={controller.diffrax_controller.step_controller.dtmax}")
