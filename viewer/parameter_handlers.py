"""
Parameter update handlers for the CFD viewer.
Handles updates to simulation parameters like Reynolds number, grid resolution,
advection schemes, pressure solvers, LES settings, and timesteps.
"""

import jax
import jax.numpy as jnp
import gc
from typing import Optional

from solver import GridParams
from viewer.state import store, set_reynolds_number, set_u_inf, set_nu


class ParameterHandlers:
    """Mixin class providing parameter update methods for the viewer."""
    
    def update_reynolds_number(self) -> None:
        """Apply new flow parameters using constraint resolution."""
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        
        # Store new values before reset
        new_U = self.control_panel.u_input.value()
        new_nu = self.control_panel.nu_input.value()
        new_Re = self.control_panel.re_input.value()
        lock_U = self.control_panel.lock_u_cb.isChecked()
        lock_nu = self.control_panel.lock_nu_cb.isChecked()
        lock_Re = self.control_panel.lock_re_cb.isChecked()
        
        # Store current LES settings to preserve them
        current_use_les = self.solver.sim_params.use_les
        current_les_model = self.solver.sim_params.les_model

        try:
            # Update constraint locks
            self.solver.flow.constraints.lock_U = lock_U
            self.solver.flow.constraints.lock_nu = lock_nu
            self.solver.flow.constraints.lock_Re = lock_Re

            # Apply stored new values from UI
            print(f"UI values: U={new_U}, ν={new_nu}, Re={new_Re}")

            # Compute characteristic length
            if self.solver.sim_params.obstacle_type == 'naca_airfoil':
                L = self.solver.sim_params.naca_chord
            else:
                L = 2.0 * self.solver.geom.radius

            # Compute derived value based on which parameter is unlocked
            # If exactly 2 are locked, derive the third
            # If 1 is locked, derive the other 2 from the locked one and user input
            locked_count = sum([lock_U, lock_nu, lock_Re])
            if locked_count == 2:
                if not lock_U:
                    new_U = new_nu * new_Re / L
                    self.control_panel.u_input.setValue(new_U)
                elif not lock_nu:
                    new_nu = new_U * L / new_Re
                    self.control_panel.nu_input.setValue(new_nu)
                elif not lock_Re:
                    new_Re = new_U * L / new_nu
                    self.control_panel.re_input.setValue(new_Re)
            elif locked_count == 1:
                if lock_U:
                    # U is locked, derive ν and Re from U
                    # Use user's Re input to compute ν
                    new_nu = new_U * L / new_Re
                    self.control_panel.nu_input.setValue(new_nu)
                elif lock_nu:
                    # ν is locked, derive U and Re from ν
                    # Use user's Re input to compute U
                    new_U = new_nu * new_Re / L
                    self.control_panel.u_input.setValue(new_U)
                elif lock_Re:
                    # Re is locked, derive U and ν from Re
                    # Use user's U input to compute ν
                    new_nu = new_U * L / new_Re
                    self.control_panel.nu_input.setValue(new_nu)

            # Warn about high velocities that may cause instability
            if new_U > 5.0:
                print(f"WARNING: High inlet velocity U={new_U:.2f} m/s may cause numerical instability.")
                print(f"Consider using a smaller timestep or reducing velocity. Current grid dx={self.solver.grid.dx:.4f} m")
                print(f"Recommended CFL-based dt for U={new_U:.2f} m/s: ~{0.2 * self.solver.grid.dx / new_U:.6f} s")

            # Dispatch Redux actions to update store state (Redux is now single source of truth)
            store.dispatch(set_reynolds_number(new_Re))
            store.dispatch(set_u_inf(new_U))
            store.dispatch(set_nu(new_nu))
            
            # Update vorticity plot title with new parameters
            if hasattr(self, 'flow_viz'):
                naca = self.solver.sim_params.naca_airfoil if hasattr(self.solver.sim_params, 'naca_airfoil') else 'N/A'
                aoa = self.solver.sim_params.naca_angle if hasattr(self.solver.sim_params, 'naca_angle') else 0.0
                self.flow_viz.update_vorticity_title(new_Re, new_U, naca, aoa)

            # Recompute eps_multiplier based on new Re if auto_eps_multiplier is enabled
            if self.solver.sim_params.auto_eps_multiplier:
                from solver.params import compute_eps_multiplier
                self.solver.sim_params.eps_multiplier = compute_eps_multiplier(new_Re)
                self.solver.sim_params.eps = self.solver.sim_params.eps_multiplier * self.solver.grid.dx
                print(f"Auto-updated eps_multiplier = {self.solver.sim_params.eps_multiplier} from Re = {new_Re:.1f}")

                # Recompute mask with new epsilon
                self.solver.mask = self.solver._compute_mask()
                print(f"Recomputed mask with new ε = {self.solver.sim_params.eps:.4f}")

            # Recalculate dt based on new velocity for stability
            if self.solver.sim_params.adaptive_dt:
                # Disable adaptive dt for high velocities - use fixed dt instead
                if new_U >= 3.0:
                    self.solver.sim_params.adaptive_dt = False
                    print(f"Disabling adaptive dt for high velocity U={new_U:.2f} m/s")

                    if new_U > 5.0:
                        cfl_target = 0.2  # Increased from 0.05 for faster simulation
                        dt_max = 0.005  # Increased from 0.001
                    else:
                        cfl_target = 0.3  # Increased from 0.1
                        dt_max = 0.005  # Increased from 0.002

                    dx = self.solver.grid.dx
                    dy = self.solver.grid.dy
                    dt_cfl = cfl_target * min(dx, dy) / (new_U + 1e-8)
                    dt_diffusion = 0.25 * min(dx**2, dy**2) / new_nu

                    self.solver.dt = min(dt_cfl, dt_diffusion, dt_max)
                    self.solver.dt = max(self.solver.dt, self.solver.sim_params.dt_min)
                    print(f"Using fixed dt for U={new_U:.2f} m/s: dt={self.solver.dt:.6f} (CFL={cfl_target})")
                else:
                    if new_U > 5.0:
                        cfl_target = 0.05  # Very conservative for very high velocities
                    elif new_U > 3.0:
                        cfl_target = 0.1   # Conservative for high velocities
                    elif new_U > 1.5:
                        cfl_target = 0.15  # Conservative for moderate velocities
                    else:
                        cfl_target = 0.3   # Normal for low velocities

                    # Further reduce CFL target for low viscosities (high Reynolds numbers)
                    if new_nu < 1e-4:
                        cfl_target = min(cfl_target, 0.02)  # More aggressive reduction for very low viscosity
                    elif new_nu < 1e-3:
                        cfl_target = min(cfl_target, 0.05)   # Moderate reduction for low viscosity
                    elif new_nu < 2e-3:
                        cfl_target = min(cfl_target, 0.1)    # Additional reduction for moderate-low viscosity

                    # Add Reynolds number awareness - even with moderate velocity, high Re needs small CFL
                    if hasattr(self.solver.flow, 'Re') and self.solver.flow.Re > 500:
                        cfl_target = min(cfl_target, 0.1)   # Conservative for Re > 500
                    if hasattr(self.solver.flow, 'Re') and self.solver.flow.Re > 1000:
                        cfl_target = min(cfl_target, 0.1)    # Conservative for Re > 1000
                    if hasattr(self.solver.flow, 'Re') and self.solver.flow.Re > 2000:
                        cfl_target = min(cfl_target, 0.1)    # Conservative for Re > 2000

                    dx = self.solver.grid.dx
                    dy = self.solver.grid.dy
                    dt_cfl = cfl_target * min(dx, dy) / (new_U + 1e-8)
                    dt_diffusion = 0.25 * min(dx**2, dy**2) / new_nu

                    # Reduce dt_max for low viscosity and high velocity
                    dt_max = self.solver.sim_params.dt_max
                    if new_nu < 1e-4:
                        dt_max = min(dt_max, 0.001)
                    elif new_nu < 1e-3:
                        dt_max = min(dt_max, 0.002)
                    elif new_nu < 2e-3:
                        dt_max = min(dt_max, 0.003)
                    if new_U > 5.0:
                        dt_max = min(dt_max, 0.001)  # Very conservative for U>5m/s
                    elif new_U > 3.0:
                        dt_max = min(dt_max, 0.002)  # Conservative for U>3m/s
                    elif new_U > 1.5:
                        dt_max = min(dt_max, 0.003)  # Conservative for U>1.5m/s

                    self.solver.dt = min(dt_cfl, dt_diffusion, dt_max)
                    self.solver.dt = max(self.solver.dt, self.solver.sim_params.dt_min)
                    print(f"Recalculated dt for U={new_U:.2f} m/s: dt={self.solver.dt:.6f} (CFL={cfl_target})")

            # Update GUI to show the derived parameter (not the locked ones)
            if not self.solver.flow.constraints.lock_U:
                self.control_panel.u_input.setValue(self.solver.flow.U_inf)
            if not self.solver.flow.constraints.lock_nu:
                self.control_panel.nu_input.setValue(self.solver.flow.nu)
            if not self.solver.flow.constraints.lock_Re:
                self.control_panel.re_input.setValue(int(self.solver.flow.Re))

            self.solver._step_jit = jax.jit(self.solver._step)

            # Restore LES settings that were preserved
            self.solver.sim_params.use_les = current_use_les
            self.solver.sim_params.les_model = current_les_model
            self.solver.sim_params.dynamic_smagorinsky = (current_les_model == "dynamic_smagorinsky")

            # Suggest grid refinement for high Re
            if new_Re > 15000 and self.solver.grid.nx < 1024:
                print(f"WARNING: Re={new_Re:.0f} on {self.solver.grid.nx}×{self.solver.grid.ny} grid may be unstable.")
                print(f"Suggested grid: {min(2048, self.solver.grid.nx*2)}×{min(768, self.solver.grid.ny*2)}")

            # Reinitialize flow state with new parameters
            if self.solver.sim_params.flow_type == 'von_karman':
                self.solver._initialize_von_karman_flow()
            elif self.solver.sim_params.flow_type == 'lid_driven_cavity':
                self.solver._initialize_cavity_flow()
            elif self.solver.sim_params.flow_type == 'taylor_green':
                self.solver._initialize_taylor_green_flow()

            # Preserve user-specified dt - do not recalculate when updating Re
            # The dt is set during solver initialization and should remain constant

            self.solver.iteration = 0
            
            print(f"Flow parameters applied")
            print(f"  Constraints: U={self.solver.flow.constraints.lock_U}, nu={self.solver.flow.constraints.lock_nu}, Re={self.solver.flow.constraints.lock_Re}")
            print(f"  Resolved: U={self.solver.flow.U_inf:.3f}, nu={self.solver.flow.nu:.6f}, Re={self.solver.flow.Re:.1f}, L={self.solver.flow.L_char:.3f}")
            
            # Update spinboxes with resolved values
            self.control_panel.u_input.setValue(self.solver.flow.U_inf)
            self.control_panel.nu_input.setValue(self.solver.flow.nu)
            self.control_panel.re_input.setValue(int(self.solver.flow.Re))
            
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
            
        except Exception as e:
            print(f"Error updating flow parameters: {e}")
            import traceback
            traceback.print_exc()
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def on_lock_u_changed(self, state) -> None:
        """Handle U lock checkbox change."""
        # Lock validation removed - allow any number of locks
        pass

    def on_lock_nu_changed(self, state) -> None:
        """Handle nu lock checkbox change."""
        # Lock validation removed - allow any number of locks
        pass

    def on_lock_re_changed(self, state) -> None:
        """Handle Re lock checkbox change."""
        # Lock validation removed - allow any number of locks
        pass
    
    def update_precision(self) -> None:
        """Update JAX precision setting and restart application."""
        precision = self.control_panel.precision_combo.currentText()
        
        # Map precision to JAX config
        enable_x64 = (precision == "float64")
        
        # Modify the JAX config in solver/config.py
        try:
            import os
            import sys
            config_file = os.path.join(os.path.dirname(__file__), "..", "solver", "config.py")
            
            # Read the file with UTF-8 encoding
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the jax_enable_x64 line (handle both True/False cases)
            import re
            pattern = r"jax\.config\.update\('jax_enable_x64', (True|False)\)"
            new_config = f"jax.config.update('jax_enable_x64', {enable_x64})"
            
            if re.search(pattern, content):
                content = re.sub(pattern, new_config, content)
                
                # Write back with UTF-8 encoding
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"\n✓ Precision changed to {precision}")
                print(f"✓ JAX config updated")
                print(f"✓ Restarting application...")
                
                # Clean up and restart
                self.close()
                
                # Restart the application
                import subprocess
                subprocess.Popen([sys.executable] + sys.argv)
                
            else:
                print(f"\n✗ Could not find JAX config line in {config_file}")
                print("Please manually change: jax.config.update('jax_enable_x64', {enable_x64})")
                
        except Exception as e:
            print(f"\n✗ Error updating precision: {e}")
            print(f"Please manually change in solver/config.py:")
            print(f"  jax.config.update('jax_enable_x64', {enable_x64})")
    
    def update_grid_resolution(self) -> None:
        """Change the simulation grid resolution with uniform voxel scaling."""
        # Get custom grid dimensions from spinboxes
        grid_nx = self.control_panel.grid_x_spinbox.value()
        grid_ny = self.control_panel.grid_y_spinbox.value()
        current_flow = self.solver.sim_params.flow_type
        
        # Calculate domain size to maintain uniform grid spacing (dx = dy)
        # Use base grid spacing from 512x128 with domain 20.0x5.0
        base_dx = 20.0 / 512  # Base grid spacing in x
        base_dy = 5.0 / 128   # Base grid spacing in y
        
        # Use the smaller spacing to ensure uniform voxels (no stretching)
        uniform_spacing = min(base_dx, base_dy)
        
        # Calculate domain dimensions based on uniform spacing
        grid_lx = grid_nx * uniform_spacing
        grid_ly = grid_ny * uniform_spacing
        
        # Handle periodic domains
        if current_flow == 'taylor_green':
            grid_lx, grid_ly = 2 * jnp.pi, 2 * jnp.pi
        
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        
        try:
            # Clear ALL JAX caches before grid change
            jax.clear_caches()

            # Re-import multigrid solver to force recompilation with new grid dimensions
            import importlib
            import pressure_solvers.multigrid_solver as mg_module
            importlib.reload(mg_module)
            from pressure_solvers import poisson_multigrid
            # Update solver's reference to reloaded multigrid solver
            import solver
            solver.poisson_multigrid = poisson_multigrid

            # Clear existing JIT compilations
            if hasattr(self.solver, '_step_jit'):
                delattr(self.solver, '_step_jit')
            
            # Clear ALL arrays including mask and scalar field to prevent shape mismatches
            if hasattr(self.solver, 'u'):
                delattr(self.solver, 'u')
            if hasattr(self.solver, 'v'):
                delattr(self.solver, 'v')
            if hasattr(self.solver, 'u_prev'):
                delattr(self.solver, 'u_prev')
            if hasattr(self.solver, 'v_prev'):
                delattr(self.solver, 'v_prev')
            if hasattr(self.solver, 'mask'):
                delattr(self.solver, 'mask')
            if hasattr(self.solver, 'current_pressure'):
                delattr(self.solver, 'current_pressure')
            if hasattr(self.solver, 'c'):
                delattr(self.solver, 'c')
            
            # Clear JIT cache to remove compiled functions with old shapes
            if hasattr(self.solver, '_jit_cache'):
                self.solver._jit_cache = {}
            
            # Force garbage collection to ensure old arrays are freed
            gc.collect()
            for _ in range(3):
                gc.collect()
            
            # Update grid
            self.solver.grid = GridParams(nx=grid_nx, ny=grid_ny, lx=grid_lx, ly=grid_ly)
            
            # Recreate coordinates
            x = jnp.linspace(0, grid_lx, grid_nx)
            y = jnp.linspace(0, grid_ly, grid_ny)
            self.solver.grid.X, self.solver.grid.Y = jnp.meshgrid(x, y, indexing='ij')
            
            # Position obstacles correctly for new domain size FIRST
            # Cylinder/NACA: centered in Y, 1/4 from left in X
            if current_flow == 'von_karman':
                cylinder_x = grid_lx * 0.25  # 1/4 from left
                cylinder_y = grid_ly * 0.5   # Centered in Y
                
                # Scale obstacle size proportionally with domain
                scale_factor = min(grid_lx / 20.0, grid_ly / 3.75)  # Scale relative to base domain
                
                # Update cylinder radius (scale from base radius 0.18 for reference domain)
                base_radius = 0.18  # Base radius for reference domain (20.0x3.75)
                scaled_radius = base_radius * scale_factor
                self.solver.geom.radius = jnp.array([scaled_radius])
                
                # Update cylinder position
                self.solver.geom.center_x = jnp.array([cylinder_x])
                self.solver.geom.center_y = jnp.array([cylinder_y])
                
                # Update NACA position if using NACA airfoil
                if hasattr(self.solver.sim_params, 'naca_airfoil') and self.solver.sim_params.naca_airfoil:
                    # Scale x-position as percentage of domain width (25% of lx)
                    x_percentage = 0.25  # 25% from left
                    self.solver.sim_params.naca_x = x_percentage * grid_lx

                    # Scale y-position as percentage of domain height (50% of ly)
                    y_percentage = 0.5  # Centered in Y
                    self.solver.sim_params.naca_y = y_percentage * grid_ly

                    # Scale chord length as percentage of domain width (15% of lx)
                    chord_percentage = 0.15  # 15% of domain width
                    self.solver.sim_params.naca_chord = chord_percentage * grid_lx
            
            # Recreate mask AFTER updating obstacle positions
            self.solver.mask = self.solver._compute_mask()
            
            # Reinitialize flow with new grid dimensions
            if current_flow == 'lid_driven_cavity':
                self.solver._initialize_cavity_flow()
            elif current_flow == 'channel_flow':
                self.solver._initialize_channel_flow()
            elif current_flow == 'backward_step':
                self.solver._initialize_backward_step_flow()
            elif current_flow == 'taylor_green':
                self.solver._initialize_taylor_green_flow()
            else:
                self.solver._initialize_von_karman_flow()
            
            # Reinitialize scalar field (dye concentration) with new grid dimensions
            self.solver.c = jnp.zeros((grid_nx, grid_ny))
            
            # Recreate previous velocity arrays with new dimensions
            if hasattr(self.solver, 'u'):
                self.solver.u_prev = jnp.copy(self.solver.u)
            if hasattr(self.solver, 'v'):
                self.solver.v_prev = jnp.copy(self.solver.v)
            
            # Recompile ALL JIT functions with error handling
            try:
                self.solver._step_jit = jax.jit(self.solver._step)
                # Import required functions for JIT compilation
                from solver import vorticity, divergence
                self.solver._vorticity = jax.jit(vorticity, static_argnums=(2, 3))
                self.solver._divergence = jax.jit(divergence, static_argnums=(2, 3))
            except Exception as e:
                print(f"Error recompiling JIT functions: {e}")
                # Fallback: create minimal JIT functions
                self.solver._step_jit = jax.jit(self.solver._step)
            
            # Update visualization - now handles outline recreation internally
            try:
                self.flow_viz.update_plots_for_new_grid(grid_nx, grid_ny, grid_lx, grid_ly)
            except Exception as viz_error:
                print(f"Warning: Failed to update visualization: {viz_error}")
                print("Attempting minimal visualization update...")
                # Fallback: try to recreate just the basic plots
                try:
                    self.flow_viz.setup_plots(grid_nx, grid_ny, grid_lx, grid_ly)
                except Exception as basic_error:
                    print(f"Error: Even basic plot recreation failed: {basic_error}")
            
            # Update simulation controller shared buffers for new grid size
            try:
                self.sim_controller.update_grid_size(grid_nx, grid_ny)
            except Exception as controller_error:
                print(f"Warning: Failed to update simulation controller: {controller_error}")
            
            print(f"Grid updated to {grid_nx}x{grid_ny} ({grid_lx}x{grid_ly})")
            print(f"Flow reinitialized and JIT functions recompiled")
            
            # Update NACA chord range based on new domain size
            max_chord = min(grid_lx * 0.5, grid_ly * 0.6, 5.0)  # Max 50% of width, 60% of height, or 5.0
            self.control_panel.set_chord_range_for_domain(max_chord)
            
            # ULTRA-CONSERVATIVE: Minimal initialization to prevent any crashes
            try:
                # Use the most basic step function possible
                if hasattr(self.solver, '_step'):
                    self.solver._step_jit = self.solver._step
                
                # Clear everything
                gc.collect()
                jax.clear_caches()
                
                # Reset simulation to very basic state
                self.solver.iteration = 0
            except Exception as ultra_error:
                print(f"Warning: Ultra-conservative mode failed: {ultra_error}")
                print("Attempting emergency restart...")
                # Last resort: restart with minimal settings
                try:
                    self.solver.iteration = 0
                    self.solver.dt = 0.001  # Very conservative timestep
                except Exception as emergency_error:
                    print(f"Emergency restart failed: {emergency_error}")
            
            # Thread safety: ensure proper cleanup before starting new simulation
            try:
                if hasattr(self, 'sim_controller') and self.sim_controller:
                    self.sim_controller.stop_simulation()
                    # Wait for thread to fully stop
                    import time
                    time.sleep(0.1)  # Brief pause for thread cleanup
            except Exception as thread_error:
                print(f"Warning: Thread cleanup failed: {thread_error}")
            
            print(f"Grid resolution set to {grid_nx} x {grid_ny}")
            
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
            
        except Exception as e:
            print(f"Error updating grid resolution: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency recovery: try to restore basic functionality
            try:
                print("Attempting emergency recovery...")
                # Recreate basic JIT functions
                self.solver._step_jit = jax.jit(self.solver._step)
                print("Emergency recovery: basic JIT functions restored")
            except Exception as recovery_error:
                print(f"Emergency recovery failed: {recovery_error}")
            
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def update_cylinder_radius(self) -> None:
        """Update cylinder radius."""
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        
        try:
            # Get new radius from UI
            new_radius = self.control_panel.cylinder_radius_spinbox.value()
            
            # Update cylinder radius in solver
            old_radius = float(self.solver.geom.radius.item()) if hasattr(self.solver.geom.radius, 'item') else float(self.solver.geom.radius)
            self.solver.geom.radius = jnp.array(new_radius)  # Store as 0D array
            
            # Clear JAX caches before recompiling with new geometry
            jax.clear_caches()
            
            # Recompute mask with new radius
            self.solver.mask = self.solver._compute_mask()
            
            # Recompile solver
            self.solver._step_jit = jax.jit(self.solver._step)
            
            # Don't reset simulation - just recompile and continue
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
            
        except Exception as e:
            print(f"Error updating cylinder radius: {e}")
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def update_cylinder_array_params(self) -> None:
        """Update cylinder array diameter and spacing."""
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        
        try:
            # Get new parameters from UI
            new_diameter = self.control_panel.cylinder_diameter_spinbox.value()
            new_spacing = self.control_panel.cylinder_spacing_spinbox.value()
            
            # Update cylinder array parameters in solver
            if hasattr(self.solver.sim_params, 'cylinder_diameter'):
                self.solver.sim_params.cylinder_diameter = new_diameter
            else:
                self.solver.sim_params.cylinder_diameter = new_diameter
            
            if hasattr(self.solver.sim_params, 'cylinder_spacing'):
                self.solver.sim_params.cylinder_spacing = new_spacing
            else:
                self.solver.sim_params.cylinder_spacing = new_spacing
            
            # Clear JAX caches before recompiling with new geometry
            jax.clear_caches()
            
            # Recompute mask with new parameters
            self.solver.mask = self.solver._compute_mask()
            
            # Recompile solver
            self.solver._step_jit = jax.jit(self.solver._step)
            
            # Update obstacle outlines
            if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer:
                self.obstacle_renderer.update_obstacle_outlines(self.solver, force_update=True)
            
            # Don't reset simulation - just recompile and continue
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
            
            print(f"Cylinder array updated: diameter={new_diameter}, spacing={new_spacing}")
            
        except Exception as e:
            print(f"Error updating cylinder array parameters: {e}")
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def update_epsilon(self) -> None:
        """Update epsilon multiplier for mask smoothness."""
        # Stop everything
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        self.is_paused = False

        try:
            # Get new eps_multiplier from UI (slider value divided by 100)
            slider_value = self.control_panel.epsilon_slider.value()
            new_eps_multiplier = float(slider_value) / 100.0

            # Safety clamp
            if new_eps_multiplier > 10:
                print(f"WARNING: eps_multiplier {new_eps_multiplier:.2f} too large, clamping to 10")
                new_eps_multiplier = 10.0
                self.control_panel.epsilon_slider.setValue(int(new_eps_multiplier * 100))
                self.control_panel.epsilon_label.setText(f"{new_eps_multiplier:.2f}")

            # Update epsilon in solver
            if hasattr(self.solver, 'sim_params'):
                self.solver.sim_params.eps_multiplier = new_eps_multiplier
            else:
                print("Error: solver does not have sim_params")
                return
            self.solver.sim_params.eps = self.solver.sim_params.eps_multiplier * self.solver.grid.dx

            # Recompute mask
            self.solver.mask = self.solver._compute_mask()
            print(f"Mask recomputed with eps={self.solver.sim_params.eps:.6f} (eps_multiplier={new_eps_multiplier})")
            print(f"Mask shape: {self.solver.mask.shape}, min: {self.solver.mask.min():.6f}, max: {self.solver.mask.max():.6f}")

            # Clear JAX caches
            jax.clear_caches()

            # Recompile solver
            self.solver._step_jit = jax.jit(self.solver._step)

            # Reset flow state
            if self.solver.sim_params.flow_type == 'von_karman':
                self.solver._initialize_von_karman_flow()
            elif self.solver.sim_params.flow_type == 'lid_driven_cavity':
                self.solver._initialize_cavity_flow()
            elif self.solver.sim_params.flow_type == 'taylor_green':
                self.solver._initialize_taylor_green_flow()

            # Reset iteration and history
            self.solver.iteration = 0
            self.solver.u_prev = jnp.copy(self.solver.u)
            self.solver.v_prev = jnp.copy(self.solver.v)
            self.solver.history = {'time': [], 'dt': [], 'drag': [], 'lift': [],
                                  # Change metrics (not error)
                                  'l2_change': [], 'rms_change': [], 'l2_change_u': [], 'l2_change_v': [], 'max_change': [], 'change_99p': [], 'rel_change': [],
                                  # Continuity metrics
                                  'rms_divergence': [], 'l2_divergence': [],
                                  # Airfoil metrics
                                  'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 'Cp_min': [], 'wake_deficit': []}}

            # Enable start button
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)

        except Exception as e:
            print(f"Error updating epsilon: {e}")
            import traceback
            traceback.print_exc()
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def update_metrics_frame_skip(self) -> None:
        """Update the metrics frame skip value."""
        frame_skip = self.info_panel.metrics_frame_skip_input.value()
        self.solver.metrics_frame_skip = frame_skip
        print(f"Metrics frame skip updated to {frame_skip} (compute every {frame_skip} frames)")
    
    def on_metrics_checkbox_changed(self, state) -> None:
        """Handle metrics checkbox state change."""
        is_checked = (state == 2)  # Qt.CheckState.Checked
        if is_checked:
            # Start metrics worker if not running
            if self.sim_controller.metrics_worker is None:
                self.sim_controller.start_metrics()
                print("Metrics computation enabled - worker started")
        else:
            # Stop metrics worker
            if self.sim_controller.metrics_worker is not None:
                self.sim_controller.stop_metrics()
                print("Metrics computation disabled - worker stopped")
            # Clear error metrics labels
            if self.info_panel:
                self.info_panel.clear_error_metrics()
            # Clear airfoil metrics labels
            if self.info_panel:
                self.info_panel.clear_airfoil_metrics()
            # Clear error plot
            if hasattr(self, 'flow_viz') and self.flow_viz:
                self.flow_viz.clear_error_plot()

    def inject_dye(self) -> None:
        """Inject dye at the specified position with the specified amount."""
        x_pos = self.control_panel.dye_x_input.value()
        y_pos = self.control_panel.dye_y_input.value()
        amount = self.control_panel.dye_amount_slider.value() / 100.0  # Convert from 0-100 to 0.0-1.0

        self.solver.inject_dye(x_pos, y_pos, amount)
        print(f"Dye injected at ({x_pos:.2f}, {y_pos:.2f}) with amount {amount:.2f}")

    def update_les_settings(self) -> None:
        """Update LES/SGS model settings."""
        # Store new settings before reset
        use_les = self.control_panel.les_checkbox.isChecked()
        les_model = self.control_panel.les_model_combo.currentText()
        
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        
        # Store current Reynolds settings to preserve them
        current_U = self.solver.flow.U_inf
        current_nu = self.solver.flow.nu
        current_Re = self.solver.flow.Re
        # Store current obstacle type to preserve it
        current_obstacle_type = self.solver.sim_params.obstacle_type
        
        try:
            # Update solver parameters with stored values
            self.solver.sim_params.use_les = use_les
            self.solver.sim_params.les_model = les_model
            
            # Set dynamic_smagorinsky based on model selection
            self.solver.sim_params.dynamic_smagorinsky = (les_model == "dynamic_smagorinsky")
            
            # Recompile JIT functions if LES state changed
            if hasattr(self.solver, '_step_jit'):
                delattr(self.solver, '_step_jit')
            
            jax.clear_caches()
            
            # Recompile step function with new LES settings using get_step_jit
            self.solver._step_jit = self.solver.get_step_jit()
            # Recompile other JIT functions
            from solver import vorticity, divergence
            self.solver._vorticity = jax.jit(vorticity, static_argnums=(2, 3))
            
            # Restore Reynolds settings that were preserved
            self.solver.flow.U_inf = current_U
            self.solver.flow.nu = current_nu
            self.solver.flow.Re = current_Re
            self.solver._divergence = jax.jit(divergence, static_argnums=(2, 3))
            
            # Restore obstacle type that was preserved
            self.solver.sim_params.obstacle_type = current_obstacle_type
            
            print(f"LES settings updated: use_les={use_les}, model={les_model}")
            
            # Update UI controls to show applied values
            self.control_panel.les_checkbox.setChecked(use_les)
            self.control_panel.les_model_combo.setCurrentText(les_model)
            
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
            
        except Exception as e:
            print(f"Error updating LES settings: {e}")
            import traceback
            traceback.print_exc()
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)

    def update_vcycles(self) -> None:
        """Update multigrid V-cycles setting."""
        vcycles = self.control_panel.vcycles_slider.value()
        
        # Update solver parameter
        self.solver.sim_params.multigrid_v_cycles = vcycles
        
        # Clear JIT cache since this affects the pressure solver
        import jax
        jax.clear_caches()
        if hasattr(self.solver, '_jit_cache'):
            self.solver._jit_cache.clear()
        if hasattr(self.solver, '_step_jit'):
            delattr(self.solver, '_step_jit')
        
        # Recompile step function with new V-cycles
        self.solver._step_jit = self.solver.get_step_jit()
        
        print(f"Multigrid V-cycles updated to {vcycles}")
    
    def update_timestep(self) -> None:
        """Set a fixed timestep value."""
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        
        try:
            new_dt = self.control_panel.dt_spinbox.value()
            
            # Clear ALL JAX caches before changing timestep
            jax.clear_caches()
            
            # Clear any existing JIT compilations
            if hasattr(self.solver, '_step_jit'):
                delattr(self.solver, '_step_jit')
            
            # Update timestep
            self.solver.set_fixed_dt(new_dt)
            self.control_panel.dt_spinbox.setValue(self.solver.dt)
            self.control_panel.adaptive_dt_checkbox.setChecked(False)
            
            # Force recompilation
            self.solver._step_jit = jax.jit(self.solver._step)

        except Exception as dt_error:
            print(f"Error setting timestep: {dt_error}")
            # Ensure UI is in a usable state
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def update_frame_skip(self) -> None:
        """Change the frame skip setting for simulation."""
        frame_skip = self.control_panel.frame_skip_spinbox.value()
        self.config.viz_config.frame_skip = frame_skip
        print(f"Frame skip set to {frame_skip}")
    
    def update_visualization_fps(self) -> None:
        """Change the visualization refresh rate."""
        vis_fps = self.control_panel.vis_fps_input.value()
        self.config.viz_config.target_vis_fps = vis_fps
        if hasattr(self, 'refresh_timer'):
            self.refresh_timer.setInterval(int(1000 / vis_fps))
        print(f"Visualization rate set to {vis_fps} FPS")
