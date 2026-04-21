"""
Flow type manager for the CFD viewer.
Handles flow type selection and application.
"""

import jax
import time
from typing import Optional
from viewer.state import store


class FlowManager:
    """Mixin class providing flow type management methods for the viewer."""
    
    def setup_store_subscription(self) -> None:
        """
        Set up Redux store subscription to sync solver with store state.
        This creates unidirectional data flow: Store → Solver
        """
        def store_subscriber(state):
            """Handle store state changes and apply to solver."""
            if hasattr(self, 'solver') and self.solver is not None:
                # Check if obstacle type changed
                current_obstacle_type = getattr(self.solver.sim_params, 'obstacle_type', None)
                new_obstacle_type = state.obstacle.obstacle_type
                
                if current_obstacle_type != new_obstacle_type:
                    print(f"[STORE SYNC] Applying obstacle type change: {current_obstacle_type} -> {new_obstacle_type}")
                    # Call solver API to handle obstacle type change (solver manages heavy logic)
                    self.solver.set_obstacle_type(new_obstacle_type)
                    
                    # Update info panel
                    if hasattr(self, 'display_manager') and self.display_manager is not None:
                        self.display_manager._update_solver_info()
                
                # Check if obstacle position changed (for live preview)
                current_naca_x = getattr(self.solver.sim_params, 'naca_x', None)
                new_naca_x = state.obstacle.naca_x
                current_naca_y = getattr(self.solver.sim_params, 'naca_y', None)
                new_naca_y = state.obstacle.naca_y
                current_cylinder_center_x = getattr(self.solver.geom, 'center_x', None)
                new_cylinder_center_x = state.obstacle.cylinder_center_x
                current_cylinder_center_y = getattr(self.solver.geom, 'center_y', None)
                new_cylinder_center_y = state.obstacle.cylinder_center_y
                current_cow_x = getattr(self.solver.sim_params, 'cow_x', None)
                new_cow_x = state.obstacle.cow_x
                current_cow_y = getattr(self.solver.sim_params, 'cow_y', None)
                new_cow_y = state.obstacle.cow_y
                current_three_cylinder_x = getattr(self.solver.sim_params, 'cylinder_x', None)
                new_three_cylinder_x = state.obstacle.three_cylinder_x
                current_three_cylinder_y = getattr(self.solver.sim_params, 'cylinder_y', None)
                new_three_cylinder_y = state.obstacle.three_cylinder_y
                
                position_changed = False
                if new_obstacle_type == 'naca_airfoil':
                    if current_naca_x != new_naca_x or current_naca_y != new_naca_y:
                        position_changed = True
                        print(f"[STORE SYNC] NACA position changed: x {current_naca_x}->{new_naca_x}, y {current_naca_y}->{new_naca_y}")
                        if new_naca_x is not None:
                            self.solver.sim_params.naca_x = new_naca_x
                        if new_naca_y is not None:
                            self.solver.sim_params.naca_y = new_naca_y
                elif new_obstacle_type == 'cylinder':
                    if current_cylinder_center_x != new_cylinder_center_x or current_cylinder_center_y != new_cylinder_center_y:
                        position_changed = True
                        print(f"[STORE SYNC] Cylinder position changed: x {current_cylinder_center_x}->{new_cylinder_center_x}, y {current_cylinder_center_y}->{new_cylinder_center_y}")
                        import jax.numpy as jnp
                        if new_cylinder_center_x is not None:
                            self.solver.geom.center_x = jnp.array(new_cylinder_center_x)
                        if new_cylinder_center_y is not None:
                            self.solver.geom.center_y = jnp.array(new_cylinder_center_y)
                elif new_obstacle_type == 'cow':
                    if current_cow_x != new_cow_x or current_cow_y != new_cow_y:
                        position_changed = True
                        print(f"[STORE SYNC] Cow position changed: x {current_cow_x}->{new_cow_x}, y {current_cow_y}->{new_cow_y}")
                        if new_cow_x is not None:
                            self.solver.sim_params.cow_x = new_cow_x
                        if new_cow_y is not None:
                            self.solver.sim_params.cow_y = new_cow_y
                        # Clear JAX caches to force recompilation of cow mask with new position
                        import jax
                        jax.clear_caches()
                elif new_obstacle_type == 'three_cylinder_array':
                    if current_three_cylinder_x != new_three_cylinder_x or current_three_cylinder_y != new_three_cylinder_y:
                        position_changed = True
                        print(f"[STORE SYNC] Three cylinder array position changed: x {current_three_cylinder_x}->{new_three_cylinder_x}, y {current_three_cylinder_y}->{new_three_cylinder_y}")
                        if new_three_cylinder_x is not None:
                            self.solver.sim_params.cylinder_x = new_three_cylinder_x
                        if new_three_cylinder_y is not None:
                            self.solver.sim_params.cylinder_y = new_three_cylinder_y
                
                if position_changed:
                    # Recompute mask with new position
                    print(f"[STORE SYNC] Recomputing mask for position change")
                    self.solver.mask = self.solver._compute_mask()
                    
                    # Update obstacle preview for live feedback
                    print(f"[STORE SYNC] Updating obstacle outlines")
                    if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer:
                        self.obstacle_renderer.update_obstacle_outlines(self.solver, force_update=True)
                
                # Check if simulation parameters changed
                current_re = self.solver.flow.Re
                new_re = state.simulation.reynolds_number
                current_u_inf = self.solver.flow.U_inf
                new_u_inf = state.simulation.u_inf
                current_nu = self.solver.flow.nu
                new_nu = state.simulation.nu
                
                if current_re != new_re or current_u_inf != new_u_inf or current_nu != new_nu:
                    print(f"[STORE SYNC] Applying simulation params: Re {current_re}->{new_re}, U {current_u_inf}->{new_u_inf}, ν {current_nu}->{new_nu}")
                    # Apply simulation parameter changes to solver
                    self.solver.flow.Re = new_re
                    self.solver.flow.U_inf = new_u_inf
                    self.solver.flow.nu = new_nu
        
        # Subscribe to store changes
        self._store_unsubscribe = store.subscribe(store_subscriber)
        print("[STORE] Store subscription set up in FlowManager")
    
    def cleanup_store_subscription(self) -> None:
        """Clean up store subscription when viewer is destroyed."""
        if hasattr(self, '_store_unsubscribe'):
            self._store_unsubscribe()
            print("[STORE] Store subscription cleaned up")
    
    def on_flow_type_selected(self, flow_type: str) -> None:
        """Handle flow type selection changes."""
        if hasattr(self.control_panel, 'show_naca_controls'):
            if hasattr(self.control_panel, 'obstacle_button_group'):
                # Update radio button selection based on current obstacle type
                if self.solver.sim_params.obstacle_type == 'cylinder':
                    self.control_panel.cylinder_radio.setChecked(True)
                elif self.solver.sim_params.obstacle_type == 'naca_airfoil':
                    self.control_panel.naca_radio.setChecked(True)
                elif self.solver.sim_params.obstacle_type == 'cow':
                    self.control_panel.cow_radio.setChecked(True)
                elif self.solver.sim_params.obstacle_type == 'three_cylinder_array':
                    self.control_panel.cylinder_array_radio.setChecked(True)
        
        # NACA controls are now handled in _load_initial_state
        self._apply_flow_type()
    
    def _apply_flow_type(self) -> None:
        """Apply the selected flow type to the simulation."""
        selected_flow = self.control_panel.flow_combo.currentText()
        
        # CRITICAL: Stop simulation completely before flow type change
        print(f"Stopping simulation for flow type change to {selected_flow}...")
        self.refresh_timer.stop()
        
        # Force stop simulation with multiple attempts
        if hasattr(self, 'sim_controller'):
            self.sim_controller.stop_simulation()
            # Wait a moment for thread to stop
            time.sleep(0.1)
            # Ensure it's really stopped
            if hasattr(self.sim_controller, 'running') and self.sim_controller.running:
                self.sim_controller.stop_simulation()
                time.sleep(0.1)
        
        try:
            self.solver.apply_flow_type(selected_flow)
            
            # Update coordinates (grid coordinates are already created in apply_flow_type)
            grid_nx = self.solver.grid.nx
            grid_ny = self.solver.grid.ny
            grid_lx = self.solver.grid.lx
            grid_ly = self.solver.grid.ly
            print(f"DEBUG: After apply_flow_type, solver grid is {grid_nx}x{grid_ny}")
            
            # Note: X, Y coordinates and flow initialization are already done in solver.apply_flow_type()
            
            # Update grid spinboxes to match the actual solver grid dimensions
            self.control_panel.grid_x_spinbox.setValue(self.solver.grid.nx)
            self.control_panel.grid_y_spinbox.setValue(self.solver.grid.ny)
            
            # Recompile solver
            self.solver.mask = self.solver._compute_mask()
            jax.clear_caches()
            self.solver._step_jit = jax.jit(self.solver._step)
            
            # Stop simulation completely before recreating shared buffers
            if hasattr(self, 'sim_controller'):
                self.sim_controller.stop_simulation()
                # Clear latest_data to prevent stale buffer references
                self.sim_controller.latest_data = None
                time.sleep(0.2)  # Give time for simulation to fully stop
            
            # Update simulation worker solver reference to new solver with new grid
            if hasattr(self, 'sim_controller') and hasattr(self.sim_controller, 'simulation_worker'):
                if self.sim_controller.simulation_worker is not None:
                    self.sim_controller.simulation_worker.solver = self.solver
            
            # Recreate shared memory buffers with new grid dimensions
            if hasattr(self, 'sim_controller') and hasattr(self.sim_controller, 'simulation_worker'):
                if self.sim_controller.simulation_worker is not None:
                    self.sim_controller.simulation_worker.recreate_shared_buffers(grid_nx, grid_ny)
            
            # Allow adaptive dt for all flow types
            if selected_flow == 'lid_driven_cavity' and self.solver.sim_params.adaptive_dt:
                print("LDC: Adaptive timestep enabled")
            
            # Update visualization
            actual_nx = self.solver.grid.nx
            actual_ny = self.solver.grid.ny
            actual_lx = self.solver.grid.lx
            actual_ly = self.solver.grid.ly
            
            # Force reinitialize solver arrays with current grid dimensions
            if hasattr(self.solver, 'u'):
                delattr(self.solver, 'u')
            if hasattr(self.solver, 'v'):
                delattr(self.solver, 'v')
            
            # Reinitialize based on current flow type
            if self.solver.sim_params.flow_type == 'lid_driven_cavity':
                self.solver._initialize_cavity_flow()
            elif self.solver.sim_params.flow_type == 'taylor_green':
                self.solver._initialize_taylor_green_flow()
            else:
                self.solver._initialize_von_karman_flow()
            
            # Verify the arrays have correct dimensions
            if hasattr(self.solver, 'u') and hasattr(self.solver, 'v'):
                if self.solver.u.shape != (actual_nx, actual_ny):
                    print(f"ERROR: Arrays still have wrong shape after reinitialization!")
            
            # Update NACA chord range based on domain size
            max_chord = min(actual_lx * 0.5, actual_ly * 0.6, 5.0)  # Max 50% of width, 60% of height, or 5.0
            self.control_panel.set_chord_range_for_domain(max_chord)
            
            # Recreate visualization to ensure grid dimensions are properly updated
            try:
                self.plot_widget.clear()  # Clear old visualization from plot widget
                from viewer.visualization import FlowVisualization
                self.flow_viz = FlowVisualization(self.plot_widget, solver=self.solver)
                
                # Reinitialize colormaps
                self.flow_viz.set_initial_colormaps(
                    velocity_colormap=self.config.viz_config.default_velocity_colormap,
                    vorticity_colormap=self.config.viz_config.default_vorticity_colormap
                )
                
                print(f"DEBUG: Calling update_plots_for_new_grid with nx={actual_nx}, ny={actual_ny}")
                # Update visualization with new grid
                self.flow_viz.update_plots_for_new_grid(actual_nx, actual_ny, actual_lx, actual_ly)
                print(f"DEBUG: After update_plots_for_new_grid, flow_viz current_nx={self.flow_viz.current_nx}, current_ny={self.flow_viz.current_ny}")
                
                # Initialize image items with dummy data and levels to prevent levels error
                import numpy as np
                dummy_data = np.zeros((actual_nx, actual_ny), dtype=np.float32)
                if self.flow_viz.vel_img is not None:
                    self.flow_viz.vel_img.setImage(dummy_data, levels=self.flow_viz.vel_levels, autoLevels=False)
                if self.flow_viz.vort_img is not None:
                    self.flow_viz.vort_img.setImage(dummy_data, levels=self.flow_viz.vort_levels, autoLevels=False)
                
                # Update info_panel visualization reference
                if hasattr(self, 'info_panel') and self.info_panel:
                    self.info_panel.set_visualization(self.flow_viz)
                
                # Update display_manager flow_viz reference
                if hasattr(self, 'sim_controller') and hasattr(self.sim_controller, 'display_manager'):
                    self.sim_controller.display_manager.flow_viz = self.flow_viz
            except Exception as viz_error:
                print(f"ERROR during visualization recreation: {viz_error}")
                import traceback
                traceback.print_exc()
            
            # Scaling is now handled internally when plots are updated
            
            # Toggle inverse design visibility
            if hasattr(self, 'inverse_dock') and self.inverse_dock:
                self.inverse_dock.setVisible(selected_flow == 'von_karman')
            
            print(f"Flow type changed to {selected_flow}")
            
            # Restart simulation after flow type change
            if hasattr(self, 'sim_controller'):
                # Pass stored callbacks if available, otherwise None
                callbacks = self.sim_controller.callbacks if hasattr(self.sim_controller, 'callbacks') else None
                self.sim_controller.start_simulation(callbacks)
            
            self.control_panel.start_btn.setEnabled(False)
            self.control_panel.pause_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error changing flow type: {e}")
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def on_obstacle_type_selected(self, obstacle_type: str) -> None:
        """
        Handle obstacle type selection changes.
        
        NOTE: This method is kept for backward compatibility but the actual
        solver update is now handled by the Redux store subscription.
        This method only handles UI updates.
        """
        if hasattr(self.control_panel, 'show_naca_controls'):
            is_naca = obstacle_type == 'naca_airfoil'
            is_cylinder = obstacle_type == 'cylinder'
            self.control_panel.show_naca_controls(is_naca)

            # Show/hide cylinder widget
            if hasattr(self.control_panel, 'cylinder_widget'):
                self.control_panel.cylinder_widget.setVisible(is_cylinder)

        # Update x-position slider to reflect current obstacle's x-position
        if hasattr(self.control_panel, 'x_position_slider'):
            grid_lx = self.solver.grid.lx
            if obstacle_type == 'cylinder':
                current_x = float(self.solver.geom.center_x.item()) if hasattr(self.solver.geom.center_x, 'item') else float(self.solver.geom.center_x)
            elif obstacle_type == 'naca_airfoil':
                current_x = getattr(self.solver.sim_params, 'naca_x', grid_lx * 0.25)
            elif obstacle_type == 'cow':
                current_x = getattr(self.solver.sim_params, 'cow_x', grid_lx * 0.25)
            elif obstacle_type == 'three_cylinder_array':
                current_x = getattr(self.solver.sim_params, 'cylinder_x', grid_lx * 0.25)
            else:
                current_x = grid_lx * 0.25
            
            percentage = int((current_x / grid_lx) * 100)
            self.control_panel.x_position_slider.setValue(percentage)
            self.control_panel.x_position_label.setText(f"{percentage}%")
        
        # Update y-position slider to reflect current obstacle's y-position
        if hasattr(self.control_panel, 'y_position_slider'):
            grid_ly = self.solver.grid.ly
            if obstacle_type == 'cylinder':
                current_y = float(self.solver.geom.center_y.item()) if hasattr(self.solver.geom.center_y, 'item') else float(self.solver.geom.center_y)
            elif obstacle_type == 'naca_airfoil':
                current_y = getattr(self.solver.sim_params, 'naca_y', grid_ly * 0.5)
            elif obstacle_type == 'cow':
                current_y = getattr(self.solver.sim_params, 'cow_y', grid_ly * 0.5)
            elif obstacle_type == 'three_cylinder_array':
                current_y = getattr(self.solver.sim_params, 'cylinder_y', grid_ly * 0.5)
            else:
                current_y = grid_ly * 0.5
            
            percentage = int((current_y / grid_ly) * 100)
            self.control_panel.y_position_slider.setValue(percentage)
            self.control_panel.y_position_label.setText(f"{percentage}%")
        
        # Update obstacle outlines to reflect new obstacle type
        if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer:
            self.obstacle_renderer.update_obstacle_outlines(self.solver, force_update=True)
        
        # Update vorticity plot title to reflect new obstacle type
        if hasattr(self, 'flow_viz') and self.flow_viz:
            re = self.solver.flow.Re
            u_inlet = self.solver.flow.U_inf
            naca = self.solver.sim_params.naca_airfoil if hasattr(self.solver.sim_params, 'naca_airfoil') else 'N/A'
            aoa = self.solver.sim_params.naca_angle if hasattr(self.solver.sim_params, 'naca_angle') else 0.0
            self.flow_viz.update_vorticity_title(re, u_inlet, naca, aoa)
        
        print(f"Obstacle type changed to {obstacle_type}, mask recomputed")
    
    def apply_x_position_change(self, percentage: int) -> None:
        """Apply x-position change from slider (percentage of domain width)."""
        import jax.numpy as jnp
        obstacle_type = self.solver.sim_params.obstacle_type if hasattr(self.solver.sim_params, 'obstacle_type') else 'cylinder'
        grid_lx = self.solver.grid.lx
        x_position = (percentage / 100.0) * grid_lx
        
        if obstacle_type == 'cylinder':
            # Update cylinder center_x - use jax array to match original format
            self.solver.geom.center_x = jnp.array(x_position)
        elif obstacle_type == 'naca_airfoil':
            # Update NACA x-position
            if hasattr(self.solver.sim_params, 'naca_x'):
                self.solver.sim_params.naca_x = x_position
        elif obstacle_type == 'cow':
            # Update cow x-position
            if hasattr(self.solver.sim_params, 'cow_x'):
                self.solver.sim_params.cow_x = x_position
            else:
                self.solver.sim_params.cow_x = x_position
        elif obstacle_type == 'three_cylinder_array':
            # Update cylinder array x-position
            if hasattr(self.solver.sim_params, 'cylinder_x'):
                self.solver.sim_params.cylinder_x = x_position
            else:
                self.solver.sim_params.cylinder_x = x_position
        
        # Recompute mask with new position
        self.solver.mask = self.solver._compute_mask()
        
        # Update obstacle outlines if renderer exists (force update for real-time slider feedback)
        if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer:
            self.obstacle_renderer.update_obstacle_outlines(self.solver, force_update=True)
    
    def apply_y_position_change(self, percentage: int) -> None:
        """Apply y-position change from slider (percentage of domain height)."""
        import jax.numpy as jnp
        obstacle_type = self.solver.sim_params.obstacle_type if hasattr(self.solver.sim_params, 'obstacle_type') else 'cylinder'
        grid_ly = self.solver.grid.ly
        y_position = (percentage / 100.0) * grid_ly
        
        if obstacle_type == 'cylinder':
            # Update cylinder center_y - use jax array to match original format
            self.solver.geom.center_y = jnp.array(y_position)
        elif obstacle_type == 'naca_airfoil':
            # Update NACA y-position
            if hasattr(self.solver.sim_params, 'naca_y'):
                self.solver.sim_params.naca_y = y_position
        elif obstacle_type == 'cow':
            # Update cow y-position
            if hasattr(self.solver.sim_params, 'cow_y'):
                self.solver.sim_params.cow_y = y_position
            else:
                self.solver.sim_params.cow_y = y_position
        elif obstacle_type == 'three_cylinder_array':
            # Update cylinder array y-position
            if hasattr(self.solver.sim_params, 'cylinder_y'):
                self.solver.sim_params.cylinder_y = y_position
            else:
                self.solver.sim_params.cylinder_y = y_position
        
        # Recompute mask with new position
        self.solver.mask = self.solver._compute_mask()
        
        # Update obstacle outlines if renderer exists (force update for real-time slider feedback)
        if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer:
            self.obstacle_renderer.update_obstacle_outlines(self.solver, force_update=True)
    
    def on_simulation_started(self) -> None:
        """Lock position sliders when simulation starts."""
        if hasattr(self.control_panel, 'x_position_slider'):
            self.control_panel.x_position_slider.setEnabled(False)
        if hasattr(self.control_panel, 'y_position_slider'):
            self.control_panel.y_position_slider.setEnabled(False)
    
    def on_simulation_stopped(self) -> None:
        """Unlock position sliders when simulation stops."""
        if hasattr(self.control_panel, 'x_position_slider'):
            self.control_panel.x_position_slider.setEnabled(True)
        if hasattr(self.control_panel, 'y_position_slider'):
            self.control_panel.y_position_slider.setEnabled(True)
    
    def apply_hyper_viscosity_change(self, nu_hyper_ratio: float) -> None:
        """Apply hyperviscosity ratio change to solver."""
        self.solver.nu_hyper_ratio = nu_hyper_ratio
        print(f"Hyper-viscosity ratio updated to {nu_hyper_ratio:.3f}")
        # Clear JIT cache since nu_hyper_ratio is a static argument
        import jax
        jax.clear_caches()
        if hasattr(self.solver, '_step_jit'):
            delattr(self.solver, '_step_jit')
        # Recreate _step_jit to ensure it exists for next step
        self.solver._step_jit = self.solver.get_step_jit()

    def apply_wall_boundary_condition(self, is_slip: bool) -> None:
        """Apply wall boundary condition change (slip vs no-slip)."""
        self.solver.slip_walls = is_slip
        print(f"Wall boundary condition: {'Slip' if is_slip else 'No-slip'}")
        # Clear JIT cache to recompile with new boundary condition
        import jax
        jax.clear_caches()
        if hasattr(self.solver, '_step_jit'):
            delattr(self.solver, '_step_jit')
        self.solver._step_jit = self.solver.get_step_jit()
