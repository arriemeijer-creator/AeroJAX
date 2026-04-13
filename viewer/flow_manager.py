"""
Flow type manager for the CFD viewer.
Handles flow type selection and application.
"""

import jax
import time
from typing import Optional


class FlowManager:
    """Mixin class providing flow type management methods for the viewer."""
    
    def on_flow_type_selected(self, flow_type: str) -> None:
        """Handle flow type selection changes."""
        if hasattr(self.control_panel, 'show_naca_controls'):
            if hasattr(self.control_panel, 'obstacle_combo'):
                self.control_panel.obstacle_combo.setCurrentText(self.solver.sim_params.obstacle_type)
        
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
            
            # Update advection scheme GUI based on flow type
            if selected_flow == 'lid_driven_cavity':
                # Restrict advection schemes for LDC to spectral and weno5 only
                current_scheme = self.solver.sim_params.advection_scheme
                if current_scheme not in ['spectral', 'weno5']:
                    print(f"LDC: Switching from {current_scheme} to weno5 (LDC-compatible)")
                    self.solver.sim_params.advection_scheme = 'weno5'
                    self.control_panel.scheme_combo.setCurrentText('weno5')
                # Update GUI to only show LDC-compatible schemes
                self.control_panel.scheme_combo.clear()
                self.control_panel.scheme_combo.addItems(['spectral', 'weno5'])
                self.control_panel.scheme_combo.setCurrentText('weno5')
            else:
                # Restore all schemes for non-LDC flows
                self.control_panel.scheme_combo.clear()
                self.control_panel.scheme_combo.addItems(['rk3', 'tvd', 'weno5', 'spectral'])
            
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
            max_chord = min(actual_lx * 0.3, actual_ly * 0.4)  # Max 30% of domain width, 40% of height
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
                self.sim_controller.start_simulation()
            
            self.control_panel.start_btn.setEnabled(False)
            self.control_panel.pause_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error changing flow type: {e}")
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def on_obstacle_type_selected(self, obstacle_type: str) -> None:
        """Handle obstacle type selection changes."""
        if hasattr(self.control_panel, 'show_naca_controls'):
            is_naca = obstacle_type == 'naca_airfoil'
            self.control_panel.show_naca_controls(is_naca)
            
            # Remove auto-apply - let user click Apply button manually
            # This prevents unwanted resets during angle updates
            print(f"Obstacle type changed to: {obstacle_type}")
            if is_naca:
                print("Click 'Apply NACA' button to apply changes")
