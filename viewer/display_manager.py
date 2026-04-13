"""
Display manager for the CFD viewer.
Handles display updates, visualization refresh, and info panel updates.
"""

import time
import numpy as np
from typing import Dict, Any


class DisplayManager:
    """Mixin class providing display update methods for the viewer."""
    
    def refresh_display(self) -> None:
        """Update the display with latest simulation data."""
        try:
            data = self.sim_controller.get_latest_data()
            if data is None:
                # Don't try to access removed info_label
                return
            
            # Get visualization data
            shared_buffers = data.get('shared_buffers', {})
            velocity_buffer = shared_buffers.get('vel_mag')
            vorticity_buffer = shared_buffers.get('vort')
            
            # Data is available - proceed with visualization update
            
            # Update plots with error handling
            try:
                self.flow_viz.update_visualization(
                    velocity_buffer.array if velocity_buffer else None,
                    vorticity_buffer.array if vorticity_buffer else None,
                    self.config.viz_config.show_velocity,
                    self.config.viz_config.show_vorticity
                )
                
                # Update scalar visualization if enabled
                if self.solver.sim_params.use_scalar and hasattr(self.solver, 'c'):
                    # Continuous dye injection while button is held
                    if self.inject_dye_pressed:
                        self.inject_dye_at_slider_position()
                    
                    scalar_data = np.array(self.solver.c)  # No transpose - use same orientation as velocity
                    # Add rect parameter to map grid coordinates to physical coordinates
                    rect = (0, 0, self.solver.grid.lx, self.solver.grid.ly)
                    self.flow_viz.scalar_img.setImage(scalar_data, levels=[0, 1], autoLevels=False, rect=rect)
                
                # Update L2 change plot (formerly L2 error)
                if hasattr(self.solver, 'history') and 'l2_change' in self.solver.history:
                    l2_changes = self.solver.history['l2_change']
                    l2_changes_u = self.solver.history.get('l2_change_u', [])
                    l2_changes_v = self.solver.history.get('l2_change_v', [])
                    times = self.solver.history['time']
                    if l2_changes and times:
                        # Get all change metrics from solver history
                        max_changes = self.solver.history.get('max_change', [])
                        rel_changes = self.solver.history.get('rel_change', [])
                        max_divergence = self.solver.history.get('max_divergence', [])
                        
                        # Ensure time and change arrays have same length
                        if len(times) >= len(l2_changes):
                            time_to_use = times[-1]
                        else:
                            time_to_use = times[-1] if times else 0.0
                        
                        # Only update error plot if metrics checkbox is checked
                        if hasattr(self, 'info_panel') and self.info_panel is not None and self.info_panel.diagnostics_checkbox.isChecked():
                            try:
                                # Always pass all change metrics, using L2 change as fallback for missing values
                                max_change = max_changes[-1] if max_changes and len(max_changes) > 0 else l2_changes[-1]
                                rel_change = rel_changes[-1] if rel_changes and len(rel_changes) > 0 else l2_changes[-1]
                                max_div = max_divergence[-1] if max_divergence and len(max_divergence) > 0 else 0.0
                                l2_change_u = l2_changes_u[-1] if l2_changes_u and len(l2_changes_u) > 0 else l2_changes[-1]
                                l2_change_v = l2_changes_v[-1] if l2_changes_v and len(l2_changes_v) > 0 else l2_changes[-1]

                                self.flow_viz.update_l2_error(
                                    l2_changes[-1], time_to_use,
                                    max_change, rel_change, l2_change_u, l2_change_v
                                )
                            except (IndexError, ValueError):
                                # Skip update if arrays are empty or mismatched
                                pass
                
                # Update InfoPanel with change metrics (copyable with Ctrl+C)
                if hasattr(self, 'info_panel') and self.info_panel is not None:
                    # Only update plots if metrics checkbox is checked
                    if self.info_panel.diagnostics_checkbox.isChecked():
                        try:
                            max_change = max_changes[-1] if max_changes and len(max_changes) > 0 else l2_changes[-1]
                            rel_change = rel_changes[-1] if rel_changes and len(rel_changes) > 0 else l2_changes[-1]
                            l2_change_u = l2_changes_u[-1] if l2_changes_u and len(l2_changes_u) > 0 else l2_changes[-1]
                            l2_change_v = l2_changes_v[-1] if l2_changes_v and len(l2_changes_v) > 0 else l2_changes[-1]
                            self.info_panel.update_error_metrics(
                                l2_changes[-1], max_change, rel_change, l2_change_u, l2_change_v
                            )
                        except (IndexError, ValueError):
                            # Skip update if arrays are empty or mismatched
                            pass
                
                # Update airfoil metrics if available
                if 'airfoil_metrics' in self.solver.history:
                    airfoil_data = self.solver.history['airfoil_metrics']
                    if airfoil_data['CL']:
                        airfoil_x = self.solver.sim_params.naca_x if hasattr(self.solver.sim_params, 'naca_x') else 2.5
                        chord_length = self.solver.sim_params.naca_chord if hasattr(self.solver.sim_params, 'naca_chord') else 3.0
                        self.info_panel.update_airfoil_metrics(
                            cl=airfoil_data['CL'][-1],
                            stagnation=airfoil_data['stagnation_x'][-1] if airfoil_data['stagnation_x'] else None,
                            separation=airfoil_data['separation_x'][-1] if airfoil_data['separation_x'] else None,
                            cp_min=airfoil_data['Cp_min'][-1] if airfoil_data['Cp_min'] else None,
                            wake_deficit=airfoil_data['wake_deficit'][-1] if airfoil_data['wake_deficit'] else None,
                            cd=airfoil_data['CD'][-1] if airfoil_data['CD'] else None,
                            airfoil_x=airfoil_x,
                            chord_length=chord_length
                        )
                        
            except Exception as viz_error:
                print(f"Warning: Visualization update failed: {viz_error}")
                # Continue running even if visualization fails
                
            # Update obstacle outlines periodically
            try:
                if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer:
                    self.obstacle_renderer.update_obstacle_outlines(self.solver)
            except Exception as outline_error:
                print(f"Warning: Outline update failed: {outline_error}")
                # Continue running even if outline update fails
                
        except Exception as refresh_error:
            print(f"Warning: Display refresh failed: {refresh_error}")
            # Don't crash the application, just continue
        
        # Update overlays with error handling
        try:
            if self.obstacle_renderer is not None:
                self.obstacle_renderer.update_obstacle_outlines(self.solver)
        except Exception as outline_error:
            print(f"Warning: Overlay outline update failed: {outline_error}")
            
        try:
            self.sdf_viz.update_sdf_visualization(self.solver)
        except Exception as sdf_error:
            print(f"Warning: SDF visualization update failed: {sdf_error}")

        # Update status information
        # Use control_panel labels if available
        if hasattr(self, 'control_panel') and self.control_panel is not None:
            self.control_panel.sim_time_label.setText(f"Time: {data['time']:.3f}")
            self.control_panel.dt_label.setText(f"dt: {self.solver.dt:.4f}")
            self.control_panel.max_div_label.setText(f"{data['max_divergence']:.6f}")

            # Update dt_spinbox to show current adaptive dt when adaptive dt is enabled
            if self.solver.sim_params.adaptive_dt:
                self.control_panel.dt_spinbox.blockSignals(True)
                self.control_panel.dt_spinbox.setValue(self.solver.dt)
                self.control_panel.dt_spinbox.blockSignals(False)
        
        # Handle video recording
        if self.recording_manager.is_recording and velocity_buffer is not None:
            self.recording_manager.capture_frame(velocity_buffer.array)
        
        # Update frame rate counter
        self._update_frame_rate_display()
        
        # Clear update flag for next frame
        self.sim_controller.reset_update_flag()
    
    def _update_frame_rate_display(self) -> None:
        """Calculate and display the current visualization frame rate."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_update_time)
            # Use control_panel label for viz FPS
            if hasattr(self, 'control_panel') and self.control_panel is not None:
                self.control_panel.viz_fps_label.setText(f"Vis FPS: {fps:.1f}")
            elif hasattr(self, 'info_panel') and self.info_panel is not None:
                self.info_panel.viz_fps_label.setText(f"Vis FPS: {fps:.1f}")
            
            # Update plot titles with both sim and viz FPS
            if hasattr(self, 'flow_viz') and self.flow_viz:
                self.flow_viz.update_plot_titles_with_fps(self.current_sim_fps, fps)
            
            self.frame_count = 0
            self.last_fps_update_time = current_time
    
    def _update_solver_info(self) -> None:
        """Refresh the solver configuration display."""
        pressure_solver = self.solver.sim_params.pressure_solver
        advection_scheme = self.solver.sim_params.advection_scheme
        # Use control_panel label for solver status
        if hasattr(self, 'control_panel') and self.control_panel is not None:
            self.control_panel.solver_status_label.setText(
                f"Solver: {pressure_solver} | Scheme: {advection_scheme}"
            )
        elif hasattr(self, 'info_panel') and self.info_panel is not None:
            self.info_panel.solver_label.setText(
                f"Solver: {pressure_solver} | Scheme: {advection_scheme}"
            )
    
    def update_simulation_fps_display(self, fps: float) -> None:
        """Update the simulation frame rate display."""
        # Store sim FPS for plot titles
        self.current_sim_fps = fps
        # Use control_panel label for sim FPS
        if hasattr(self, 'control_panel') and self.control_panel is not None:
            self.control_panel.sim_fps_label.setText(f"Sim FPS: {fps}")
        elif hasattr(self, 'info_panel') and self.info_panel is not None:
            self.info_panel.sim_fps_label.setText(f"Sim FPS: {fps}")
    
    def handle_metrics_data(self, metrics_data: Dict[str, Any]) -> None:
        """Process metrics data from the separate metrics thread."""
        error_metrics = metrics_data.get('error_metrics')
        airfoil_metrics = metrics_data.get('airfoil_metrics')
        
        # Update solver history with async metrics
        if error_metrics:
            self.solver.history['l2_change'].append(error_metrics['l2_change'])
            self.solver.history['l2_change_u'].append(error_metrics['l2_change_u'])
            self.solver.history['l2_change_v'].append(error_metrics['l2_change_v'])
            self.solver.history['max_change'].append(error_metrics['max_change'])
            self.solver.history['rel_change'].append(error_metrics['rel_change'])
            self.solver.history['max_divergence'].append(error_metrics['max_divergence'])
            self.solver.history['l2_divergence'].append(error_metrics['l2_divergence'])
        
        if airfoil_metrics:
            self.solver.history['airfoil_metrics']['CL'].append(airfoil_metrics['CL'])
            self.solver.history['airfoil_metrics']['CD'].append(airfoil_metrics['CD'])
            self.solver.history['airfoil_metrics']['stagnation_x'].append(airfoil_metrics['stagnation_x'])
            self.solver.history['airfoil_metrics']['separation_x'].append(airfoil_metrics['separation_x'])
            self.solver.history['airfoil_metrics']['Cp_min'].append(airfoil_metrics['Cp_min'])
            self.solver.history['airfoil_metrics']['wake_deficit'].append(airfoil_metrics['wake_deficit'])
    
    def handle_simulation_data(self, data: Dict[str, Any]) -> None:
        """Process new data from the simulation engine."""
        # Debug LDC flow progress with explosion detection
        if hasattr(self.solver, 'sim_params') and self.solver.sim_params.flow_type == 'lid_driven_cavity':
            iteration = data.get('iteration', 0)
            u_data = data.get('u', [])
            v_data = data.get('v', [])
            
            if u_data is not None and v_data is not None and len(u_data) > 0:
                # Check for NaN values (explosion indicator)
                u_has_nan = np.any(np.isnan(u_data))
                v_has_nan = np.any(np.isnan(v_data))
                
                if u_has_nan or v_has_nan:
                    print(f"🚨 LDC EXPLOSION DETECTED at iteration {iteration}!")
                    print(f"   u_has_nan: {u_has_nan}, v_has_nan: {v_has_nan}")
                    print(f"   Adaptive dt: {self.solver.sim_params.adaptive_dt}")
                    print(f"   Current dt: {self.solver.dt}")
                    print(f"   Reynolds number: {self.solver.flow.Re}")
                    # Stop simulation immediately
                    self.sim_controller.stop_simulation()
                    return
                
                # Check for explosion (very large values)
                u_max = float(np.max(np.abs(u_data)))
                v_max = float(np.max(np.abs(v_data)))
                
                if u_max > 100.0 or v_max > 100.0:  # Explosion threshold
                    print(f"🚨 LDC EXPLOSION DETECTED at iteration {iteration}!")
                    print(f"   u_max: {u_max:.6f}, v_max: {v_max:.6f}")
                    print(f"   Adaptive dt: {self.solver.sim_params.adaptive_dt}")
                    print(f"   Current dt: {self.solver.dt}")
                    print(f"   Reynolds number: {self.solver.flow.Re}")
                    # Stop simulation immediately
                    self.sim_controller.stop_simulation()
                    return
                
                # Regular debugging every 10 iterations
                if iteration % 10 == 0:
                    u_mean = float(np.mean(u_data))
                    v_mean = float(np.mean(v_data))
                    print(f"DEBUG LDC Iter {iteration}: u_max={u_max:.6f}, v_max={v_max:.6f}, u_mean={u_mean:.6f}, v_mean={v_mean:.6f}")
        self.sim_controller.on_simulation_data_ready(data)
