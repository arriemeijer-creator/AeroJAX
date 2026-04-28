"""
Display manager for the CFD viewer.
Handles display updates, visualization refresh, and info panel updates.
"""

import time
import numpy as np
import csv
import os
from typing import Dict, Any


class DisplayManager:
    """Mixin class providing display update methods for the viewer."""

    def _init_csv_logging(self):
        """Initialize CSV logging for metrics."""
        self.csv_log_interval = 10  # Log every 10 airfoil metric samples
        self.airfoil_metric_count = 0
        
        # Visualization profiling data storage
        self.viz_profiling_data = {
            'viz_total': 0.0,
            'velocity': 0.0,
            'vorticity': 0.0,
            'pressure': 0.0
        }

        # Get angle of attack from NACA airfoil parameters if available
        if hasattr(self.solver, 'sim_params') and hasattr(self.solver.sim_params, 'naca_angle'):
            aoa = self.solver.sim_params.naca_angle
            print(f"DEBUG CSV: Using naca_angle = {aoa}")
        elif hasattr(self.solver, 'sim_params') and hasattr(self.solver.sim_params, 'angle_of_attack'):
            aoa = self.solver.sim_params.angle_of_attack
            print(f"DEBUG CSV: Using angle_of_attack = {aoa}")
        else:
            aoa = 10.0
            print(f"DEBUG CSV: Using default angle = {aoa}")

        self.csv_file_path = f"AoA={aoa:.0f}.csv"

        # Create CSV file with headers only if it doesn't exist
        headers = [
            'Time', 'Iteration', 'Sim FPS',
            'L2 Error', 'Max Error', 'Rel Error',
            'L2 U Error', 'L2 V Error',
            'CL', 'CD', 'Stagnation', 'Separation', 'Cp_min', 'Wake Deficit',
            'Strouhal', 'Avg CL (t=8s)', 'Avg CD (t=8s)'
        ]

        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"CSV logging initialized: {self.csv_file_path}")
        else:
            print(f"CSV logging appending to existing file: {self.csv_file_path}")

    def _log_metrics_to_csv(self, error_metrics, airfoil_metrics, current_time, iteration, sim_fps):
        """Log current metrics to CSV file."""
        try:
            from solver.metrics import detect_strouhal_stability

            # Pressure diagnostics (runs in main thread)
            if hasattr(self.solver, 'current_pressure') and hasattr(self.solver, 'grid'):
                p = self.solver.current_pressure
                u = self.solver.u if hasattr(self.solver, 'u') else None
                v = self.solver.v if hasattr(self.solver, 'v') else None

                print(f"\n=== PRESSURE DIAGNOSTICS (iter {iteration}) ===")
                print(f"P min: {p.min():.8f}, max: {p.max():.8f}, mean: {p.mean():.8f}")

                if u is not None and v is not None:
                    # Check grid type and use appropriate operators
                    grid_type = getattr(self.solver.sim_params, 'grid_type', 'collocated')
                    if grid_type == 'mac':
                        from solver.operators_mac import divergence_nonperiodic_staggered, vorticity_nonperiodic_staggered
                        div = divergence_nonperiodic_staggered(u, v, self.solver.grid.dx, self.solver.grid.dy)
                        vort = vorticity_nonperiodic_staggered(u, v, self.solver.grid.dx, self.solver.grid.dy)
                        # Skip du_dx and dv_dy for MAC grid diagnostics
                        du_dx = None
                        dv_dy = None
                    else:
                        from solver.operators import divergence_nonperiodic, grad_x_nonperiodic, grad_y_nonperiodic, vorticity_nonperiodic
                        div = divergence_nonperiodic(u, v, self.solver.grid.dx, self.solver.grid.dy)
                        du_dx = grad_x_nonperiodic(u, self.solver.grid.dx)
                        dv_dy = grad_y_nonperiodic(v, self.solver.grid.dy)
                        vort = vorticity_nonperiodic(u, v, self.solver.grid.dx, self.solver.grid.dy)
                    
                    dt = self.solver.dt
                    rhs = div / dt
                    print(f"Div min: {div.min():.8f}, max: {div.max():.8f}, mean: {div.mean():.8f}")
                    print(f"Div max abs: {np.max(np.abs(div)):.6f}")
                    print(f"u max: {u.max():.4f}, v max: {v.max():.4f}")
                    if du_dx is not None and dv_dy is not None:
                        print(f"du/dx max: {du_dx.max():.4f}, dv/dy max: {dv_dy.max():.4f}")
                    print(f"dt: {dt:.6f}")

                    # Compute circulation-based CL
                    if hasattr(self.solver, 'mask') and hasattr(self.solver.flow, 'U_inf'):
                        from solver.metrics import compute_CL_circulation
                        mask = self.solver.mask
                        chord = getattr(self.solver.sim_params, 'naca_chord', 3.0)
                        cl_circ = float(compute_CL_circulation(vort, mask, self.solver.grid.dx, self.solver.grid.dy,
                                                                self.solver.flow.U_inf, chord, fluid_threshold=0.95))
                        print(f"CL_circulation: {cl_circ:.4f}")

            # Calculate Strouhal and time-averaged values
            cl_history = np.array(self.solver.history['airfoil_metrics']['CL'])
            cd_history = np.array(self.solver.history['airfoil_metrics']['CD'])
            # Use stored time for airfoil metrics
            times_for_cl = np.array(self.solver.history['airfoil_metrics'].get('time', []))

            # Calculate Strouhal and time-averaged values
            if len(cl_history) > 0 and len(times_for_cl) > 0 and len(times_for_cl) == len(cl_history):
                if len(cl_history) > 20:
                    is_stable, strouhal, dominant_freq = detect_strouhal_stability(
                        cl_history, times_for_cl,
                        self.solver.flow.U_inf,
                        getattr(self.solver.sim_params, 'naca_chord', 3.0)
                    )
                else:
                    strouhal = 0.0

                # Calculate time-averaged CL/CD from t=8s
                start_time = 8.0
                if current_time > start_time:
                    start_idx = np.searchsorted(times_for_cl, start_time)
                    if start_idx < len(cl_history):
                        avg_cl = np.mean(cl_history[start_idx:])
                        avg_cd = np.mean(cd_history[start_idx:])
                    else:
                        avg_cl = np.mean(cl_history)
                        avg_cd = np.mean(cd_history)
                else:
                    avg_cl = np.mean(cl_history)
                    avg_cd = np.mean(cd_history)
            else:
                strouhal = 0.0
                avg_cl = airfoil_metrics['CL'] if airfoil_metrics else 0.0
                avg_cd = airfoil_metrics['CD'] if airfoil_metrics else 0.0

            # Format stagnation and separation
            if airfoil_metrics:
                stagnation_x = airfoil_metrics['stagnation_x']
                separation_x = airfoil_metrics['separation_x']
                chord_length = getattr(self.solver.sim_params, 'naca_chord', 3.0)
                if stagnation_x is not None and chord_length > 0:
                    stag_rel = stagnation_x / chord_length
                else:
                    stag_rel = 0.0
                if separation_x is not None and chord_length > 0:
                    sep_rel = separation_x / chord_length
                else:
                    sep_rel = 0.0
            else:
                stag_rel = 0.0
                sep_rel = 0.0

            # Prepare row data
            row = [
                f"{current_time:.3f}",
                iteration,
                f"{sim_fps:.1f}",
                f"{error_metrics['l2_change']:.3e}" if error_metrics else "N/A",
                f"{error_metrics['max_change']:.3e}" if error_metrics else "N/A",
                f"{error_metrics['rel_change']:.3%}" if error_metrics else "N/A",
                f"{error_metrics['l2_change_u']:.3e}" if error_metrics else "N/A",
                f"{error_metrics['l2_change_v']:.3e}" if error_metrics else "N/A",
                f"{airfoil_metrics['CL']:.3f}" if airfoil_metrics else "N/A",
                f"{airfoil_metrics['CD']:.3f}" if airfoil_metrics else "N/A",
                f"{stag_rel:.3f}",
                f"{sep_rel:.3f}",
                f"{airfoil_metrics['Cp_min']:.2f}" if airfoil_metrics else "N/A",
                f"{airfoil_metrics['wake_deficit']:.3f}" if airfoil_metrics else "N/A",
                f"{strouhal:.3f}",
                f"{avg_cl:.3f}",
                f"{avg_cd:.3f}"
            ]

            # Write to CSV
            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"Logged metrics to CSV at t={current_time:.2f}s (iteration {iteration})")

        except Exception as e:
            print(f"Error logging to CSV: {e}")
            import traceback
            traceback.print_exc()

    def save_csv_to_file(self, file_path):
        """Save all current metrics history to a CSV file."""
        try:
            from solver.metrics import detect_strouhal_stability

            # Write metadata header
            metadata_rows = [
                f"# NACA Airfoil Type: {getattr(self.solver.sim_params, 'naca_airfoil', 'N/A')}",
                f"# Angle of Attack: {getattr(self.solver.sim_params, 'naca_angle', 0.0):.1f}°",
                f"# Reynolds Number: {self.solver.flow.Re:.0f}",
                f"# U_inf: {self.solver.flow.U_inf:.2f} m/s",
                f"# Grid Size: {self.solver.grid.nx}x{self.solver.grid.ny}",
                f"# LES: {'ON' if getattr(self.solver.sim_params, 'use_les', False) else 'OFF'}",
                ""
            ]

            # Headers
            headers = [
                'Time', 'Iteration', 'Sim FPS',
                'L2 Error', 'Max Error', 'Rel Error',
                'L2 U Error', 'L2 V Error',
                'CL', 'CD', 'Stagnation (x/c)', 'Separation (x/c)', 'Cp_min', 'Wake Deficit',
                'Strouhal', 'Avg CL (t=8s)', 'Avg CD (t=8s)'
            ]

            # Get all data from history
            times = self.solver.history['time']
            l2_errors = self.solver.history['l2_change']
            max_errors = self.solver.history['max_change']
            rel_errors = self.solver.history['rel_change']
            l2_u_errors = self.solver.history['l2_change_u']
            l2_v_errors = self.solver.history['l2_change_v']

            cl_history = np.array(self.solver.history['airfoil_metrics']['CL'])
            cd_history = np.array(self.solver.history['airfoil_metrics']['CD'])
            stagnation_history = self.solver.history['airfoil_metrics']['stagnation_x']
            separation_history = self.solver.history['airfoil_metrics']['separation_x']
            cp_min_history = self.solver.history['airfoil_metrics']['Cp_min']
            wake_deficit_history = self.solver.history['airfoil_metrics']['wake_deficit']
            # Use stored time for airfoil metrics
            times_for_cl = np.array(self.solver.history['airfoil_metrics'].get('time', []))

            chord_length = getattr(self.solver.sim_params, 'naca_chord', 3.0)
            naca_x = getattr(self.solver.sim_params, 'naca_x', 0.0)

            # Only save rows where airfoil metrics exist
            num_airfoil_metrics = len(cl_history)
            rows = []

            # Calculate Strouhal and time-averaged values once for the full history
            if num_airfoil_metrics > 20 and len(times_for_cl) > 0 and len(times_for_cl) == len(cl_history):
                is_stable, strouhal, dominant_freq = detect_strouhal_stability(
                    cl_history, times_for_cl,
                    self.solver.flow.U_inf,
                    chord_length
                )

                # Time-averaged from t=8s
                start_time = 8.0
                start_idx = np.searchsorted(times_for_cl, start_time)
                if start_idx < len(cl_history) and start_idx > 0:
                    avg_cl = np.mean(cl_history[start_idx:])
                    avg_cd = np.mean(cd_history[start_idx:])
                else:
                    avg_cl = np.mean(cl_history)
                    avg_cd = np.mean(cd_history)
            else:
                strouhal = 0.0
                avg_cl = np.mean(cl_history) if len(cl_history) > 0 else 0.0
                avg_cd = np.mean(cd_history) if len(cd_history) > 0 else 0.0

            # Build rows only for time points where airfoil metrics exist
            for i in range(num_airfoil_metrics):
                # Map airfoil metric index to time index
                time_ratio = len(times) / num_airfoil_metrics
                time_idx = int(i * time_ratio)
                if time_idx >= len(times):
                    time_idx = len(times) - 1

                t = times[time_idx]

                # Get corresponding error metrics
                l2_err = l2_errors[time_idx] if time_idx < len(l2_errors) else "N/A"
                max_err = max_errors[time_idx] if time_idx < len(max_errors) else "N/A"
                rel_err = rel_errors[time_idx] if time_idx < len(rel_errors) else "N/A"
                l2_u_err = l2_u_errors[time_idx] if time_idx < len(l2_u_errors) else "N/A"
                l2_v_err = l2_v_errors[time_idx] if time_idx < len(l2_v_errors) else "N/A"

                # Get airfoil metrics
                cl = cl_history[i] if i < len(cl_history) else "N/A"
                cd = cd_history[i] if i < len(cd_history) else "N/A"
                stag = stagnation_history[i] if i < len(stagnation_history) else None
                sep = separation_history[i] if i < len(separation_history) else None
                cp_min = cp_min_history[i] if i < len(cp_min_history) else "N/A"
                wake = wake_deficit_history[i] if i < len(wake_deficit_history) else "N/A"

                # Format stagnation and separation normalized to chord length (0 to 1)
                stag_rel = (stag - naca_x) / chord_length if stag is not None and chord_length > 0 else 0.0
                sep_rel = (sep - naca_x) / chord_length if sep is not None and chord_length > 0 else 0.0

                # Sim FPS - use current value
                sim_fps = getattr(self, 'current_sim_fps', 0.0)

                row = [
                    f"{t:.3f}",
                    time_idx,
                    f"{sim_fps:.1f}",
                    f"{l2_err:.3e}" if isinstance(l2_err, float) else l2_err,
                    f"{max_err:.3e}" if isinstance(max_err, float) else max_err,
                    f"{rel_err:.3%}" if isinstance(rel_err, float) else rel_err,
                    f"{l2_u_err:.3e}" if isinstance(l2_u_err, float) else l2_u_err,
                    f"{l2_v_err:.3e}" if isinstance(l2_v_err, float) else l2_v_err,
                    f"{cl:.3f}" if isinstance(cl, float) else cl,
                    f"{cd:.3f}" if isinstance(cd, float) else cd,
                    f"{stag_rel:.3f}",
                    f"{sep_rel:.3f}",
                    f"{cp_min:.2f}" if isinstance(cp_min, float) else cp_min,
                    f"{wake:.3f}" if isinstance(wake, float) else wake,
                    f"{strouhal:.3f}",
                    f"{avg_cl:.3f}",
                    f"{avg_cd:.3f}"
                ]
                rows.append(row)

            # Write to file
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write metadata header
                for row in metadata_rows:
                    f.write(row + '\n')
                # Write column headers
                writer.writerow(headers)
                # Write data rows
                writer.writerows(rows)

            print(f"Saved {len(rows)} metrics rows to {file_path}")
            return True

        except Exception as e:
            print(f"Error saving CSV to file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def refresh_display(self) -> None:
        """Update the display with latest simulation data."""
        import time
        t_start = time.time()
        
        try:
            data = self.sim_controller.get_latest_data()
            if data is None:
                # Don't try to access removed info_label
                return
            
            t_get_data = time.time()
            
            # TEMPORARY: Get arrays directly instead of from shared buffers
            vel_mag_data = data.get('vel_mag')
            vort_data = data.get('vort')
            div_data = data.get('div')
            
            # Get visualization data (commented out temporarily)
            # shared_buffers = data.get('shared_buffers', {})
            # velocity_buffer = shared_buffers.get('vel_mag')
            # vorticity_buffer = shared_buffers.get('vort')
            
            t_get_buffers = time.time()
            
            # Data is available - proceed with visualization update
            
            # Update plots with error handling
            try:
                # Get pressure data from solver
                pressure_data = self.solver.current_pressure if hasattr(self.solver, 'current_pressure') else None

                t_update_viz = time.time()
                self.flow_viz.update_visualization(
                    vel_mag_data if vel_mag_data is not None else None,
                    vort_data if vort_data is not None else None,
                    pressure_data,
                    div_data if div_data is not None else None,
                    self.config.viz_config.show_velocity,
                    self.config.viz_config.show_vorticity,
                    self.config.viz_config.show_pressure,
                    self.config.viz_config.show_dye
                )
                t_after_viz = time.time()
            except Exception as viz_error:
                import traceback
                print(f"Warning: Visualization update failed: {viz_error}")
                traceback.print_exc()
                # Continue running even if visualization fails
                
                # Update scalar visualization (dye)
                t_dye_start = time.time()
                if self.config.viz_config.show_dye and hasattr(self.solver, 'c'):
                    # Continuous dye injection while button is held
                    if self.inject_dye_pressed:
                        self.inject_dye_at_slider_position()

                    if self.flow_viz.use_particles:
                        # Update particles with velocity field
                        # Update multiple times to match simulation speed (frame skip)
                        frame_skip = self.sim_controller.update_every if hasattr(self.sim_controller, 'update_every') else 1
                        u_np = np.array(self.solver.u)
                        v_np = np.array(self.solver.v)
                        X_np = np.array(self.solver.grid.X)
                        Y_np = np.array(self.solver.grid.Y)
                        domain_bounds = (0, self.solver.grid.lx, 0, self.solver.grid.ly)
                        
                        # Update particles multiple times to match simulation steps
                        for _ in range(frame_skip):
                            self.flow_viz.update_particles(u_np, v_np, X_np, Y_np, 
                                                         self.solver.grid.dx, self.solver.grid.dy, 
                                                         self.solver.dt, domain_bounds)
                    else:
                        # Update dye field
                        scalar_data = np.array(self.solver.c)  # No transpose - use same orientation as velocity
                        # Add rect parameter to map grid coordinates to physical coordinates
                        rect = (0, 0, self.solver.grid.lx, self.solver.grid.ly)
                        self.flow_viz.scalar_img.setImage(scalar_data, levels=[0, 1], autoLevels=False, rect=rect)
                t_dye_end = time.time()
                
                # Update L2 change plot (formerly L2 error)
                t_l2_start = time.time()
                if hasattr(self.solver, 'history') and 'l2_change' in self.solver.history:
                    l2_changes = self.solver.history['l2_change']
                    rms_changes = self.solver.history.get('rms_change', [])
                    change_99p_list = self.solver.history.get('change_99p', [])
                    l2_changes_u = self.solver.history.get('l2_change_u', [])
                    l2_changes_v = self.solver.history.get('l2_change_v', [])
                    times = self.solver.history['time']
                    # Initialize change metrics to prevent UnboundLocalError
                    max_changes = []
                    rel_changes = []
                    rms_divergence = []
                    time_to_use = 0.0

                    if l2_changes and times:
                        # Get all change metrics from solver history
                        max_changes = self.solver.history.get('max_change', [])
                        rel_changes = self.solver.history.get('rel_change', [])
                        rms_divergence = self.solver.history.get('rms_divergence', [])

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
                                rms_div = rms_divergence[-1] if rms_divergence and len(rms_divergence) > 0 else 0.0
                                l2_change_u = l2_changes_u[-1] if l2_changes_u and len(l2_changes_u) > 0 else l2_changes[-1]
                                l2_change_v = l2_changes_v[-1] if l2_changes_v and len(l2_changes_v) > 0 else l2_changes[-1]
                                rms_change = rms_changes[-1] if rms_changes and len(rms_changes) > 0 else 0.0
                                change_99p = change_99p_list[-1] if change_99p_list and len(change_99p_list) > 0 else 0.0

                                self.flow_viz.update_l2_error(
                                    l2_changes[-1], time_to_use,
                                    max_change, rel_change, l2_change_u, l2_change_v,
                                    rms_change, change_99p
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
                            rms_change = rms_changes[-1] if rms_changes and len(rms_changes) > 0 else 0.0
                            change_99p = change_99p_list[-1] if change_99p_list and len(change_99p_list) > 0 else 0.0
                            self.info_panel.update_error_metrics(
                                l2_changes[-1], rms_change, max_change, change_99p, rel_change, l2_change_u, l2_change_v
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

                        # Get time-averaged values if available
                        avg_cl = None
                        avg_cd = None
                        strouhal = None
                        if 'time_averaged' in self.solver.history and self.solver.history['time_averaged']['CL']:
                            avg_cl = self.solver.history['time_averaged']['CL'][-1]
                            avg_cd = self.solver.history['time_averaged']['CD'][-1]
                            strouhal = self.solver.history['time_averaged']['strouhal'][-1]

                        # Determine obstacle type for display - check sim_params first
                        flow_type = getattr(self.solver.sim_params, 'flow_type', 'von_karman')
                        if flow_type == 'von_karman':
                            # Check sim_params.obstacle_type first for explicit obstacle type
                            obstacle_type = getattr(self.solver.sim_params, 'obstacle_type', 'cylinder')
                            
                            # If obstacle_type is naca_airfoil, verify with naca_airfoil param
                            if obstacle_type == 'naca_airfoil':
                                naca_airfoil = getattr(self.solver.sim_params, 'naca_airfoil', None)
                                if naca_airfoil is None or not naca_airfoil.startswith('NACA'):
                                    # Fallback to geometry check if naca_airfoil param is invalid
                                    if hasattr(self.solver.geom, 'radius') and self.solver.geom.radius > 0:
                                        obstacle_type = 'cylinder'
                        else:
                            obstacle_type = getattr(self.solver.sim_params, 'obstacle_type', 'unknown')

                        # Get cylinder diameter and center if needed
                        cylinder_diameter = 0.36
                        cylinder_center_x = 5.0
                        if obstacle_type == 'cylinder':
                            radius = getattr(self.solver.geom, 'radius', 0.18)
                            cylinder_diameter = float(2.0 * radius)
                            # Handle both 0D and 1D arrays for center_x
                            center_x_array = self.solver.geom.center_x
                            if hasattr(center_x_array, 'ndim') and center_x_array.ndim > 0:
                                cylinder_center_x = float(center_x_array[0])
                            else:
                                cylinder_center_x = float(center_x_array)

                        self.info_panel.update_airfoil_metrics(
                            cl=airfoil_data['CL'][-1],
                            stagnation=airfoil_data['stagnation_x'][-1] if airfoil_data['stagnation_x'] else None,
                            separation=airfoil_data['separation_x'][-1] if airfoil_data['separation_x'] else None,
                            cp_min=airfoil_data['Cp_min'][-1] if airfoil_data['Cp_min'] else None,
                            wake_deficit=airfoil_data['wake_deficit'][-1] if airfoil_data['wake_deficit'] else None,
                            cd=airfoil_data['CD'][-1] if airfoil_data['CD'] else None,
                            avg_cl=avg_cl,
                            avg_cd=avg_cd,
                            strouhal=strouhal,
                            airfoil_x=airfoil_x,
                            chord_length=chord_length,
                            obstacle_type=obstacle_type,
                            cylinder_diameter=cylinder_diameter,
                            cylinder_center_x=cylinder_center_x
                        )
                t_l2_end = time.time()
                        
            except Exception as viz_error:
                print(f"Warning: Visualization update failed: {viz_error}")
                # Continue running even if visualization fails
                
            # Update obstacle outlines periodically
            t_outline_start = time.time()
            try:
                if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer:
                    self.obstacle_renderer.update_obstacle_outlines(self.solver)
            except Exception as outline_error:
                print(f"Warning: Outline update failed: {outline_error}")
                # Continue running even if outline update fails
            t_outline_end = time.time()
                
        except Exception as viz_error:
            print(f"Warning: Visualization update failed: {viz_error}")
            # Continue running even if visualization fails
                
        except Exception as refresh_error:
            print(f"Warning: Display refresh failed: {refresh_error}")
            # Don't crash the application, just continue
        
        # Update overlays with error handling
        t_overlay_start = time.time()
        try:
            if self.obstacle_renderer is not None:
                self.obstacle_renderer.update_obstacle_outlines(self.solver)
        except Exception as outline_error:
            print(f"Warning: Overlay outline update failed: {outline_error}")
        t_overlay_end = time.time()
            
        t_sdf_start = time.time()
        try:
            self.sdf_viz.update_sdf_visualization(self.solver)
        except Exception as sdf_error:
            print(f"Warning: SDF visualization update failed: {sdf_error}")
        t_sdf_end = time.time()
        
        t_total = time.time() - t_start
        
        # Print profiling every 60 frames
        if not hasattr(self, '_viz_frame_count'):
            self._viz_frame_count = 0
        self._viz_frame_count += 1

        # Update status information
        # Use control_panel labels if available
        if hasattr(self, 'control_panel') and self.control_panel is not None:
            self.control_panel.sim_time_label.setText(f"Time: {data['time']:.3f}")
            self.control_panel.dt_label.setText(f"dt: {self.solver.dt:.4f}")
            self.control_panel.max_div_label.setText(f"RMS Divergence: {data['rms_divergence']:.6f}")

            # Update dt_spinbox to show current adaptive dt when adaptive dt is enabled
            if self.solver.sim_params.adaptive_dt:
                self.control_panel.dt_spinbox.blockSignals(True)
                self.control_panel.dt_spinbox.setValue(self.solver.dt)
                self.control_panel.dt_spinbox.blockSignals(False)

        # Also update info_panel labels if available (for copy button)
        if hasattr(self, 'info_panel') and self.info_panel is not None:
            self.info_panel.div_label.setText(f"Divergence: {data['rms_divergence']:.6f}")
            # Update dt value
            self.info_panel.dt_value_label.setText(f"DT: {self.solver.dt:.6f}")
            # Update dt mode
            dt_mode = "Adaptive" if self.solver.sim_params.adaptive_dt else "Fixed"
            self.info_panel.dt_mode_label.setText(f"DT Mode: {dt_mode}")
        
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
        # Also update info_panel label for copy button
        if hasattr(self, 'info_panel') and self.info_panel is not None:
            self.info_panel.solver_label.setText(
                f"Solver: {pressure_solver} | Scheme: {advection_scheme}"
            )
            # Update additional solver info
            self.info_panel.grid_size_label.setText(f"Grid: {self.solver.grid.nx}x{self.solver.grid.ny}")
            self.info_panel.reynolds_label.setText(f"Re: {self.solver.flow.Re:.0f}")
            self.info_panel.advection_scheme_label.setText(f"Scheme: {advection_scheme}")

            # Get hyper-viscosity from control panel
            if hasattr(self.control_panel, 'hyper_viscosity_label'):
                hyper_visc = self.control_panel.hyper_viscosity_label.text()
                self.info_panel.hyper_viscosity_label.setText(f"Hyper ν: {hyper_visc}")

            # Get mask epsilon from solver (actual value used)
            actual_epsilon = getattr(self.solver.sim_params, 'eps', 0.01)
            self.info_panel.mask_epsilon_label.setText(f"Mask ε: {actual_epsilon:.2e}")

            # Get LES status from control panel
            if hasattr(self.control_panel, 'les_checkbox'):
                les_enabled = self.control_panel.les_checkbox.isChecked()
                les_status = "On" if les_enabled else "Off"
                self.info_panel.les_status_label.setText(f"LES: {les_status}")
                if les_enabled:
                    les_model = self.control_panel.les_model_combo.currentText()
                    self.info_panel.les_model_label.setText(f"LES Model: {les_model}")
                else:
                    self.info_panel.les_model_label.setText("LES Model: --")

            # Get obstacle info - check geometry directly for cylinder vs airfoil
            flow_type = getattr(self.solver.sim_params, 'flow_type', 'von_karman')
            if flow_type == 'von_karman':
                # Check sim_params.obstacle_type first for explicit obstacle type
                obstacle_type = getattr(self.solver.sim_params, 'obstacle_type', 'cylinder')
                
                # If obstacle_type is naca_airfoil, verify with naca_airfoil param
                if obstacle_type == 'naca_airfoil':
                    naca_airfoil = getattr(self.solver.sim_params, 'naca_airfoil', None)
                    if naca_airfoil is None or not naca_airfoil.startswith('NACA'):
                        # Fallback to geometry check if naca_airfoil param is invalid
                        if hasattr(self.solver.geom, 'radius') and self.solver.geom.radius > 0:
                            obstacle_type = 'cylinder'
            else:
                obstacle_type = getattr(self.solver.sim_params, 'obstacle_type', 'unknown')

            # Debug output
            print(f"DEBUG _update_solver_info: obstacle_type={obstacle_type}, flow_type={flow_type}")

            if obstacle_type == 'naca_airfoil':
                naca_airfoil = getattr(self.solver.sim_params, 'naca_airfoil', 'NACA 0012')
                naca_angle = getattr(self.solver.sim_params, 'naca_angle', 0.0)
                naca_chord = getattr(self.solver.sim_params, 'naca_chord', 0.0)
                naca_x = getattr(self.solver.sim_params, 'naca_x', 0.0)
                naca_y = getattr(self.solver.sim_params, 'naca_y', 0.0)
                self.info_panel.obstacle_type_label.setText(f"Obstacle: {naca_airfoil}")
                self.info_panel.obstacle_aoa_label.setText(f"AoA: {naca_angle:.1f}°")
                self.info_panel.obstacle_chord_label.setText(f"Chord: {naca_chord:.3f}")
                self.info_panel.obstacle_x_label.setText(f"X: {naca_x:.3f}")
                self.info_panel.obstacle_y_label.setText(f"Y: {naca_y:.3f}")
                self.info_panel.obstacle_diameter_label.setText("Diameter: N/A")
            elif obstacle_type == 'cylinder':
                radius = getattr(self.solver.geom, 'radius', 0.0)
                diameter = float(2.0 * radius)
                # Handle both 0D and 1D arrays for center_x and center_y
                center_x_array = self.solver.geom.center_x
                center_y_array = self.solver.geom.center_y
                # Use ndim to check if array is 0D or 1D
                if hasattr(center_x_array, 'ndim') and center_x_array.ndim > 0:
                    center_x = float(center_x_array[0])
                else:
                    center_x = float(center_x_array)
                if hasattr(center_y_array, 'ndim') and center_y_array.ndim > 0:
                    center_y = float(center_y_array[0])
                else:
                    center_y = float(center_y_array)
                self.info_panel.obstacle_type_label.setText("Obstacle: Cylinder")
                self.info_panel.obstacle_aoa_label.setText("AoA: N/A")
                self.info_panel.obstacle_chord_label.setText("Chord: N/A")
                self.info_panel.obstacle_x_label.setText(f"X: {center_x:.3f}")
                self.info_panel.obstacle_y_label.setText(f"Y: {center_y:.3f}")
                self.info_panel.obstacle_diameter_label.setText(f"Diameter: {diameter:.3f}")
            else:
                self.info_panel.obstacle_type_label.setText(f"Obstacle: {obstacle_type}")
                self.info_panel.obstacle_aoa_label.setText("AoA: N/A")
                self.info_panel.obstacle_chord_label.setText("Chord: N/A")
                self.info_panel.obstacle_x_label.setText("X: N/A")
                self.info_panel.obstacle_y_label.setText("Y: N/A")
                self.info_panel.obstacle_diameter_label.setText("Diameter: N/A")
    
    def update_simulation_fps_display(self, fps: float) -> None:
        """Update the simulation frame rate display."""
        # Store sim FPS for plot titles
        self.current_sim_fps = fps
        # Use control_panel label for sim FPS
        if hasattr(self, 'control_panel') and self.control_panel is not None:
            self.control_panel.sim_fps_label.setText(f"Sim FPS: {fps}")
        # Also update info_panel label for copy button
        if hasattr(self, 'info_panel') and self.info_panel is not None:
            self.info_panel.sim_fps_label.setText(f"Sim FPS: {fps}")
    
    def handle_profiling_update(self, solver_ms: float, interp_ms: float, total_ms: float, sim_fps: float) -> None:
        """Handle profiling data update and update overlay."""
        # Get visualization timing data if available
        viz_data = None
        if hasattr(self.flow_viz, '_latest_viz_timing'):
            viz_data = self.flow_viz._latest_viz_timing
        
        self.flow_viz.update_profiling_overlay(solver_ms, interp_ms, total_ms, sim_fps, viz_data)
    
    def toggle_profiling_overlay(self, state: int) -> None:
        """Toggle profiling overlay visibility."""
        visible = (state == 2)  # Qt.CheckState.Checked
        self.flow_viz.set_profiling_visible(visible)
    
    def handle_metrics_data(self, metrics_data: Dict[str, Any]) -> None:
        """Process metrics data from the separate metrics thread."""
        error_metrics = metrics_data.get('error_metrics')
        airfoil_metrics = metrics_data.get('airfoil_metrics')

        # Initialize CSV logging on first call
        if not hasattr(self, 'csv_file_path'):
            self._init_csv_logging()

        # Update solver history with async metrics
        if error_metrics:
            self.solver.history['l2_change'].append(error_metrics['l2_change'])
            self.solver.history['rms_change'].append(error_metrics['rms_change'])
            self.solver.history['l2_change_u'].append(error_metrics['l2_change_u'])
            self.solver.history['l2_change_v'].append(error_metrics['l2_change_v'])
            self.solver.history['max_change'].append(error_metrics['max_change'])
            self.solver.history['change_99p'].append(error_metrics['change_99p'])
            self.solver.history['rel_change'].append(error_metrics['rel_change'])
            self.solver.history['rms_divergence'].append(error_metrics['rms_divergence'])
            self.solver.history['l2_divergence'].append(error_metrics['l2_divergence'])

            # Increment counter and log to CSV every N samples (log from t=0)
            self.airfoil_metric_count += 1
            if self.airfoil_metric_count % self.csv_log_interval == 0:
                current_time = self.solver.history['time'][-1] if self.solver.history.get('time') else 0.0
                iteration = len(self.solver.history['time']) - 1
                sim_fps = getattr(self, 'current_sim_fps', 0.0)
                self._log_metrics_to_csv(error_metrics, airfoil_metrics, current_time, iteration, sim_fps)

        if airfoil_metrics:
            self.solver.history['airfoil_metrics']['CL'].append(airfoil_metrics['CL'])
            self.solver.history['airfoil_metrics']['CD'].append(airfoil_metrics['CD'])
            self.solver.history['airfoil_metrics']['stagnation_x'].append(airfoil_metrics['stagnation_x'])
            self.solver.history['airfoil_metrics']['separation_x'].append(airfoil_metrics['separation_x'])
            self.solver.history['airfoil_metrics']['Cp_min'].append(airfoil_metrics['Cp_min'])
            self.solver.history['airfoil_metrics']['wake_deficit'].append(airfoil_metrics['wake_deficit'])
            if 'strouhal' in airfoil_metrics:
                self.solver.history['airfoil_metrics']['strouhal'].append(airfoil_metrics['strouhal'])
            else:
                # Ensure strouhal list exists and append 0.0 if not provided
                if 'strouhal' not in self.solver.history['airfoil_metrics']:
                    self.solver.history['airfoil_metrics']['strouhal'] = []
                self.solver.history['airfoil_metrics']['strouhal'].append(0.0)

            # Update UI with airfoil metrics
            if hasattr(self, 'info_panel') and self.info_panel is not None:
                chord_length = getattr(self.solver.sim_params, 'naca_chord', 2.0)
                airfoil_x = getattr(self.solver.sim_params, 'naca_x', 2.5)
                obstacle_type = getattr(self.solver.sim_params, 'obstacle_type', 'naca_airfoil')
                cylinder_diameter = float(2.0 * self.solver.geom.radius) if hasattr(self.solver.geom, 'radius') else 0.36
                cylinder_center_x = float(self.solver.geom.center_x) if hasattr(self.solver.geom, 'center_x') else 5.0

                self.info_panel.update_airfoil_metrics(
                    cl=airfoil_metrics['CL'],
                    cd=airfoil_metrics['CD'],
                    stagnation=airfoil_metrics['stagnation_x'],
                    separation=airfoil_metrics['separation_x'],
                    cp_min=airfoil_metrics['Cp_min'],
                    wake_deficit=airfoil_metrics['wake_deficit'],
                    strouhal=airfoil_metrics.get('strouhal', 0.0),
                    airfoil_x=airfoil_x,
                    chord_length=chord_length,
                    obstacle_type=obstacle_type,
                    cylinder_diameter=cylinder_diameter,
                    cylinder_center_x=cylinder_center_x
                )

        # Update UI with error metrics
        if error_metrics and hasattr(self, 'info_panel') and self.info_panel is not None:
            self.info_panel.update_error_metrics(
                l2_error=error_metrics['l2_change'],
                rms_change=error_metrics['rms_change'],
                max_error=error_metrics['max_change'],
                change_99p=error_metrics['change_99p'],
                rel_error=error_metrics['rel_change'],
                l2_u_error=error_metrics['l2_change_u'],
                l2_v_error=error_metrics['l2_change_v']
            )

            # Update error metrics plot
            if hasattr(self, 'flow_viz') and self.flow_viz is not None:
                current_time = self.solver.history['time'][-1] if self.solver.history.get('time') else 0.0
                self.flow_viz.update_l2_error(
                    error_metrics['l2_change'],
                    current_time,
                    error_metrics['max_change'],
                    error_metrics['rel_change'],
                    error_metrics['l2_change_u'],
                    error_metrics['l2_change_v'],
                    error_metrics['rms_change'],
                    error_metrics['change_99p']
                )
    
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
