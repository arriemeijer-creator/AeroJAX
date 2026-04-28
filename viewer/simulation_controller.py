"""
Simulation controller for Baseline Navier-Stokes Viewer
Handles simulation threading, data management, and worker communication
"""

import sys
import time
import threading
import queue
import numpy as np
import jax.numpy as jnp
import jax
import multiprocessing.shared_memory as shm
from PyQt6.QtCore import QObject, pyqtSignal, Qt


class SharedData:
    """Shared memory buffer for zero-copy data transfer"""
    def __init__(self, shape, dtype=np.float32):
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)  # Convert to Python int
        self.shm = shm.SharedMemory(create=True, size=size)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
    
    def cleanup(self):
        """Clean up shared memory"""
        if hasattr(self, 'shm'):
            self.shm.close()
            self.shm.unlink()


class MetricsWorker(QObject):
    """Separate thread for metrics computation to avoid blocking simulation"""
    metrics_ready = pyqtSignal(dict)  # Signal when new metrics are ready

    def __init__(self, solver):
        super().__init__()
        self.solver = solver
        self.running = False
        self.paused = False
        self.data_queue = queue.Queue(maxsize=2)  # Buffer of 2 frames
        self.thread = None
        self.frame_count = 0

    def start(self):
        """Start the metrics computation thread"""
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self.run_metrics, daemon=True)
        self.thread.start()
        print("Metrics worker started")

    def stop(self):
        """Stop the metrics computation thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("Metrics worker stopped")

    def pause(self):
        """Pause the metrics computation"""
        self.paused = True

    def resume(self):
        """Resume the metrics computation"""
        self.paused = False

    def enqueue_data(self, u, v, pressure, mask, iteration):
        """Enqueue simulation data for metrics computation"""
        try:
            if self.data_queue.full():
                # Drop oldest frame if queue is full
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
            self.data_queue.put((u, v, pressure, mask, iteration), block=False)
        except queue.Full:
            pass  # Drop frame if queue is full

    def run_metrics(self):
        """Main loop for metrics computation"""
        import numpy as np
        from solver.metrics import find_stagnation_point, find_separation_point, compute_forces_ibm, get_airfoil_surface_mask
        from solver.operators import laplacian_nonperiodic_x, divergence_nonperiodic

        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue

            try:
                # Get data from queue with timeout
                u, v, pressure, mask, iteration = self.data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Convert JAX arrays to numpy for computation
                u_np = np.array(u)
                v_np = np.array(v)
                pressure_np = np.array(pressure)
                mask_np = np.array(mask)
                
                # Compute pressure residual to check solver convergence
                dx = self.solver.grid.dx
                dy = self.solver.grid.dy
                dt = self.solver.dt
                grid_type = getattr(self.solver.sim_params, 'grid_type', 'collocated')
                if grid_type == 'mac':
                    from solver.operators_mac import divergence_nonperiodic_staggered
                    div_star = divergence_nonperiodic_staggered(u, v, dx, dy)
                else:
                    div_star = divergence_nonperiodic(u, v, dx, dy)
                rhs = div_star / dt
                laplacian_p = laplacian_nonperiodic_x(pressure, dx, dy)
                pressure_residual = np.linalg.norm(np.array(laplacian_p) - np.array(rhs))
                rhs_norm = np.linalg.norm(np.array(rhs))
                relative_residual = pressure_residual / (rhs_norm + 1e-10)
                
                if iteration < 5 or iteration % 1000 == 0:
                    print(f"Pressure residual (iter {iteration}): {pressure_residual:.8e}, relative: {relative_residual:.8e}")
                    if relative_residual > 1e-3:
                        print(f"WARNING: Relative pressure residual > 1e-3 - solver may not be converging!")

                # For MAC grid, interpolate velocities to cell centers before computing metrics
                grid_type = getattr(self.solver.sim_params, 'grid_type', 'collocated')
                # Save original staggered arrays for divergence computation (convert to numpy first)
                u_np_staggered = np.array(u_np) if grid_type == 'mac' else None
                v_np_staggered = np.array(v_np) if grid_type == 'mac' else None
                if grid_type == 'mac':
                    # u is staggered in x (nx+1, ny), interpolate in x: 0.5 * (u[:-1, :] + u[1:, :])
                    u_np = 0.5 * (u_np[:-1, :] + u_np[1:, :])
                    # v is staggered in y (nx, ny+1), interpolate in y: 0.5 * (v[:, :-1] + v[:, 1:])
                    v_np = 0.5 * (v_np[:, :-1] + v_np[:, 1:])

                # Check if we should compute metrics based on frame skip (match solver behavior)
                should_compute_metrics = (iteration % self.solver.metrics_frame_skip == 0) if self.solver.metrics_frame_skip > 1 else True

                # Compute error metrics (only on frames matching frame skip)
                if self.solver.iteration > 0 and should_compute_metrics:
                    u_prev_np = np.array(self.solver.u_prev)
                    v_prev_np = np.array(self.solver.v_prev)

                    # For MAC grid, interpolate previous velocities to cell centers (current already interpolated above)
                    if grid_type == 'mac':
                        # u is staggered in x (nx+1, ny), interpolate in x: 0.5 * (u[:-1, :] + u[1:, :])
                        u_prev_np = 0.5 * (u_prev_np[:-1, :] + u_prev_np[1:, :])
                        # v is staggered in y (nx, ny+1), interpolate in y: 0.5 * (v[:, :-1] + v[:, 1:])
                        v_prev_np = 0.5 * (v_prev_np[:, :-1] + v_prev_np[:, 1:])

                    delta_u = u_np - u_prev_np
                    delta_v = v_np - v_prev_np

                    dx = float(self.solver.grid.dx)
                    dy = float(self.solver.grid.dy)

                    l2_delta_u = np.sqrt(np.sum(delta_u**2) * dx * dy)
                    l2_delta_v = np.sqrt(np.sum(delta_v**2) * dx * dy)
                    l2_delta_total = np.sqrt(l2_delta_u**2 + l2_delta_v**2)

                    max_delta_u = np.max(np.abs(delta_u))
                    max_delta_v = np.max(np.abs(delta_v))
                    max_delta_total = np.maximum(max_delta_u, max_delta_v)

                    # Calculate velocity change magnitude
                    if self.solver.sim_params.grid_type == 'mac':
                        # For MAC grid, skip delta_mag computation due to shape issues
                        delta_mag = np.zeros((self.solver.grid.nx, self.solver.grid.ny))
                    else:
                        delta_mag = np.sqrt(delta_u**2 + delta_v**2)

                    u_rms = np.sqrt(np.sum(u_np**2) * dx * dy / (self.solver.grid.nx * self.solver.grid.ny))
                    v_rms = np.sqrt(np.sum(v_np**2) * dx * dy / (self.solver.grid.nx * self.solver.grid.ny))
                    vel_rms = np.sqrt(u_rms**2 + v_rms**2) + 1e-8

                    rel_delta = l2_delta_total / (vel_rms * np.sqrt(float(self.solver.grid.lx) * float(self.solver.grid.ly)))

                    # Compute divergence only in pure fluid region (mask > 0.99) to exclude IBM transition zone
                    if self.solver.sim_params.grid_type == 'mac':
                        # Use proper staggered divergence for MAC grid with original staggered arrays
                        from solver.operators_mac import divergence_nonperiodic_staggered
                        div = divergence_nonperiodic_staggered(u_np_staggered, v_np_staggered, dx, dy)
                    else:
                        # Use collocated divergence for collocated grid
                        div_x = np.gradient(u_np, dx, axis=0)
                        div_y = np.gradient(v_np, dy, axis=1)
                        div = div_x + div_y
                    fluid_mask = (mask_np > 0.99)  # Changed from 0.5 to 0.99 to exclude transition zone
                    div_fluid = div * fluid_mask
                    div_rms = np.sqrt(np.sum(div_fluid**2) / (np.sum(fluid_mask) + 1e-8))
                    l2_div = np.sqrt(np.sum(div_fluid**2) * dx * dy)

                    error_metrics = {
                        'l2_change': float(l2_delta_total),
                        'rms_change': float(l2_delta_total / np.sqrt(self.solver.grid.nx * self.solver.grid.ny)),
                        'max_change': float(max_delta_total),
                        'change_99p': float(np.percentile(delta_mag, 99)),
                        'rel_change': float(rel_delta),
                        'l2_change_u': float(l2_delta_u),
                        'l2_change_v': float(l2_delta_v),
                        'rms_divergence': float(div_rms),  # Now stores RMS
                        'l2_divergence': float(l2_div),
                        'iteration': iteration
                    }
                else:
                    error_metrics = {
                        'l2_change': 0.0,
                        'rms_change': 0.0,
                        'max_change': 0.0,
                        'change_99p': 0.0,
                        'rel_change': 0.0,
                        'l2_change_u': 0.0,
                        'l2_change_v': 0.0,
                        'rms_divergence': 0.0,  # Now stores RMS divergence
                        'l2_divergence': 0.0,
                        'iteration': iteration
                    }

                # Compute airfoil metrics if enabled and frame skip allows
                airfoil_metrics = None
                self.frame_count += 1
                if self.solver.compute_airfoil_metrics and self.solver.sim_params.flow_type == 'von_karman' and should_compute_metrics:
                    try:
                        X_np = np.array(self.solver.grid.X)
                        Y_np = np.array(self.solver.grid.Y)
                        # u_np and v_np are already interpolated to cell centers for MAC grid above

                        # Compute vorticity for circulation-based force calculation
                        from solver.operators import vorticity, vorticity_nonperiodic
                        grid_type = getattr(self.solver.sim_params, 'grid_type', 'collocated')
                        if grid_type == 'mac':
                            from solver.operators_mac import vorticity_staggered, vorticity_nonperiodic_staggered
                            if self.solver.sim_params.flow_type == 'von_karman' or self.solver.sim_params.flow_type == 'lid_driven_cavity':
                                w_np = np.array(vorticity_nonperiodic_staggered(u, v, dx, dy))
                            else:
                                w_np = np.array(vorticity_staggered(u, v, dx, dy))
                        else:
                            if self.solver.sim_params.flow_type == 'von_karman' or self.solver.sim_params.flow_type == 'lid_driven_cavity':
                                w_np = np.array(vorticity_nonperiodic(u_np, v_np, dx, dy))
                            else:
                                w_np = np.array(vorticity(u_np, v_np, dx, dy))

                        stag_x = find_stagnation_point(u_np, v_np, mask_np, pressure_np, X_np, dx)
                        sep_x = find_separation_point(u_np, v_np, mask_np, X_np, dx, dy)

                        chord_length = getattr(self.solver.sim_params, 'naca_chord', 2.0)
                        airfoil_x = getattr(self.solver.sim_params, 'naca_x', 5.0)
                        airfoil_y = getattr(self.solver.sim_params, 'naca_y', 2.5)

                        # Use circulation-based force calculation (IBM-appropriate)
                        cl, cd = compute_forces_ibm(u_np, v_np, w_np, X_np, Y_np, mask_np,
                                                  dx, dy, self.solver.flow.U_inf,
                                                  chord_length, airfoil_x, airfoil_y,
                                                  self.solver.grid.lx,
                                                  grid_type=grid_type)

                        rho = 1.0
                        surface = get_airfoil_surface_mask(mask_np, dx, threshold=0.1)
                        p_inf = 0.0
                        q_inf = 0.5 * rho * self.solver.flow.U_inf**2
                        cp = (pressure_np - p_inf) / q_inf
                        cp_surface = np.where(surface, cp, np.inf)
                        cp_min = float(np.min(cp_surface))

                        airfoil_x = getattr(self.solver.sim_params, 'naca_x', 2.5)
                        wake_x = airfoil_x + chord_length
                        wake_x_idx = int(wake_x / dx)
                        wake_deficit = 0.0
                        if 0 <= wake_x_idx < self.solver.grid.nx:
                            u_wake = u_np[wake_x_idx, :]
                            wake_deficit = float(self.solver.flow.U_inf - np.mean(u_wake[mask_np[wake_x_idx, :] > 0.5]))

                        airfoil_metrics = {
                            'CL': cl,
                            'CD': cd,
                            'stagnation_x': float(stag_x),
                            'separation_x': float(sep_x),
                            'Cp_min': cp_min,
                            'wake_deficit': wake_deficit,
                            'strouhal': 0.0,  # Will be updated when stability is detected
                            'iteration': iteration
                        }
                    except Exception as e:
                        print(f"Error computing airfoil metrics: {e}")
                        airfoil_metrics = None

                # Only emit metrics signal when we actually computed metrics (not on skipped frames)
                if should_compute_metrics:
                    metrics_data = {
                        'error_metrics': error_metrics,
                        'airfoil_metrics': airfoil_metrics
                    }
                    self.metrics_ready.emit(metrics_data)

            except Exception as e:
                print(f"Error in metrics computation: {e}")
                import traceback
                traceback.print_exc()


class SimulationWorker(QObject):
    """Separate thread for simulation computation using Python threading"""
    data_ready = pyqtSignal(object)  # Signal when new data is ready
    fps_update = pyqtSignal(int)     # Signal for FPS updates
    profiling_update = pyqtSignal(float, float, float, float)  # Signal for profiling data (solver_ms, interp_ms, total_ms, sim_fps)

    def __init__(self, solver, control_panel=None, info_panel=None, metrics_worker=None, flow_viz=None):
        super().__init__()
        self.solver = solver
        self.control_panel = control_panel
        self.info_panel = info_panel
        self.metrics_worker = metrics_worker
        self.flow_viz = flow_viz
        self.running = False
        self.paused = False
        self.data_queue = queue.Queue(maxsize=2)  # Buffer of 2 frames
        self.simulation_speed = 1.0  # Simulation speed multiplier
        self.thread = None  # Python thread instead of Qt thread
        
        # Frame skipping
        self.simulation_step_counter = 0
        self.update_every = 1
        
        # FPS tracking for simulation
        self.sim_fps_counter = 0
        self.last_sim_fps_time = time.time()
        
        # Initialize shared memory buffers for zero-copy transfer
        nx, ny = solver.grid.nx, solver.grid.ny
        self.shared_buffers = {
            'u': SharedData((nx, ny), np.float32),
            'v': SharedData((nx, ny), np.float32),
            'vort': SharedData((nx, ny), np.float32),
            'vel_mag': SharedData((nx, ny), np.float32)
        }
        
    def start(self):
        """Start the simulation thread"""
        try:
            if self.thread is None or not self.thread.is_alive():
                self.running = True
                self.thread = threading.Thread(target=self.run_simulation, daemon=True)
                self.thread.start()
                print("Simulation thread started")
        except Exception as e:
            print(f"ERROR: Failed to start simulation thread: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def run_simulation(self):
        """Main simulation loop in separate thread"""
        step_count = 0
        print("Starting simulation loop...")
        
        # Track iteration rate directly
        iteration_start_time = time.time()
        
        while self.running:
            try:
                # If paused, wait and continue
                if self.paused:
                    time.sleep(0.01)
                    continue
                
                step_count += 1
                
                # Check if we should stop (more responsive)
                if not self.running:
                    print("Simulation stop requested")
                    break
                
                # Run simulation step with coefficient computation for airfoil metrics
                t_solver_start = time.time()
                try:
                    # Get diagnostics setting from GUI
                    compute_diagnostics = self.info_panel.diagnostics_checkbox.isChecked() if self.info_panel else True
                    
                    # Always pass compute_diagnostics=False to avoid blocking - metrics computed in separate thread
                    # Only compute divergence if the divergence plot is visible
                    compute_div = hasattr(self, 'flow_viz') and hasattr(self.flow_viz, 'div_plot') and self.flow_viz.div_plot.isVisible()
                    u, v, vort, div = self.solver.step_for_visualization(
                        compute_divergence=compute_div,
                        compute_drag_lift=True,
                        compute_diagnostics=False  # Metrics computed in separate thread
                    )
                except Exception as step_error:
                    print(f"ERROR: Simulation step failed: {step_error}")
                    import traceback
                    traceback.print_exc()
                    
                    # Check for specific LDC-related errors
                    if "lid_driven_cavity" in str(step_error).lower() or "ldc" in str(step_error).lower():
                        print("LDC-specific error detected, stopping simulation...")
                        break
                    
                    time.sleep(0.01)  # Brief pause before retry
                    continue  # Skip this step but continue simulation
                t_solver_end = time.time()
                
                # Enqueue data for metrics worker if enabled and worker exists
                if compute_diagnostics and self.metrics_worker:
                    self.metrics_worker.enqueue_data(
                        self.solver.u,
                        self.solver.v,
                        self.solver.current_pressure,
                        self.solver.mask,
                        self.solver.iteration
                    )
                
                # Get divergence from solver history (already computed in JIT step)
                if hasattr(self.solver, 'history') and 'rms_divergence' in self.solver.history and self.solver.history['rms_divergence']:
                    div_rms = self.solver.history['rms_divergence'][-1]
                else:
                    div_rms = 0.0  # Fallback if not available
                
                # Interpolate MAC grid velocities to cell centers for visualization
                t_interp_start = time.time()
                grid_type = getattr(self.solver.sim_params, 'grid_type', 'collocated')
                solver_type = getattr(self.solver.sim_params, 'solver_type', 'navier_stokes')
                
                # LBM always uses collocated grid, regardless of grid_type setting
                if solver_type == 'lattice_boltzmann' or grid_type == 'collocated':
                    # Collocated grid - use directly (LBM case)
                    u_display = u
                    v_display = v
                    vort_display = vort
                    try:
                        vel_mag = jnp.sqrt(u**2 + v**2)
                    except Exception as vel_error:
                        print(f"ERROR: Velocity magnitude calculation failed: {vel_error}")
                        vel_mag = jnp.zeros_like(u)  # Fallback value
                    vel_mag_display = vel_mag
                elif grid_type == 'mac':
                    # MAC grid with traditional NS solver - interpolate staggered velocities
                    u_center = 0.5 * (u[1:, :] + u[:-1, :])
                    v_center = 0.5 * (v[:, 1:] + v[:, :-1])
                    u_display = u_center
                    v_display = v_center
                    # Compute velocity magnitude from interpolated velocities
                    vel_mag = jnp.sqrt(u_center**2 + v_center**2)
                    vel_mag_display = vel_mag
                    # Interpolate vorticity to cell centers if needed
                    if vort.shape == (self.solver.grid.nx + 1, self.solver.grid.ny):
                        vort_display = 0.5 * (vort[1:, :] + vort[:-1, :])
                    elif vort.shape == (self.solver.grid.nx, self.solver.grid.ny + 1):
                        vort_display = 0.5 * (vort[:, 1:] + vort[:, :-1])
                    else:
                        vort_display = vort
                else:
                    # Default to collocated treatment
                    u_display = u
                    v_display = v
                    vort_display = vort
                    try:
                        vel_mag = jnp.sqrt(u**2 + v**2)
                    except Exception as vel_error:
                        print(f"ERROR: Velocity magnitude calculation failed: {vel_error}")
                        vel_mag = jnp.zeros_like(u)  # Fallback value
                    vel_mag_display = vel_mag
                t_interp_end = time.time()
                
                # Create metadata dict with arrays directly
                t_queue_start = time.time()
                data = {
                    'time': self.solver.iteration * self.solver.dt,
                    'iteration': self.solver.iteration,
                    'rms_divergence': div_rms,
                    'u': np.asarray(u_display, dtype=np.float32),
                    'v': np.asarray(v_display, dtype=np.float32),
                    'vort': np.asarray(vort_display, dtype=np.float32),
                    'vel_mag': np.asarray(vel_mag_display, dtype=np.float32),
                    'div': np.asarray(div, dtype=np.float32) if div is not None else None,
                }
                t_queue_end = time.time()
                
                # Emit signal for UI update (every iteration for smooth visualization)
                t_signal_start = time.time()
                if self.running:
                    try:
                        self.data_ready.emit(data)
                    except Exception as emit_error:
                        print(f"ERROR: Signal emission failed: {emit_error}")
                        import traceback
                        traceback.print_exc()
                t_signal_end = time.time()
                
                # Simulation FPS counter
                self.sim_fps_counter += 1
                if time.time() - self.last_sim_fps_time > 1.0:
                    try:
                        self.fps_update.emit(self.sim_fps_counter)
                        # Emit profiling data (every second)
                        solver_ms = (t_solver_end - t_solver_start) * 1000
                        interp_ms = (t_interp_end - t_interp_start) * 1000
                        total_ms = (t_signal_end - t_solver_start) * 1000
                        self.profiling_update.emit(solver_ms, interp_ms, total_ms, self.sim_fps_counter)
                    except Exception as fps_error:
                        print(f"ERROR: FPS update failed: {fps_error}")
                    self.sim_fps_counter = 0
                    self.last_sim_fps_time = time.time()
                
            except Exception as e:
                if not self.running:  # Error during shutdown is OK
                    break
                print(f"FATAL ERROR: Simulation loop crashed: {e}")
                import traceback
                traceback.print_exc()
                break  # Exit the loop on fatal error
        
        print("Simulation thread stopped")
    
    def pause(self):
        """Pause the simulation"""
        self.paused = True

    def resume(self):
        """Resume the simulation"""
        self.paused = False

    def recreate_shared_buffers(self, new_nx, new_ny):
        """Recreate shared memory buffers for new grid dimensions"""
        # Clean up old buffers
        if hasattr(self, 'shared_buffers'):
            for buffer in self.shared_buffers.values():
                buffer.cleanup()
        
        # Create new buffers with updated dimensions
        self.shared_buffers = {
            'u': SharedData((new_nx, new_ny), np.float32),
            'v': SharedData((new_nx, new_ny), np.float32),
            'vort': SharedData((new_nx, new_ny), np.float32),
            'vel_mag': SharedData((new_nx, new_ny), np.float32)
        }

    def stop_simulation(self):
        """Stop the simulation thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)  # Wait up to 2 seconds
            if self.thread.is_alive():
                print("Warning: Simulation thread did not stop gracefully")

        # Clean up shared memory
        if hasattr(self, 'shared_buffers'):
            for buffer in self.shared_buffers.values():
                try:
                    buffer.cleanup()
                except Exception as e:
                    print(f"Warning: Buffer cleanup error: {e}")
    
    def pause_simulation(self):
        """Pause the simulation without stopping the thread"""
        self.paused = True
        print("Simulation paused")
    
    def resume_simulation(self):
        """Resume the simulation from paused state"""
        self.paused = False
        print("Simulation resumed")


class SimulationController:
    """Controls simulation execution and data flow"""
    
    def __init__(self, solver, control_panel=None, info_panel=None, flow_viz=None):
        self.solver = solver
        self.control_panel = control_panel
        self.info_panel = info_panel
        self.flow_viz = flow_viz
        self.simulation_worker = None
        self.metrics_worker = None
        self.latest_data = None
        self.latest_metrics = None
        self.callbacks = None  # Store callbacks for reconnecting signals
        
        # Frame skipping controls (kept for compatibility with existing code)
        self.simulation_step_counter = 0
        self.update_every = 1  # Default to 1 (no skipping)
        self.should_update_visualization = False
        
        # Performance tracking
        self.frame_counter = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def update_grid_size(self, new_nx, new_ny):
        """Update simulation worker for new grid size - COMPLETE RESTART"""
        # Force complete shutdown
        self.stop_simulation()
        
        # Clear all references to force garbage collection
        if self.simulation_worker:
            try:
                if hasattr(self.simulation_worker, 'shared_buffers'):
                    for buffer in self.simulation_worker.shared_buffers.values():
                        buffer.cleanup()
            except Exception as cleanup_error:
                print(f"Warning: Buffer cleanup error: {cleanup_error}")
            finally:
                self.simulation_worker = None
        
        # Clear latest data to prevent stale references
        self.latest_data = None
        
        # Force garbage collection
        import gc
        gc.collect()
        import jax
        jax.clear_caches()

    def start_simulation(self, callbacks):
        """Start simulation in separate thread"""
        # Store callbacks for reconnecting signals when metrics worker is restarted
        self.callbacks = callbacks
        
        try:
            # COMPLETELY stop any existing simulation
            if self.simulation_worker is not None:
                old_worker = self.simulation_worker
                old_worker.stop_simulation()

                # Disconnect signals to prevent callbacks to deleted objects
                try:
                    old_worker.data_ready.disconnect()
                    old_worker.fps_update.disconnect()
                except Exception:
                    pass  # Signals might not be connected

                # Wait for thread to finish with proper synchronization
                import time
                if old_worker.thread and old_worker.thread.is_alive():
                    for _ in range(50):  # 5 seconds max
                        if not old_worker.thread.is_alive():
                            break
                        time.sleep(0.1)
                    if old_worker.thread.is_alive():
                        print("Warning: Old simulation thread did not stop in 5 seconds")

                # Clear reference
                self.simulation_worker = None
            
            # Clear stale data before restart
            self.latest_data = None

            # Start metrics worker if not already running
            if self.metrics_worker is None:
                self.metrics_worker = MetricsWorker(self.solver)
                self.metrics_worker.start()
                print("Metrics worker started")

            # Create fresh simulation worker
            self.simulation_worker = SimulationWorker(self.solver, self.control_panel, self.info_panel, self.metrics_worker, self.flow_viz)

            # Connect signals
            if 'data_ready' in callbacks:
                self.simulation_worker.data_ready.connect(
                    callbacks['data_ready'], Qt.ConnectionType.QueuedConnection
                )
            if 'fps_update' in callbacks:
                self.simulation_worker.fps_update.connect(
                    callbacks['fps_update'], Qt.ConnectionType.QueuedConnection
                )
            if 'profiling_update' in callbacks:
                self.simulation_worker.profiling_update.connect(
                    callbacks['profiling_update'], Qt.ConnectionType.QueuedConnection
                )
            if 'metrics_ready' in callbacks and self.metrics_worker:
                self.metrics_worker.metrics_ready.connect(
                    callbacks['metrics_ready'], Qt.ConnectionType.QueuedConnection
                )

            # Start the thread
            self.simulation_worker.start()

            print("Simulation started in separate thread")

        except Exception as e:
            print(f"Error starting simulation: {e}")
            import traceback
            traceback.print_exc()
            self.simulation_worker = None
    
    def full_reset(self):
        """Completely reset the simulation controller for parameter changes."""

        # Stop and cleanup existing worker
        if self.simulation_worker is not None:
            self.simulation_worker.stop_simulation()
            self.simulation_worker = None

        # Clear latest data
        self.latest_data = None

        # Reset counters
        self.simulation_step_counter = 0
        self.should_update_visualization = False

        # Force garbage collection
        import gc
        gc.collect()

        # Clear JAX caches
        import jax
        jax.clear_caches()

    def start_metrics(self):
        """Start metrics worker"""
        if self.metrics_worker is None:
            self.metrics_worker = MetricsWorker(self.solver)
            self.metrics_worker.start()
            
            # Reconnect metrics_ready signal if callbacks are available
            if self.callbacks and 'metrics_ready' in self.callbacks:
                self.metrics_worker.metrics_ready.connect(
                    self.callbacks['metrics_ready'], Qt.ConnectionType.QueuedConnection
                )
                print("Metrics worker started and signal reconnected")
            else:
                print("Metrics worker started (no callbacks to reconnect)")
            
            # Update simulation worker's reference to the new metrics worker
            if self.simulation_worker:
                self.simulation_worker.metrics_worker = self.metrics_worker
                print("Simulation worker's metrics_worker reference updated")
        else:
            print("Metrics worker already running")

    def stop_metrics(self):
        """Stop metrics worker"""
        if self.metrics_worker:
            self.metrics_worker.stop()
            self.metrics_worker = None
            print("Metrics worker stopped")

    def stop_simulation(self):
        """Stop the simulation and metrics worker"""
        if self.simulation_worker:
            self.simulation_worker.stop_simulation()
        if self.metrics_worker:
            self.metrics_worker.stop()
        print("Simulation stopped")
    
    def pause_simulation(self):
        """Pause simulation without stopping the thread"""
        if self.simulation_worker:
            self.simulation_worker.pause()
        if self.metrics_worker:
            self.metrics_worker.pause()
    
    def resume_simulation(self):
        """Resume simulation from paused state"""
        if self.simulation_worker:
            self.simulation_worker.resume()
        if self.metrics_worker:
            self.metrics_worker.resume()
    
    def on_simulation_data_ready(self, data):
        """Handle new data from simulation thread"""
        # Store the latest data
        self.latest_data = data
        
        # Increment simulation step counter for frame skipping
        self.simulation_step_counter += 1
        
        # Frame skipping: only update visualization every N simulation steps
        if self.simulation_step_counter % self.update_every == 0:
            # This is a frame that should be visualized
            self.should_update_visualization = True
        else:
            # Skip this frame
            self.should_update_visualization = False
    
    def set_frame_skip(self, update_every):
        """Set frame skip setting"""
        self.update_every = update_every
    
    def get_latest_data(self):
        """Get the latest simulation data"""
        return self.latest_data
    
    def should_update(self):
        """Check if visualization should be updated"""
        return self.should_update_visualization
    
    def reset_update_flag(self):
        """Reset the visualization update flag"""
        self.should_update_visualization = False


class RecordingManager:
    """Manages video recording functionality"""
    
    def __init__(self):
        self.is_recording = False
        self.recorded_frames = []
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.is_recording:
            self.is_recording = True
            self.recorded_frames = []
            print("Started recording video...")
            return "Stop Recording"
        else:
            self.is_recording = False
            print(f"Stopped recording. Captured {len(self.recorded_frames)} frames.")
            return "Record Video"
    
    def capture_frame(self, frame_data):
        """Capture a frame for recording"""
        if self.is_recording:
            try:
                # Normalize to 0-255 for video
                frame_norm = ((frame_data - frame_data.min()) / 
                             (frame_data.max() - frame_data.min()) * 255).astype(np.uint8)
                self.recorded_frames.append(frame_norm)
            except:
                pass  # Skip frame if capture fails
    
    def save_video(self, parent_widget=None):
        """Save recorded frames as video"""
        if not self.recorded_frames:
            print("No frames to save!")
            return
            
        try:
            import imageio
            from PyQt6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                parent_widget, "Save Video", "flow_simulation.mp4", "Video Files (*.mp4 *.avi)"
            )
            if filename:
                print(f"Saving video with {len(self.recorded_frames)} frames...")
                imageio.mimsave(filename, self.recorded_frames, fps=30)
                print(f"Video saved as {filename}")
        except ImportError:
            print("imageio not installed. Install with: pip install imageio")
        except Exception as e:
            print(f"Error saving video: {e}")
    
    def has_frames(self):
        """Check if there are frames to save"""
        return len(self.recorded_frames) > 0
    
    def get_frame_count(self):
        """Get number of recorded frames"""
        return len(self.recorded_frames)


class DataExporter:
    """Handles data export functionality"""
    
    @staticmethod
    def export_simulation_data(solver, history_data=None):
        """Export simulation data to files"""
        try:
            import datetime
            import json
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get current data
            u_np = np.array(solver.u)
            v_np = np.array(solver.v)
            
            # Get vorticity from cache if available, otherwise compute
            if hasattr(solver, 'current_vorticity') and solver.current_vorticity is not None:
                vort_np = np.array(solver.current_vorticity)
            else:
                vort_np = np.zeros_like(u_np)  # Fallback
            
            # Export to CSV files
            np.savetxt(f'velocity_u_{timestamp}.csv', u_np, delimiter=',')
            np.savetxt(f'velocity_v_{timestamp}.csv', v_np, delimiter=',')
            np.savetxt(f'vorticity_{timestamp}.csv', vort_np, delimiter=',')
            
            # Export history data if provided
            if history_data:
                DataExporter._export_history_data(history_data, timestamp)
            
            # Export grid information
            DataExporter._export_grid_info(solver, timestamp)
            
            print(f"Data exported successfully with timestamp {timestamp}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    @staticmethod
    def _export_history_data(history_data, timestamp):
        """Export history data to CSV"""
        try:
            time_data = history_data.get('time', [])
            enst_data = history_data.get('enstrophy', [])
            drag_data = history_data.get('drag', [])
            lift_data = history_data.get('lift', [])
            
            if time_data and len(time_data) > 0:
                history_array = np.column_stack([
                    time_data[:len(time_data)],
                    enst_data[:len(time_data)], 
                    drag_data[:len(time_data)], 
                    lift_data[:len(time_data)]
                ])
                np.savetxt(f'history_{timestamp}.csv', history_array, delimiter=',',
                          header='time,enstrophy,drag,lift')
        except Exception as e:
            print(f"Error exporting history data: {e}")
    
    @staticmethod
    def _export_grid_info(solver, timestamp):
        """Export grid information to JSON"""
        try:
            import json
            
            grid_info = {
                'nx': solver.grid.nx,
                'ny': solver.grid.ny,
                'lx': solver.grid.lx,
                'ly': solver.grid.ly,
                'dx': solver.grid.dx,
                'dy': solver.grid.dy,
                'flow_type': solver.sim_params.flow_type,
                'Re': solver.flow.Re,
                'U_inf': solver.flow.U_inf,
                'nu': solver.flow.nu,
                'dt': solver.dt,
                'iteration': solver.iteration
            }
            
            with open(f'grid_info_{timestamp}.json', 'w') as f:
                json.dump(grid_info, f, indent=2)
        except Exception as e:
            print(f"Error exporting grid info: {e}")
