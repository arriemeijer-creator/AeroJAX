"""
Visualization components for Baseline Navier-Stokes Viewer
Separates visualization logic from UI and main viewer logic
"""

import numpy as np
import jax.numpy as jnp
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6 import sip
import scipy.ndimage
from .obstacle_renderer import ObstacleRenderer
from .particle_system import ParticleSystem


def _upscale_for_display(data, scale_factor=2):
    """Upscale data for smooth visualization using bilinear interpolation"""
    if scale_factor <= 1:
        return data
    return scipy.ndimage.zoom(data, scale_factor, order=1, mode='nearest')


def _get_cell_centered_velocities(u, v, grid_type='collocated'):
    """
    Get velocities at cell centers for visualization.
    For collocated grid: u and v are already at cell centers.
    For MAC staggered grid: interpolate u and v from faces to cell centers.
    """
    if grid_type == 'mac':
        # Interpolate staggered velocities to cell centers
        # u: (nx+1, ny) at x-faces -> (nx, ny) at centers
        u_center = 0.5 * (u[1:, :] + u[:-1, :])
        # v: (nx, ny+1) at y-faces -> (nx, ny) at centers
        v_center = 0.5 * (v[:, 1:] + v[:, :-1])
        return u_center, v_center
    else:
        # Collocated grid: already at cell centers
        return u, v


class FlowVisualization:
    """Handles flow field visualization (velocity, vorticity)"""
    
    def __init__(self, plot_widget, solver=None, control_panel=None, skip_initial_setup=False):
        self.plot_widget = plot_widget
        self.solver = solver  # Store solver reference
        self.control_panel = control_panel  # Store control panel reference for checkbox access
        self.vel_plot = None
        self.vort_plot = None
        self.pressure_plot = None
        self.upscale_factor = 1  # Default upscale factor for smooth visualization (1x = no upscaling)
        # CL and CD plots removed for stability
        self.cl_plot = None
        self.cd_plot = None
        self.l2_plot = None  # Add L2 error plot
        self.vel_img = None
        self.vort_img = None
        self.scalar_img = None
        self.vel_outline = None
        self.vort_outline = None
        self.scalar_outline = None
        self.pressure_outline = None
        self.vel_sdf = None
        self.vort_sdf = None
        self.l2_curve = None  # L2 error curve data
        
        # Stagnation and separation markers (vertical lines)
        self.stag_line_vel = None
        self.sep_line_vel = None
        self.stag_line_vort = None
        self.sep_line_vort = None
        self.show_stagnation_marker = True
        self.show_separation_marker = True
        
        # Performance settings
        self.vel_levels = [0.0, 0.02]  # Appropriate range for LDC flow (lid velocity = 0.01)
        self.vort_levels = [-1.0, 1.0]  # Appropriate range for LDC vorticity (max ~0.64)
        self.level_update_counter = 0
        self.level_update_interval = 1  # Update every frame
        self.manual_bounds_set = False  # Flag to prevent auto-fit override
        
        # Streamline settings
        self.show_streamlines = False
        self.streamline_items = []
        self.streamline_update_interval = 5  # Update every 5 frames to maintain performance
        self.streamline_counter = 0
        self.streamline_density = 16  # Number of seed points per dimension
        
        # Quiver settings
        self.show_quivers = False
        self.quiver_items = []
        self.quiver_update_interval = 5  # Update every 5 frames to maintain performance
        self.quiver_counter = 0
        self.quiver_density = 20  # Number of quivers per dimension
        
        # Particle system settings
        self.particle_system = ParticleSystem(max_particles=1000)
        self.particle_scatter = None
        self.use_particles = False  # Toggle between dye and particles
        
        # CL and CD plots removed - no coefficient tracking needed
        
        # Initialize obstacle renderer
        self.obstacle_renderer = None
        
        # Profiling overlay text item
        self.profiling_text = None
        self.profiling_label = None  # QLabel overlay
        self.profiling_visible = False
        
        # Initialize visualization timing data
        self._latest_viz_timing = {
            'velocity': 0.0,
            'vorticity': 0.0,
            'pressure': 0.0,
            'viz_total': 0.0
        }
        
        # Initialize grid dimensions from solver if available, otherwise use defaults
        if solver is not None and hasattr(solver, 'grid'):
            self.current_nx = solver.grid.nx
            self.current_ny = solver.grid.ny
            self.current_lx = solver.grid.lx
            self.current_ly = solver.grid.ly
            self.current_y_min = 0.0
            self.current_y_max = solver.grid.ly
        else:
            # Default dimensions (will be updated in setup_plots or update_plots_for_new_grid)
            self.current_nx = 512
            self.current_ny = 192
            self.current_lx = 20.0
            self.current_ly = 4.5
            self.current_y_min = 0.0
            self.current_y_max = 4.5
        
        # Only setup plots if not skipping (e.g., when recreating for grid update)
        if not skip_initial_setup:
            self.setup_plots()
            self.set_initial_colormaps()
    
    def update_profiling_overlay(self, solver_ms, interp_ms, total_ms, sim_fps, viz_data=None):
        """Update profiling overlay text"""
        text = f"Sim FPS: {sim_fps:.1f}\nSolver: {solver_ms:.2f}ms\nInterp: {interp_ms:.2f}ms\nTotal: {total_ms:.2f}ms"
        
        # Always add visualization timing (use 0 if not available yet)
        if viz_data is None:
            viz_data = self._latest_viz_timing
        
        text += f"\n\nViz Total: {viz_data.get('viz_total', 0):.2f}ms"
        text += f"\nVel: {viz_data.get('velocity', 0):.2f}ms"
        text += f"\nVort: {viz_data.get('vorticity', 0):.2f}ms"
        text += f"\nPress: {viz_data.get('pressure', 0):.2f}ms"
        
        if self.profiling_label:
            self.profiling_label.setText(text)
    
    def set_profiling_visible(self, visible):
        """Toggle profiling overlay visibility"""
        self.profiling_visible = visible
        if self.profiling_label:
            self.profiling_label.setVisible(visible)
    
    def set_initial_colormaps(self, velocity_colormap='viridis', vorticity_colormap='RdBu', divergence_colormap='seismic', pressure_colormap='RdBu'):
        """Set initial colormaps for velocity, vorticity, divergence, and pressure plots"""
        try:
            # Set initial velocity colormap
            try:
                vel_colormap = pg.colormap.get(velocity_colormap)
                if vel_colormap is None:
                    vel_colormap = pg.colormap.getFromMatplotlib(velocity_colormap)
            except:
                vel_colormap = pg.colormap.getFromMatplotlib(velocity_colormap)
                
            if vel_colormap is not None:
                vel_lut = vel_colormap.getLookupTable()
                if self.vel_img is not None:
                    self.vel_img.setLookupTable(vel_lut)
            
            # Set initial vorticity colormap
            try:
                vort_colormap = pg.colormap.get(vorticity_colormap)
                if vort_colormap is None:
                    vort_colormap = pg.colormap.getFromMatplotlib(vorticity_colormap)
            except:
                vort_colormap = pg.colormap.getFromMatplotlib(vorticity_colormap)
            
            # Initialize vorticity plot title with current solver parameters
            if hasattr(self, 'solver') and self.solver is not None:
                re = self.solver.flow.Re
                u_inlet = self.solver.flow.U_inf
                naca = self.solver.sim_params.naca_airfoil if hasattr(self.solver.sim_params, 'naca_airfoil') else 'N/A'
                aoa = self.solver.sim_params.naca_angle if hasattr(self.solver.sim_params, 'naca_angle') else 0.0
                self.update_vorticity_title(re, u_inlet, naca, aoa)
            
            # Set colormaps on colorbars
            if hasattr(self, 'vel_colorbar') and self.vel_colorbar:
                try:
                    vel_colormap = pg.colormap.get(velocity_colormap)
                    if vel_colormap is None:
                        vel_colormap = pg.colormap.getFromMatplotlib(velocity_colormap)
                    if vel_colormap is not None:
                        self.vel_colorbar.setColorMap(vel_colormap)
                except:
                    pass
            
            if hasattr(self, 'vort_colorbar') and self.vort_colorbar:
                try:
                    vort_colormap = pg.colormap.get(vorticity_colormap)
                    if vort_colormap is None:
                        vort_colormap = pg.colormap.getFromMatplotlib(vorticity_colormap)
                    if vort_colormap is not None:
                        self.vort_colorbar.setColorMap(vort_colormap)
                except:
                    pass

            # Set initial pressure colormap
            try:
                pressure_colormap_obj = pg.colormap.get(pressure_colormap)
                if pressure_colormap_obj is None:
                    pressure_colormap_obj = pg.colormap.getFromMatplotlib(pressure_colormap)
            except:
                pressure_colormap_obj = pg.colormap.getFromMatplotlib(pressure_colormap)

            if pressure_colormap_obj is not None:
                pressure_lut = pressure_colormap_obj.getLookupTable()
                if self.pressure_img is not None:
                    self.pressure_img.setLookupTable(pressure_lut)

            # Set pressure colormap on colorbar
            if hasattr(self, 'pressure_colorbar') and self.pressure_colorbar:
                try:
                    pressure_colormap_obj = pg.colormap.get(pressure_colormap)
                    if pressure_colormap_obj is None:
                        pressure_colormap_obj = pg.colormap.getFromMatplotlib(pressure_colormap)
                    if pressure_colormap_obj is not None:
                        self.pressure_colorbar.setColorMap(pressure_colormap_obj)
                except:
                    pass

            # Set initial divergence colormap
            try:
                div_colormap = pg.colormap.get(divergence_colormap)
                if div_colormap is None:
                    div_colormap = pg.colormap.getFromMatplotlib(divergence_colormap)
            except:
                div_colormap = pg.colormap.getFromMatplotlib(divergence_colormap)

            if div_colormap is not None:
                div_lut = div_colormap.getLookupTable()
                if self.div_img is not None:
                    self.div_img.setLookupTable(div_lut)

            # Set divergence colormap on colorbar
            if hasattr(self, 'div_colorbar') and self.div_colorbar:
                try:
                    div_colormap_obj = pg.colormap.get(divergence_colormap)
                    if div_colormap_obj is None:
                        div_colormap_obj = pg.colormap.getFromMatplotlib(divergence_colormap)
                    if div_colormap_obj is not None:
                        self.div_colorbar.setColorMap(div_colormap_obj)
                except:
                    pass


            if vort_colormap is not None:
                vort_lut = vort_colormap.getLookupTable()
                if self.vort_img is not None:
                    self.vort_img.setLookupTable(vort_lut)
                
        except Exception as e:
            print(f"Error setting initial colormaps: {e}")
    
    def setup_plots(self, nx=512, ny=192, lx=20.0, ly=7.5):
        """Setup velocity, vorticity, and coefficient plots"""
        # Only clear on initial setup, not when called from update_plots_for_new_grid
        # (update_plots_for_new_grid handles clearing before calling this)
        if not hasattr(self, 'vel_plot') or self.vel_plot is None:
            if hasattr(self, 'plot_widget') and self.plot_widget is not None:
                self.plot_widget.clear()
        
        # Row 0: Velocity (spans 2 columns by default, 1 when divergence is visible) | Divergence (right)
        self.vel_plot = self.plot_widget.addPlot(title="Velocity Magnitude", row=0, col=0, colspan=2)
        self.vel_plot.setLabel('left', 'y')
        self.vel_plot.setLabel('bottom', 'x')

        self.div_plot = self.plot_widget.addPlot(title="Divergence", row=0, col=1, colspan=1)
        self.div_plot.setLabel('left', 'y')
        self.div_plot.setLabel('bottom', 'x')
        self.div_plot.setVisible(False)  # Hidden by default

        # Row 1: Vorticity (left) | Pressure (right)
        self.vort_plot = self.plot_widget.addPlot(title="Vorticity", row=1, col=0, colspan=1)
        self.vort_plot.setLabel('left', 'y')
        self.vort_plot.setLabel('bottom', 'x')

        self.pressure_plot = self.plot_widget.addPlot(title="Pressure", row=1, col=1, colspan=1)
        self.pressure_plot.setLabel('left', 'y')
        self.pressure_plot.setLabel('bottom', 'x')

        # Row 2: Dye (left) | Error metrics (right)
        # Create scalar (dye) plot
        self.scalar_plot = self.plot_widget.addPlot(title="Dye Concentration", row=2, col=0, colspan=1)
        self.scalar_plot.setLabel('left', 'y')
        self.scalar_plot.setLabel('bottom', 'x')
        self.scalar_img = pg.ImageItem()
        self.scalar_img.setLevels([0, 1])  # Set levels immediately to prevent float input type error
        self.scalar_plot.addItem(self.scalar_img)
        
        # Set white background for scalar plot
        view_box = self.scalar_plot.getViewBox()
        view_box.setBackgroundColor('w')  # White background
        
        # Add dye injection position marker (circle)
        self.dye_marker = pg.ScatterPlotItem(size=15, pen=pg.mkPen('r', width=2), brush=pg.mkBrush(255, 0, 0, 150))
        self.scalar_plot.addItem(self.dye_marker)
        self.dye_marker.setVisible(False)  # Initially hidden
        
        # Add particle scatter plot item
        self.particle_scatter = pg.ScatterPlotItem(size=2, pen=pg.mkPen('b', width=1), brush=pg.mkBrush(0, 0, 255, 200))
        self.scalar_plot.addItem(self.particle_scatter)
        self.particle_scatter.setVisible(False)  # Initially hidden
        
        # Use black colormap for dye on white background
        try:
            # Create black colormap (white=0, black=1)
            pos = np.array([0.0, 1.0])
            color = np.array([
                [255, 255, 255, 255],  # White for 0 (no dye)
                [0, 0, 0, 255]          # Black for 1 (maximum dye)
            ], dtype=np.ubyte)
            black_cmap = pg.ColorMap(pos, color)
            self.scalar_img.setLookupTable(black_cmap.getLookupTable())
            self.scalar_img.setLevels([0, 1])
        except:
            # Fallback to gray colormap if custom fails
            try:
                gray_cmap = pg.colormap.get('gray')
                lut = gray_cmap.getLookupTable()
                inverted_lut = 255 - lut  # Invert for white background
                self.scalar_img.setLookupTable(inverted_lut)
            except:
                pass
        
        # Create enhanced error plot with multiple metrics (right column)
        self.l2_plot = self.plot_widget.addPlot(title="Error Metrics", row=2, col=1, colspan=1)
        self.l2_plot.setLabel('left', 'Error')
        self.l2_plot.setLabel('bottom', 'Time')
        self.l2_plot.showGrid(x=True, y=True)
        self.l2_plot.setLogMode(y=True)  # Log scale for better visualization
        self.l2_plot.setXRange(0, 1)  # Start with range from 0, will expand as data grows

        # Add error legend positioned on the RIGHT side
        legend = self.l2_plot.addLegend(offset=(-10, 10))  # 10px from right, 10px from top
        legend.anchor(itemPos=(1, 0), parentPos=(1, 0))   # Anchor to top-right corner
        
        # Initialize error tracking arrays and curves
        self.l2_curve = None
        self.max_error_curve = None
        self.rel_error_curve = None
        self.l2_u_curve = None
        self.l2_v_curve = None
        self.rms_change_curve = None
        self.change_99p_curve = None
        self.error_times = []
        self.l2_errors = []
        self.max_errors = []
        self.rel_errors = []
        self.l2_u_errors = []
        self.l2_v_errors = []
        self.rms_change_errors = []
        self.change_99p_errors = []
        
        # FPS tracking
        self.current_sim_fps = 0.0
        self.current_viz_fps = 0.0
        
        # CL and CD plots removed for stability and performance
        
        # Create profiling overlay as QLabel on plot widget (more reliable than TextItem)
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtCore import Qt
        self.profiling_label = QLabel(self.plot_widget)
        self.profiling_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: #00BFFF;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.profiling_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.profiling_label.move(20, 20)  # Top-left corner
        self.profiling_label.setVisible(False)
        
        # Get actual y-bounds
        y_min = 0.0
        y_max = ly
        
        if hasattr(self, 'solver') and self.solver is not None:
            try:
                if hasattr(self.solver, 'grid') and hasattr(self.solver.grid, 'y'):
                    y_coords = np.array(self.solver.grid.y)
                    y_min = y_coords.min()
                    y_max = y_coords.max()
            except:
                pass
        
        # Set plot ranges - CRITICAL: These must match the image rect
        padding_x = lx * 0.1
        padding_y = (y_max - y_min) * 0.1
        
        # Clamp values to prevent overflow
        x_min = max(-padding_x, -1e10)
        x_max = min(lx + padding_x, 1e10)
        y_min_clamped = max(y_min - padding_y, -1e10)
        y_max_clamped = min(y_max + padding_y, 1e10)
        
        try:
            self.vel_plot.setXRange(x_min, x_max)
            self.vel_plot.setYRange(y_min_clamped, y_max_clamped)
            self.div_plot.setXRange(x_min, x_max)
            self.div_plot.setYRange(y_min_clamped, y_max_clamped)
            self.vort_plot.setXRange(x_min, x_max)
            self.vort_plot.setYRange(y_min_clamped, y_max_clamped)
            self.pressure_plot.setXRange(x_min, x_max)
            self.pressure_plot.setYRange(y_min_clamped, y_max_clamped)
            self.scalar_plot.setXRange(x_min, x_max)
            self.scalar_plot.setYRange(y_min_clamped, y_max_clamped)
        except Exception as e:
            print(f"Warning: Failed to set plot ranges: {e}")
            # Fallback to simple ranges
            self.vel_plot.setXRange(0, lx)
            self.vel_plot.setYRange(0, ly)
            self.div_plot.setXRange(0, lx)
            self.div_plot.setYRange(0, ly)
            self.vort_plot.setXRange(0, lx)
            self.vort_plot.setYRange(0, ly)
            self.pressure_plot.setXRange(0, lx)
            self.pressure_plot.setYRange(0, ly)
            self.scalar_plot.setXRange(0, lx)
            self.scalar_plot.setYRange(0, ly)

        # Lock aspect ratio
        self.vel_plot.setAspectLocked(True)
        self.div_plot.setAspectLocked(True)
        self.vort_plot.setAspectLocked(True)
        self.pressure_plot.setAspectLocked(True)
        
        # Create image items
        self.vel_img = pg.ImageItem()
        self.div_img = pg.ImageItem()
        self.vort_img = pg.ImageItem()
        self.pressure_img = pg.ImageItem()

        # Initialize fixed colormap levels to prevent jumping during startup
        self.vel_levels = [0.0, 2.5]  # Fixed velocity range (0 to 2.5)
        self.div_levels = [-10.0, 10.0]  # Fixed divergence range (-10 to 10)
        self.vort_levels = [-5.0, 5.0]  # Fixed vorticity range (-5 to 5)
        self.pressure_levels = [-1.0, 1.0]  # Fixed pressure range (-1 to 1)
        self.level_update_counter = 0
        self.level_update_interval = 1  # Update every frame

        # Set initial colormaps
        self._setup_colormaps()
        
        # Add colorbars - both will use dynamic ranges based on U_inf
        # Initialize with default ranges, will be updated dynamically
        self.vel_colorbar = pg.ColorBarItem(values=(0, 1.5), width=20)
        self.vel_colorbar.setImageItem(self.vel_img, insert_in=self.vel_plot)

        self.div_colorbar = pg.ColorBarItem(values=(-10.0, 10.0), width=20)
        self.div_colorbar.setImageItem(self.div_img, insert_in=self.div_plot)

        self.vort_colorbar = pg.ColorBarItem(values=(-5.0, 5.0), width=20)
        self.vort_colorbar.setImageItem(self.vort_img, insert_in=self.vort_plot)

        self.pressure_colorbar = pg.ColorBarItem(values=(-1.0, 1.0), width=20)
        self.pressure_colorbar.setImageItem(self.pressure_img, insert_in=self.pressure_plot)

        self.scalar_colorbar = pg.ColorBarItem(values=(0, 1), width=20)
        self.scalar_colorbar.setImageItem(self.scalar_img, insert_in=self.scalar_plot)
        
        # Create SDF items before adding to plots
        try:
            self.vel_sdf = pg.ImageItem()
            self.div_sdf = pg.ImageItem()
            self.vort_sdf = pg.ImageItem()
        except Exception as e:
            print(f"Warning: Failed to create SDF items: {e}")
            self.vel_sdf = None
            self.div_sdf = None
            self.vort_sdf = None
        
        # Add image items FIRST (so they're in the background)
        self.vel_plot.addItem(self.vel_img)
        self.div_plot.addItem(self.div_img)
        self.vort_plot.addItem(self.vort_img)
        self.pressure_plot.addItem(self.pressure_img)

        # Initialize divergence image with zeros so it shows up when toggled on
        if hasattr(self, 'current_nx') and hasattr(self, 'current_ny'):
            zeros = np.zeros((self.current_nx, self.current_ny), dtype=np.float32)
            # Set image with explicit bounds
            from PyQt6.QtCore import QRectF
            rect = QRectF(
                0,                          # x (left) - physical coordinate
                self.current_y_min,          # y (bottom) - physical coordinate
                self.current_lx,             # width - PHYSICAL width (lx)
                self.current_y_max - self.current_y_min  # height - PHYSICAL height (ly)
            )
            self.div_img.setImage(zeros, autoLevels=False, rect=rect)
            self.div_colorbar.setLevels(self.div_levels)

        # Add SDF items SECOND (so they're between images and outlines) - only if they exist
        if self.vel_sdf is not None:
            self.vel_plot.addItem(self.vel_sdf)
        if self.div_sdf is not None:
            self.div_plot.addItem(self.div_sdf)
        if self.vort_sdf is not None:
            self.vort_plot.addItem(self.vort_sdf)
        
        # Add overlay items THIRD (so they're on top of everything)
        self._add_overlay_items()
        
        # Add stagnation and separation markers (vertical lines)
        self.stag_line_vel = pg.InfiniteLine(angle=90, pen=pg.mkPen('lime', width=2, style=Qt.PenStyle.DashLine))
        self.sep_line_vel = pg.InfiniteLine(angle=90, pen=pg.mkPen('red', width=2, style=Qt.PenStyle.DashLine))
        self.stag_line_vort = pg.InfiniteLine(angle=90, pen=pg.mkPen('lime', width=2, style=Qt.PenStyle.DashLine))
        self.sep_line_vort = pg.InfiniteLine(angle=90, pen=pg.mkPen('red', width=2, style=Qt.PenStyle.DashLine))
        self.vel_plot.addItem(self.stag_line_vel)
        self.vel_plot.addItem(self.sep_line_vel)
        self.vort_plot.addItem(self.stag_line_vort)
        self.vort_plot.addItem(self.sep_line_vort)

        # Add legends for velocity and vorticity plots using sample graphics items
        self.vel_legend = pg.LegendItem(offset=(80, 10))
        self.vel_legend.setParentItem(self.vel_plot)
        # Create sample curve items for legend (InfiniteLine doesn't work with legend.addItem)
        stag_sample = pg.PlotCurveItem(pen=pg.mkPen('lime', width=2, style=Qt.PenStyle.DashLine))
        sep_sample = pg.PlotCurveItem(pen=pg.mkPen('red', width=2, style=Qt.PenStyle.DashLine))
        self.vel_legend.addItem(stag_sample, 'Stagnation')
        self.vel_legend.addItem(sep_sample, 'Separation')

        self.vort_legend = pg.LegendItem(offset=(80, 10))
        self.vort_legend.setParentItem(self.vort_plot)
        # Create sample curve items for legend
        stag_sample_vort = pg.PlotCurveItem(pen=pg.mkPen('lime', width=2, style=Qt.PenStyle.DashLine))
        sep_sample_vort = pg.PlotCurveItem(pen=pg.mkPen('red', width=2, style=Qt.PenStyle.DashLine))
        self.vort_legend.addItem(stag_sample_vort, 'Stagnation')
        self.vort_legend.addItem(sep_sample_vort, 'Separation')

        # Hide markers and legends by default until airfoil_metrics are available
        self.stag_line_vel.setVisible(False)
        self.sep_line_vel.setVisible(False)
        self.stag_line_vort.setVisible(False)
        self.sep_line_vort.setVisible(False)
        self.vel_legend.setVisible(False)
        self.vort_legend.setVisible(False)
        
        # Configure performance
        self._configure_performance_settings()
        
        # Initialize obstacle renderer - only if outline items exist
        if self.vel_outline is not None and self.vort_outline is not None and self.scalar_outline is not None and self.pressure_outline is not None:
            try:
                self.obstacle_renderer = ObstacleRenderer(self.vel_outline, self.div_outline, self.vort_outline, self.scalar_outline, self.pressure_outline)
            except Exception as e:
                print(f"Warning: Failed to create obstacle renderer in setup_plots: {e}")
                self.obstacle_renderer = None
        else:
            self.obstacle_renderer = None
        
        # Configure plots
        for plot in [self.vel_plot, self.div_plot, self.vort_plot, self.pressure_plot]:
            plot.hideButtons()
            plot.enableAutoRange(False)
            plot.setAutoVisible(y=False)

        # Add cursor callout functionality
        self._setup_cursor_callouts()
        
        # Store grid parameters for later use
        self.current_nx = nx
        self.current_ny = ny
        self.current_lx = lx
        self.current_ly = ly
        self.current_y_min = y_min
        self.current_y_max = y_max
    
    def _setup_colormaps(self):
        """Setup initial colormaps for plots"""
        try:
            # Use matplotlib colormap loader for better compatibility
            plasma_colormap = pg.colormap.get('plasma')
            rdbu_colormap = pg.colormap.getFromMatplotlib('RdBu')
            
            if plasma_colormap is not None:
                plasma_lut = plasma_colormap.getLookupTable()
            else:
                plasma_colormap = pg.colormap.getFromMatplotlib('plasma')
                plasma_lut = plasma_colormap.getLookupTable()
                
            if rdbu_colormap is not None:
                rdbu_lut = rdbu_colormap.getLookupTable()
            else:
                # Fallback to a built-in colormap
                rdbu_colormap = pg.colormap.get('viridis')
                rdbu_lut = rdbu_colormap.getLookupTable()
                
        except Exception as e:
            print(f"Warning: Error loading colormaps, using fallbacks: {e}")
            plasma_lut = pg.colormap.get('plasma').getLookupTable()
            rdbu_lut = pg.colormap.get('viridis').getLookupTable()
        
        self.vel_img.setLookupTable(plasma_lut)
        self.vort_img.setLookupTable(rdbu_lut)

        # Use a diverging colormap for pressure (RdBu is good for pressure)
        self.pressure_img.setLookupTable(rdbu_lut)
    
    def _configure_performance_settings(self):
        """Configure performance optimization settings"""
        self.vel_img.autoDownsample = True
        self.vort_img.autoDownsample = True
        self.pressure_img.autoDownsample = True
        self.vel_img.downsampleMethod = 'average'
        self.vort_img.downsampleMethod = 'average'
        self.pressure_img.downsampleMethod = 'average'

    def _setup_cursor_callouts(self):
        """Setup cursor callouts with crosshair and data readout for plots"""
        # Create crosshair cursors
        self.vel_crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.vel_crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.div_crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.div_crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.vort_crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.vort_crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.pressure_crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.pressure_crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))

        # Add crosshairs to plots
        self.vel_plot.addItem(self.vel_crosshair_v, ignoreBounds=True)
        self.vel_plot.addItem(self.vel_crosshair_h, ignoreBounds=True)
        self.div_plot.addItem(self.div_crosshair_v, ignoreBounds=True)
        self.div_plot.addItem(self.div_crosshair_h, ignoreBounds=True)
        self.vort_plot.addItem(self.vort_crosshair_v, ignoreBounds=True)
        self.vort_plot.addItem(self.vort_crosshair_h, ignoreBounds=True)
        self.pressure_plot.addItem(self.pressure_crosshair_v, ignoreBounds=True)
        self.pressure_plot.addItem(self.pressure_crosshair_h, ignoreBounds=True)

        # Create data readout labels
        self.vel_data_label = pg.TextItem(anchor=(0, 1), color='w', fill=(0, 0, 0, 180))
        self.div_data_label = pg.TextItem(anchor=(0, 1), color='w', fill=(0, 0, 0, 180))
        self.vort_data_label = pg.TextItem(anchor=(0, 1), color='w', fill=(0, 0, 0, 180))
        self.pressure_data_label = pg.TextItem(anchor=(0, 1), color='w', fill=(0, 0, 0, 180))

        # Add labels to plots
        self.vel_plot.addItem(self.vel_data_label, ignoreBounds=True)
        self.div_plot.addItem(self.div_data_label, ignoreBounds=True)
        self.vort_plot.addItem(self.vort_data_label, ignoreBounds=True)
        self.pressure_plot.addItem(self.pressure_data_label, ignoreBounds=True)

        # Store current data for cursor readout
        self.current_vel_data = None
        self.current_div_data = None
        self.current_vort_data = None
        self.current_pressure_data = None

        # Store current cursor positions for live updates
        self.vel_cursor_pos = None
        self.div_cursor_pos = None
        self.vort_cursor_pos = None
        self.pressure_cursor_pos = None

        # Connect mouse move events to update crosshair and data readout
        self.vel_plot.scene().sigMouseMoved.connect(self._on_vel_mouse_moved)
        self.div_plot.scene().sigMouseMoved.connect(self._on_div_mouse_moved)
        self.vort_plot.scene().sigMouseMoved.connect(self._on_vort_mouse_moved)
        self.pressure_plot.scene().sigMouseMoved.connect(self._on_pressure_mouse_moved)

        # Hide crosshairs initially
        self.vel_crosshair_v.setVisible(False)
        self.vel_crosshair_h.setVisible(False)
        self.div_crosshair_v.setVisible(False)
        self.div_crosshair_h.setVisible(False)
        self.vort_crosshair_v.setVisible(False)
        self.vort_crosshair_h.setVisible(False)
        self.pressure_crosshair_v.setVisible(False)
        self.pressure_crosshair_h.setVisible(False)
        self.vel_data_label.setVisible(False)
        self.div_data_label.setVisible(False)
        self.vort_data_label.setVisible(False)
        self.pressure_data_label.setVisible(False)

    def _on_vel_mouse_moved(self, pos):
        """Handle mouse movement on velocity plot"""
        if self.vel_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.vel_plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Store cursor position for live updates
            self.vel_cursor_pos = (x, y)

            # Update crosshair position
            self.vel_crosshair_v.setPos(x)
            self.vel_crosshair_h.setPos(y)
            self.vel_crosshair_v.setVisible(True)
            self.vel_crosshair_h.setVisible(True)

            # Update data readout
            self._update_vel_readout(x, y)
        else:
            # Hide crosshairs when mouse leaves plot
            self.vel_crosshair_v.setVisible(False)
            self.vel_crosshair_h.setVisible(False)
            self.vel_data_label.setVisible(False)
            self.vel_cursor_pos = None

    def _on_div_mouse_moved(self, pos):
        """Handle mouse movement on divergence plot"""
        if self.div_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.div_plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Store cursor position for live updates
            self.div_cursor_pos = (x, y)

            # Update crosshair position
            self.div_crosshair_v.setPos(x)
            self.div_crosshair_h.setPos(y)
            self.div_crosshair_v.setVisible(True)
            self.div_crosshair_h.setVisible(True)

            # Update data readout
            self._update_div_readout(x, y)
        else:
            # Hide crosshairs when mouse leaves plot
            self.div_crosshair_v.setVisible(False)
            self.div_crosshair_h.setVisible(False)
            self.div_data_label.setVisible(False)
            self.div_cursor_pos = None

    def _on_vort_mouse_moved(self, pos):
        """Handle mouse movement on vorticity plot"""
        if self.vort_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.vort_plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Store cursor position for live updates
            self.vort_cursor_pos = (x, y)

            # Update crosshair position
            self.vort_crosshair_v.setPos(x)
            self.vort_crosshair_h.setPos(y)
            self.vort_crosshair_v.setVisible(True)
            self.vort_crosshair_h.setVisible(True)

            # Update data readout
            self._update_vort_readout(x, y)
        else:
            # Hide crosshairs when mouse leaves plot
            self.vort_crosshair_v.setVisible(False)
            self.vort_crosshair_h.setVisible(False)
            self.vort_data_label.setVisible(False)
            self.vort_cursor_pos = None

    def _on_pressure_mouse_moved(self, pos):
        """Handle mouse movement on pressure plot"""
        if self.pressure_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.pressure_plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Store cursor position for live updates
            self.pressure_cursor_pos = (x, y)

            # Update crosshair position
            self.pressure_crosshair_v.setPos(x)
            self.pressure_crosshair_h.setPos(y)
            self.pressure_crosshair_v.setVisible(True)
            self.pressure_crosshair_h.setVisible(True)

            # Update data readout
            self._update_pressure_readout(x, y)
        else:
            # Hide crosshairs when mouse leaves plot
            self.pressure_crosshair_v.setVisible(False)
            self.pressure_crosshair_h.setVisible(False)
            self.pressure_data_label.setVisible(False)
            self.pressure_cursor_pos = None

    def _update_div_readout(self, x, y):
        """Update divergence readout at given position using bilinear interpolation"""
        if self.current_div_data is not None:
            # Convert physical coordinates to continuous grid indices
            if hasattr(self, 'current_nx') and hasattr(self, 'current_lx'):
                fx = (x / self.current_lx) * self.current_nx
                fy = (y / self.current_ly) * self.current_ny

                # Get floor indices for lower-left corner
                ix = int(fx)
                iy = int(fy)

                # Check bounds (need to be at least 1 cell away from edge for bilinear)
                if 0 <= ix < self.current_nx - 1 and 0 <= iy < self.current_ny - 1:
                    # Fractional offsets within the cell
                    dx = fx - ix
                    dy = fy - iy

                    # Get values from 4 neighboring cells
                    v00 = self.current_div_data[ix, iy]
                    v10 = self.current_div_data[ix + 1, iy]
                    v01 = self.current_div_data[ix, iy + 1]
                    v11 = self.current_div_data[ix + 1, iy + 1]

                    # Bilinear interpolation
                    div_value = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11
                    self.div_data_label.setText(f"Div: {div_value:.4f}\nPos: ({x:.2f}, {y:.2f})")
                    self.div_data_label.setPos(x + 0.5, y - 0.5)
                    self.div_data_label.setVisible(True)
                    return

        # If no data or out of bounds, hide label
        self.div_data_label.setVisible(False)

    def _update_vel_readout(self, x, y):
        """Update velocity readout at given position using bilinear interpolation"""
        if self.current_vel_data is not None:
            # Convert physical coordinates to continuous grid indices
            if hasattr(self, 'current_nx') and hasattr(self, 'current_lx'):
                fx = (x / self.current_lx) * self.current_nx
                fy = (y / self.current_ly) * self.current_ny

                # Get floor indices for lower-left corner
                ix = int(fx)
                iy = int(fy)

                # Check bounds (need to be at least 1 cell away from edge for bilinear)
                if 0 <= ix < self.current_nx - 1 and 0 <= iy < self.current_ny - 1:
                    # Fractional offsets within the cell
                    dx = fx - ix
                    dy = fy - iy

                    # Get values from 4 neighboring cells
                    v00 = self.current_vel_data[ix, iy]
                    v10 = self.current_vel_data[ix + 1, iy]
                    v01 = self.current_vel_data[ix, iy + 1]
                    v11 = self.current_vel_data[ix + 1, iy + 1]

                    # Bilinear interpolation
                    vel_value = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11
                    self.vel_data_label.setText(f"Vel: {vel_value:.4f} m/s\nPos: ({x:.2f}, {y:.2f})")
                    self.vel_data_label.setPos(x + 0.5, y - 0.5)
                    self.vel_data_label.setVisible(True)
                    return

        # If no data or out of bounds, hide label
        self.vel_data_label.setVisible(False)

    def _update_vort_readout(self, x, y):
        """Update vorticity readout at given position using bilinear interpolation"""
        if self.current_vort_data is not None:
            # Convert physical coordinates to continuous grid indices
            if hasattr(self, 'current_nx') and hasattr(self, 'current_lx'):
                fx = (x / self.current_lx) * self.current_nx
                fy = (y / self.current_ly) * self.current_ny

                # Get floor indices for lower-left corner
                ix = int(fx)
                iy = int(fy)

                # Check bounds (need to be at least 1 cell away from edge for bilinear)
                if 0 <= ix < self.current_nx - 1 and 0 <= iy < self.current_ny - 1:
                    # Fractional offsets within the cell
                    dx = fx - ix
                    dy = fy - iy

                    # Get values from 4 neighboring cells
                    v00 = self.current_vort_data[ix, iy]
                    v10 = self.current_vort_data[ix + 1, iy]
                    v01 = self.current_vort_data[ix, iy + 1]
                    v11 = self.current_vort_data[ix + 1, iy + 1]

                    # Bilinear interpolation
                    vort_value = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11
                    self.vort_data_label.setText(f"Vort: {vort_value:.4f} 1/s\nPos: ({x:.2f}, {y:.2f})")
                    self.vort_data_label.setPos(x + 0.5, y - 0.5)
                    self.vort_data_label.setVisible(True)
                    return

        # If no data or out of bounds, hide label
        self.vort_data_label.setVisible(False)

    def _update_pressure_readout(self, x, y):
        """Update pressure readout at given position using bilinear interpolation"""
        if self.current_pressure_data is not None:
            # Convert physical coordinates to continuous grid indices
            if hasattr(self, 'current_nx') and hasattr(self, 'current_lx'):
                fx = (x / self.current_lx) * self.current_nx
                fy = (y / self.current_ly) * self.current_ny

                # Get floor indices for lower-left corner
                ix = int(fx)
                iy = int(fy)

                # Check bounds (need to be at least 1 cell away from edge for bilinear)
                if 0 <= ix < self.current_nx - 1 and 0 <= iy < self.current_ny - 1:
                    # Fractional offsets within the cell
                    dx = fx - ix
                    dy = fy - iy

                    # Get values from 4 neighboring cells
                    v00 = self.current_pressure_data[ix, iy]
                    v10 = self.current_pressure_data[ix + 1, iy]
                    v01 = self.current_pressure_data[ix, iy + 1]
                    v11 = self.current_pressure_data[ix + 1, iy + 1]

                    # Bilinear interpolation
                    pressure_value = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11
                    self.pressure_data_label.setText(f"Pres: {pressure_value:.4f}\nPos: ({x:.2f}, {y:.2f})")
                    self.pressure_data_label.setPos(x + 0.5, y - 0.5)
                    self.pressure_data_label.setVisible(True)
                    return

        # If no data or out of bounds, hide label
        self.pressure_data_label.setVisible(False)
    
    def _add_overlay_items(self):
        """Add overlay items for outlines and SDF visualization"""
        # Use QGraphicsPolygonItem for filled airfoil mask (black)
        try:
            from PyQt6.QtWidgets import QGraphicsPolygonItem
            from PyQt6.QtGui import QPolygonF, QBrush, QColor
            from PyQt6.QtCore import QPointF
            
            # Create filled polygon items for obstacle mask
            self.vel_outline = QGraphicsPolygonItem()
            self.vel_outline.setBrush(QBrush(QColor(200, 200, 200, 255)))  # Light grey for velocity plot
            self.vel_outline.setPen(pg.mkPen('k', width=1))

            self.div_outline = QGraphicsPolygonItem()
            self.div_outline.setBrush(QBrush(QColor(0, 0, 0, 255)))  # Black with full opacity
            self.div_outline.setPen(pg.mkPen('k', width=1))

            self.vort_outline = QGraphicsPolygonItem()
            self.vort_outline.setBrush(QBrush(QColor(0, 0, 0, 255)))  # Black with full opacity
            self.vort_outline.setPen(pg.mkPen('k', width=1))

            self.scalar_outline = QGraphicsPolygonItem()
            self.scalar_outline.setBrush(QBrush(QColor(0, 0, 0, 255)))  # Black with full opacity
            self.scalar_outline.setPen(pg.mkPen('k', width=1))

            self.pressure_outline = QGraphicsPolygonItem()
            self.pressure_outline.setBrush(QBrush(QColor(0, 0, 0, 255)))  # Black with full opacity
            self.pressure_outline.setPen(pg.mkPen('k', width=1))

            # Add outline items to plots (on top of images)
            if hasattr(self, 'vel_plot') and self.vel_plot is not None:
                self.vel_plot.addItem(self.vel_outline)
                self.vel_outline.setVisible(True)
            if hasattr(self, 'div_plot') and self.div_plot is not None:
                self.div_plot.addItem(self.div_outline)
                self.div_outline.setVisible(True)
            if hasattr(self, 'vort_plot') and self.vort_plot is not None:
                self.vort_plot.addItem(self.vort_outline)
                self.vort_outline.setVisible(True)
            if hasattr(self, 'scalar_plot') and self.scalar_plot is not None:
                self.scalar_plot.addItem(self.scalar_outline)
                self.scalar_outline.setVisible(True)
            if hasattr(self, 'pressure_plot') and self.pressure_plot is not None:
                self.pressure_plot.addItem(self.pressure_outline)
                self.pressure_outline.setVisible(True)

        except Exception as e:
            print(f"Warning: Failed to create outline items: {e}")
            self.vel_outline = None
            self.div_outline = None
            self.vort_outline = None
            self.scalar_outline = None
            self.pressure_outline = None
    
    def update_plots_for_new_grid(self, nx, ny, lx, ly):
        """Update plots when grid size changes - just update ranges, don't recreate plots"""
        
        # Get actual y-bounds
        y_min = 0.0
        y_max = ly
        
        if hasattr(self, 'solver') and self.solver is not None:
            try:
                if hasattr(self.solver, 'grid') and hasattr(self.solver.grid, 'y'):
                    y_coords = np.array(self.solver.grid.y)
                    y_min = y_coords.min()
                    y_max = y_coords.max()
            except:
                pass
        
        # Store current grid parameters
        self.current_nx = nx
        self.current_ny = ny
        self.current_lx = lx
        self.current_ly = ly
        self.current_y_min = y_min
        self.current_y_max = y_max
        
        # Update plot ranges without recreating plots
        try:
            if self.vel_plot is not None:
                self.vel_plot.setXRange(0, lx)
                self.vel_plot.setYRange(y_min, y_max)
            if self.vort_plot is not None:
                self.vort_plot.setXRange(0, lx)
                self.vort_plot.setYRange(y_min, y_max)
            if self.pressure_plot is not None:
                self.pressure_plot.setXRange(0, lx)
                self.pressure_plot.setYRange(y_min, y_max)
            if self.scalar_plot is not None:
                self.scalar_plot.setXRange(0, lx)
                self.scalar_plot.setYRange(y_min, y_max)
            if self.div_plot is not None:
                self.div_plot.setXRange(0, lx)
                self.div_plot.setYRange(y_min, y_max)
        except Exception as e:
            print(f"Warning: Failed to update plot ranges: {e}")
        
        # Recreate obstacle outlines with new grid dimensions
        self._add_overlay_items()
        
        # CRITICAL FIX: Recreate obstacle renderer only after all items are created
        try:
            if hasattr(self, 'vel_outline') and self.vel_outline is not None and \
               hasattr(self, 'vort_outline') and self.vort_outline is not None and \
               hasattr(self, 'scalar_outline') and self.scalar_outline is not None and \
               hasattr(self, 'pressure_outline') and self.pressure_outline is not None:
                self.obstacle_renderer = ObstacleRenderer(self.vel_outline, self.vort_outline, self.scalar_outline, self.pressure_outline)
                
                # Update outlines immediately with current solver
                if hasattr(self, 'solver') and self.solver is not None:
                    try:
                        self.obstacle_renderer.update_obstacle_outlines(self.solver)
                    except Exception as outline_error:
                        print(f"Warning: Failed to update outlines after grid change: {outline_error}")
                else:
                    print("Warning: No solver available for outline update")
            else:
                print("Warning: Outline items not available after grid change")
        except Exception as renderer_error:
            print(f"Warning: Failed to recreate obstacle renderer: {renderer_error}")
            # Fallback to simple ranges
            self.vel_plot.setXRange(0, lx)
            self.vel_plot.setYRange(0, ly)
            self.vort_plot.setXRange(0, lx)
            self.vort_plot.setYRange(0, ly)
            self.pressure_plot.setXRange(0, lx)
            self.pressure_plot.setYRange(0, ly)
            if self.scalar_plot is not None:
                self.scalar_plot.setXRange(0, lx)
                self.scalar_plot.setYRange(0, ly)
        
        # Don't reset levels - keep existing levels to prevent sudden jumps
        self.level_update_counter = 0
        self.level_update_interval = 1  # Update every frame
    
    def update_visualization(self, vel_mag_data, vort_data, pressure_data=None, div_data=None, show_velocity=True, show_vorticity=True, show_pressure=True, show_dye=True):
        """Update visualization with new data"""
        import time
        t_start = time.time()

        try:
            # Check if data dimensions match expected grid dimensions
            if vel_mag_data is not None and hasattr(self, 'current_nx'):
                if vel_mag_data.shape != (self.current_nx, self.current_ny):
                    print(f"Warning: Data shape {vel_mag_data.shape} doesn't match grid {self.current_nx}x{self.current_ny}, skipping visualization update")
                    return
        except Exception as e:
            print(f"Error in update_visualization: {e}")
            return

        # Store current data for cursor readout (always store, even if not visible)
        if vel_mag_data is not None:
            self.current_vel_data = np.array(vel_mag_data)
        if vort_data is not None:
            self.current_vort_data = np.array(vort_data)
        if pressure_data is not None:
            self.current_pressure_data = np.array(pressure_data)
        if div_data is not None:
            self.current_div_data = np.array(div_data)

        # Update cursor readouts if cursor is stationary (live updates)
        if self.vel_cursor_pos is not None:
            x, y = self.vel_cursor_pos
            self._update_vel_readout(x, y)
        if self.div_cursor_pos is not None:
            x, y = self.div_cursor_pos
            self._update_div_readout(x, y)
        if self.vort_cursor_pos is not None:
            x, y = self.vort_cursor_pos
            self._update_vort_readout(x, y)
        if self.pressure_cursor_pos is not None:
            x, y = self.pressure_cursor_pos
            self._update_pressure_readout(x, y)
        
        # Update velocity plot
        t_vel_start = time.time()
        if show_velocity and self.vel_img is not None and vel_mag_data is not None:
            
            # Check if lid_driven_cavity flow type needs rotation
            is_lid_driven_cavity = (hasattr(self.solver, 'sim_params') and 
                                   hasattr(self.solver.sim_params, 'flow_type') and
                                   self.solver.sim_params.flow_type == 'lid_driven_cavity')
            
            # Simulation: vel_mag_data[nx][ny] where nx=x, ny=y
            # PyQtGraph: image[ny][nx] where rows=y, cols=x
            # For LDC, lid is at top (last y-index in simulation), so don't transpose
            vel_data_correct = vel_mag_data
            
            # Convert to float32 to prevent levels error
            if vel_data_correct.dtype != np.float32:
                vel_data_correct = vel_data_correct.astype(np.float32)
            
            # CRITICAL: Set the image with explicit rectangle bounds
            # Rectangle is (x, y, width, height) where (x,y) is bottom-left corner
            # IMPORTANT: Use PHYSICAL dimensions, not grid dimensions
            rect = QRectF(
                0,                          # x (left) - physical coordinate
                self.current_y_min,          # y (bottom) - physical coordinate
                self.current_lx,             # width - PHYSICAL width (lx)
                self.current_y_max - self.current_y_min  # height - PHYSICAL height (ly)
            )

            # Apply logarithmic scaling if checkbox is enabled
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'log_colorscale_checkbox'):
                use_log = self.control_panel.log_colorscale_checkbox.isChecked()
            else:
                use_log = True  # Default to log scaling if checkbox not available

            if use_log:
                # Filter out invalid values before log1p
                vel_valid = np.nan_to_num(vel_data_correct, nan=0.0, posinf=0.0, neginf=0.0)
                vel_valid = np.maximum(vel_valid, 0.0)  # Ensure non-negative for log1p
                vel_display = np.log1p(vel_valid)  # log(1 + x) to handle zero values
            else:
                vel_display = vel_data_correct

            # Apply spatial weighting if checkbox is enabled
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'spatial_colorscale_checkbox'):
                use_spatial = self.control_panel.spatial_colorscale_checkbox.isChecked()
            else:
                use_spatial = True  # Default to spatial weighting if checkbox not available

            if use_spatial:
                # Create y-coordinate array from 0 to ly
                y_coords = np.linspace(0, self.current_ly, self.current_ny)
                # Normalize to -1 to 1 range centered at 0
                y_normalized = (y_coords - self.current_ly / 2) / (self.current_ly / 2)
                # Weight function: 1.0 at center, decreases towards walls
                # Use cosine shape: weight = 0.5 * (1 + cos(pi * y_normalized))
                weight_y = 0.5 * (1 + np.cos(np.pi * y_normalized))
                # Broadcast weight to match data shape (nx, ny)
                weight_grid = np.tile(weight_y, (self.current_nx, 1))
                # Apply weighting
                vel_display = vel_display * weight_grid
            
            # Upscale for smooth visualization (display only, physics unchanged)
            t_upscale_start = time.time()
            vel_display = _upscale_for_display(vel_display, scale_factor=self.upscale_factor)
            t_upscale_end = time.time()

            # Set image with explicit bounds
            t_setimg_start = time.time()
            self.vel_img.setImage(vel_display, autoLevels=False, rect=rect)
            t_setimg_end = time.time()

            # Only update colorbar levels if adaptive scaling is enabled
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'adaptive_colorscale_checkbox'):
                use_adaptive = self.control_panel.adaptive_colorscale_checkbox.isChecked()
            else:
                use_adaptive = True  # Default to adaptive if checkbox not available

            if use_adaptive:
                # Calculate actual data range for adaptive color scaling
                vel_min = float(np.nanmin(vel_display))
                vel_max = float(np.nanmax(vel_display))

                # Add small padding to avoid extremes at color boundaries
                vel_range = vel_max - vel_min
                if vel_range > 0:
                    vel_min_padded = vel_min - 0.05 * vel_range
                    vel_max_padded = vel_max + 0.05 * vel_range
                else:
                    vel_min_padded = vel_min
                    vel_max_padded = vel_max + 1e-6  # Ensure non-zero range

                self.vel_colorbar.setLevels([vel_min_padded, vel_max_padded])  # Update colorbar
                # Colorbar will automatically update image levels via setImageItem connection
        t_vel_end = time.time()

        # Update divergence plot
        t_div_start = time.time()
        if self.div_img is not None and div_data is not None:
            div_data_correct = div_data.astype(np.float32) if div_data.dtype != np.float32 else div_data

            # Set image with explicit bounds
            rect = QRectF(
                0,                          # x (left) - physical coordinate
                self.current_y_min,          # y (bottom) - physical coordinate
                self.current_lx,             # width - PHYSICAL width (lx)
                self.current_y_max - self.current_y_min  # height - PHYSICAL height (ly)
            )

            self.div_img.setImage(div_data_correct, autoLevels=False, rect=rect)
            self.div_colorbar.setLevels(self.div_levels)
        t_div_end = time.time()

        # Update vorticity plot
        t_vort_start = time.time()
        if show_vorticity and self.vort_img is not None and vort_data is not None:

            # Check if lid_driven_cavity flow type needs rotation
            is_lid_driven_cavity = (hasattr(self.solver, 'sim_params') and 
                                   hasattr(self.solver.sim_params, 'flow_type') and
                                   self.solver.sim_params.flow_type == 'lid_driven_cavity')

            # Simulation: vort_data[nx][ny] where nx=x, ny=y
            # PyQtGraph: image[ny][nx] where rows=y, cols=x
            # For LDC, lid is at top (last y-index in simulation), so don't transpose
            vort_data_correct = vort_data
            
            # Convert to float32 to prevent levels error
            if vort_data_correct.dtype != np.float32:
                vort_data_correct = vort_data_correct.astype(np.float32)

            # Exclude wall boundaries from colormap level calculation
            # This prevents the colormap from focusing on high wall vorticity
            # instead of the interesting vorticity around the airfoil and wake
            if vort_data_correct.shape[0] > 4:  # Ensure we have enough rows
                # Exclude top and bottom boundary rows (walls)
                vort_data_for_levels = vort_data_correct[1:-1, :]
            else:
                vort_data_for_levels = vort_data_correct

            # Use the same rect approach as velocity plot for consistent scaling
            rect = QRectF(
                0,                          # x (left) - physical coordinate
                self.current_y_min,          # y (bottom) - physical coordinate
                self.current_lx,             # width - PHYSICAL width (lx)
                self.current_y_max - self.current_y_min  # height - PHYSICAL height (ly)
            )

            # Apply logarithmic scaling if checkbox is enabled
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'log_colorscale_checkbox'):
                use_log = self.control_panel.log_colorscale_checkbox.isChecked()
            else:
                use_log = True  # Default to log scaling if checkbox not available

            if use_log:
                # For log scale, use signed log: sign(x) * log(1 + |x|)
                vort_valid = np.nan_to_num(vort_data_correct, nan=0.0, posinf=0.0, neginf=0.0)
                vort_sign = np.sign(vort_valid)
                vort_abs = np.abs(vort_valid)
                vort_display = vort_sign * np.log1p(vort_abs)
            else:
                vort_display = vort_data_correct

            # Apply spatial weighting if checkbox is enabled
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'spatial_colorscale_checkbox'):
                use_spatial = self.control_panel.spatial_colorscale_checkbox.isChecked()
            else:
                use_spatial = True  # Default to spatial weighting if checkbox not available

            if use_spatial:
                # Create y-coordinate array from 0 to ly
                y_coords = np.linspace(0, self.current_ly, self.current_ny)
                # Normalize to -1 to 1 range centered at 0
                y_normalized = (y_coords - self.current_ly / 2) / (self.current_ly / 2)
                # Weight function: 1.0 at center, decreases towards walls
                # Use cosine shape: weight = 0.5 * (1 + cos(pi * y_normalized))
                weight_y = 0.5 * (1 + np.cos(np.pi * y_normalized))
                # Broadcast weight to match data shape (nx, ny)
                weight_grid = np.tile(weight_y, (self.current_nx, 1))
                # Apply weighting
                vort_display = vort_display * weight_grid
            
            # Upscale for smooth visualization (display only, physics unchanged)
            vort_display = _upscale_for_display(vort_display, scale_factor=self.upscale_factor)

            # Set image with explicit bounds
            self.vort_img.setImage(vort_display, autoLevels=False, rect=rect)

            # Only update colorbar levels if adaptive scaling is enabled
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'adaptive_colorscale_checkbox'):
                use_adaptive = self.control_panel.adaptive_colorscale_checkbox.isChecked()
            else:
                use_adaptive = True  # Default to adaptive if checkbox not available

            if use_adaptive:
                # Calculate actual data range for adaptive color scaling
                vort_min = float(np.nanmin(vort_display))
                vort_max = float(np.nanmax(vort_display))

                # Add small padding to avoid extremes at color boundaries
                vort_range = vort_max - vort_min
                if vort_range > 0:
                    vort_min_padded = vort_min - 0.05 * vort_range
                    vort_max_padded = vort_max + 0.05 * vort_range
                else:
                    vort_min_padded = vort_min
                    vort_max_padded = vort_max + 1e-6  # Ensure non-zero range

                self.vort_colorbar.setLevels([vort_min_padded, vort_max_padded])  # Update colorbar
                # Colorbar will automatically update image levels via setImageItem connection
        t_vort_end = time.time()
        
        # Update pressure plot
        t_press_start = time.time()
        if show_pressure and self.pressure_img is not None and pressure_data is not None:
            # Check if lid_driven_cavity flow type needs rotation
            is_lid_driven_cavity = (hasattr(self.solver, 'sim_params') and 
                                   hasattr(self.solver.sim_params, 'flow_type') and
                                   self.solver.sim_params.flow_type == 'lid_driven_cavity')
            
            # Simulation: pressure_data[nx][ny] where nx=x, ny=y
            # PyQtGraph: image[ny][nx] where rows=y, cols=x
            # For LDC, lid is at top (last y-index in simulation), so don't transpose
            pressure_data_correct = pressure_data

            # Convert to float32 to prevent levels error
            if pressure_data_correct.dtype != np.float32:
                pressure_data_correct = pressure_data_correct.astype(np.float32)

            # Use the same rect approach for consistent scaling
            rect = QRectF(
                0,                          # x (left) - physical coordinate
                self.current_y_min,          # y (bottom) - physical coordinate
                self.current_lx,             # width - PHYSICAL width (lx)
                self.current_y_max - self.current_y_min  # height - PHYSICAL height (ly)
            )

            # Upscale for smooth visualization
            pressure_display = _upscale_for_display(pressure_data_correct, scale_factor=self.upscale_factor)

            # Set image with explicit bounds
            self.pressure_img.setImage(pressure_display, autoLevels=False, rect=rect)

            # Update pressure colorbar levels (adaptive scaling)
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'adaptive_colorscale_checkbox'):
                use_adaptive = self.control_panel.adaptive_colorscale_checkbox.isChecked()
            else:
                use_adaptive = True  # Default to adaptive

            if use_adaptive:
                # Calculate actual data range for adaptive color scaling
                pressure_min = float(np.nanmin(pressure_display))
                pressure_max = float(np.nanmax(pressure_display))

                # Add small padding to avoid extremes at color boundaries
                pressure_range = pressure_max - pressure_min
                if pressure_range > 0:
                    pressure_min_padded = pressure_min - 0.05 * pressure_range
                    pressure_max_padded = pressure_max + 0.05 * pressure_range
                else:
                    pressure_min_padded = pressure_min
                    pressure_max_padded = pressure_max + 1e-6  # Ensure non-zero range

                self.pressure_colorbar.setLevels([pressure_min_padded, pressure_max_padded])
        t_press_end = time.time()
        
        # Store visualization timing data for overlay (no console print)
        if not hasattr(self, '_viz_detail_frame_count'):
            self._viz_detail_frame_count = 0
        self._viz_detail_frame_count += 1
        
        viz_timing = {
            'velocity': (t_vel_end - t_vel_start) * 1000,
            'vorticity': (t_vort_end - t_vort_start) * 1000,
            'pressure': (t_press_end - t_press_start) * 1000,
            'viz_total': (time.time() - t_start) * 1000
        }
        
        # Store as instance variable for access
        self._latest_viz_timing = viz_timing
        
        # Update levels cache
        self.level_update_counter += 1
        # DISABLED: Dynamic level adjustment to isolate the snapping issue
        # Using fixed levels instead
        # if self.level_update_counter % self.level_update_interval == 0:
        
        # Update obstacle outlines (less frequent)
        if self.obstacle_renderer is not None and self.solver is not None:
            if self.level_update_counter % 5 == 0:  # Update every 5 frames instead of every frame
                self.obstacle_renderer.update_obstacle_outlines(self.solver)
        
        # Update streamlines (less frequent for performance)
        if self.show_streamlines and self.solver is not None:
            self.streamline_counter += 1
            if self.streamline_counter % self.streamline_update_interval == 0:
                # Get u and v from solver, interpolate to cell centers if MAC grid
                grid_type = getattr(self.solver.sim_params, 'grid_type', 'collocated')
                u_data, v_data = _get_cell_centered_velocities(self.solver.u, self.solver.v, grid_type)
                self._update_streamlines(u_data, v_data)
        
        # Update quivers (less frequent for performance)
        if self.show_quivers and self.solver is not None:
            self.quiver_counter += 1
            if self.quiver_counter % self.quiver_update_interval == 0:
                # Get u and v from solver, interpolate to cell centers if MAC grid
                grid_type = getattr(self.solver.sim_params, 'grid_type', 'collocated')
                u_data, v_data = _get_cell_centered_velocities(self.solver.u, self.solver.v, grid_type)
                self._update_quivers(u_data, v_data)
        
        # Update stagnation and separation markers (only for von_karman flow)
        if (hasattr(self.solver, 'sim_params') and
            self.solver.sim_params.flow_type == 'von_karman' and
            hasattr(self.solver, 'history') and
            'airfoil_metrics' in self.solver.history):
            metrics = self.solver.history['airfoil_metrics']
            if metrics['stagnation_x'] and metrics['separation_x']:
                stag_x = metrics['stagnation_x'][-1]
                sep_x = metrics['separation_x'][-1]

                # Update velocity plot lines
                if self.stag_line_vel is not None:
                    self.stag_line_vel.setPos(stag_x)
                    self.stag_line_vel.setVisible(self.show_stagnation_marker)
                if self.sep_line_vel is not None:
                    self.sep_line_vel.setPos(sep_x)
                    self.sep_line_vel.setVisible(self.show_separation_marker)

                # Update vorticity plot lines
                if self.stag_line_vort is not None:
                    self.stag_line_vort.setPos(stag_x)
                    self.stag_line_vort.setVisible(self.show_stagnation_marker)
                if self.sep_line_vort is not None:
                    self.sep_line_vort.setPos(sep_x)
                    self.sep_line_vort.setVisible(self.show_separation_marker)

                # Show legends if markers are visible
                if self.vel_legend is not None:
                    self.vel_legend.setVisible(self.show_stagnation_marker or self.show_separation_marker)
                if self.vort_legend is not None:
                    self.vort_legend.setVisible(self.show_stagnation_marker or self.show_separation_marker)
        else:
            # Hide markers and legends for non-von_karman flows
            if self.stag_line_vel is not None:
                self.stag_line_vel.setVisible(False)
            if self.sep_line_vel is not None:
                self.sep_line_vel.setVisible(False)
            if self.stag_line_vort is not None:
                self.stag_line_vort.setVisible(False)
            if self.sep_line_vort is not None:
                self.sep_line_vort.setVisible(False)
            if self.vel_legend is not None:
                self.vel_legend.setVisible(False)
            if self.vort_legend is not None:
                self.vort_legend.setVisible(False)
    
    def _update_streamlines(self, u_data, v_data):
        """Compute and update streamlines using numpy"""
        try:
            # Clear existing streamlines from vorticity plot
            for item in self.streamline_items:
                try:
                    if item.scene() is not None:
                        self.vort_plot.removeItem(item)
                except:
                    pass
            self.streamline_items.clear()
            
            # Convert to numpy once
            if hasattr(u_data, 'toArray'):
                u_np = np.array(u_data.toArray())
            else:
                u_np = np.array(u_data)
            
            if hasattr(v_data, 'toArray'):
                v_np = np.array(v_data.toArray())
            else:
                v_np = np.array(v_data)
            
            # Get mask (not used currently)
            mask_data = self.solver.mask if hasattr(self.solver, 'mask') else None
            if mask_data is not None:
                if hasattr(mask_data, 'toArray'):
                    mask_np = np.array(mask_data.toArray())
                else:
                    mask_np = np.array(mask_data)
            else:
                mask_np = None
            
            # Get grid dimensions
            nx, ny = u_np.shape
            lx = self.current_lx if hasattr(self, 'current_lx') else 4.0
            y_min = self.current_y_min if hasattr(self, 'current_y_min') else 0.0
            y_max = self.current_y_max if hasattr(self, 'current_y_max') else 2.0
            
            # Adaptive seeding based on vorticity magnitude
            # Get vorticity field from solver
            vort_data = self.solver.vorticity if hasattr(self.solver, 'vorticity') else None
            if vort_data is not None:
                if hasattr(vort_data, 'toArray'):
                    vort_np = np.array(vort_data.toArray())
                else:
                    vort_np = np.array(vort_data)
                
                # Compute vorticity magnitude
                vort_mag = np.abs(vort_np)
                
                # Normalize vorticity to get probability weights
                vort_max = np.max(vort_mag)
                if vort_max > 0:
                    vort_weight = vort_mag / vort_max
                else:
                    vort_weight = np.ones_like(vort_mag) * 0.1
                
                # Add minimum weight to ensure some streamlines everywhere
                vort_weight = vort_weight * 0.9 + 0.1
                
                # Generate seed points based on vorticity-weighted sampling
                seed_x_list = []
                seed_y_list = []
                num_seeds = self.streamline_density * self.streamline_density
                
                # Grid coordinates
                x_coords = np.linspace(0, lx, nx)
                y_coords = np.linspace(y_min, y_max, ny)
                
                # Flatten weights and coordinates
                vort_weight_flat = vort_weight.flatten()
                x_coords_flat, y_coords_flat = np.meshgrid(x_coords, y_coords, indexing='ij')
                x_coords_flat = x_coords_flat.flatten()
                y_coords_flat = y_coords_flat.flatten()
                
                # Normalize weights to probabilities
                prob = vort_weight_flat / np.sum(vort_weight_flat)
                
                # Sample seed points based on vorticity weights
                indices = np.random.choice(len(prob), size=num_seeds, p=prob, replace=True)
                seed_x_list = x_coords_flat[indices]
                seed_y_list = y_coords_flat[indices]
                
                seed_x_np = seed_x_list
                seed_y_np = seed_y_list
            else:
                # Fallback to uniform grid if vorticity not available
                seed_x_np = np.linspace(lx * 0.1, lx * 0.9, self.streamline_density)
                seed_y_np = np.linspace(y_min + (y_max - y_min) * 0.1, y_min + (y_max - y_min) * 0.9, self.streamline_density)
            
            # Integration parameters
            max_steps = 100
            ds = lx / nx * 2  # Step size
            
            # Compute all streamlines using numpy
            streamlines = self._compute_streamlines_numpy(seed_x_np, seed_y_np, u_np, v_np, mask_np, lx, y_min, y_max, nx, ny, max_steps, ds)
            
            # Draw streamlines on vorticity plot (black lines)
            for streamline_x, streamline_y in streamlines:
                curve = pg.PlotCurveItem(streamline_x, streamline_y, 
                                        pen=pg.mkPen('k', width=1, alpha=0.7))
                self.vort_plot.addItem(curve)
                self.streamline_items.append(curve)
                        
        except Exception as e:
            print(f"Error updating streamlines: {e}")
    
    def _compute_streamlines_numpy(self, seed_x_np, seed_y_np, u_np, v_np, mask_np, lx, y_min, y_max, nx, ny, max_steps, ds):
        """Compute streamlines using numpy"""
        streamlines = []
        ly = y_max - y_min  # Height of domain
        
        for sx in seed_x_np:
            for sy in seed_y_np:
                # Initialize streamline
                cx, cy = sx, sy
                streamline_x = [cx]
                streamline_y = [cy]
                
                for _ in range(max_steps):
                    # Find nearest grid point (convert physical coordinates to grid indices)
                    ix = int(cx / lx * nx)
                    iy = int((cy - y_min) / ly * ny)
                    
                    # Check bounds
                    if ix < 0 or ix >= nx - 1 or iy < 0 or iy >= ny - 1:
                        break
                    
                    # Bilinear interpolation of velocity
                    fx = (cx / lx * nx) - ix
                    fy = (cy / ly * ny) - iy
                    
                    u_interp = ((1-fx)*(1-fy)*u_np[ix,iy] + fx*(1-fy)*u_np[ix+1,iy] + 
                               (1-fx)*fy*u_np[ix,iy+1] + fx*fy*u_np[ix+1,iy+1])
                    v_interp = ((1-fx)*(1-fy)*v_np[ix,iy] + fx*(1-fy)*v_np[ix+1,iy] + 
                               (1-fx)*fy*v_np[ix,iy+1] + fx*fy*v_np[ix+1,iy+1])
                    
                    # Normalize velocity
                    vel_mag = np.sqrt(u_interp**2 + v_interp**2)
                    if vel_mag < 1e-6:
                        break
                    
                    # Step forward
                    cx += (u_interp / vel_mag) * ds
                    cy += (v_interp / vel_mag) * ds
                    
                    streamline_x.append(cx)
                    streamline_y.append(cy)
                    
                    # Check if out of bounds
                    if cx < 0 or cx > lx or cy < y_min or cy > y_max:
                        break
                
                streamlines.append((np.array(streamline_x), np.array(streamline_y)))
        
        return streamlines
    
    def toggle_streamlines(self, enabled: bool):
        """Toggle streamline display"""
        self.show_streamlines = enabled
        if not enabled:
            # Clear existing streamlines from vorticity plot
            for item in self.streamline_items:
                try:
                    if item.scene() is not None:
                        self.vort_plot.removeItem(item)
                except:
                    pass
            self.streamline_items.clear()
    
    def _update_quivers(self, u_data, v_data):
        """Update quiver arrows using numpy and PyQtGraph ArrowItem"""
        try:
            # Clear existing quivers from scalar plot
            for item in self.quiver_items:
                try:
                    if hasattr(item, 'scene') and item.scene() is not None:
                        self.scalar_plot.removeItem(item)
                except:
                    pass
                # Also try to remove directly if scene check fails
                try:
                    self.scalar_plot.removeItem(item)
                except:
                    pass
            self.quiver_items.clear()
            
            # Convert to numpy arrays
            if hasattr(u_data, 'toArray'):
                u_np = np.array(u_data.toArray())
            else:
                u_np = np.array(u_data)
            
            if hasattr(v_data, 'toArray'):
                v_np = np.array(v_data.toArray())
            else:
                v_np = np.array(v_data)
            
            # Get grid dimensions
            nx, ny = u_np.shape
            lx = self.current_lx if hasattr(self, 'current_lx') else 4.0
            y_min = self.current_y_min if hasattr(self, 'current_y_min') else 0.0
            y_max = self.current_y_max if hasattr(self, 'current_y_max') else 2.0
            
            # Create uniform grid of quiver positions
            x_indices = np.linspace(0, nx-1, self.quiver_density, dtype=int)
            y_indices = np.linspace(0, ny-1, self.quiver_density, dtype=int)
            
            # Get mask to avoid placing quivers inside obstacles
            mask_data = self.solver.mask if hasattr(self.solver, 'mask') else None
            if mask_data is not None:
                if hasattr(mask_data, 'toArray'):
                    mask_np = np.array(mask_data.toArray())
                else:
                    mask_np = np.array(mask_data)
            else:
                mask_np = np.ones((nx, ny), dtype=bool)
            
            # Create quivers at grid points
            for ix in x_indices:
                for iy in y_indices:
                    # Skip if inside obstacle
                    if not mask_np[ix, iy]:
                        continue
                    
                    # Get physical coordinates
                    x = (ix / nx) * lx
                    y = y_min + (iy / ny) * (y_max - y_min)
                    
                    # Get velocity components
                    u = u_np[ix, iy]
                    v = v_np[ix, iy]
                    
                    # Calculate velocity magnitude
                    vel_mag = np.sqrt(u**2 + v**2)
                    
                    # Skip if velocity is too small
                    if vel_mag < 1e-6:
                        continue
                    
                    # Get velocity color based on magnitude
                    vel_min, vel_max = self.vel_levels
                    vel_norm = (vel_mag - vel_min) / (vel_max - vel_min + 1e-6)
                    vel_norm = np.clip(vel_norm, 0, 1)
                    
                    # Map to viridis colormap (approximate RGB values)
                    # Simple viridis-like mapping
                    if vel_norm < 0.25:
                        # Purple to blue
                        r = int(68 + (59 - 68) * vel_norm * 4)
                        g = int(1 + (84 - 1) * vel_norm * 4)
                        b = int(84 + (251 - 84) * vel_norm * 4)
                    elif vel_norm < 0.5:
                        # Blue to teal
                        t = (vel_norm - 0.25) * 4
                        r = int(59 + (33 - 59) * t)
                        g = int(84 + (144 - 84) * t)
                        b = int(251 + (140 - 251) * t)
                    elif vel_norm < 0.75:
                        # Teal to yellow-green
                        t = (vel_norm - 0.5) * 4
                        r = int(33 + (253 - 33) * t)
                        g = int(144 + (231 - 144) * t)
                        b = int(140 + (37 - 140) * t)
                    else:
                        # Yellow-green to yellow
                        t = (vel_norm - 0.75) * 4
                        r = int(253 + (252 - 253) * t)
                        g = int(231 + (225 - 231) * t)
                        b = int(37 + (8 - 37) * t)
                    
                    arrow_color = (r, g, b)
                    
                    # Calculate arrow angle (add 180 to reverse direction)
                    angle = np.degrees(np.arctan2(v, u)) + 180
                    
                    # Scale arrow size based on velocity magnitude
                    arrow_scale = min(vel_mag * 2, 5)  # Scale factor
                    head_len = arrow_scale * 0.9  # Bigger head
                    head_width = arrow_scale * 0.8  # Bigger head
                    
                    # Calculate arrow shaft length (much shorter for cleaner look)
                    shaft_length = arrow_scale * 0.2  # Very short shaft
                    
                    # Calculate arrow end point
                    dx = shaft_length * np.cos(np.radians(angle))
                    dy = shaft_length * np.sin(np.radians(angle))
                    end_x = x + dx
                    end_y = y + dy
                    
                    # Create arrow shaft line with velocity color
                    arrow_line = pg.PlotCurveItem([x, end_x], [y, end_y], 
                                                pen=pg.mkPen(arrow_color, width=1))
                    self.scalar_plot.addItem(arrow_line)
                    self.quiver_items.append(arrow_line)
                    
                    # Create arrow head at start point (pointing toward end) with velocity color
                    arrow = pg.ArrowItem(pos=(x, y), angle=angle, 
                                      headLen=head_len, headWidth=head_width,
                                      pen=pg.mkPen(arrow_color, width=1), 
                                      brush=pg.mkBrush(arrow_color))
                    self.scalar_plot.addItem(arrow)
                    self.quiver_items.append(arrow)
                        
        except Exception as e:
            print(f"Error updating quivers: {e}")
    
    def toggle_quivers(self, enabled: bool):
        """Toggle quiver display"""
        self.show_quivers = enabled
        if not enabled:
            # Clear existing quivers from scalar plot
            for item in self.quiver_items:
                try:
                    if item.scene() is not None:
                        self.scalar_plot.removeItem(item)
                except:
                    pass
            self.quiver_items.clear()
    
    def update_coefficients(self, cl_value: float, cd_value: float, time_value: float) -> None:
        """Coefficient plots removed - no implementation needed"""
        pass

    def clear_error_plot(self) -> None:
        """Clear error plot curves"""
        if hasattr(self, 'l2_plot') and self.l2_plot is not None:
            try:
                # Clear all curves
                self.l2_curve = None
                self.max_error_curve = None
                self.rel_error_curve = None
                self.l2_u_curve = None
                self.l2_v_curve = None
                # Clear plot data
                self.l2_plot.clear()
            except:
                pass

    def update_l2_error(self, l2_error_value: float, time_value: float,
                       max_error_value: float = None, rel_error_value: float = None,
                       l2_error_u: float = None, l2_error_v: float = None,
                       rms_change: float = None, change_99p: float = None) -> None:
        """Update error plot with multiple metrics"""
        if hasattr(self, 'l2_plot') and self.l2_plot is not None:
            try:
                # Initialize error curves on first call
                if self.l2_curve is None:
                    self.l2_curve = self.l2_plot.plot(pen=pg.mkPen('r', width=2), name='L2 Error', connect='all')
                    self.max_error_curve = self.l2_plot.plot(pen=pg.mkPen('b', width=1), name='Max Error', connect='all')
                    self.rel_error_curve = self.l2_plot.plot(pen=pg.mkPen('g', width=1), name='Rel Error', connect='all')
                    self.l2_u_curve = self.l2_plot.plot(pen=pg.mkPen('m', width=1, style=Qt.PenStyle.DashLine), name='L2 U', connect='all')
                    self.l2_v_curve = self.l2_plot.plot(pen=pg.mkPen('c', width=1, style=Qt.PenStyle.DashLine), name='L2 V', connect='all')
                    self.rms_change_curve = self.l2_plot.plot(pen=pg.mkPen('k', width=2), name='RMS Change', connect='all')
                    self.change_99p_curve = self.l2_plot.plot(pen=pg.mkPen('y', width=2), name='Change 99p', connect='all')

                # Accumulate data with array length synchronization
                self.error_times.append(time_value)
                self.l2_errors.append(l2_error_value)

                # Always append to all arrays to maintain synchronized lengths
                self.max_errors.append(max_error_value if max_error_value is not None else None)
                self.rel_errors.append(rel_error_value if rel_error_value is not None else None)
                self.l2_u_errors.append(l2_error_u if l2_error_u is not None else None)
                self.l2_v_errors.append(l2_error_v if l2_error_v is not None else None)
                
                if rms_change is not None:
                    # Normalize RMS Change to be in similar range as other error metrics
                    # Divide by 100 to scale it down (RMS Change is typically 100x larger than L2 Error)
                    normalized_rms_change = rms_change * 100.0
                    self.rms_change_errors.append(normalized_rms_change)
                else:
                    self.rms_change_errors.append(None)
                
                self.change_99p_errors.append(change_99p if change_99p is not None else None)

                # Update all curves with synchronized data arrays
                self.l2_curve.setData(self.error_times, self.l2_errors)
                self.max_error_curve.setData(self.error_times, self.max_errors)
                self.rel_error_curve.setData(self.error_times, self.rel_errors)
                self.l2_u_curve.setData(self.error_times, self.l2_u_errors)
                self.l2_v_curve.setData(self.error_times, self.l2_v_errors)
                self.rms_change_curve.setData(self.error_times, self.rms_change_errors)
                self.change_99p_curve.setData(self.error_times, self.change_99p_errors)

                # Update x-range to show all data from 0 to current time
                if self.error_times:
                    max_time = max(self.error_times)
                    self.l2_plot.setXRange(0, max_time * 1.1)  # Add 10% padding

            except Exception as e:
                print(f"Error updating L2 error plot: {e}")
                # Fallback to simple L2 error plot
                if self.l2_curve is None:
                    self.l2_curve = self.l2_plot.plot(pen=pg.mkPen('r', width=2), name='L2 Error', connect='all')
                    self.error_times = []
                    self.l2_errors = []

                self.error_times.append(time_value)
                self.l2_errors.append(l2_error_value)
                self.l2_curve.setData(self.error_times, self.l2_errors)
    
    def clear_error_plot(self):
        """Clear error plot data arrays and reset curves."""
        self.error_times = []
        self.l2_errors = []
        self.max_errors = []
        self.rel_errors = []
        self.l2_u_errors = []
        self.l2_v_errors = []
        self.rms_change_errors = []
        self.change_99p_errors = []
        # Reset curves to None so they'll be recreated on next update
        self.l2_curve = None
        self.max_error_curve = None
        self.rel_error_curve = None
        self.l2_u_curve = None
        self.l2_v_curve = None
        self.rms_change_curve = None
        self.change_99p_curve = None
        # Reset Velocity plot title
        if hasattr(self, 'vel_plot') and self.vel_plot:
            self.vel_plot.setTitle("Velocity Magnitude")
    
    def update_plot_titles_with_fps(self, sim_fps: float, viz_fps: float):
        """Update FPS in Velocity plot title."""
        self.current_sim_fps = sim_fps
        self.current_viz_fps = viz_fps
        if hasattr(self, 'vel_plot') and self.vel_plot:
            self.vel_plot.setTitle(f"Velocity Magnitude (Sim - {sim_fps:.1f} FPS | Viz - {viz_fps:.1f} FPS)")
    
    def update_vorticity_title(self, re: float, u_inlet: float, naca: str, aoa: float):
        """Update Vorticity plot title with flow parameters."""
        if hasattr(self, 'vort_plot') and self.vort_plot:
            # Check if obstacle is cow FIRST (before using naca parameter)
            obstacle_type = self.solver.sim_params.obstacle_type if hasattr(self.solver.sim_params, 'obstacle_type') else None
            
            if obstacle_type == 'cow':
                naca_display = "Cow"
                aoa_display = "AoA - Probably?"
            elif obstacle_type == 'cylinder':
                naca_display = "Cylinder"
                aoa_display = "N/A"
            else:
                # Use HTML subscript for "inlet"
                # Check if naca already contains "NACA" to avoid duplication
                naca_display = naca if naca.upper().startswith('NACA') else f"NACA {naca}"
                aoa_display = f"{aoa:.1f}° AoA"
            
            title = f"Vorticity (Re = {re:.0f} | U<sub>inlet</sub> = {u_inlet:.2f} m/s | {naca_display} | {aoa_display})"
            self.vort_plot.setTitle(title)
    
    # update_coefficient_plots_with_metrics method removed - no coefficient plots
    
    def change_velocity_colormap(self, colormap_name):
        """Change colormap for velocity plot - simplified robust version"""
        try:
            # Basic validation
            if not colormap_name or not isinstance(colormap_name, str):
                return
            
            if self.vel_img is None:
                return
            
            # Simple colormap change without complex operations
            try:
                # First try direct PyQtGraph colormap
                colormap = pg.colormap.get(colormap_name)
                if colormap is not None:
                    lut = colormap.getLookupTable()
                    self.vel_img.setLookupTable(lut)
                    print(f"Velocity colormap changed to {colormap_name}")
                else:
                    # Try matplotlib colormap
                    try:
                        colormap = pg.colormap.getFromMatplotlib(colormap_name)
                        if colormap is not None:
                            lut = colormap.getLookupTable()
                            self.vel_img.setLookupTable(lut)
                            print(f"Velocity colormap changed to {colormap_name} (matplotlib)")
                    except:
                        print(f"Colormap '{colormap_name}' not available")
            except:
                # If colormap change fails, silently ignore to prevent hang
                pass
            
            # Try matplotlib colormaps as fallback
            try:
                colormap = pg.colormap.getFromMatplotlib(colormap_name)
                if colormap is not None:
                    lut = colormap.getLookupTable()
                    self.vel_img.setLookupTable(lut)
                    # Update colorbar colormap
                    if hasattr(self, 'vel_colorbar') and self.vel_colorbar:
                        self.vel_colorbar.setColorMap(colormap)
                    print(f"Velocity colormap changed to {colormap_name} (matplotlib)")
                    return
            except:
                pass
            
            print(f"Warning: Could not set velocity colormap to {colormap_name}")
            
        except Exception as e:
            print(f"Error changing velocity colormap: {e}")
    
    def change_vorticity_colormap(self, colormap_name):
        """Change colormap for vorticity plot - simplified robust version"""
        try:
            # Basic validation
            if not colormap_name or not isinstance(colormap_name, str):
                return

            if self.vort_img is None:
                return

            # Simple colormap change without complex operations
            try:
                # First try direct PyQtGraph colormap
                colormap = pg.colormap.get(colormap_name)
                if colormap is not None:
                    lut = colormap.getLookupTable()
                    self.vort_img.setLookupTable(lut)
                    print(f"Vorticity colormap changed to {colormap_name}")
                    return
            except Exception as e:
                pass

            # Try matplotlib colormap
            try:
                colormap = pg.colormap.getFromMatplotlib(colormap_name)
                if colormap is not None:
                    lut = colormap.getLookupTable()
                    self.vort_img.setLookupTable(lut)
                    # Update colorbar colormap
                    if hasattr(self, 'vort_colorbar') and self.vort_colorbar:
                        self.vort_colorbar.setColorMap(colormap)
                    print(f"Vorticity colormap changed to {colormap_name} (matplotlib)")
                    return
            except Exception as e:
                pass

        except:
            # Top-level protection - never re-raise
            pass

    def change_pressure_colormap(self, colormap_name):
        """Change colormap for pressure plot - simplified robust version"""
        try:
            # Basic validation
            if not colormap_name or not isinstance(colormap_name, str):
                return

            if self.pressure_img is None:
                return

            # Simple colormap change without complex operations
            try:
                # First try direct PyQtGraph colormap
                colormap = pg.colormap.get(colormap_name)
                if colormap is not None:
                    lut = colormap.getLookupTable()
                    self.pressure_img.setLookupTable(lut)
                    print(f"Pressure colormap changed to {colormap_name}")
                    return
            except Exception as e:
                pass

            # Try matplotlib colormap
            try:
                colormap = pg.colormap.getFromMatplotlib(colormap_name)
                if colormap is not None:
                    lut = colormap.getLookupTable()
                    self.pressure_img.setLookupTable(lut)
                    # Update colorbar colormap
                    if hasattr(self, 'pressure_colorbar') and self.pressure_colorbar:
                        self.pressure_colorbar.setColorMap(colormap)
                    print(f"Pressure colormap changed to {colormap_name} (matplotlib)")
                    return
            except Exception as e:
                pass

        except:
            # Top-level protection - never re-raise
            pass
    
    def auto_fit_velocity(self):
        """Auto-fit velocity plot to full domain"""
        try:
            if self.vel_plot is not None:
                # Only auto-fit if manual bounds haven't been set
                if not self.manual_bounds_set:
                    # Get the current solver's domain size
                    if hasattr(self, 'solver') and self.solver is not None:
                        lx, ly = self.solver.grid.lx, self.solver.grid.ly
                        
                        # Get actual y-bounds
                        y_min = 0.0
                        y_max = ly
                        if hasattr(self.solver, 'grid') and hasattr(self.solver.grid, 'y'):
                            try:
                                y_coords = np.array(self.solver.grid.y)
                                y_min = y_coords.min()
                                y_max = y_coords.max()
                            except:
                                pass
                        
                        padding_x = lx * 0.1  # 10% padding
                        padding_y = (y_max - y_min) * 0.1  # 10% padding based on actual y range
                        
                        self.vel_plot.setXRange(-padding_x, lx + padding_x)
                        self.vel_plot.setYRange(y_min - padding_y, y_max + padding_y)
                else:
                    pass
        except:
            pass
    
    def auto_fit_vorticity(self):
        """Auto-fit vorticity plot to full domain"""
        try:
            if self.vort_plot is not None:
                # Only auto-fit if manual bounds haven't been set
                if not self.manual_bounds_set:
                    # Get the current solver's domain size
                    if hasattr(self, 'solver') and self.solver is not None:
                        lx, ly = self.solver.grid.lx, self.solver.grid.ly
                        
                        # Get actual y-bounds (same as velocity plot)
                        y_min = 0.0
                        y_max = ly
                        if hasattr(self.solver, 'grid') and hasattr(self.solver.grid, 'y'):
                            try:
                                y_coords = np.array(self.solver.grid.y)
                                y_min = y_coords.min()
                                y_max = y_coords.max()
                            except:
                                pass
                        
                        padding_x = lx * 0.1  # 10% padding
                        padding_y = (y_max - y_min) * 0.1  # 10% padding based on actual y range
                        
                        self.vort_plot.setXRange(-padding_x, lx + padding_x)
                        self.vort_plot.setYRange(y_min - padding_y, y_max + padding_y)
                else:
                    pass
        except:
            pass
    
    def auto_fit_pressure(self):
        """Auto-fit pressure plot to full domain"""
        try:
            if self.pressure_plot is not None:
                if hasattr(self, 'solver') and self.solver is not None:
                    lx, ly = self.solver.grid.lx, self.solver.grid.ly
                    
                    # Get actual y-bounds
                    y_min = 0.0
                    y_max = ly
                    if hasattr(self.solver, 'grid') and hasattr(self.solver.grid, 'y'):
                        try:
                            y_coords = np.array(self.solver.grid.y)
                            y_min = y_coords.min()
                            y_max = y_coords.max()
                        except:
                            pass
                    
                    padding_x = lx * 0.1
                    padding_y = (y_max - y_min) * 0.1
                    
                    self.pressure_plot.setXRange(-padding_x, lx + padding_x)
                    self.pressure_plot.setYRange(y_min - padding_y, y_max + padding_y)
        except:
            pass
    
    def auto_fit_dye(self):
        """Auto-fit dye plot to full domain"""
        try:
            if self.scalar_plot is not None:
                if hasattr(self, 'solver') and self.solver is not None:
                    lx, ly = self.solver.grid.lx, self.solver.grid.ly
                    
                    # Get actual y-bounds
                    y_min = 0.0
                    y_max = ly
                    if hasattr(self.solver, 'grid') and hasattr(self.solver.grid, 'y'):
                        try:
                            y_coords = np.array(self.solver.grid.y)
                            y_min = y_coords.min()
                            y_max = y_coords.max()
                        except:
                            pass
                    
                    padding_x = lx * 0.1
                    padding_y = (y_max - y_min) * 0.1
                    
                    self.scalar_plot.setXRange(-padding_x, lx + padding_x)
                    self.scalar_plot.setYRange(y_min - padding_y, y_max + padding_y)
        except:
            pass
    
    def auto_fit_all(self):
        """Auto-fit all plots to data"""
        try:
            self.auto_fit_velocity()
            self.auto_fit_vorticity()
            self.auto_fit_pressure()
            self.auto_fit_dye()
        except:
            pass
    
    def reset_plot_ranges(self, nx=512, ny=192, lx=20.0, ly=7.5):
        """Reset plot ranges to original domain bounds"""
        try:
            # Get actual y-bounds from solver if available
            y_min = 0.0
            y_max = ly
            
            if hasattr(self, 'solver') and self.solver is not None:
                try:
                    if hasattr(self.solver, 'grid') and hasattr(self.solver.grid, 'y'):
                        y_coords = np.array(self.solver.grid.y)
                        y_min = y_coords.min()
                        y_max = y_coords.max()
                except:
                    pass
            
            # Set plot ranges with padding
            padding_x = lx * 0.05  # 5% padding
            padding_y = (y_max - y_min) * 0.05  # 5% padding based on actual y range
            
            self.vel_plot.setXRange(-padding_x, lx + padding_x)
            self.vel_plot.setYRange(y_min - padding_y, y_max + padding_y)
            self.vort_plot.setXRange(-padding_x, lx + padding_x)
            self.vort_plot.setYRange(y_min - padding_y, y_max + padding_y)
            print(f"Plot ranges reset to domain bounds: Y=[{y_min:.3f}, {y_max:.3f}]")
        except:
            pass
    
    def set_visibility(self, show_velocity=True, show_vorticity=True):
        """Set visibility of plots"""
        if self.vel_plot is not None:
            self.vel_plot.setVisible(show_velocity)
        if self.vort_plot is not None:
            self.vort_plot.setVisible(show_vorticity)
    
    def set_divergence_visibility(self, show_divergence=True):
        """Set visibility of divergence plot and adjust velocity plot colspan"""
        if self.div_plot is not None:
            self.div_plot.setVisible(show_divergence)
        if self.vel_plot is not None:
            # Adjust velocity plot colspan by re-adding to layout with new span
            layout = self.plot_widget.ci.layout
            vel_item = layout.itemAt(0, 0)
            if vel_item is not None:
                layout.removeItem(vel_item)
                if show_divergence:
                    layout.addItem(vel_item, 0, 0, 1, 1)
                else:
                    layout.addItem(vel_item, 0, 0, 1, 2)
    
    def set_pressure_visibility(self, show_pressure=True):
        """Set visibility of pressure plot"""
        if self.pressure_plot is not None:
            self.pressure_plot.setVisible(show_pressure)
    
    def set_dye_visibility(self, show_dye=True):
        """Set visibility of dye plot"""
        if self.scalar_plot is not None:
            self.scalar_plot.setVisible(show_dye)
        if self.scalar_img is not None:
            self.scalar_img.setVisible(show_dye)
    
    def set_particle_mode(self, use_particles: bool):
        """Toggle between dye and particle mode"""
        self.use_particles = use_particles
        self.particle_system.enabled = use_particles
        
        # Toggle visibility
        if self.scalar_img is not None:
            self.scalar_img.setVisible(not use_particles)
        if self.particle_scatter is not None:
            self.particle_scatter.setVisible(use_particles)
        
        # Clear particles when switching to dye mode
        if not use_particles:
            self.particle_system.clear()
            if self.particle_scatter is not None:
                self.particle_scatter.setData([], [])
    
    def update_particles(self, u: np.ndarray, v: np.ndarray, X: np.ndarray, Y: np.ndarray,
                        dx: float, dy: float, dt: float, domain_bounds: tuple):
        """Update particle positions and render them"""
        if not self.use_particles:
            return
            
        # Update particle positions
        self.particle_system.update(u, v, X, Y, dx, dy, dt, domain_bounds)
        
        # Get particle positions and update scatter plot
        positions = self.particle_system.get_positions()
        if len(positions) > 0 and self.particle_scatter is not None:
            self.particle_scatter.setData(positions[:, 0], positions[:, 1])
        elif self.particle_scatter is not None:
            self.particle_scatter.setData([], [])
    
    def inject_particles(self, x: float, y: float, count: int = 10):
        """Inject particles at specified location"""
        if self.use_particles:
            self.particle_system.inject_particles(x, y, count)
    
    def update_dye_marker(self, x_pos: float, y_pos: float):
        """Update the dye injection marker position"""
        if hasattr(self, 'dye_marker') and self.dye_marker is not None:
            self.dye_marker.setData([x_pos], [y_pos])
            self.dye_marker.setVisible(True)
    
    def clear(self):
        """Clear all visualization items"""
        try:
            if self.vel_img is not None:
                self.vel_img.clear()
            if self.vort_img is not None:
                self.vort_img.clear()
            if self.vel_outline is not None:
                self.vel_outline.clear()
            if self.vort_outline is not None:
                self.vort_outline.clear()
            if self.scalar_outline is not None:
                self.scalar_outline.clear()
            if self.pressure_outline is not None:
                self.pressure_outline.clear()
            if self.vel_sdf is not None:
                self.vel_sdf.clear()
            if self.vort_sdf is not None:
                self.vort_sdf.clear()
            # Coefficient plots removed - no clearing needed
        except:
            pass  # Ignore cleanup errors

