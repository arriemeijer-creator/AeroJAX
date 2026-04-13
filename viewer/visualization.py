"""
Visualization components for Baseline Navier-Stokes Viewer
Separates visualization logic from UI and main viewer logic
"""

import numpy as np
import jax.numpy as jnp
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6 import sip


class FlowVisualization:
    """Handles flow field visualization (velocity, vorticity)"""
    
    def __init__(self, plot_widget, solver=None):
        self.plot_widget = plot_widget
        self.solver = solver  # Store solver reference
        self.vel_plot = None
        self.vort_plot = None
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
        
        # CL and CD plots removed - no coefficient tracking needed
        
        # Initialize obstacle renderer
        self.obstacle_renderer = None
        
        self.setup_plots()
        self.set_initial_colormaps()
    
    def set_initial_colormaps(self, velocity_colormap='viridis', vorticity_colormap='RdBu'):
        """Set initial colormaps for velocity and vorticity plots"""
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
            
                
            if vort_colormap is not None:
                vort_lut = vort_colormap.getLookupTable()
                if self.vort_img is not None:
                    self.vort_img.setLookupTable(vort_lut)
                
        except Exception as e:
            print(f"Error setting initial colormaps: {e}")
    
    def setup_plots(self, nx=512, ny=96, lx=20.0, ly=4.5):
        """Setup velocity, vorticity, and coefficient plots"""
        # Create velocity and vorticity plots (top row) - span full width
        self.vel_plot = self.plot_widget.addPlot(title="Velocity Magnitude", row=0, col=0, colspan=2)
        self.vel_plot.setLabel('left', 'y')
        self.vel_plot.setLabel('bottom', 'x')
        
        # Create vorticity plot - span full width
        self.vort_plot = self.plot_widget.addPlot(title="Vorticity", row=1, col=0, colspan=2)
        self.vort_plot.setLabel('left', 'y')
        self.vort_plot.setLabel('bottom', 'x')
        
        # Create scalar (dye) plot - span full width
        self.scalar_plot = self.plot_widget.addPlot(title="Dye Concentration", row=2, col=0, colspan=2)
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
        
        # Create enhanced error plot with multiple metrics
        self.l2_plot = self.plot_widget.addPlot(title="Error Metrics", row=3, col=0)
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
        self.error_times = []
        self.l2_errors = []
        self.max_errors = []
        self.rel_errors = []
        self.l2_u_errors = []
        self.l2_v_errors = []
        
        # FPS tracking
        self.current_sim_fps = 0.0
        self.current_viz_fps = 0.0
        
        # CL and CD plots removed for stability and performance
        
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
        
        # Configure scalar plot (after y_min/y_max are defined)
        self.scalar_plot.setXRange(-0.5, lx + 0.5)
        self.scalar_plot.setYRange(y_min - 0.2, y_max + 0.2)
        self.scalar_plot.setAspectLocked(True)
        self.scalar_plot.hideButtons()
        self.scalar_plot.enableAutoRange(False)
        self.scalar_plot.hide()  # Hide by default until enabled
        
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
            self.vort_plot.setXRange(x_min, x_max)
            self.vort_plot.setYRange(y_min_clamped, y_max_clamped)
        except Exception as e:
            print(f"Warning: Failed to set plot ranges: {e}")
            # Fallback to simple ranges
            self.vel_plot.setXRange(0, lx)
            self.vel_plot.setYRange(0, ly)
            self.vort_plot.setXRange(0, lx)
            self.vort_plot.setYRange(0, ly)
        
        # Lock aspect ratio
        self.vel_plot.setAspectLocked(True)
        self.vort_plot.setAspectLocked(True)
        
        # Create image items
        self.vel_img = pg.ImageItem()
        self.vort_img = pg.ImageItem()

        # Initialize fixed colormap levels to prevent jumping during startup
        self.vel_levels = [0.0, 2.5]  # Fixed velocity range (0 to 2.5)
        self.vort_levels = [-5.0, 5.0]  # Fixed vorticity range (-5 to 5)
        self.level_update_counter = 0
        self.level_update_interval = 1  # Update every frame

        # Set initial colormaps
        self._setup_colormaps()
        
        # Add colorbars using the correct approach with setImageItem and insert_in
        self.vel_colorbar = pg.ColorBarItem(values=(0, 2.5), width=20)
        self.vel_colorbar.setImageItem(self.vel_img, insert_in=self.vel_plot)
        
        self.vort_colorbar = pg.ColorBarItem(values=(-5, 5), width=20)
        self.vort_colorbar.setImageItem(self.vort_img, insert_in=self.vort_plot)
        
        # Create SDF items before adding to plots
        try:
            self.vel_sdf = pg.ImageItem()
            self.vort_sdf = pg.ImageItem()
        except Exception as e:
            print(f"Warning: Failed to create SDF items: {e}")
            self.vel_sdf = None
            self.vort_sdf = None
        
        # Add image items FIRST (so they're in the background)
        self.vel_plot.addItem(self.vel_img)
        self.vort_plot.addItem(self.vort_img)
        
        # Add SDF items SECOND (so they're between images and outlines) - only if they exist
        if self.vel_sdf is not None:
            self.vel_plot.addItem(self.vel_sdf)
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
        if self.vel_outline is not None and self.vort_outline is not None and self.scalar_outline is not None:
            try:
                self.obstacle_renderer = ObstacleRenderer(self.vel_outline, self.vort_outline, self.scalar_outline)
            except Exception as e:
                print(f"Warning: Failed to create obstacle renderer in setup_plots: {e}")
                self.obstacle_renderer = None
        else:
            self.obstacle_renderer = None
        
        # Configure plots
        for plot in [self.vel_plot, self.vort_plot]:
            plot.hideButtons()
            plot.enableAutoRange(False)
            plot.setAutoVisible(y=False)
        
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
    
    def _configure_performance_settings(self):
        """Configure performance optimization settings"""
        self.vel_img.autoDownsample = True
        self.vort_img.autoDownsample = True
        self.vel_img.downsampleMethod = 'average'
        self.vort_img.downsampleMethod = 'average'
    
    def _add_overlay_items(self):
        """Add overlay items for outlines and SDF visualization"""
        # Use QGraphicsPolygonItem for filled airfoil mask (black)
        try:
            from PyQt6.QtWidgets import QGraphicsPolygonItem
            from PyQt6.QtGui import QPolygonF, QBrush, QColor
            from PyQt6.QtCore import QPointF
            
            # Create filled polygon items for obstacle mask
            self.vel_outline = QGraphicsPolygonItem()
            self.vel_outline.setBrush(QBrush(QColor(0, 0, 0, 255)))  # Black with full opacity
            self.vel_outline.setPen(pg.mkPen('k', width=1))
            
            self.vort_outline = QGraphicsPolygonItem()
            self.vort_outline.setBrush(QBrush(QColor(0, 0, 0, 255)))  # Black with full opacity
            self.vort_outline.setPen(pg.mkPen('k', width=1))
            
            self.scalar_outline = QGraphicsPolygonItem()
            self.scalar_outline.setBrush(QBrush(QColor(0, 0, 0, 255)))  # Black with full opacity
            self.scalar_outline.setPen(pg.mkPen('k', width=1))
            
            # Add outline items to plots (on top of images)
            if hasattr(self, 'vel_plot') and self.vel_plot is not None:
                self.vel_plot.addItem(self.vel_outline)
            if hasattr(self, 'vort_plot') and self.vort_plot is not None:
                self.vort_plot.addItem(self.vort_outline)
            if hasattr(self, 'scalar_plot') and self.scalar_plot is not None:
                self.scalar_plot.addItem(self.scalar_outline)
                
        except Exception as e:
            print(f"Warning: Failed to create outline items: {e}")
            self.vel_outline = None
            self.vort_outline = None
            self.scalar_outline = None
    
    def update_plots_for_new_grid(self, nx, ny, lx, ly):
        """Update plots when grid size changes"""
        
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
        
        # Clear and recreate plots
        self.plot_widget.clear()
        
        # Reset outline references to prevent accessing deleted items
        self.vel_outline = None
        self.vort_outline = None
        self.scalar_outline = None
        self.obstacle_renderer = None
        
        # Reset marker references
        self.stag_line_vel = None
        self.sep_line_vel = None
        self.stag_line_vort = None
        self.sep_line_vort = None
        
        # Recreate plots (this will call setup_plots again)
        self.setup_plots(nx, ny, lx, ly)
        
        # CRITICAL FIX: Recreate obstacle renderer only after all items are created
        try:
            if hasattr(self, 'vel_outline') and self.vel_outline is not None and \
               hasattr(self, 'vort_outline') and self.vort_outline is not None and \
               hasattr(self, 'scalar_outline') and self.scalar_outline is not None:
                self.obstacle_renderer = ObstacleRenderer(self.vel_outline, self.vort_outline, self.scalar_outline)
                
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
                self.obstacle_renderer = None
        except Exception as renderer_error:
            print(f"Warning: Failed to create obstacle renderer after grid change: {renderer_error}")
            self.obstacle_renderer = None
        
        # Reset plot ranges
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
            self.vort_plot.setXRange(x_min, x_max)
            self.vort_plot.setYRange(y_min_clamped, y_max_clamped)
        except Exception as e:
            print(f"Warning: Failed to set plot ranges in update_plots_for_new_grid: {e}")
            # Fallback to simple ranges
            self.vel_plot.setXRange(0, lx)
            self.vel_plot.setYRange(0, ly)
            self.vort_plot.setXRange(0, lx)
            self.vort_plot.setYRange(0, ly)
        
        # Don't reset levels - keep existing levels to prevent sudden jumps
        self.level_update_counter = 0
        self.level_update_interval = 1  # Update every frame
    
    def update_visualization(self, vel_mag_data, vort_data, show_velocity=True, show_vorticity=True):
        """Update visualization with new data"""
        try:
            # Check if data dimensions match expected grid dimensions
            if vel_mag_data is not None and hasattr(self, 'current_nx'):
                if vel_mag_data.shape != (self.current_nx, self.current_ny):
                    print(f"Warning: Data shape {vel_mag_data.shape} doesn't match grid {self.current_nx}x{self.current_ny}, skipping visualization update")
                    return
        except Exception as e:
            print(f"Error in update_visualization: {e}")
            return
        
        # Update velocity plot
        if show_velocity and self.vel_img is not None and vel_mag_data is not None:
            
            # CRITICAL FIX: Try NOT transposing - maybe the data is already correct for PyQtGraph
            # Simulation: vel_mag_data[nx][ny] where nx=x, ny=y
            # PyQtGraph: image[ny][nx] where rows=y, cols=x
            # Maybe the simulation data is already in the right format!
            if vel_mag_data.shape == (self.current_nx, self.current_ny):
                # Data is (nx, ny) - try WITHOUT transpose first
                vel_data_correct = vel_mag_data
            elif vel_mag_data.shape == (self.current_ny, self.current_nx):
                # Data is (ny, nx) - use as is
                vel_data_correct = vel_mag_data
            else:
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
            
            # Set image with explicit bounds (always use fixed levels to prevent jumping)
            self.vel_img.setImage(vel_data_correct, levels=self.vel_levels, autoLevels=False, rect=rect)
        
        # Update vorticity plot
        if show_vorticity and self.vort_img is not None and vort_data is not None:

            # CRITICAL FIX: Try NOT transposing - maybe the data is already correct for PyQtGraph
            # Simulation: vort_data[nx][ny] where nx=x, ny=y
            # PyQtGraph: image[ny][nx] where rows=y, cols=x
            # Maybe the simulation data is already in the right format!
            if vort_data.shape == (self.current_nx, self.current_ny):
                # Data is (nx, ny) - try WITHOUT transpose first
                vort_data_correct = vort_data
            elif vort_data.shape == (self.current_ny, self.current_nx):
                # Data is (ny, nx) - use as is
                vort_data_correct = vort_data
            else:
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

            # Set image with explicit bounds (always use fixed levels to prevent jumping)
            # Display full vorticity field (including walls), but use levels from interior
            self.vort_img.setImage(vort_data_correct, levels=self.vort_levels, autoLevels=False, rect=rect)
        
        # Update levels cache
        self.level_update_counter += 1
        # DISABLED: Dynamic level adjustment to isolate the snapping issue
        # Using fixed levels instead
        # if self.level_update_counter % self.level_update_interval == 0:
        
        # Update obstacle outlines (less frequent)
        if self.obstacle_renderer is not None and self.solver is not None:
            if self.level_update_counter % 5 == 0:  # Update every 5 frames instead of every frame
                self.obstacle_renderer.update_obstacle_outlines(self.solver)
        
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
                       l2_error_u: float = None, l2_error_v: float = None) -> None:
        """Update error plot with multiple metrics"""
        if hasattr(self, 'l2_plot') and self.l2_plot is not None:
            try:
                # Initialize error curves on first call
                if self.l2_curve is None:
                    self.l2_curve = self.l2_plot.plot(pen=pg.mkPen('r', width=2), name='L2 Error')
                    self.max_error_curve = self.l2_plot.plot(pen=pg.mkPen('b', width=1), name='Max Error')
                    self.rel_error_curve = self.l2_plot.plot(pen=pg.mkPen('g', width=1), name='Rel Error')
                    self.l2_u_curve = self.l2_plot.plot(pen=pg.mkPen('m', width=1, style=Qt.PenStyle.DashLine), name='L2 U')
                    self.l2_v_curve = self.l2_plot.plot(pen=pg.mkPen('c', width=1, style=Qt.PenStyle.DashLine), name='L2 V')
                
                # Accumulate data with array length synchronization
                self.error_times.append(time_value)
                self.l2_errors.append(l2_error_value)
                
                # Update L2 error curve with all data points (interconnects sparse points)
                self.l2_curve.setData(self.error_times, self.l2_errors)
                
                # Handle additional metrics if available
                if max_error_value is not None:
                    self.max_errors.append(max_error_value)
                    self.max_error_curve.setData(self.error_times, self.max_errors)
                
                if rel_error_value is not None:
                    self.rel_errors.append(rel_error_value)
                    self.rel_error_curve.setData(self.error_times, self.rel_errors)
                
                if l2_error_u is not None:
                    self.l2_u_errors.append(l2_error_u)
                    self.l2_u_curve.setData(self.error_times, self.l2_u_errors)
                
                if l2_error_v is not None:
                    self.l2_v_errors.append(l2_error_v)
                    self.l2_v_curve.setData(self.error_times, self.l2_v_errors)
                
                # Update x-range to show all data from 0 to current time
                if self.error_times:
                    max_time = max(self.error_times)
                    self.l2_plot.setXRange(0, max_time * 1.1)  # Add 10% padding
                    
            except Exception as e:
                print(f"Error updating L2 error plot: {e}")
                # Fallback to simple L2 error plot
                if self.l2_curve is None:
                    self.l2_curve = self.l2_plot.plot(pen=pg.mkPen('r', width=2), name='L2 Error')
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
        # Reset curves to None so they'll be recreated on next update
        self.l2_curve = None
        self.max_error_curve = None
        self.rel_error_curve = None
        self.l2_u_curve = None
        self.l2_v_curve = None
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
            # Use HTML subscript for "inlet"
            # Check if naca already contains "NACA" to avoid duplication
            naca_display = naca if naca.upper().startswith('NACA') else f"NACA {naca}"
            title = f"Vorticity (Re = {re:.0f} | U<sub>inlet</sub> = {u_inlet:.2f} m/s | {naca_display} | {aoa:.1f}° AoA)"
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
    
    def auto_fit_both(self):
        """Auto-fit both plots to data"""
        try:
            self.auto_fit_velocity()
            self.auto_fit_vorticity()
        except:
            pass
    
    def reset_plot_ranges(self, nx=512, ny=96, lx=20.0, ly=4.5):
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
            if self.vel_sdf is not None:
                self.vel_sdf.clear()
            if self.vort_sdf is not None:
                self.vort_sdf.clear()
            # Coefficient plots removed - no clearing needed
        except:
            pass  # Ignore cleanup errors


class ObstacleRenderer:
    """Handles rendering of obstacles (cylinder, NACA airfoils)"""
    
    def __init__(self, vel_outline, vort_outline, scalar_outline):
        self.vel_outline = vel_outline
        self.vort_outline = vort_outline
        self.scalar_outline = scalar_outline
        self.naca_available = self._check_naca_availability()
    
    def _check_naca_availability(self):
        """Check if NACA airfoils are available"""
        try:
            from solver.naca_airfoils import NACA_AIRFOILS
            return True
        except ImportError:
            return False
    
    def update_obstacle_outlines(self, solver):
        """Update obstacle outlines based on current solver geometry"""
        # Rate limit updates to prevent error cascades
        if not hasattr(self, '_last_naca_update_time'):
            self._last_naca_update_time = 0
            self._naca_error_count = 0
            self._last_naca_error_designation = None
        
        import time
        current_time = time.time()
        if current_time - self._last_naca_update_time < 0.2:  # 200ms minimum between updates
            return
        
        self._last_naca_update_time = current_time
        if not hasattr(solver, 'sim_params') or solver is None:
            return
        
        try:
            # Check flow type and obstacle type
            if solver.sim_params.flow_type != 'von_karman':
                # Clear outlines for non-von_karman flows
                from PyQt6.QtGui import QPolygonF
                from PyQt6.QtCore import QPointF
                empty_polygon = QPolygonF()
                if (self.vel_outline is not None and 
                    hasattr(self.vel_outline, 'setPolygon') and 
                    not sip.isdeleted(self.vel_outline)):
                    self.vel_outline.setPolygon(empty_polygon)
                if (self.vort_outline is not None and 
                    hasattr(self.vort_outline, 'setPolygon') and 
                    not sip.isdeleted(self.vort_outline)):
                    self.vort_outline.setPolygon(empty_polygon)
                if (self.scalar_outline is not None and 
                    hasattr(self.scalar_outline, 'setPolygon') and 
                    not sip.isdeleted(self.scalar_outline)):
                    self.scalar_outline.setPolygon(empty_polygon)
                return
            
            # Get obstacle parameters
            obstacle_type = getattr(solver.sim_params, 'obstacle_type', 'cylinder')
            
            if obstacle_type == 'naca_airfoil':
                # Draw NACA airfoil outline
                self._draw_naca_outline(solver)
            else:
                # Draw cylinder outline
                self._draw_cylinder_outline(solver)
        except Exception as e:
            print(f"Error in update_obstacle_outlines: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_cylinder_outline(self, solver):
        """Draw cylinder outline"""
        try:
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            
            # Get cylinder parameters
            center_x = float(solver.geom.center_x.item()) if hasattr(solver.geom.center_x, 'item') else float(solver.geom.center_x)
            center_y = float(solver.geom.center_y.item()) if hasattr(solver.geom.center_y, 'item') else float(solver.geom.center_y)
            radius = float(solver.geom.radius.item()) if hasattr(solver.geom.radius, 'item') else float(solver.geom.radius)
            
            # Create circle points
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = center_x + radius * np.cos(theta)
            y_circle = center_y + radius * np.sin(theta)
            
            # Create QPolygonF from points
            polygon_points = [QPointF(x, y) for x, y in zip(x_circle, y_circle)]
            polygon = QPolygonF(polygon_points)
            
            # Check if outline items exist and are properly connected to plots
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPolygon') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPolygon(polygon)
            
            if (self.vort_outline is not None and 
                hasattr(self.vort_outline, 'setPolygon') and 
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPolygon(polygon)
            
            if (self.scalar_outline is not None and 
                hasattr(self.scalar_outline, 'setPolygon') and 
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPolygon(polygon)
                
        except Exception as e:
            print(f"Error drawing cylinder outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_naca_outline(self, solver):
        """Draw NACA airfoil outline"""
        if not self.naca_available:
            return
        
        try:
            sim = solver.sim_params
            
            # Get NACA parameters
            designation = sim.naca_airfoil
            chord = sim.naca_chord
            angle = sim.naca_angle
            pos_x = sim.naca_x
            pos_y = sim.naca_y
            
            # Extract digits from designation
            digits = ''.join(filter(str.isdigit, designation))
            
            if len(digits) == 4:
                # 4-digit airfoil
                from solver.naca_airfoils import generate_naca_4digit, parse_naca_4digit
                
                try:
                    m, p, t = parse_naca_4digit(designation)
                    # Validate parameters to prevent NaN
                    if p < 0 or p >= 1:
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid camber position p={p} for 4-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    if t <= 0 or t > 0.5:
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid thickness t={t} for 4-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    
                    # Special handling for symmetric airfoils (p=0)
                    if p == 0.0:
                        # For symmetric airfoils, upper and lower surfaces are just thickness distribution
                        x_norm = np.linspace(0, 1, 100)
                        
                        # Thickness distribution (same as in naca_airfoils.py)
                        yt = 5 * t * (0.2969 * np.sqrt(np.abs(x_norm)) - 0.1260 * x_norm - 
                                      0.3516 * x_norm**2 + 0.2843 * x_norm**3 - 0.1015 * x_norm**4)
                        
                        # For symmetric airfoil: yc = 0, theta = 0
                        xu = x_norm
                        yu = yt  # Upper surface
                        xl = x_norm  
                        yl = -yt  # Lower surface (negative)
                    else:
                        # Use regular NACA generation for cambered airfoils
                        x_norm = np.linspace(0, 1, 100)
                        xu, yu, xl, yl = generate_naca_4digit(jnp.array(x_norm), m, p, t)
                        
                        # Convert to numpy
                        xu, yu = np.array(xu), np.array(yu)
                        xl, yl = np.array(xl), np.array(yl)
                except Exception as e:
                    print(f"NACA Error: Failed to generate 4-digit airfoil: {e}")
                    return
                
            elif len(digits) == 5:
                # 5-digit airfoil
                from solver.naca_airfoils import generate_naca_5digit, parse_naca_5digit
                
                try:
                    cl, p, m, t = parse_naca_5digit(designation)
                    # Validate parameters to prevent NaN
                    # p=0 is valid for reflexed 5-digit airfoils (e.g., NACA 23012)
                    if p < 0 or p > 1:
                        # Only print error once per designation
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid camber position p={p} for 5-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    x_norm = np.linspace(0, 1, 100)
                    xu, yu, xl, yl = generate_naca_5digit(jnp.array(x_norm), cl, p, m, t)
                    
                    # Convert to numpy
                    xu, yu = np.array(xu), np.array(yu)
                    xl, yl = np.array(xl), np.array(yl)
                except Exception as e:
                    print(f"NACA Error: Failed to generate 5-digit airfoil: {e}")
                    return
            
            # Check for NaN values
            if np.any(np.isnan(xu)) or np.any(np.isnan(yu)) or np.any(np.isnan(xl)) or np.any(np.isnan(yl)):
                # Only print error once per designation
                if self._last_naca_error_designation != designation:
                    print(f"NACA Error: NaN values detected in generated coordinates for {designation}")
                    print(f"  Parameters: digits={digits}, len={len(digits)}")
                    if len(digits) == 4:
                        print(f"  4-digit: m={m}, p={p}, t={t}")
                    elif len(digits) == 5:
                        print(f"  5-digit: cl={cl}, p={p}, m={m}, t={t}")
                    self._last_naca_error_designation = designation
                return
            
            # Scale
            xu, yu = xu * chord, yu * chord
            xl, yl = xl * chord, yl * chord
            
            # Rotate
            angle_rad = np.radians(angle)
            xu_rot = xu * np.cos(angle_rad) - yu * np.sin(angle_rad)
            yu_rot = xu * np.sin(angle_rad) + yu * np.cos(angle_rad)
            xl_rot = xl * np.cos(angle_rad) - yl * np.sin(angle_rad)
            yl_rot = xl * np.sin(angle_rad) + yl * np.cos(angle_rad)
            
            # Translate (position should be leading edge, not center)
            # FIXED: Remove chord/2 offset to align with mask which uses leading edge position
            xu_final = xu_rot + pos_x
            yu_final = yu_rot + pos_y
            xl_final = xl_rot + pos_x
            yl_final = yl_rot + pos_y
            
            # Combine upper and lower surfaces
            x_outline = np.concatenate([xu_final, xl_final[::-1], [xu_final[0]]])
            y_outline = np.concatenate([yu_final, yl_final[::-1], [yu_final[0]]])
            
            # Final check for NaN values
            if np.any(np.isnan(x_outline)) or np.any(np.isnan(y_outline)):
                print("NACA Error: NaN values in final outline coordinates")
                return
            
            # Check if outline items are still valid before setting data
            try:
                from PyQt6.QtGui import QPolygonF
                from PyQt6.QtCore import QPointF
                
                # Create QPolygonF from points
                polygon_points = [QPointF(x, y) for x, y in zip(x_outline, y_outline)]
                polygon = QPolygonF(polygon_points)
                
                if (self.vel_outline is not None and 
                    hasattr(self.vel_outline, 'setPolygon') and 
                    not sip.isdeleted(self.vel_outline)):
                    self.vel_outline.setPolygon(polygon)
                    
                if (self.vort_outline is not None and 
                    hasattr(self.vort_outline, 'setPolygon') and 
                    not sip.isdeleted(self.vort_outline)):
                    self.vort_outline.setPolygon(polygon)
                    
                if (self.scalar_outline is not None and 
                    hasattr(self.scalar_outline, 'setPolygon') and 
                    not sip.isdeleted(self.scalar_outline)):
                    self.scalar_outline.setPolygon(polygon)
            except RuntimeError as e:
                if "has been deleted" in str(e):
                    print("Warning: Outline plot items deleted during cleanup")
                else:
                    raise
            except Exception as e:
                print(f"Error drawing NACA outline: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error drawing NACA outline: {e}")
            import traceback
            traceback.print_exc()


class SDFVisualization:
    """Handles Signed Distance Function visualization"""
    
    def __init__(self, vel_sdf, vort_sdf, parent_vis=None):
        self.vel_sdf = vel_sdf
        self.vort_sdf = vort_sdf
        self.parent_vis = parent_vis  # Reference to FlowVisualization for bounds
        self.is_visible = False  # Track visibility state - hidden by default (using filled polygon instead)
    
    def set_visibility(self, is_visible):
        """Set visibility of SDF overlay"""
        self.is_visible = is_visible
        if self.vel_sdf is not None:
            self.vel_sdf.setVisible(is_visible)
        if self.vort_sdf is not None:
            self.vort_sdf.setVisible(is_visible)
    
    def update_sdf_visualization(self, solver):
        """Update SDF visualization overlay"""
        if not hasattr(solver, 'mask') or solver is None:
            return
    
        try:
            # Check if ImageItem objects are still valid before using them
            if (self.vel_sdf is None or self.vort_sdf is None or
                not hasattr(self.vel_sdf, 'setImage') or not hasattr(self.vort_sdf, 'setImage')):
                return
            
            # Get actual mask from solver
            mask_array = solver.mask
            
            # Convert to numpy if it's a JAX array
            if hasattr(mask_array, 'toArray'):
                mask_array = mask_array.toArray()
            elif not isinstance(mask_array, np.ndarray):
                mask_array = np.array(mask_array)
            
            # Properly threshold the mask - values > 0.5 are solid, <= 0.5 are fluid
            binary_mask = (mask_array > 0.5).astype(np.float64)
            
            # Create visualization array (0 for fluid, 1 for solid)
            sdf_viz = binary_mask.copy()
            
            # Create custom colormap for clear mask visualization
            sdf_lut = np.zeros((256, 4), dtype=np.uint8)
            # Fluid regions: completely transparent
            sdf_lut[:128, :] = [0, 0, 0, 0]  # Fully transparent
            # Solid regions: solid grey with full opacity
            sdf_lut[128:, :] = [128, 128, 128, 255]  # Solid grey
            
            # IMPORTANT: Set bounds rectangle so image aligns properly
            # Get grid dimensions from solver
            if self.parent_vis is not None and hasattr(self.parent_vis, 'current_lx'):
                lx = self.parent_vis.current_lx
                y_min = self.parent_vis.current_y_min
                y_max = self.parent_vis.current_y_max
            else:
                # Fallback to default dimensions
                lx = 20.0
                y_min = 0.0
                y_max = 4.5
                
            rect = QRectF(0, y_min, lx, y_max - y_min)
            
            # Update SDF visualization
            try:
                if self.vel_sdf is not None and hasattr(self.vel_sdf, 'setImage'):
                    self.vel_sdf.setImage(sdf_viz, 
                                         lookupTable=sdf_lut, 
                                         autoLevels=False,
                                         rect=rect)  # CRITICAL: Set bounds
                    self.vel_sdf.setLevels([0, 1])  # Binary mask range
                    self.vel_sdf.setOpacity(1.0)  # FULL opacity for solid appearance
                    self.vel_sdf.setVisible(self.is_visible)
                    
            except Exception as e:
                pass
                
            try:
                if self.vort_sdf is not None and hasattr(self.vort_sdf, 'setImage'):
                    self.vort_sdf.setImage(sdf_viz, 
                                          lookupTable=sdf_lut, 
                                          autoLevels=False,
                                          rect=rect)  # CRITICAL: Set bounds
                    self.vort_sdf.setLevels([0, 1])
                    self.vort_sdf.setOpacity(1.0)  # FULL opacity
                    self.vort_sdf.setVisible(self.is_visible)
                    
            except Exception as e:
                pass
            
        except Exception as e:
            if "has been deleted" not in str(e):
                print(f"Error updating SDF visualization: {e}")
    
    def set_visibility(self, visible=True):
        """Set SDF visualization visibility"""
        self.is_visible = visible  # Track the state
        try:
            if self.vel_sdf is not None and hasattr(self.vel_sdf, 'setVisible'):
                self.vel_sdf.setVisible(visible)
            if self.vort_sdf is not None and hasattr(self.vort_sdf, 'setVisible'):
                self.vort_sdf.setVisible(visible)
        except:
            pass  # Silently handle deletion errors
    
    def clear(self):
        """Clear SDF visualization"""
        try:
            if self.vel_sdf is not None and hasattr(self.vel_sdf, 'clear'):
                self.vel_sdf.clear()
            if self.vort_sdf is not None and hasattr(self.vort_sdf, 'clear'):
                self.vort_sdf.clear()
        except:
            pass  # Silently handle cleanup errors
