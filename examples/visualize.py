"""
Baseline Viewer - PyQtGraph Visualization for Clean Baseline Solver
Optimized for cylinder flow with vortex shedding visualization

Author: Arno Meijer
Version: 1.0 (Baseline Viewer)
"""

import sys
import numpy as np
import jax.numpy as jnp
import jax
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout, QFileDialog, QLineEdit, QSpinBox
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap
import pyqtgraph as pg
from dataclasses import dataclass
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from PIL import Image
import io

# Import baseline solver
try:
    from baseline_clean import BaselineSolver, GridParams, FlowParams, GeometryParams, EGCEParams
    print("Successfully imported BaselineSolver")
except ImportError as e:
    print(f"Failed to import BaselineSolver: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# PyQtGraph Visualization
# ----------------------------------------------------------------------

class BaselineViewer(QMainWindow):
    """Real-time visualization of baseline Navier-Stokes simulation."""
    
    def __init__(self, solver: BaselineSolver):
        super().__init__()
        self.solver = solver
        self.is_recording = False
        self.recorded_frames = []
        
        # Set up main window
        self.setWindowTitle("Baseline Navier-Stokes Solver - Vortex Shedding")
        self.setGeometry(100, 100, 1400, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create control buttons
        self.create_control_buttons(layout)
        
        # Create plot widgets
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        
        # Set up plots with lighter background
        self.setup_plots()
        
        # Set white background for the entire plot widget
        self.plot_widget.setBackground('w')
        
        # Set up timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)  # Update every 1ms
        
        # Info label
        self.info_label = QLabel("Starting baseline simulation...")
        layout.addWidget(self.info_label)
        
    def create_control_buttons(self, layout):
        """Create control buttons for start/pause/reset and recording."""
        button_layout = QHBoxLayout()
        
        # Control buttons
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        self.record_btn = QPushButton("Record Video")
        self.save_btn = QPushButton("Save Video")
        
        # Reynolds number input
        re_label = QLabel("Re:")
        self.re_input = QSpinBox()
        self.re_input.setRange(10, 1000)
        self.re_input.setValue(int(self.solver.flow.Re))
        self.re_input.setSingleStep(10)
        self.re_input.setSuffix(" ")
        self.apply_re_btn = QPushButton("Apply Re")
        
        # Connect buttons to functions
        self.start_btn.clicked.connect(self.start_simulation)
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.reset_btn.clicked.connect(self.reset_simulation)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.save_btn.clicked.connect(self.save_video)
        self.apply_re_btn.clicked.connect(self.apply_reynolds)
        
        # Add buttons and inputs to layout
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(re_label)
        button_layout.addWidget(self.re_input)
        button_layout.addWidget(self.apply_re_btn)
        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.save_btn)
        
        # Add button layout to main layout
        layout.addLayout(button_layout)
        
        # Initially disable pause and save buttons
        self.pause_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
    def apply_reynolds(self):
        """Apply new Reynolds number and reset simulation."""
        new_re = self.re_input.value()
        
        # Pause simulation
        self.timer.stop()
        
        # Update Reynolds number in solver
        self.solver.flow.Re = float(new_re)
        self.solver.flow.nu = self.solver.flow.U_inf * 2.0 * float(self.solver.geom.radius) / new_re
        
        # Recompile JIT functions with new viscosity
        self.solver._step_jit = jax.jit(self.solver._step)
        
        # Reset simulation with new Reynolds number
        self.reset_simulation()
        
        print(f"Reynolds number changed to {new_re}")
        print(f"New kinematic viscosity: {self.solver.flow.nu:.6f}")
        print(f"JIT functions recompiled with new viscosity")
        
    def start_simulation(self):
        """Start the simulation."""
        self.timer.start(1)
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        
    def pause_simulation(self):
        """Pause the simulation."""
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        
    def reset_simulation(self):
        """Reset simulation to initial conditions."""
        self.timer.stop()
        
        # Reset solver
        self.solver.iteration = 0
        self.solver.u = jnp.ones((self.solver.grid.nx, self.solver.grid.ny)) * self.solver.flow.U_inf
        self.solver.v = jnp.zeros((self.solver.grid.nx, self.solver.grid.ny))
        
        # Add stronger initial perturbation to trigger vortex shedding
        np.random.seed(42)
        u_perturbation = 0.05 * np.random.randn(self.solver.grid.nx, self.solver.grid.ny)
        v_perturbation = 0.05 * np.random.randn(self.solver.grid.nx, self.solver.grid.ny)
        self.solver.u = self.solver.u + u_perturbation
        self.solver.v = self.solver.v + v_perturbation
        
        # Clear recording
        self.recorded_frames = []
        self.is_recording = False
        self.record_btn.setText("Record Video")
        self.save_btn.setEnabled(False)
        
        # Reset buttons
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        
        # Clear plots
        self.vel_img.clear()
        self.vort_img.clear()
        
        print(f"Reset simulation with perturbation strength 0.05")
        
    def toggle_recording(self):
        """Toggle video recording on/off."""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.recorded_frames = []
            self.record_btn.setText("Stop Recording")
            self.save_btn.setEnabled(False)
            print("Started recording...")
        else:
            # Stop recording
            self.is_recording = False
            self.record_btn.setText("Record Video")
            self.save_btn.setEnabled(True)
            print(f"Stopped recording. Captured {len(self.recorded_frames)} frames.")
            
    def save_video(self):
        """Save recorded frames as a video/GIF."""
        if not self.recorded_frames:
            print("No frames to save!")
            return
            
        # Ask user for save location
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "GIF Files (*.gif);;All Files (*)"
        )
        
        if filename:
            if not filename.endswith('.gif'):
                filename += '.gif'
                
            print(f"Saving {len(self.recorded_frames)} frames to {filename}...")
            
            # Save as GIF
            self.recorded_frames[0].save(
                filename,
                save_all=True,
                append_images=self.recorded_frames[1:],
                duration=50,  # 50ms per frame (20 FPS)
                loop=0
            )
            print(f"Video saved to {filename}")
            
    def capture_frame(self):
        """Capture current plot as an image."""
        if self.is_recording:
            # Grab the plot widget as a pixmap
            pixmap = self.plot_widget.grab()
            
            # Convert to PIL Image
            qimage = pixmap.toImage()
            w, h = qimage.width(), qimage.height()
            ptr = qimage.bits()
            ptr.setsize(h * w * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
            
            # Convert RGBA to RGB
            rgb_arr = arr[:, :, :3]
            pil_image = Image.fromarray(rgb_arr, 'RGB')
            self.recorded_frames.append(pil_image)
        
    def setup_plots(self):
        """Initialize velocity and vorticity plots."""
        # Velocity magnitude plot (top)
        self.vel_plot = self.plot_widget.addPlot(title="Velocity Magnitude", row=0, col=0)
        self.vel_plot.setLabel('left', 'y')
        self.vel_plot.setLabel('bottom', 'x')
        self.vel_plot.setAspectLocked(True)
        
        # Vorticity plot (bottom)
        self.vort_plot = self.plot_widget.addPlot(title="Vorticity", row=1, col=0)
        self.vort_plot.setLabel('left', 'y')
        self.vort_plot.setLabel('bottom', 'x')
        self.vort_plot.setAspectLocked(True)
        
        # Initialize image items
        nx, ny = self.solver.grid.nx, self.solver.grid.ny
        lx, ly = self.solver.grid.lx, self.solver.grid.ly
        
        # Create image items with proper scaling
        self.vel_img = pg.ImageItem()
        self.vort_img = pg.ImageItem()
        
        # Apply plasma colormap for professional visualization
        self.vel_img.setLookupTable(pg.colormap.get('plasma').getLookupTable())
        self.vort_img.setLookupTable(pg.colormap.get('plasma').getLookupTable())
        
        # Set image scaling to match physical domain (rotated 90 degrees clockwise)
        scale_x = lx/nx
        scale_y = ly/ny
        # Swap scaling for rotation and apply to correct axes
        self.vel_img.setScale(scale_x)  # X scaling
        self.vel_img.setPos(0, 0)
        
        self.vort_img.setScale(scale_x)  # X scaling  
        self.vort_img.setPos(0, 0)
        
        # Add to plots
        self.vel_plot.addItem(self.vel_img)
        self.vort_plot.addItem(self.vort_img)
        
        # Set plot ranges to physical domain
        self.vel_plot.setXRange(0, lx)
        self.vel_plot.setYRange(0, ly)
        self.vort_plot.setXRange(0, lx)
        self.vort_plot.setYRange(0, ly)
        
    def update(self):
        """Update visualization with solver data."""
        try:
            # Step solver
            u, v, vort, ke, enst, drag, lift = self.solver.step()
            
            # Convert to numpy for display
            u_np = np.array(u)
            v_np = np.array(v)
            vort_np = np.array(vort)
            
            # Compute velocity magnitude
            vel_mag = np.sqrt(u_np**2 + v_np**2)
            
            # Update images (no transpose for clockwise rotation)
            self.vel_img.setImage(vel_mag, levels=(0, 2))
            self.vort_img.setImage(vort_np, levels=(-5, 5))
            
            # Capture frame if recording
            self.capture_frame()
            
            # Update info
            step = self.solver.iteration
            time = step * self.solver.dt
            self.info_label.setText(
                f"Step: {step:6d} | Time: {time:.2f}s | "
                f"KE: {ke:.4f} | Drag: {drag:.4f} | Lift: {lift:+.4f}"
            )
            
        except Exception as e:
            print(f"Visualization error: {e}")
            self.timer.stop()
            
    def closeEvent(self, event):
        """Handle window close event."""
        print("Visualization stopped by user")
        self.timer.stop()
        super().closeEvent(event)

def run_pyqtgraph_visualization(solver: BaselineSolver):
    """Start PyQtGraph visualization."""
    print("Starting baseline visualization...")
    print(f"Physical domain: X=[{solver.grid.lx:.1f}, {solver.grid.lx:.1f}]m")
    print(f"Physical domain: Y=[0.0, {solver.grid.ly:.1f}]m")
    print(f"Grid resolution: {solver.grid.nx}x{solver.grid.ny}")
    print(f"Reynolds number: {solver.flow.Re:.1f}")
    print(f"Mode: Baseline")
    print("(Close window to stop simulation)")
    
    app = QApplication(sys.argv)
    viewer = BaselineViewer(solver)
    viewer.show()
    sys.exit(app.exec())

# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------

def main():
    """Test baseline viewer with correct scaling."""
    print("=" * 60)
    print("Baseline Viewer - Clean Navier-Stokes Solver")
    print("=" * 60)
    print("Optimized for vortex shedding visualization")
    
    # Create solver instance with optimal parameters
    grid = GridParams(nx=512, ny=96, lx=20.0, ly=4.5)  # High resolution for vortex shedding
    flow = FlowParams(Re=300.0, U_inf=1.0)  # Higher Reynolds number
    geom = GeometryParams(center_x=jnp.array(2.5), center_y=jnp.array(2.25), radius=jnp.array(0.18))
    egce = EGCEParams(Cs=0.17, eps=0.05)
    
    # Initialize solver
    solver = BaselineSolver(grid, flow, geom, egce, dt=0.002)
    
    # Print scaling info for debugging
    X, Y = solver.grid.X, solver.grid.Y
    print(f"\nScaling Information:")
    print(f"  Physical domain: X=[{float(X[0, 0]):.1f}, {float(X[-1, 0]):.1f}]m")
    print(f"  Physical domain: Y=[{float(Y[0, 0]):.1f}, {float(Y[0, -1]):.1f}]m")
    print(f"  Grid spacing: dx={solver.grid.dx:.4f}m, dy={solver.grid.dy:.4f}m")
    print(f"  Total domain size: {(float(X[-1, 0]) - float(X[0, 0])):.1f}m x {(float(Y[0, -1]) - float(Y[0, 0])):.1f}m")
    
    print(f"\nInitial perturbation already added by BaselineSolver")
    print(f"Ready for vortex shedding visualization")
    
    # Run visualization
    run_pyqtgraph_visualization(solver)

if __name__ == "__main__":
    main()
