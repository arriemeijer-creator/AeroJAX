"""
Visualization for Differential Backpropagation GUI
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt
from typing import Optional, Dict, Any


class InverseDesignVisualization(QWidget):
    """Visualization panel for inverse design optimization"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Create tab widget for different visualizations
        self.tabs = pg.QtWidgets.QTabWidget()
        
        # Flow visualization tab
        self.flow_tab = QWidget()
        self._setup_flow_tab()
        self.tabs.addTab(self.flow_tab, "Flow Field")
        
        # Optimization history tab
        self.history_tab = QWidget()
        self._setup_history_tab()
        self.tabs.addTab(self.history_tab, "Optimization History")
        
        # Airfoil shape tab
        self.shape_tab = QWidget()
        self._setup_shape_tab()
        self.tabs.addTab(self.shape_tab, "Airfoil Shape")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def _setup_flow_tab(self):
        """Setup flow field visualization"""
        layout = QVBoxLayout()
        
        # Velocity magnitude plot
        self.vel_plot = pg.PlotItem(title="Velocity Magnitude")
        self.vel_image = pg.ImageItem()
        self.vel_plot.addItem(self.vel_image)
        self.vel_plot.setAspectLocked(True)
        
        # Vorticity plot
        self.vort_plot = pg.PlotItem(title="Vorticity")
        self.vort_image = pg.ImageItem()
        self.vort_plot.addItem(self.vort_image)
        self.vort_plot.setAspectLocked(True)
        
        # Create graphics layout
        gl = pg.GraphicsLayoutWidget()
        gl.addItem(self.vel_plot, row=0, col=0)
        gl.addItem(self.vort_plot, row=0, col=1)
        
        layout.addWidget(gl)
        self.flow_tab.setLayout(layout)
    
    def _setup_history_tab(self):
        """Setup optimization history visualization"""
        layout = QVBoxLayout()
        
        # Loss history
        self.loss_plot = pg.PlotItem(title="Loss History")
        self.loss_curve = self.loss_plot.plot(pen='r')
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setLabel('bottom', 'Iteration')
        self.loss_plot.showGrid(x=True, y=True)
        
        # Cl history
        self.cl_plot = pg.PlotItem(title="Cl History")
        self.cl_curve = self.cl_plot.plot(pen='b')
        self.cl_target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine))
        self.cl_plot.addItem(self.cl_target_line)
        self.cl_plot.setLabel('left', 'Cl')
        self.cl_plot.setLabel('bottom', 'Iteration')
        self.cl_plot.showGrid(x=True, y=True)
        
        # Cd history
        self.cd_plot = pg.PlotItem(title="Cd History")
        self.cd_curve = self.cd_plot.plot(pen='m')
        self.cd_target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine))
        self.cd_plot.addItem(self.cd_target_line)
        self.cd_plot.setLabel('left', 'Cd')
        self.cd_plot.setLabel('bottom', 'Iteration')
        self.cd_plot.showGrid(x=True, y=True)
        
        # Create graphics layout
        gl = pg.GraphicsLayoutWidget()
        gl.addItem(self.loss_plot, row=0, col=0)
        gl.addItem(self.cl_plot, row=0, col=1)
        gl.addItem(self.cd_plot, row=1, col=0)
        
        layout.addWidget(gl)
        self.history_tab.setLayout(layout)
    
    def _setup_shape_tab(self):
        """Setup airfoil shape visualization"""
        layout = QVBoxLayout()
        
        # Airfoil shape plot
        self.shape_plot = pg.PlotItem(title="Airfoil Shape")
        self.shape_curve = self.shape_plot.plot(pen='b', symbol='o')
        self.shape_plot.setAspectLocked(True)
        self.shape_plot.setLabel('left', 'Y/c')
        self.shape_plot.setLabel('bottom', 'X/c')
        self.shape_plot.showGrid(x=True, y=True)
        
        # Create graphics widget
        gl = pg.GraphicsLayoutWidget()
        gl.addItem(self.shape_plot)
        
        layout.addWidget(gl)
        self.shape_tab.setLayout(layout)
    
    def update_flow_field(self, u: np.ndarray, v: np.ndarray, vort: np.ndarray):
        """Update flow field visualization"""
        # Compute velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Update velocity image
        self.vel_image.setImage(vel_mag.T)
        
        # Update vorticity image
        self.vort_image.setImage(vort.T)
        
        # Set colormaps
        self.vel_image.setLookupTable(pg.colormap.get('inferno').getLookupTable())
        self.vort_image.setLookupTable(pg.colormap.get('RdBu').getLookupTable())
    
    def update_history(self, history: Dict[str, list]):
        """Update optimization history plots"""
        iterations = np.arange(len(history['loss']))
        
        # Update loss
        self.loss_curve.setData(iterations, history['loss'])
        
        # Update Cl
        if 'cl' in history and len(history['cl']) > 0:
            self.cl_curve.setData(iterations, history['cl'])
        
        # Update Cd
        if 'cd' in history and len(history['cd']) > 0:
            self.cd_curve.setData(iterations, history['cd'])
    
    def update_target_lines(self, target_cl: Optional[float] = None, 
                           target_cd: Optional[float] = None):
        """Update target value lines"""
        if target_cl is not None:
            self.cl_target_line.setPos(target_cl)
        if target_cd is not None:
            self.cd_target_line.setPos(target_cd)
    
    def update_airfoil_shape(self, x_upper: np.ndarray, y_upper: np.ndarray,
                            x_lower: np.ndarray, y_lower: np.ndarray):
        """Update airfoil shape visualization"""
        # Combine upper and lower surfaces
        x = np.concatenate([x_upper, x_lower[::-1]])
        y = np.concatenate([y_upper, y_lower[::-1]])
        
        self.shape_curve.setData(x, y)


class MetricsDisplay(QWidget):
    """Display for optimization metrics"""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Current metrics
        metrics_group = QFrame()
        metrics_group.setFrameStyle(QFrame.Shape.StyledPanel)
        metrics_layout = QVBoxLayout()
        
        self.iteration_label = QLabel("Iteration: 0")
        self.iteration_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.iteration_label)
        
        self.loss_label = QLabel("Loss: 0.0000")
        self.loss_label.setStyleSheet("font-size: 12px; color: red;")
        metrics_layout.addWidget(self.loss_label)
        
        self.cl_label = QLabel("Cl: 0.000")
        metrics_layout.addWidget(self.cl_label)
        
        self.cd_label = QLabel("Cd: 0.000")
        metrics_layout.addWidget(self.cd_label)
        
        self.strouhal_label = QLabel("Strouhal: 0.000")
        metrics_layout.addWidget(self.strouhal_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Best metrics
        best_group = QFrame()
        best_group.setFrameStyle(QFrame.Shape.StyledPanel)
        best_layout = QVBoxLayout()
        
        best_label = QLabel("Best Solution")
        best_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        best_layout.addWidget(best_label)
        
        self.best_loss_label = QLabel("Best Loss: 0.0000")
        self.best_loss_label.setStyleSheet("font-size: 12px; color: green;")
        best_layout.addWidget(self.best_loss_label)
        
        self.best_params_label = QLabel("Params: -")
        best_layout.addWidget(self.best_params_label)
        
        best_group.setLayout(best_layout)
        layout.addWidget(best_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_metrics(self, iteration: int, loss: float, cl: float, 
                      cd: float, strouhal: Optional[float] = None):
        """Update current metrics display"""
        self.iteration_label.setText(f"Iteration: {iteration}")
        self.loss_label.setText(f"Loss: {loss:.6f}")
        self.cl_label.setText(f"Cl: {cl:.4f}")
        self.cd_label.setText(f"Cd: {cd:.4f}")
        if strouhal is not None:
            self.strouhal_label.setText(f"Strouhal: {strouhal:.4f}")
        else:
            self.strouhal_label.setText("Strouhal: -")
    
    def update_best_metrics(self, best_loss: float, best_params: Dict[str, float]):
        """Update best solution display"""
        self.best_loss_label.setText(f"Best Loss: {best_loss:.6f}")
        params_str = ", ".join([f"{k}: {v:.3f}" for k, v in best_params.items()])
        self.best_params_label.setText(f"Params: {params_str}")
