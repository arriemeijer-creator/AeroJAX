"""
GUI overlay markers for LDC validation visualization.

This module provides visualization components for displaying benchmark validation
results on top of flow field plots in the AeroJAX viewer.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from typing import Optional, Tuple
from .ldc_validator import LDCValidator, ValidationError


class LDCValidationOverlay:
    """
    GUI overlay for LDC validation results.
    
    Renders markers for:
    - Ghia benchmark vortex center (fixed, green)
    - Simulation vortex center (moving, red)
    - Error vector line connecting both (yellow)
    
    Usage:
        overlay = LDCValidationOverlay(validator, plot_widget)
        overlay.update()  # Call after validator.compute()
        overlay.set_visible(True/False)
    """
    
    def __init__(self, validator: LDCValidator, plot_widget):
        """
        Initialize validation overlay.
        
        Args:
            validator: LDCValidator instance with computed results
            plot_widget: pyqtgraph PlotWidget to add overlays to
        """
        self.validator = validator
        self.plot_widget = plot_widget
        
        # Create marker items
        self.ref_marker = pg.ScatterPlotItem(
            size=20,
            pen=pg.mkPen('g', width=3),
            brush=pg.mkBrush(0, 255, 0, 150),
            symbol='x'
        )
        
        self.sim_marker = pg.ScatterPlotItem(
            size=20,
            pen=pg.mkPen('r', width=3),
            brush=pg.mkBrush(255, 0, 0, 150),
            symbol='o'
        )
        
        self.error_line = pg.PlotCurveItem(
            pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine)
        )
        
        # Error text label
        self.error_text = pg.TextItem(
            anchor=(0, 1),
            color='y',
            fill=(0, 0, 0, 180)
        )
        
        # Add items to plot
        self.plot_widget.addItem(self.ref_marker)
        self.plot_widget.addItem(self.sim_marker)
        self.plot_widget.addItem(self.error_line)
        self.plot_widget.addItem(self.error_text)
        
        # Initially hidden
        self.set_visible(False)
    
    def update(self) -> None:
        """
        Update overlay markers based on current validation results.
        
        Call this after validator.compute() to refresh the visualization.
        """
        vortex_center = self.validator.get_vortex_center()
        error = self.validator.get_error()
        
        if vortex_center is None or error is None:
            self.set_visible(False)
            return
        
        # Get domain dimensions
        lx = self.validator.lx
        ly = self.validator.ly
        
        # Reference marker (Ghia)
        x_ref, y_ref = self.validator.get_reference_center()
        self.ref_marker.setData([x_ref * lx], [y_ref * ly])
        
        # Simulation marker
        x_sim, y_sim = vortex_center.x, vortex_center.y
        self.sim_marker.setData([x_sim * lx], [y_sim * ly])
        
        # Error vector line
        self.error_line.setData(
            [x_ref * lx, x_sim * lx],
            [y_ref * ly, y_sim * ly]
        )
        
        # Error text
        error_str = (
            f"Re={self.validator.Re}\n"
            f"Ref: ({x_ref:.3f}, {y_ref:.3f})\n"
            f"Sim: ({x_sim:.3f}, {y_sim:.3f})\n"
            f"Error: {error.l2_distance:.4f}"
        )
        self.error_text.setText(error_str)
        self.error_text.setPos(x_sim * lx + 0.05 * lx, y_sim * ly + 0.05 * ly)
        
        self.set_visible(True)
    
    def set_visible(self, visible: bool) -> None:
        """
        Set overlay visibility.
        
        Args:
            visible: True to show overlays, False to hide
        """
        self.ref_marker.setVisible(visible)
        self.sim_marker.setVisible(visible)
        self.error_line.setVisible(visible)
        self.error_text.setVisible(visible)
    
    def remove(self) -> None:
        """Remove overlay items from plot."""
        self.plot_widget.removeItem(self.ref_marker)
        self.plot_widget.removeItem(self.sim_marker)
        self.plot_widget.removeItem(self.error_line)
        self.plot_widget.removeItem(self.error_text)
