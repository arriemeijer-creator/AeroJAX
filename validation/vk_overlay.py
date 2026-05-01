"""
Von Karman vortex visualization overlay.

This module provides visualization overlays for von Karman vortex tracking,
including x markers for vortex centers and vertical dashed lines for
Strouhal frequency visualization.
"""

import numpy as np
from typing import Optional
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor


class VKVortexOverlay:
    """
    Visualization overlay for von Karman vortex tracking.

    Displays vortex centers as 'x' markers and vertical dashed lines
    following each vortex to visualize shedding frequency.
    """

    def __init__(self, plot_widget):
        """
        Initialize VK vortex overlay.

        Args:
            plot_widget: pyqtgraph PlotWidget to draw on
        """
        self.plot_widget = plot_widget

        # Visualization items
        self.vortex_markers = None  # ScatterPlotItem for 'x' markers at vortex centers
        self.dashed_lines = []  # List of PlotCurveItem for vertical dashed lines following vortices
        self.info_text = None  # TextItem for Strouhal number display
        self.boundary_lines = []  # List of PlotCurveItem for x_min and x_max boundaries

        # Store current vortex positions (normalized coordinates)
        self.current_vortices = []  # List of (x_norm, y_norm, sign, id) tuples

        # Physical domain bounds (from simulation configuration)
        self.domain_x_min = 0.0
        self.domain_x_max = 20.0
        self.domain_y_min = 0.0
        self.domain_y_max = 7.5

        # X-range boundaries (normalized)
        self.x_min = 0.5
        self.x_max = 1.0

        # Max vortices to display (default matches slider)
        self.max_vortices = 2

        # Create visualization items
        self._create_visualization()

        # Connect to view change signal to update lines on zoom/pan
        self.plot_widget.sigRangeChanged.connect(self._on_view_changed)

    def _normalized_to_plot(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        """
        Convert normalized coordinates (0-1) to physical domain coordinates.

        Args:
            x_norm: Normalized x coordinate (0-1)
            y_norm: Normalized y coordinate (0-1)

        Returns:
            Tuple of (x_plot, y_plot) in physical domain coordinate system
        """
        x_plot = x_norm * (self.domain_x_max - self.domain_x_min) + self.domain_x_min
        y_plot = y_norm * (self.domain_y_max - self.domain_y_min) + self.domain_y_min

        return x_plot, y_plot

    def _create_visualization(self):
        """Create visualization items for vortex tracking."""
        # Create 'x' markers for vortex centers
        # Using ScatterPlotItem with cross symbol
        self.vortex_markers = pg.ScatterPlotItem(
            symbol='x',
            size=15,
            pen=pg.mkPen('r', width=1),
            brush=pg.mkBrush(255, 0, 0, 150)
        )
        self.plot_widget.addItem(self.vortex_markers)

        # Create vertical dashed lines for tracking vortices (dynamic based on number of vortices)
        self.dashed_lines = []
        for _ in range(10):  # Support up to 10 vortices
            line = pg.PlotCurveItem(
                pen=pg.mkPen((128, 128, 128, 150), width=1, style=Qt.PenStyle.DotLine)
            )
            self.plot_widget.addItem(line)
            self.dashed_lines.append(line)

        # Create info text for Strouhal number
        self.info_text = pg.TextItem(
            text="Strouhal: --",
            color=(255, 255, 255),
            anchor=(0, 1),
            fill=(0, 0, 0, 150)
        )
        self.info_text.setPos(0.02, 0.98)
        self.plot_widget.addItem(self.info_text)

        # Create white dotted lines for x_min and x_max boundaries
        self.boundary_lines = []
        for _ in range(2):
            line = pg.PlotCurveItem(
                pen=pg.mkPen((255, 255, 255, 200), width=1, style=Qt.PenStyle.DotLine)
            )
            self.plot_widget.addItem(line)
            self.boundary_lines.append(line)

        # Initialize boundary lines
        self.update_boundary_lines(self.x_min, self.x_max)
    
    def update(self, tracking_data: dict):
        """
        Update visualization with computed tracking data.

        Args:
            tracking_data: Dictionary containing:
                - primary_vortices: List of (x_norm, y_norm, sign, id) tuples for vortex centers
                - wake_center: Current wake center y-position (normalized)
                - tracking_x: X-position for tracking line (normalized)
                - strouhal: Strouhal number (or None)
                - is_shedding: Boolean indicating if shedding is detected
        """
        primary_vortices = tracking_data.get('primary_vortices', [])
        wake_center = tracking_data.get('wake_center', 0.5)
        tracking_x = tracking_data.get('tracking_x', 0.5)
        strouhal = tracking_data.get('strouhal', None)
        is_shedding = tracking_data.get('is_shedding', False)

        # Debug: print received data
        print(f"DEBUG Overlay: Received {len(primary_vortices)} vortices, shedding={is_shedding}")

        # Get plot bounds for coordinate conversion
        view_range = self.plot_widget.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]

        # Store current vortex positions (normalized) - limit to max_vortices
        self.current_vortices = primary_vortices[:self.max_vortices]

        # Update vortex markers at actual vortex centers
        # Clear markers first to ensure old ones are removed
        self.vortex_markers.setData([], [])

        if primary_vortices:
            # Limit to self.max_vortices (safety check)
            display_vortices = primary_vortices[:self.max_vortices]

            x_coords = []
            y_coords = []
            brushes = []

            for vortex_data in display_vortices:
                # Handle both old format (x, y, sign) and new format (x, y, sign, id)
                if len(vortex_data) == 4:
                    x_norm, y_norm, sign, vortex_id = vortex_data
                else:
                    x_norm, y_norm, sign = vortex_data
                    vortex_id = None

                x_plot, y_plot = self._normalized_to_plot(x_norm, y_norm)
                x_coords.append(x_plot)
                y_coords.append(y_plot)
                # Red for positive (clockwise), blue for negative (counter-clockwise)
                brush = (255, 0, 0, 200) if sign > 0 else (0, 0, 255, 200)
                brushes.append(brush)

            self.vortex_markers.setData(x_coords, y_coords, brush=brushes)

        # Ensure we don't have more dashed lines than vortices
        for i in range(len(self.dashed_lines)):
            if i >= len(self.current_vortices):
                self.dashed_lines[i].setData([], [])

        # Clear excess dashed lines (important when max_vortices is reduced)
        for i, line in enumerate(self.dashed_lines):
            if i >= len(primary_vortices):
                line.setData([], [])

        # Update dashed lines following vortices
        self._update_dashed_lines(x_min, x_max, y_min, y_max)

        # Update Strouhal number display
        if strouhal is not None:
            shedding_status = "Shedding" if is_shedding else "No shedding"
            self.info_text.setText(f"Strouhal: {strouhal:.3f} ({shedding_status})")
        else:
            self.info_text.setText("Strouhal: -- (Insufficient data)")

    def _update_dashed_lines(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Update dashed lines based on current view range.

        Args:
            x_min, x_max: Current x view range
            y_min, y_max: Current y view range
        """
        for i, line in enumerate(self.dashed_lines):
            if i < len(self.current_vortices):
                vortex_data = self.current_vortices[i]
                # Handle both old and new format
                if len(vortex_data) == 4:
                    x_norm, y_norm, sign, vortex_id = vortex_data
                else:
                    x_norm, y_norm, sign = vortex_data
                x_plot, _ = self._normalized_to_plot(x_norm, 0)
                # Show vertical line at vortex x-position (physical coords)
                # spanning the current view range in y
                line.setData([x_plot, x_plot], [y_min, y_max])
                # Make line thin and grey dotted
                line.setPen(pg.mkPen((128, 128, 128, 150), width=1, style=Qt.PenStyle.DotLine))
            else:
                line.setData([], [])

    def _on_view_changed(self):
        """Handle view range changes (zoom/pan) to update dashed lines."""
        view_range = self.plot_widget.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]
        self._update_dashed_lines(x_min, x_max, y_min, y_max)
        self._update_boundary_lines(x_min, x_max, y_min, y_max)

    def update_boundary_lines(self, x_min_norm: float, x_max_norm: float):
        """
        Update the x_min and x_max boundary lines.

        Args:
            x_min_norm: Normalized x_min (0-1)
            x_max_norm: Normalized x_max (0-1)
        """
        self.x_min = x_min_norm
        self.x_max = x_max_norm

        view_range = self.plot_widget.viewRange()
        y_min, y_max = view_range[1]
        self._update_boundary_lines(None, None, y_min, y_max)

    def _update_boundary_lines(self, x_view_min, x_view_max, y_min, y_max):
        """
        Update boundary lines based on current view range.

        Args:
            x_view_min, x_view_max: Current x view range (ignored, uses physical coords)
            y_min, y_max: Current y view range
        """
        # Convert normalized boundaries to physical coordinates
        x_min_phys = self.x_min * (self.domain_x_max - self.domain_x_min) + self.domain_x_min
        x_max_phys = self.x_max * (self.domain_x_max - self.domain_x_min) + self.domain_x_min

        # Update x_min line
        self.boundary_lines[0].setData([x_min_phys, x_min_phys], [y_min, y_max])

        # Update x_max line
        self.boundary_lines[1].setData([x_max_phys, x_max_phys], [y_min, y_max])

    def remove(self):
        """Remove all visualization items from plot."""
        if self.vortex_markers is not None:
            self.plot_widget.removeItem(self.vortex_markers)
        for line in self.dashed_lines:
            self.plot_widget.removeItem(line)
        for line in self.boundary_lines:
            self.plot_widget.removeItem(line)
        if self.info_text is not None:
            self.plot_widget.removeItem(self.info_text)

    def force_clear_markers(self):
        """Force clear all markers and dashed lines immediately."""
        self.vortex_markers.setData([], [])
        for line in self.dashed_lines:
            line.setData([], [])
        # Also clear stored vortex positions
        self.current_vortices = []

    def set_max_vortices(self, max_vortices: int):
        """Update the maximum number of vortices to display."""
        self.max_vortices = max_vortices
