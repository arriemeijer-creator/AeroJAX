"""
Subdomain Numerical Viewer for inspecting field values.

Displays a QTableView of extracted subdomain patches with u, v, p, divergence.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTableView, QSpinBox, QCheckBox, QGroupBox, QTabWidget, QPushButton
)
from PyQt6.QtCore import Qt, QAbstractTableModel
from PyQt6.QtGui import QColor, QWheelEvent
import numpy as np
from typing import Dict, Optional


class ZoomableTableView(QTableView):
    """A QTableView that supports Ctrl + mouse wheel zoom."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_section_size = 5  # Allow much more zoom out
        self.max_section_size = 200
        self.zoom_factor = 1.2
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle wheel event for zooming when Ctrl is pressed."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom functionality
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            elif delta < 0:
                self.zoom_out()
            event.accept()
        else:
            # Normal scrolling
            super().wheelEvent(event)
    
    def zoom_in(self):
        """Zoom in by increasing section sizes."""
        current_h_size = self.horizontalHeader().defaultSectionSize()
        current_v_size = self.verticalHeader().defaultSectionSize()
        
        new_h_size = min(int(current_h_size * self.zoom_factor), self.max_section_size)
        new_v_size = min(int(current_v_size * self.zoom_factor), self.max_section_size)
        
        self.horizontalHeader().setDefaultSectionSize(new_h_size)
        self.verticalHeader().setDefaultSectionSize(new_v_size)
        
    def zoom_out(self):
        """Zoom out by decreasing section sizes."""
        current_h_size = self.horizontalHeader().defaultSectionSize()
        current_v_size = self.verticalHeader().defaultSectionSize()
        
        new_h_size = max(int(current_h_size / self.zoom_factor), self.min_section_size)
        new_v_size = max(int(current_v_size / self.zoom_factor), self.min_section_size)
        
        self.horizontalHeader().setDefaultSectionSize(new_h_size)
        self.verticalHeader().setDefaultSectionSize(new_v_size)


class SubdomainTableModel(QAbstractTableModel):
    """Model for displaying subdomain field values in QTableView."""
    
    def __init__(self, data: np.ndarray, field_name: str, parent=None):
        super().__init__(parent)
        # Transpose data so x is horizontal (columns) and y is vertical (rows)
        self.data = data.T  # Transpose to swap x and y
        self.field_name = field_name
        self.rows, self.cols = self.data.shape  # Update shape after transpose
        
    def rowCount(self, parent=None):
        return self.rows
        
    def columnCount(self, parent=None):
        return self.cols
        
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
            
        row, col = index.row(), index.column()
        value = self.data[row, col]
        
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{value:.4f}"
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Color coding based on value magnitude
            abs_val = abs(value)
            max_val = np.max(np.abs(self.data)) if np.max(np.abs(self.data)) > 0 else 1.0
            intensity = int(255 * (1 - abs_val / max_val))
            # Blue for positive, red for negative
            if value >= 0:
                return QColor(0, 0, min(255, intensity + 50))
            else:
                return QColor(min(255, intensity + 50), 0, 0)
        elif role == Qt.ItemDataRole.ToolTipRole:
            return f"{self.field_name}[{col}, {row}] = {value:.6f}"  # Swapped indices due to transpose
            
        return None
        
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return f"x={section}"  # x is now horizontal (left to right)
            else:
                return f"y={section}"  # y is now vertical (bottom to top)
        return None


class SubdomainViewer(QWidget):
    """Panel for viewing numerical values in a selected subdomain."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data: Dict[str, np.ndarray] = {}
        self.nx = 0
        self.ny = 0
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Subdomain Numerical Viewer")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Subdomain selection controls
        selection_group = QGroupBox("Subdomain Selection")
        selection_layout = QVBoxLayout()
        
        # X range
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X range:"))
        self.x_min_spin = QSpinBox()
        self.x_min_spin.setMinimum(0)
        self.x_min_spin.valueChanged.connect(self.update_subdomain)
        x_layout.addWidget(self.x_min_spin)
        x_layout.addWidget(QLabel("to"))
        self.x_max_spin = QSpinBox()
        self.x_max_spin.setMinimum(0)
        self.x_max_spin.valueChanged.connect(self.update_subdomain)
        x_layout.addWidget(self.x_max_spin)
        selection_layout.addLayout(x_layout)
        
        # Y range
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y range:"))
        self.y_min_spin = QSpinBox()
        self.y_min_spin.setMinimum(0)
        self.y_min_spin.valueChanged.connect(self.update_subdomain)
        y_layout.addWidget(self.y_min_spin)
        y_layout.addWidget(QLabel("to"))
        self.y_max_spin = QSpinBox()
        self.y_max_spin.setMinimum(0)
        self.y_max_spin.valueChanged.connect(self.update_subdomain)
        y_layout.addWidget(self.y_max_spin)
        selection_layout.addLayout(y_layout)
        
        # Auto Fit button
        self.auto_fit_btn = QPushButton("Auto Fit")
        self.auto_fit_btn.clicked.connect(self.auto_fit_bounds)
        selection_layout.addWidget(self.auto_fit_btn)
        
        # Show divergence checkbox
        self.show_divergence = QCheckBox("Show Divergence")
        self.show_divergence.setChecked(True)
        self.show_divergence.toggled.connect(self.update_tabs)
        selection_layout.addWidget(self.show_divergence)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Tab widget for different fields
        self.tabs = QTabWidget()
        
        # U-velocity tab
        self.u_table = ZoomableTableView()
        self.u_table.horizontalHeader().setDefaultSectionSize(30)
        self.u_table.verticalHeader().setDefaultSectionSize(10)
        self.tabs.addTab(self.u_table, "U Velocity")
        
        # V-velocity tab
        self.v_table = ZoomableTableView()
        self.v_table.horizontalHeader().setDefaultSectionSize(30)
        self.v_table.verticalHeader().setDefaultSectionSize(10)
        self.tabs.addTab(self.v_table, "V Velocity")
        
        # Pressure tab
        self.p_table = ZoomableTableView()
        self.p_table.horizontalHeader().setDefaultSectionSize(30)
        self.p_table.verticalHeader().setDefaultSectionSize(10)
        self.tabs.addTab(self.p_table, "Pressure")
        
        # Divergence tab
        self.div_table = ZoomableTableView()
        self.div_table.horizontalHeader().setDefaultSectionSize(30)
        self.div_table.verticalHeader().setDefaultSectionSize(10)
        self.tabs.addTab(self.div_table, "Divergence")
        
        layout.addWidget(self.tabs)
        
        # Info label
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        
    def set_grid_size(self, nx: int, ny: int):
        """Set the grid dimensions."""
        self.nx = nx
        self.ny = ny
        self.x_min_spin.setMaximum(nx - 1)
        self.x_max_spin.setMaximum(nx - 1)
        self.y_min_spin.setMaximum(ny - 1)
        self.y_max_spin.setMaximum(ny - 1)
        
        # Set default to show center region
        self.x_min_spin.setValue(max(0, nx // 2 - 5))
        self.x_max_spin.setValue(min(nx - 1, nx // 2 + 5))
        self.y_min_spin.setValue(max(0, ny // 2 - 5))
        self.y_max_spin.setValue(min(ny - 1, ny // 2 + 5))
        
    def set_data(self, data: Dict[str, np.ndarray]):
        """Set the field data for the current snapshot."""
        self.current_data = data
        
        if 'u' in data:
            self.nx, self.ny = data['u'].shape
            
            # Check if this is the first time data is being set
            is_first_load = (self.x_min_spin.maximum() == 0 and 
                           self.x_max_spin.maximum() == 0 and 
                           self.y_min_spin.maximum() == 0 and 
                           self.y_max_spin.maximum() == 0)
            
            if is_first_load:
                # First load - set default values to show center region
                self.x_min_spin.setMaximum(self.nx - 1)
                self.x_max_spin.setMaximum(self.nx - 1)
                self.y_min_spin.setMaximum(self.ny - 1)
                self.y_max_spin.setMaximum(self.ny - 1)
                
                # Set default to show center region
                self.x_min_spin.setValue(max(0, self.nx // 2 - 5))
                self.x_max_spin.setValue(min(self.nx - 1, self.nx // 2 + 5))
                self.y_min_spin.setValue(max(0, self.ny // 2 - 5))
                self.y_max_spin.setValue(min(self.ny - 1, self.ny // 2 + 5))
            else:
                # Preserve current spin box values
                current_x_min = self.x_min_spin.value()
                current_x_max = self.x_max_spin.value()
                current_y_min = self.y_min_spin.value()
                current_y_max = self.y_max_spin.value()
                
                # Update grid size limits without resetting values
                self.x_min_spin.setMaximum(self.nx - 1)
                self.x_max_spin.setMaximum(self.nx - 1)
                self.y_min_spin.setMaximum(self.ny - 1)
                self.y_max_spin.setMaximum(self.ny - 1)
                
                # Clamp current values to new bounds if necessary
                self.x_min_spin.setValue(min(current_x_min, self.nx - 1))
                self.x_max_spin.setValue(min(current_x_max, self.nx - 1))
                self.y_min_spin.setValue(min(current_y_min, self.ny - 1))
                self.y_max_spin.setValue(min(current_y_max, self.ny - 1))
            
        self.update_subdomain()
        
    def update_subdomain(self):
        """Update the subdomain display based on selection."""
        if not self.current_data:
            return
            
        x_min = self.x_min_spin.value()
        x_max = self.x_max_spin.value()
        y_min = self.y_min_spin.value()
        y_max = self.y_max_spin.value()
        
        # Ensure valid range
        x_min = min(x_min, x_max)
        y_min = min(y_min, y_max)
        
        # Extract subdomain
        if 'u' in self.current_data:
            u_patch = self.current_data['u'][x_min:x_max+1, y_min:y_max+1]
            u_model = SubdomainTableModel(u_patch, 'u')
            self.u_table.setModel(u_model)
            
        if 'v' in self.current_data:
            v_patch = self.current_data['v'][x_min:x_max+1, y_min:y_max+1]
            v_model = SubdomainTableModel(v_patch, 'v')
            self.v_table.setModel(v_model)
            
        if 'p' in self.current_data:
            p_patch = self.current_data['p'][x_min:x_max+1, y_min:y_max+1]
            p_model = SubdomainTableModel(p_patch, 'p')
            self.p_table.setModel(p_model)
            
        if 'divergence' in self.current_data and self.current_data['divergence'] is not None:
            div_patch = self.current_data['divergence'][x_min:x_max+1, y_min:y_max+1]
            div_model = SubdomainTableModel(div_patch, 'divergence')
            self.div_table.setModel(div_model)
            
        # Update info
        patch_size = (x_max - x_min + 1) * (y_max - y_min + 1)
        self.info_label.setText(f"Subdomain: [{x_min}:{x_max}, {y_min}:{y_max}] - {patch_size} cells")
        
        self.update_tabs()
        
    def auto_fit_bounds(self):
        """Set bounds to show full grid (0 to max for both x and y)."""
        if self.nx > 0 and self.ny > 0:
            self.x_min_spin.setValue(0)
            self.x_max_spin.setValue(self.nx - 1)
            self.y_min_spin.setValue(0)
            self.y_max_spin.setValue(self.ny - 1)
            
    def update_tabs(self):
        """Show/hide divergence tab based on checkbox."""
        if self.show_divergence.isChecked() and 'divergence' in self.current_data and self.current_data['divergence'] is not None:
            self.tabs.setTabVisible(3, True)
        else:
            self.tabs.setTabVisible(3, False)
            
    def clear(self):
        """Clear the viewer."""
        self.current_data = {}
        self.u_table.setModel(None)
        self.v_table.setModel(None)
        self.p_table.setModel(None)
        self.div_table.setModel(None)
        self.info_label.setText("No data loaded")
