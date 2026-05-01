"""
Step Navigation Panel for the trace viewer.

Provides controls for navigating through snapshots in time.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class NavigationPanel(QWidget):
    """Panel for navigating through simulation snapshots."""
    
    # Signals
    step_changed = pyqtSignal(int)  # Emitted when timestep changes
    step_forward = pyqtSignal()
    step_backward = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_step = 0
        self.total_steps = 0
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Step Navigation")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Current step display
        self.step_label = QLabel("Step: 0 / 0")
        self.step_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.step_label)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.backward_btn = QPushButton("◀ Step Back")
        self.backward_btn.clicked.connect(self.on_backward)
        self.backward_btn.setEnabled(False)
        button_layout.addWidget(self.backward_btn)
        
        self.forward_btn = QPushButton("Step Forward ▶")
        self.forward_btn.clicked.connect(self.on_forward)
        self.forward_btn.setEnabled(False)
        button_layout.addWidget(self.forward_btn)
        
        layout.addLayout(button_layout)
        
        # Timeline slider
        slider_label = QLabel("Timeline:")
        slider_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(slider_label)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)
        
        # Step spinbox for direct input
        spinbox_layout = QHBoxLayout()
        spinbox_layout.addWidget(QLabel("Go to step:"))
        self.step_spinbox = QSpinBox()
        self.step_spinbox.setMinimum(0)
        self.step_spinbox.setMaximum(0)
        self.step_spinbox.valueChanged.connect(self.on_spinbox_changed)
        spinbox_layout.addWidget(self.step_spinbox)
        layout.addLayout(spinbox_layout)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def set_total_steps(self, total: int):
        """Set total number of steps."""
        self.total_steps = total
        self.slider.setMaximum(total - 1 if total > 0 else 0)
        self.step_spinbox.setMaximum(total - 1 if total > 0 else 0)
        self.update_ui()
        
    def set_current_step(self, step: int):
        """Set current timestep."""
        self.current_step = step
        self.slider.blockSignals(True)
        self.slider.setValue(step)
        self.slider.blockSignals(False)
        self.step_spinbox.blockSignals(True)
        self.step_spinbox.setValue(step)
        self.step_spinbox.blockSignals(False)
        self.update_ui()
        
    def update_ui(self):
        """Update UI state based on current step."""
        self.step_label.setText(f"Step: {self.current_step} / {self.total_steps - 1 if self.total_steps > 0 else 0}")
        self.backward_btn.setEnabled(self.current_step > 0)
        self.forward_btn.setEnabled(self.current_step < self.total_steps - 1)
        
    def on_backward(self):
        """Handle step backward button."""
        if self.current_step > 0:
            self.set_current_step(self.current_step - 1)
            self.step_backward.emit()
            self.step_changed.emit(self.current_step)
            
    def on_forward(self):
        """Handle step forward button."""
        if self.current_step < self.total_steps - 1:
            self.set_current_step(self.current_step + 1)
            self.step_forward.emit()
            self.step_changed.emit(self.current_step)
            
    def on_slider_changed(self, value: int):
        """Handle slider value change."""
        self.set_current_step(value)
        self.step_changed.emit(self.current_step)
        
    def on_spinbox_changed(self, value: int):
        """Handle spinbox value change."""
        self.set_current_step(value)
        self.step_changed.emit(self.current_step)
