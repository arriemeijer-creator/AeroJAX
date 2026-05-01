"""
Solver Trace Panel for displaying the 5-step solver pipeline.

Shows step-by-step numerical operations with LaTeX equations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QScrollArea, QFrame, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .trace_builder import SolverTrace, TraceStep


class LaTeXRenderer(FigureCanvas):
    """Renders LaTeX equations using matplotlib mathtext."""
    
    def __init__(self, equation: str, parent=None):
        self.figure = Figure(figsize=(8, 1), dpi=100)
        super().__init__(self.figure)
        self.setParent(parent)
        self.render_equation(equation)
        
    def render_equation(self, equation: str):
        """Render LaTeX equation."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, f'${equation}$', 
                fontsize=14, ha='center', va='center')
        self.figure.tight_layout()
        self.draw()


class StepWidget(QFrame):
    """Widget displaying a single solver step."""
    
    def __init__(self, step: TraceStep, parent=None):
        super().__init__(parent)
        self.step = step
        self.init_ui()
        
    def init_ui(self):
        """Initialize the step widget UI."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: #2d2d2d;
                margin: 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Step header
        header_layout = QHBoxLayout()
        step_num = QLabel(f"Step {self.step.step_number}")
        step_num.setStyleSheet("font-weight: bold; font-size: 13px; color: #4a9eff;")
        header_layout.addWidget(step_num)
        
        step_name = QLabel(self.step.step_name)
        step_name.setStyleSheet("font-weight: bold; font-size: 13px;")
        header_layout.addWidget(step_name)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Method label
        method_label = QLabel(f"Method: {self.step.method}")
        method_label.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(method_label)
        
        # LaTeX equation
        try:
            latex_widget = LaTeXRenderer(self.step.equation_latex)
            layout.addWidget(latex_widget)
        except Exception as e:
            # Fallback to plain text if LaTeX rendering fails
            eq_label = QLabel(f"Equation: {self.step.equation_latex}")
            eq_label.setStyleSheet("font-style: italic; color: #aaa;")
            layout.addWidget(eq_label)
        
        # Description
        desc_label = QLabel(self.step.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 11px; color: #ccc;")
        layout.addWidget(desc_label)
        
        # Inputs
        if self.step.inputs:
            inputs_group = QGroupBox("Inputs")
            inputs_layout = QVBoxLayout()
            for key, value in self.step.inputs.items():
                input_label = QLabel(f"  • {key}: {value}")
                input_label.setStyleSheet("font-size: 10px; color: #aaa;")
                inputs_layout.addWidget(input_label)
            inputs_group.setLayout(inputs_layout)
            layout.addWidget(inputs_group)
        
        # Outputs
        if self.step.outputs:
            outputs_group = QGroupBox("Outputs")
            outputs_layout = QVBoxLayout()
            for key, value in self.step.outputs.items():
                output_label = QLabel(f"  • {key}: {value}")
                output_label.setStyleSheet("font-size: 10px; color: #aaa;")
                outputs_layout.addWidget(output_label)
            outputs_group.setLayout(outputs_layout)
            layout.addWidget(outputs_group)
        
        self.setLayout(layout)


class SolverTracePanel(QWidget):
    """Panel displaying the 5-step solver pipeline for current timestep."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_trace: SolverTrace = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Solver Trace Pipeline")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Scroll area for steps
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout()
        self.steps_layout.addStretch()
        self.steps_container.setLayout(self.steps_layout)
        
        scroll.setWidget(self.steps_container)
        layout.addWidget(scroll)
        
        # Metrics display
        self.metrics_label = QLabel("Metrics: No data loaded")
        self.metrics_label.setStyleSheet("font-size: 11px; color: #888; padding: 5px;")
        layout.addWidget(self.metrics_label)
        
        self.setLayout(layout)
        
    def set_trace(self, trace: SolverTrace):
        """Update the panel with a new solver trace."""
        self.current_trace = trace
        
        # Clear existing steps
        for i in reversed(range(self.steps_layout.count() - 1)):
            item = self.steps_layout.takeAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        # Add step widgets
        for step in trace.steps:
            step_widget = StepWidget(step)
            self.steps_layout.insertWidget(self.steps_layout.count() - 1, step_widget)
        
        # Update metrics
        metrics_text = "Metrics: "
        metrics_items = []
        for key, value in trace.metrics.items():
            if isinstance(value, float):
                metrics_items.append(f"{key}={value:.4f}")
            else:
                metrics_items.append(f"{key}={value}")
        self.metrics_label.setText("Metrics: " + ", ".join(metrics_items))
        
    def clear(self):
        """Clear the panel."""
        self.current_trace = None
        for i in reversed(range(self.steps_layout.count() - 1)):
            item = self.steps_layout.takeAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.metrics_label.setText("Metrics: No data loaded")
