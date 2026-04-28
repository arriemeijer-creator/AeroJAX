"""
Neural Operator Training UI Component
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox,
    QLineEdit, QFileDialog, QProgressBar
)
from PyQt6.QtCore import Qt
import os
import glob


class NeuralOperatorTraining(QWidget):
    """Neural Operator Training Controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup neural operator training UI"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Dataset Generation Section
        dataset_group = QGroupBox("Dataset Generation")
        dataset_layout = QVBoxLayout()
        
        # Number of steps
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Simulation Steps:"))
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setMinimum(1)
        self.steps_spinbox.setMaximum(100000)
        self.steps_spinbox.setValue(1000)
        steps_layout.addWidget(self.steps_spinbox)
        dataset_layout.addLayout(steps_layout)
        
        # Fields to save
        fields_label = QLabel("Fields to Save:")
        dataset_layout.addWidget(fields_label)
        
        fields_layout = QVBoxLayout()
        self.save_u_cb = QCheckBox("Velocity U")
        self.save_u_cb.setChecked(True)
        self.save_v_cb = QCheckBox("Velocity V")
        self.save_v_cb.setChecked(True)
        self.save_p_cb = QCheckBox("Pressure")
        self.save_p_cb.setChecked(True)
        self.save_mask_cb = QCheckBox("Mask")
        self.save_mask_cb.setChecked(True)
        self.save_div_cb = QCheckBox("Divergence")
        self.save_div_cb.setChecked(False)
        
        fields_layout.addWidget(self.save_u_cb)
        fields_layout.addWidget(self.save_v_cb)
        fields_layout.addWidget(self.save_p_cb)
        fields_layout.addWidget(self.save_mask_cb)
        fields_layout.addWidget(self.save_div_cb)
        dataset_layout.addLayout(fields_layout)
        
        # Output filename
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Filename:"))
        self.output_filename = QLineEdit("training_data.npz")
        output_layout.addWidget(self.output_filename)
        dataset_layout.addLayout(output_layout)
        
        # Run & Save button
        self.run_save_btn = QPushButton("Run & Save Dataset")
        self.run_save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        dataset_layout.addWidget(self.run_save_btn)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Model Selection Section
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        # Neural operator dropdown
        operator_layout = QHBoxLayout()
        operator_layout.addWidget(QLabel("Neural Operator:"))
        self.operator_combo = QComboBox()
        self.populate_operators()
        operator_layout.addWidget(self.operator_combo)
        model_layout.addLayout(operator_layout)
        
        # Model architecture selector
        arch_layout = QHBoxLayout()
        arch_layout.addWidget(QLabel("Architecture:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItem("Linear")
        self.arch_combo.addItem("NonLinear")
        self.arch_combo.addItem("Advanced")
        arch_layout.addWidget(self.arch_combo)
        model_layout.addLayout(arch_layout)
        
        # Load trained model
        load_model_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Trained Model")
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        load_model_layout.addWidget(self.load_model_btn)
        model_layout.addLayout(load_model_layout)
        
        # Loaded model path display
        self.loaded_model_path = QLineEdit("No model loaded")
        self.loaded_model_path.setReadOnly(True)
        model_layout.addWidget(self.loaded_model_path)
        
        # Pressure solver selector
        solver_layout = QHBoxLayout()
        solver_layout.addWidget(QLabel("Pressure Solver:"))
        self.pressure_solver_combo = QComboBox()
        self.pressure_solver_combo.addItem("Multigrid")
        self.pressure_solver_combo.addItem("Neural Network")
        solver_layout.addWidget(self.pressure_solver_combo)
        model_layout.addLayout(solver_layout)
        
        # Training parameters
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(10000)
        self.epochs_spinbox.setValue(100)
        epochs_layout.addWidget(self.epochs_spinbox)
        model_layout.addLayout(epochs_layout)
        
        # Cancel training button
        cancel_layout = QHBoxLayout()
        self.cancel_training_btn = QPushButton("Cancel Training")
        self.cancel_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.cancel_training_btn.setEnabled(False)
        cancel_layout.addWidget(self.cancel_training_btn)
        model_layout.addLayout(cancel_layout)
        
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setMinimum(1e-6)
        self.lr_spinbox.setMaximum(1.0)
        self.lr_spinbox.setDecimals(6)
        self.lr_spinbox.setValue(1e-3)
        lr_layout.addWidget(self.lr_spinbox)
        model_layout.addLayout(lr_layout)
        
        batch_size_layout = QHBoxLayout()
        batch_size_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setMinimum(1)
        self.batch_size_spinbox.setMaximum(1000)
        self.batch_size_spinbox.setValue(32)
        batch_size_layout.addWidget(self.batch_size_spinbox)
        model_layout.addLayout(batch_size_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training Section
        training_group = QGroupBox("Training")
        training_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        training_layout.addWidget(self.progress_bar)
        
        # Train button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        training_layout.addWidget(self.train_btn)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def populate_operators(self):
        """Populate neural operator dropdown with files from neural_operators folder"""
        self.operator_combo.clear()
        
        # Get the neural_operators directory
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        neural_operators_dir = os.path.join(current_dir, "neural_operators")
        
        if os.path.exists(neural_operators_dir):
            # Find all Python files in the directory
            operator_files = glob.glob(os.path.join(neural_operators_dir, "*.py"))
            
            for file in operator_files:
                filename = os.path.basename(file)
                # Skip __init__.py
                if filename != "__init__.py":
                    # Remove .py extension
                    operator_name = filename[:-3]
                    self.operator_combo.addItem(operator_name)
        
        if self.operator_combo.count() == 0:
            self.operator_combo.addItem("No operators found")
    
    def get_training_config(self):
        """Get training configuration from UI"""
        return {
            'steps': self.steps_spinbox.value(),
            'save_fields': {
                'u': self.save_u_cb.isChecked(),
                'v': self.save_v_cb.isChecked(),
                'p': self.save_p_cb.isChecked(),
                'mask': self.save_mask_cb.isChecked(),
                'divergence': self.save_div_cb.isChecked()
            },
            'output_filename': self.output_filename.text(),
            'operator': self.operator_combo.currentText(),
            'architecture': self.arch_combo.currentText(),
            'epochs': self.epochs_spinbox.value(),
            'learning_rate': self.lr_spinbox.value(),
            'batch_size': self.batch_size_spinbox.value()
        }
    
    def get_loaded_model_path(self):
        """Get the path of the loaded model"""
        path = self.loaded_model_path.text()
        return path if path != "No model loaded" else None
