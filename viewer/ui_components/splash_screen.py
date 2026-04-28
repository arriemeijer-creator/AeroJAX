"""
Custom splash screen with progress bar.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class SplashScreen(QWidget):
    """Custom splash screen with image and progress bar."""
    
    def __init__(self, image_path: str):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Load and display splash image
        self.image_label = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(
            int(pixmap.width() * 0.75), 
            int(pixmap.height() * 0.75), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 5px;
                text-align: center;
                background-color: #e0e0e0;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: white; font-size: 12px; background-color: #333; padding: 5px;")
        self.status_label.setText("Initializing...")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        self.setFixedSize(pixmap.width(), pixmap.height() + 30)  # Image + progress bar + status
    
    def update_progress(self, value: int, message: str = ""):
        """Update progress bar and status message."""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
