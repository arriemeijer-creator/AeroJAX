"""
Collapsible GroupBox widget for space-saving UI
Allows users to collapse/expand sections to save vertical space
"""

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QFrame
)
from PyQt6.QtCore import Qt


class CollapsibleGroupBox(QFrame):
    """A collapsible group box that can be collapsed and expanded by clicking on the title"""
    
    def __init__(self, title="", parent=None, start_collapsed=True):
        super().__init__(parent)
        self._is_collapsed = start_collapsed
        self._content_widget = None
        self._toggle_button = None
        self._original_layout = None
        self._title_label = None
        self._title_text = title
        self._setup_collapsible_ui(start_collapsed)
        
        # Add styling for discrete container look
        self.setStyleSheet("""
            CollapsibleGroupBox {
                border: 1px solid #888;
                border-radius: 4px;
                background-color: #f5f5f5;
                margin: 2px;
            }
        """)
    
    def _setup_collapsible_ui(self, start_collapsed):
        """Setup the collapsible UI with a toggle button"""
        # Create a custom layout that includes a toggle button
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Title bar with toggle button
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: transparent;")
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(8, 6, 8, 6)
        title_bar_layout.setSpacing(8)
        
        # Toggle button with arrow
        self._toggle_button = QPushButton("▼")
        self._toggle_button.setFixedSize(16, 16)
        self._toggle_button.setFlat(True)
        self._toggle_button.setStyleSheet("""
            QPushButton {
                border: none;
                padding: 0px;
                font-size: 10px;
                background: transparent;
            }
        """)
        self._toggle_button.clicked.connect(self.toggle_collapse)
        
        # Title label
        self._title_label = QLabel(self._title_text)
        self._title_label.setStyleSheet("font-weight: 600;")
        
        title_bar_layout.addWidget(self._toggle_button)
        title_bar_layout.addWidget(self._title_label)
        title_bar_layout.addStretch()
        
        # Content container - will hold the original layout
        self._content_widget = QWidget()
        self._content_widget.setStyleSheet("background-color: transparent;")
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(8, 4, 8, 8)
        self._content_layout.setSpacing(4)
        
        # Hide content initially if starting collapsed
        if start_collapsed:
            self._content_widget.setVisible(False)
            self._toggle_button.setText("▶")
        
        # Add to main layout
        main_layout.addWidget(title_bar)
        main_layout.addWidget(self._content_widget)
    
    def setLayout(self, layout):
        """Override setLayout to wrap the layout in the content widget"""
        self._original_layout = layout
        self._content_layout.addLayout(layout)
        
        # Content is already hidden initially if start_collapsed was True
        # No need to re-apply collapse state here
    
    def toggle_collapse(self):
        """Toggle the collapsed state"""
        self.set_collapsed(not self._is_collapsed)
    
    def set_collapsed(self, collapsed: bool):
        """Set the collapsed state"""
        if self._is_collapsed == collapsed:
            return

        self._is_collapsed = collapsed

        # Update toggle button
        self._toggle_button.setText("▶" if collapsed else "▼")

        # Show/hide content
        self._content_widget.setVisible(not collapsed)

        # Adjust size
        if collapsed:
            self.setMaximumHeight(35)
        else:
            self.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
    
    def is_collapsed(self) -> bool:
        """Return whether the group box is collapsed"""
        return self._is_collapsed
    
    def setTitle(self, title: str):
        """Set the title text"""
        self._title_text = title
        if self._title_label:
            self._title_label.setText(title)
