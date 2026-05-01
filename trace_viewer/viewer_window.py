"""
Main Trace Viewer Window for AeroJAX solver traces.

Combines navigation, solver trace, and subdomain viewer panels.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QSplitter, QMenuBar, QFileDialog, QMessageBox, QLabel
)
from PyQt6.QtCore import Qt
import sys

from .snapshot import Snapshot, SnapshotSeries
from .snapshot_utils import (
    load_snapshot, load_snapshot_series, load_snapshots_from_directory,
    validate_snapshot_directory
)
from .trace_builder import TraceBuilder
from .navigation_panel import NavigationPanel
from .solver_trace_panel import SolverTracePanel
from .subdomain_viewer import SubdomainViewer


class TraceViewerWindow(QMainWindow):
    """Main window for the solver trace viewer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot_series: SnapshotSeries = None
        self.current_step = 0
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        """Initialize the UI layout."""
        self.setWindowTitle("AeroJAX Solver Trace Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Navigation
        self.navigation_panel = NavigationPanel()
        self.navigation_panel.setMinimumWidth(200)
        self.navigation_panel.setMaximumWidth(250)
        self.navigation_panel.step_changed.connect(self.on_step_changed)
        
        # Center panel: Solver Trace
        self.trace_panel = SolverTracePanel()
        self.trace_panel.setMinimumWidth(400)
        
        # Right panel: Subdomain Viewer
        self.subdomain_viewer = SubdomainViewer()
        self.subdomain_viewer.setMinimumWidth(350)
        
        # Add panels to splitter
        splitter.addWidget(self.navigation_panel)
        splitter.addWidget(self.trace_panel)
        splitter.addWidget(self.subdomain_viewer)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        
        main_layout.addWidget(splitter)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Status bar
        self.status_label = QLabel("No snapshot series loaded")
        self.statusBar().addWidget(self.status_label)
        
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_series_action = file_menu.addAction("Load Snapshot Series...")
        load_series_action.triggered.connect(self.load_snapshot_series)
        
        load_dir_action = file_menu.addAction("Load from Directory...")
        load_dir_action.triggered.connect(self.load_from_directory)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        reset_layout_action = view_menu.addAction("Reset Layout")
        reset_layout_action.triggered.connect(self.reset_layout)
        
    def apply_dark_theme(self):
        """Apply a dark theme to the window."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #2d2d2d;
                color: #e0e0e0;
                font-family: Segoe UI, Arial, sans-serif;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #5d5d5d;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #3d3d3d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a9eff;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSpinBox {
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                color: #e0e0e0;
                padding: 5px 10px;
                border: 1px solid #555;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #4a9eff;
                color: white;
            }
            QTableView {
                background-color: #1e1e1e;
                gridline-color: #444;
                color: #e0e0e0;
            }
            QTableView::item {
                padding: 2px;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: #e0e0e0;
                padding: 3px;
                border: 1px solid #555;
            }
            QScrollArea {
                border: none;
                background-color: #2d2d2d;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border-bottom: 1px solid #555;
            }
            QMenuBar::item {
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #4a9eff;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #555;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #4a9eff;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: #888;
            }
        """)
        
    def load_snapshot_series(self):
        """Load a snapshot series file."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Snapshot Series",
            "",
            "Snapshot Files (*.pkl);;All Files (*)"
        )
        
        if filepaths:
            try:
                if len(filepaths) == 1:
                    # Single file - use existing logic
                    self.snapshot_series = load_snapshot_series(filepaths[0])
                else:
                    # Multiple files - load all and create a series
                    snapshots = []
                    for filepath in filepaths:
                        snapshot = load_snapshot(filepath)
                        snapshots.append(snapshot)
                    
                    # Sort snapshots by iteration if available, otherwise by filename
                    try:
                        snapshots.sort(key=lambda s: s.iteration)
                    except:
                        # If iteration not available, sort by filename order
                        # Create pairs of (filepath, snapshot) and sort by filepath
                        filepath_snapshot_pairs = list(zip(filepaths, snapshots))
                        filepath_snapshot_pairs.sort(key=lambda pair: pair[0])  # Sort by filepath
                        snapshots = [pair[1] for pair in filepath_snapshot_pairs]
                    
                    self.snapshot_series = SnapshotSeries(snapshots=snapshots)
                
                self.initialize_viewer()
                self.status_label.setText(f"Loaded {len(self.snapshot_series)} snapshots")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load snapshot series: {str(e)}")
                
    def load_from_directory(self):
        """Load snapshots from a directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Load Snapshots from Directory",
            ""
        )
        
        if directory:
            if not validate_snapshot_directory(directory):
                QMessageBox.warning(
                    self,
                    "Invalid Directory",
                    "Directory does not contain valid snapshot files."
                )
                return
                
            try:
                snapshots = load_snapshots_from_directory(directory)
                print(f"Loaded {len(snapshots)} snapshots from directory")
                print(f"Type of snapshots: {type(snapshots)}")
                if len(snapshots) > 0:
                    print(f"Type of first snapshot: {type(snapshots[0])}")
                
                # Ensure snapshots is a list
                if not isinstance(snapshots, list):
                    print(f"Converting single snapshot to list")
                    snapshots = [snapshots]
                
                self.snapshot_series = SnapshotSeries(snapshots=snapshots)
                print(f"Created SnapshotSeries with {len(self.snapshot_series)} snapshots")
                self.initialize_viewer()
                self.status_label.setText(f"Loaded {len(self.snapshot_series)} snapshots from {directory}")
            except Exception as e:
                import traceback
                print(f"Error loading snapshots: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Failed to load snapshots: {str(e)}")
                
    def initialize_viewer(self):
        """Initialize the viewer with loaded snapshot series."""
        if self.snapshot_series is None or len(self.snapshot_series) == 0:
            return
            
        # Set up navigation panel
        self.navigation_panel.set_total_steps(len(self.snapshot_series))
        self.navigation_panel.set_current_step(0)
        
        # Load first snapshot
        self.load_snapshot(0)
        
    def load_snapshot(self, step: int):
        """Load and display a snapshot at the given step."""
        if self.snapshot_series is None:
            return
            
        if step < 0 or step >= len(self.snapshot_series):
            return
            
        self.current_step = step
        snapshot = self.snapshot_series[step]
        
        # Build trace
        trace_builder = TraceBuilder(snapshot)
        trace = trace_builder.build_trace()
        
        # Update trace panel
        self.trace_panel.set_trace(trace)
        
        # Update subdomain viewer
        subdomain_data = {
            'u': snapshot.u,
            'v': snapshot.v,
            'p': snapshot.p,
            'divergence': trace.reconstructed_fields.get('divergence'),
            'mask': snapshot.mask
        }
        self.subdomain_viewer.set_data(subdomain_data)
        
    def on_step_changed(self, step: int):
        """Handle step change from navigation panel."""
        self.load_snapshot(step)
        
    def reset_layout(self):
        """Reset the layout to default proportions."""
        # This is a placeholder - could implement actual layout reset logic
        QMessageBox.information(self, "Reset Layout", "Layout reset functionality coming soon.")
        
    def closeEvent(self, event):
        """Handle window close event."""
        event.accept()


def main():
    """Main entry point for the trace viewer."""
    app = None
    for app_instance in QApplication.instances():
        if isinstance(app_instance, QApplication):
            app = app_instance
            break
    
    if app is None:
        app = QApplication(sys.argv)
    
    window = TraceViewerWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
