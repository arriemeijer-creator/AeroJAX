"""
Right control panel for Error Metrics and Airfoil Metrics.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PyQt6.QtCore import Qt


class RightControlPanel(QWidget):
    """Right control panel containing Error Metrics and Airfoil Metrics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.error_metrics_group = None
        self.airfoil_metrics_group = None
        self.solver_info_group = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the right control panel UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Scrollable area for metrics
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Scrollable content widget
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Metrics groups will be added here via set_metrics_groups
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.scroll_layout = scroll_layout

    def set_metrics_groups(self, error_group, airfoil_group):
        """Set the metrics groups to display"""
        self.error_metrics_group = error_group
        self.airfoil_metrics_group = airfoil_group

        # Remove from parent layout if they have one
        if error_group.parent() is not None:
            error_group.parent().layout().removeWidget(error_group)
        if airfoil_group.parent() is not None:
            airfoil_group.parent().layout().removeWidget(airfoil_group)

        # Insert them before the stretch
        self.scroll_layout.insertWidget(0, error_group)
        self.scroll_layout.insertWidget(1, airfoil_group)

    def set_solver_info_group(self, solver_info_group):
        """Set the solver info group to display"""
        self.solver_info_group = solver_info_group

        # Remove from parent layout if it has one
        if solver_info_group.parent() is not None:
            solver_info_group.parent().layout().removeWidget(solver_info_group)

        # Insert at the top (before metrics groups)
        self.scroll_layout.insertWidget(0, solver_info_group)
