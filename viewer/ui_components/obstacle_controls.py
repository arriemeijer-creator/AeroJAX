"""
Obstacle control widgets for the CFD viewer.
Handles obstacle type selection, NACA airfoil parameters, and cylinder parameters.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QRadioButton, QButtonGroup, QComboBox, QDoubleSpinBox,
                             QSlider, QPushButton, QSizePolicy, QGridLayout)
from PyQt6.QtCore import Qt
from viewer.state import store, set_obstacle_position, set_obstacle_type
from .collapsible_groupbox import CollapsibleGroupBox


class ObstacleControls(CollapsibleGroupBox):
    """Group for obstacle selection (cylinder / NACA airfoil / cow) and parameters"""

    def __init__(self, parent=None):
        super().__init__("Obstacle Configuration")
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup obstacle configuration UI"""
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Row 1: Obstacle type with radio buttons and images
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))

        self.obstacle_button_group = QButtonGroup()

        # Cylinder option
        cylinder_layout = QVBoxLayout()
        cylinder_radio = QRadioButton("Cylinder")
        cylinder_radio.setChecked(False)
        cylinder_layout.addWidget(cylinder_radio)
        cylinder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.obstacle_button_group.addButton(cylinder_radio, 0)

        # NACA airfoil option
        naca_layout = QVBoxLayout()
        naca_radio = QRadioButton("NACA Airfoil")
        naca_radio.setChecked(True)
        naca_layout.addWidget(naca_radio)
        naca_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.obstacle_button_group.addButton(naca_radio, 1)

        # Cow option
        cow_layout = QVBoxLayout()
        cow_radio = QRadioButton("Cow")
        cow_layout.addWidget(cow_radio)
        cow_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.obstacle_button_group.addButton(cow_radio, 2)

        # Three-cylinder array option
        cylinder_array_layout = QVBoxLayout()
        cylinder_array_radio = QRadioButton("3 Cylinders")
        cylinder_array_layout.addWidget(cylinder_array_radio)
        cylinder_array_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.obstacle_button_group.addButton(cylinder_array_radio, 3)

        # Store radio buttons for later access
        self.cylinder_radio = cylinder_radio
        self.naca_radio = naca_radio
        self.cow_radio = cow_radio
        self.cylinder_array_radio = cylinder_array_radio

        # Connect button group to obstacle type selection
        self.obstacle_button_group.buttonClicked.connect(self._on_obstacle_radio_changed)

        # Add to row
        row1.addLayout(cylinder_layout)
        row1.addSpacing(10)
        row1.addLayout(naca_layout)
        row1.addSpacing(10)
        row1.addLayout(cow_layout)
        row1.addSpacing(10)
        row1.addLayout(cylinder_array_layout)
        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: NACA controls (initially visible)
        self.naca_widget = QWidget()
        naca_layout = QVBoxLayout(self.naca_widget)
        naca_layout.setContentsMargins(0, 0, 0, 0)
        naca_layout.setSpacing(5)

        # Row 0: NACA combo
        naca_row = QHBoxLayout()
        naca_row.addWidget(QLabel("NACA:"))
        self.naca_combo = QComboBox()
        if self._check_naca_availability():
            from obstacles.naca_airfoils import NACA_AIRFOILS
            self.naca_combo.addItems(list(NACA_AIRFOILS.keys()))
            self.naca_combo.setCurrentText("NACA 0012")
        else:
            self.naca_combo.addItems(["NACA 0012"])
        self.naca_combo.setMaximumWidth(150)
        self.naca_combo.setMouseTracking(True)
        self.naca_combo.currentIndexChanged.connect(self._on_naca_hover)
        naca_row.addWidget(self.naca_combo)
        naca_row.addStretch()
        naca_layout.addLayout(naca_row)

        # Row 1: Chord
        chord_row = QHBoxLayout()
        chord_row.addWidget(QLabel("Chord:"))
        self.chord_spinbox = QDoubleSpinBox()
        self.chord_spinbox.setRange(0.1, 5.0)
        self.chord_spinbox.setValue(3.0)
        self.chord_spinbox.setDecimals(2)
        self.chord_spinbox.setSingleStep(0.1)
        self.chord_spinbox.setMaximumWidth(120)
        chord_row.addWidget(self.chord_spinbox)
        chord_row.addStretch()
        naca_layout.addLayout(chord_row)

        # Row 2: AoA with slider and spinbox
        aoa_row = QHBoxLayout()
        aoa_row.addWidget(QLabel("AoA:"))
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(-20.0, 20.0)
        self.angle_spinbox.setValue(-10.0)
        self.angle_spinbox.setDecimals(1)
        self.angle_spinbox.setSingleStep(1.0)
        self.angle_spinbox.setMaximumWidth(80)
        aoa_row.addWidget(self.angle_spinbox)
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(-200, 200)
        self.angle_slider.setValue(0)
        self.angle_slider.setMaximumWidth(120)
        aoa_row.addWidget(self.angle_slider)
        aoa_row.addStretch()
        naca_layout.addLayout(aoa_row)

        # Row 3: Apply button
        apply_row = QHBoxLayout()
        self.apply_naca_btn = QPushButton("Apply")
        self.apply_naca_btn.setMaximumWidth(60)
        apply_row.addWidget(self.apply_naca_btn)
        apply_row.addStretch()
        naca_layout.addLayout(apply_row)

        self.naca_widget.setVisible(True)
        layout.addWidget(self.naca_widget)

        # Row 3: Cylinder controls (initially hidden)
        self.cylinder_widget = QWidget()
        cylinder_layout = QHBoxLayout(self.cylinder_widget)
        cylinder_layout.setContentsMargins(0, 0, 0, 0)
        cylinder_layout.setSpacing(8)

        cylinder_layout.addWidget(QLabel("Radius:"))
        self.cylinder_radius_spinbox = QDoubleSpinBox()
        self.cylinder_radius_spinbox.setRange(0.05, 2.0)
        self.cylinder_radius_spinbox.setValue(0.18)
        self.cylinder_radius_spinbox.setDecimals(3)
        self.cylinder_radius_spinbox.setSingleStep(0.01)
        self.cylinder_radius_spinbox.setMaximumWidth(120)
        cylinder_layout.addWidget(self.cylinder_radius_spinbox)

        self.apply_cylinder_btn = QPushButton("Apply")
        self.apply_cylinder_btn.setMaximumWidth(60)
        cylinder_layout.addWidget(self.apply_cylinder_btn)

        cylinder_layout.addStretch()
        self.cylinder_widget.setVisible(False)
        layout.addWidget(self.cylinder_widget)

        # Row 4: Cylinder array controls (initially hidden)
        self.cylinder_array_widget = QWidget()
        cylinder_array_layout = QHBoxLayout(self.cylinder_array_widget)
        cylinder_array_layout.setContentsMargins(0, 0, 0, 0)
        cylinder_array_layout.setSpacing(8)

        cylinder_array_layout.addWidget(QLabel("Diameter:"))
        self.cylinder_diameter_spinbox = QDoubleSpinBox()
        self.cylinder_diameter_spinbox.setRange(0.1, 2.0)
        self.cylinder_diameter_spinbox.setValue(0.5)
        self.cylinder_diameter_spinbox.setDecimals(2)
        self.cylinder_diameter_spinbox.setSingleStep(0.1)
        self.cylinder_diameter_spinbox.setMaximumWidth(120)
        cylinder_array_layout.addWidget(self.cylinder_diameter_spinbox)

        cylinder_array_layout.addWidget(QLabel("Spacing:"))
        self.cylinder_spacing_spinbox = QDoubleSpinBox()
        self.cylinder_spacing_spinbox.setRange(0.1, 5.0)
        self.cylinder_spacing_spinbox.setValue(0.5)
        self.cylinder_spacing_spinbox.setDecimals(2)
        self.cylinder_spacing_spinbox.setSingleStep(0.1)
        self.cylinder_spacing_spinbox.setMaximumWidth(120)
        cylinder_array_layout.addWidget(self.cylinder_spacing_spinbox)

        self.apply_cylinder_array_btn = QPushButton("Apply")
        self.apply_cylinder_array_btn.setMaximumWidth(60)
        cylinder_array_layout.addWidget(self.apply_cylinder_array_btn)

        cylinder_array_layout.addStretch()
        self.cylinder_array_widget.setVisible(False)
        layout.addWidget(self.cylinder_array_widget)

        # Row 5: X-position slider (always visible)
        x_pos_layout = QHBoxLayout()
        x_pos_layout.addWidget(QLabel("X-Position:"))
        self.x_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_position_slider.setRange(10, 90)
        self.x_position_slider.setValue(25)
        self.x_position_slider.setMaximumWidth(200)
        x_pos_layout.addWidget(self.x_position_slider)
        self.x_position_label = QLabel("25%")
        self.x_position_label.setMinimumWidth(40)
        x_pos_layout.addWidget(self.x_position_label)
        x_pos_layout.addStretch()
        layout.addLayout(x_pos_layout)

        # Row 6: Y-position slider (always visible)
        y_pos_layout = QHBoxLayout()
        y_pos_layout.addWidget(QLabel("Y-Position:"))
        self.y_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_position_slider.setRange(10, 90)
        self.y_position_slider.setValue(50)
        self.y_position_slider.setMaximumWidth(200)
        y_pos_layout.addWidget(self.y_position_slider)
        self.y_position_label = QLabel("50%")
        self.y_position_label.setMinimumWidth(40)
        y_pos_layout.addWidget(self.y_position_label)
        y_pos_layout.addStretch()
        layout.addLayout(y_pos_layout)

        # Row 7: Custom obstacle drawing button
        draw_button_layout = QHBoxLayout()
        self.draw_custom_btn = QPushButton("Draw Custom Obstacle")
        self.draw_custom_btn.setMaximumWidth(200)
        draw_button_layout.addWidget(self.draw_custom_btn)
        draw_button_layout.addStretch()
        layout.addLayout(draw_button_layout)

        # Connect slider signals
        self.x_position_slider.valueChanged.connect(self._on_x_position_changed)
        self.y_position_slider.valueChanged.connect(self._on_y_position_changed)

        self.setLayout(layout)

        # If NACA module missing, disable controls
        if not self._check_naca_availability():
            self.naca_combo = None
            self.chord_spinbox = None
            self.angle_spinbox = None
            self.angle_slider = None
            self.apply_naca_btn = None

    def _on_obstacle_radio_changed(self, button):
        """Handle obstacle type radio button selection changes."""
        if button == self.cylinder_radio:
            obstacle_type = 'cylinder'
        elif button == self.naca_radio:
            obstacle_type = 'naca_airfoil'
        elif button == self.cow_radio:
            obstacle_type = 'cow'
        elif button == self.cylinder_array_radio:
            obstacle_type = 'three_cylinder_array'
        else:
            return

        # Dispatch Redux action to update store state
        store.dispatch(set_obstacle_type(obstacle_type))

        # Update UI controls visibility
        if hasattr(self, 'naca_widget'):
            self.naca_widget.setVisible(obstacle_type == 'naca_airfoil')
        if hasattr(self, 'cylinder_widget'):
            self.cylinder_widget.setVisible(obstacle_type == 'cylinder')
        if hasattr(self, 'cylinder_array_widget'):
            self.cylinder_array_widget.setVisible(obstacle_type == 'three_cylinder_array')

        # Notify parent viewer if available (backward compatibility)
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_obstacle_type_selected'):
                self.parent_viewer.on_obstacle_type_selected(obstacle_type)

    def _on_x_position_changed(self, value):
        """Handle x-position slider changes."""
        self.x_position_label.setText(f"{value}%")
        
        # Dispatch Redux action for live preview update
        # Access solver through parent_viewer.parent_viewer (ControlPanel -> Main Viewer)
        viewer = None
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'parent_viewer'):
                viewer = self.parent_viewer.parent_viewer
            elif hasattr(self.parent_viewer, 'solver'):
                viewer = self.parent_viewer
        
        if viewer is not None and hasattr(viewer, 'solver'):
            grid_lx = viewer.solver.grid.lx
            x_position = (value / 100.0) * grid_lx
            obstacle_type = getattr(viewer.solver.sim_params, 'obstacle_type', 'cylinder')
            
            # Get current y position
            if hasattr(self, 'y_position_slider'):
                y_value = self.y_position_slider.value()
                grid_ly = viewer.solver.grid.ly
                y_position = (y_value / 100.0) * grid_ly
            else:
                y_position = None
            
            print(f"[SLIDER] Dispatching action: x={x_position:.2f}, obstacle_type={obstacle_type}")
            # Dispatch Redux action
            store.dispatch(set_obstacle_position(obstacle_type, x_position, y_position))
        
        # Notify parent viewer (backward compatibility)
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_x_position_change'):
                self.parent_viewer.apply_x_position_change(value)

    def _on_y_position_changed(self, value):
        """Handle y-position slider changes."""
        self.y_position_label.setText(f"{value}%")
        
        # Dispatch Redux action for live preview update
        # Access solver through parent_viewer.parent_viewer (ControlPanel -> Main Viewer)
        viewer = None
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'parent_viewer'):
                viewer = self.parent_viewer.parent_viewer
            elif hasattr(self.parent_viewer, 'solver'):
                viewer = self.parent_viewer
        
        if viewer is not None and hasattr(viewer, 'solver'):
            grid_ly = viewer.solver.grid.ly
            y_position = (value / 100.0) * grid_ly
            obstacle_type = getattr(viewer.solver.sim_params, 'obstacle_type', 'cylinder')
            
            # Get current x position
            if hasattr(self, 'x_position_slider'):
                x_value = self.x_position_slider.value()
                grid_lx = viewer.solver.grid.lx
                x_position = (x_value / 100.0) * grid_lx
            else:
                x_position = None
            
            print(f"[SLIDER] Dispatching action: y={y_position:.2f}, obstacle_type={obstacle_type}")
            # Dispatch Redux action
            store.dispatch(set_obstacle_position(obstacle_type, x_position, y_position))
        
        # Notify parent viewer (backward compatibility)
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_y_position_change'):
                self.parent_viewer.apply_y_position_change(value)

    def _on_naca_hover(self, index):
        """Show airfoil preview when selection changes"""
        if not self._check_naca_availability():
            return

        designation = self.naca_combo.currentText()
        if not designation or designation == "NACA 0012":
            return

        try:
            from obstacles.naca_airfoils import NACA_AIRFOILS, parse_naca_4digit, parse_naca_5digit
            if designation not in NACA_AIRFOILS:
                return

            digits = ''.join(filter(str.isdigit, designation))
            tooltip_text = f"<b>{designation}</b><br><br>"

            if len(digits) == 4:
                m, p, t = parse_naca_4digit(designation)
                tooltip_text += f"Type: 4-digit series<br>"
                tooltip_text += f"Max camber: {m*100:.1f}%<br>"
                tooltip_text += f"Camber position: {p*100:.0f}% chord<br>"
                tooltip_text += f"Max thickness: {t*100:.1f}% chord"
            elif len(digits) == 5:
                cl, p, m, t = parse_naca_5digit(designation)
                tooltip_text += f"Type: 5-digit series<br>"
                tooltip_text += f"Design lift coeff: {cl:.2f}<br>"
                tooltip_text += f"Camber position: {p*100:.0f}% chord<br>"
                tooltip_text += f"Max camber: {m*10:.1f}%<br>"
                tooltip_text += f"Max thickness: {t*100:.1f}% chord"

            self.naca_combo.setToolTip(tooltip_text)
        except Exception as e:
            pass

    def _check_naca_availability(self):
        """Check if NACA airfoils are available"""
        try:
            from obstacles.naca_airfoils import NACA_AIRFOILS
            return True
        except ImportError:
            return False

    def set_chord_range_for_domain(self, max_chord: float):
        """Update chord spinbox range based on domain size"""
        if hasattr(self, 'chord_spinbox') and self.chord_spinbox is not None:
            self.chord_spinbox.setRange(0.1, max_chord)

    def show_naca_controls(self, show: bool) -> None:
        """Show/hide NACA controls based on obstacle selection"""
        if hasattr(self, 'naca_widget'):
            self.naca_widget.setVisible(show)
