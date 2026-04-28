"""
NACA airfoil handler for the CFD viewer.
Handles NACA airfoil configuration and angle of attack adjustments.
"""

import jax
from typing import Optional


class NACAHandler:
    """Mixin class providing NACA airfoil management methods for the viewer."""
    
    def apply_naca_airfoil_settings(self) -> None:
        """Apply NACA airfoil configuration to the simulation."""
        # Check if NACA controls are available
        if not hasattr(self.control_panel, 'chord_spinbox') or self.control_panel.chord_spinbox is None:
            print("NACA Error: chord_spinbox not available")
            return
        
        self.refresh_timer.stop()
        self.sim_controller.stop_simulation()
        
        try:
            obstacle_type = 'naca_airfoil'  # NACA is always the obstacle type when applying NACA settings
            naca_airfoil = self.control_panel.naca_combo.currentText()
            chord_length = self.control_panel.chord_spinbox.value()
            
            # Use current solver angle instead of spinbox to preserve real-time changes
            current_angle = getattr(self.solver.sim_params, 'naca_angle', 0.0)
            print(f"Apply NACA: Using current solver angle {current_angle:.1f}° instead of spinbox value {self.control_panel.angle_spinbox.value():.1f}°")
            
            # Use current solver position or calculate center
            current_x = getattr(self.solver.sim_params, 'naca_x', self.solver.grid.lx * 0.25)
            current_y = getattr(self.solver.sim_params, 'naca_y', self.solver.grid.ly * 0.5)
            
            self.solver.set_obstacle_type(
                obstacle_type,
                airfoil=naca_airfoil,
                chord=chord_length,
                angle=current_angle,  # Use current solver angle
                x=current_x,
                y=current_y
            )
            
            # Recompile solver with new geometry
            self.solver.mask = self.solver._compute_mask()
            jax.clear_caches()
            self.solver._step_jit = self.solver.get_step_jit()
            
            # Update obstacle outlines immediately
            if hasattr(self, 'obstacle_renderer') and self.obstacle_renderer:
                self.obstacle_renderer.update_obstacle_outlines(self.solver)
            
            print(f"NACA {naca_airfoil} applied: chord={chord_length}, angle={current_angle}°")
            
            # Update UI to match the applied angle
            if hasattr(self.control_panel, 'angle_spinbox') and self.control_panel.angle_spinbox is not None:
                self.control_panel.angle_spinbox.setValue(current_angle)
            if hasattr(self.control_panel, 'angle_slider') and self.control_panel.angle_slider is not None:
                slider_value = int(current_angle * 10.0)
                self.control_panel.angle_slider.setValue(slider_value)
            
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
            
        except Exception as e:
            print(f"Error applying NACA settings: {e}")
            self.control_panel.start_btn.setEnabled(True)
            self.control_panel.pause_btn.setEnabled(False)
    
    def on_angle_slider_changed(self, slider_value: int) -> None:
        """Handle real-time angle slider changes during simulation."""
        if not hasattr(self.control_panel, 'angle_slider') or not self.control_panel.angle_slider:
            return
        
        # Convert slider value (-200 to 200) to angle (-20 to +20 degrees)
        angle_of_attack = slider_value / 10.0
        
        # Update the spinbox to match
        if hasattr(self.control_panel, 'angle_spinbox'):
            self.control_panel.angle_spinbox.blockSignals(True)
            self.control_panel.angle_slider.blockSignals(True)
            
            self.control_panel.angle_spinbox.setValue(angle_of_attack)
            
            self.control_panel.angle_spinbox.blockSignals(False)
            self.control_panel.angle_slider.blockSignals(False)
        
        # Update angle and recompute mask (fast enough for smooth drag)
        # Note: Don't dispatch to Redux store - only update solver directly
        # Store should only be updated when user explicitly clicks "Apply"
        if hasattr(self.solver, 'update_naca_angle'):
            self.solver.update_naca_angle(angle_of_attack, recompute=True)
            
            # Update visualization
            self.obstacle_renderer.update_obstacle_outlines(self.solver)
            self.sdf_viz.update_sdf_visualization(self.solver)
        
        print(f"Real-time angle update: {angle_of_attack:.1f}°")
    
    def on_angle_spinbox_changed(self, spinbox_value: float) -> None:
        """Handle angle spinbox changes for bidirectional synchronization."""
        if not hasattr(self.control_panel, 'angle_slider') or not self.control_panel.angle_slider:
            return
        
        # Convert spinbox value to slider value (-20 to +20 degrees to -200 to 200)
        slider_value = int(spinbox_value * 10.0)
        
        # Update the slider to match, but block signals to prevent feedback loop
        # Block both spinbox and slider signals to prevent any feedback
        self.control_panel.angle_spinbox.blockSignals(True)
        self.control_panel.angle_slider.blockSignals(True)
        
        self.control_panel.angle_slider.setValue(slider_value)
        
        # Unblock signals
        self.control_panel.angle_spinbox.blockSignals(False)
        self.control_panel.angle_slider.blockSignals(False)
        
        # Update angle and recompute mask
        # Note: Don't dispatch to Redux store - only update solver directly
        # Store should only be updated when user explicitly clicks "Apply"
        if hasattr(self.solver, 'update_naca_angle'):
            self.solver.update_naca_angle(spinbox_value, recompute=True)
            
            # Update the outline visualization immediately
            self.obstacle_renderer.update_obstacle_outlines(self.solver)
            self.sdf_viz.update_sdf_visualization(self.solver)
        
        print(f"Spinbox angle update: {spinbox_value:.1f}°")
    
    def on_angle_slider_released(self) -> None:
        """Handle slider release to maintain angle when user stops dragging."""
        if not hasattr(self.control_panel, 'angle_slider') or not self.control_panel.angle_slider:
            return
        
        # Get the current slider value and force cache clear to ensure new angle persists
        slider_value = self.control_panel.angle_slider.value()
        angle_of_attack = slider_value / 10.0
        
        print(f"Slider released at angle: {angle_of_attack:.1f}° - clearing JAX cache to persist angle")
        
        # Force update with cache clear to ensure new angle is properly traced
        if hasattr(self.solver, 'update_naca_angle'):
            self.solver.update_naca_angle(angle_of_attack, recompute=True, clear_cache=True)
            
            # Update visualization
            self.obstacle_renderer.update_obstacle_outlines(self.solver)
            self.sdf_viz.update_sdf_visualization(self.solver)
