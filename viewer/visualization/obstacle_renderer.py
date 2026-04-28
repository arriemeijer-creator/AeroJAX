"""
Obstacle renderer for visualization
"""

import numpy as np
import jax.numpy as jnp
import pyqtgraph as pg
from PyQt6 import sip

class ObstacleRenderer:
    """Handles rendering of obstacles (cylinder, NACA airfoils)"""

    def __init__(self, vel_outline, div_outline, vort_outline, scalar_outline, pressure_outline):
        self.vel_outline = vel_outline
        self.div_outline = div_outline  # May be None
        self.vort_outline = vort_outline
        self.scalar_outline = scalar_outline
        self.pressure_outline = pressure_outline
        self.naca_available = self._check_naca_availability()
    
    def _check_naca_availability(self):
        """Check if NACA airfoils are available"""
        try:
            from obstacles.naca_airfoils import NACA_AIRFOILS
            return True
        except ImportError:
            return False
    
    def update_obstacle_outlines(self, solver, force_update=False):
        """Update obstacle outlines based on current solver geometry"""
        # Rate limit updates to prevent error cascades (unless force_update is True)
        if not force_update:
            if not hasattr(self, '_last_naca_update_time'):
                self._last_naca_update_time = 0
                self._naca_error_count = 0
                self._last_naca_error_designation = None
            
            import time
            current_time = time.time()
            if current_time - self._last_naca_update_time < 0.05:  # Reduced to 50ms for smoother slider updates
                return
            
            self._last_naca_update_time = current_time
        
        if not hasattr(solver, 'sim_params') or solver is None:
            return
        
        try:
            # Check flow type and obstacle type
            if solver.sim_params.flow_type != 'von_karman':
                # Clear outlines for non-von_karman flows
                from PyQt6.QtGui import QPolygonF
                from PyQt6.QtCore import QPointF
                empty_polygon = QPolygonF()
                if (self.vel_outline is not None and
                    hasattr(self.vel_outline, 'setPolygon') and
                    not sip.isdeleted(self.vel_outline)):
                    self.vel_outline.setPolygon(empty_polygon)
                if (self.div_outline is not None and
                    hasattr(self.div_outline, 'setPolygon') and
                    not sip.isdeleted(self.div_outline)):
                    self.div_outline.setPolygon(empty_polygon)
                if (self.vort_outline is not None and
                    hasattr(self.vort_outline, 'setPolygon') and
                    not sip.isdeleted(self.vort_outline)):
                    self.vort_outline.setPolygon(empty_polygon)
                if (self.scalar_outline is not None and
                    hasattr(self.scalar_outline, 'setPolygon') and
                    not sip.isdeleted(self.scalar_outline)):
                    self.scalar_outline.setPolygon(empty_polygon)
                if (self.pressure_outline is not None and
                    hasattr(self.pressure_outline, 'setPolygon') and
                    not sip.isdeleted(self.pressure_outline)):
                    self.pressure_outline.setPolygon(empty_polygon)
                return
            
            # Get obstacle parameters
            obstacle_type = getattr(solver.sim_params, 'obstacle_type', 'cylinder')
            
            if obstacle_type == 'naca_airfoil':
                # Draw NACA airfoil outline
                self._draw_naca_outline(solver)
            elif obstacle_type == 'cow':
                # Draw cow outline
                self._draw_cow_outline(solver)
            elif obstacle_type == 'three_cylinder_array':
                # Draw three-cylinder array outline
                self._draw_cylinder_array_outline(solver)
            elif obstacle_type == 'custom':
                # Draw custom obstacle outline
                self._draw_custom_outline(solver)
            else:
                # Draw cylinder outline
                self._draw_cylinder_outline(solver)
        except Exception as e:
            print(f"Error in update_obstacle_outlines: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_cylinder_outline(self, solver):
        """Draw cylinder outline"""
        try:
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            
            # Get cylinder parameters
            center_x = float(solver.geom.center_x.item()) if hasattr(solver.geom.center_x, 'item') else float(solver.geom.center_x)
            center_y = float(solver.geom.center_y.item()) if hasattr(solver.geom.center_y, 'item') else float(solver.geom.center_y)
            radius = float(solver.geom.radius.item()) if hasattr(solver.geom.radius, 'item') else float(solver.geom.radius)
            
            # Create circle points
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = center_x + radius * np.cos(theta)
            y_circle = center_y + radius * np.sin(theta)
            
            # Create QPolygonF from points
            polygon_points = [QPointF(x, y) for x, y in zip(x_circle, y_circle)]
            polygon = QPolygonF(polygon_points)
            
            # Check if outline items exist and are properly connected to plots
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPolygon') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPolygon(polygon)
                self.vel_outline.setVisible(True)
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPolygon') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPolygon(polygon)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPolygon') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPolygon(polygon)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPolygon') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPolygon(polygon)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPolygon') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPolygon(polygon)

        except Exception as e:
            print(f"Error drawing cylinder outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_cylinder_array_outline(self, solver):
        """Draw three-cylinder array outline"""
        try:
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            
            # Get cylinder array parameters from sim_params
            cylinder_x = getattr(solver.sim_params, 'cylinder_x', 5.0)
            cylinder_y = getattr(solver.sim_params, 'cylinder_y', solver.grid.ly / 2.0)
            cylinder_diameter = getattr(solver.sim_params, 'cylinder_diameter', 0.5)
            cylinder_spacing = getattr(solver.sim_params, 'cylinder_spacing', 0.5)
            radius = cylinder_diameter / 2.0
            spacing = cylinder_spacing  # Use dynamic spacing from sim_params
            
            # Create circle points for each of the 3 cylinders
            theta = np.linspace(0, 2*np.pi, 100)
            
            # Combine all 3 circles into one polygon
            all_points = []
            for i in range(3):
                center_x_i = cylinder_x + i * spacing
                x_circle = center_x_i + radius * np.cos(theta)
                y_circle = cylinder_y + radius * np.sin(theta)
                all_points.extend([QPointF(x, y) for x, y in zip(x_circle, y_circle)])
            
            # Create QPolygonF from points
            polygon = QPolygonF(all_points)
            
            # Check if outline items exist and are properly connected to plots
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPolygon') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPolygon(polygon)
                self.vel_outline.setVisible(True)
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPolygon') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPolygon(polygon)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPolygon') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPolygon(polygon)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPolygon') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPolygon(polygon)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPolygon') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPolygon(polygon)

        except Exception as e:
            print(f"Error drawing cylinder array outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_cow_outline(self, solver):
        """Draw cow outline using cow.py SDF with matplotlib contour extraction"""
        try:
            from PyQt6.QtGui import QPolygonF, QPen, QBrush, QColor
            from PyQt6.QtCore import QPointF, Qt
            from obstacles.cow import sdf_cow_side
            import jax
            import numpy as np
            from matplotlib import pyplot as plt
            
            # Compute cow position relative to grid bounds
            # Cow should be at 25% of domain width and grounded at bottom
            cow_x = solver.grid.lx * 0.25  # 25% of domain width
            cow_y = solver.grid.ly * 0.35  # 35% of domain height (grounded)
            
            # Compute scale factor based on grid dimensions relative to reference (20x3.75)
            ref_lx = 20.0
            ref_ly = 3.75
            scale_x = solver.grid.lx / ref_lx
            scale_y = solver.grid.ly / ref_ly
            cow_scale = (scale_x + scale_y) / 2.0  # Average of x and y scaling
            
            # Create a fine grid to sample the SDF
            nx_fine = 400
            ny_fine = 100
            lx_fine = solver.grid.lx
            ly_fine = solver.grid.ly
            
            x_fine = jnp.linspace(0, lx_fine, nx_fine)
            y_fine = jnp.linspace(0, ly_fine, ny_fine)
            X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine, indexing='ij')
            
            # Compute SDF on fine grid with cow_x, cow_y, and scale parameters
            sdf = sdf_cow_side(X_fine, Y_fine, cow_x, cow_y, cow_scale)
            sdf_np = np.array(sdf)
            
            # Transpose to match matplotlib's expectation (ny, nx)
            sdf_np = sdf_np.T
            
            # Extract contour at SDF = 0 using matplotlib
            contours = plt.contour(x_fine, y_fine, sdf_np, levels=[0])
            
            # Collect all contour points using allsegs
            all_points = []
            for seg in contours.allsegs[0]:  # allsegs[0] corresponds to level 0
                for x, y in zip(seg[:, 0], seg[:, 1]):
                    all_points.append(QPointF(float(x), float(y)))
            
            plt.close()  # Close the matplotlib figure
            
            # Create polygon from contour points
            polygon = QPolygonF(all_points)
            
            # Draw using existing outline items (same pattern as cylinder)
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPolygon') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPolygon(polygon)
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPolygon') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPolygon(polygon)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPolygon') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPolygon(polygon)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPolygon') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPolygon(polygon)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPolygon') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPolygon(polygon)

        except Exception as e:
            print(f"Error drawing cow outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_naca_outline(self, solver):
        """Draw NACA airfoil outline"""
        if not self.naca_available:
            return
        
        try:
            sim = solver.sim_params
            
            # Get NACA parameters
            designation = sim.naca_airfoil
            chord = sim.naca_chord
            angle = sim.naca_angle
            pos_x = sim.naca_x
            pos_y = sim.naca_y
            
            # Extract digits from designation
            digits = ''.join(filter(str.isdigit, designation))
            
            if len(digits) == 4:
                # 4-digit airfoil
                from obstacles.naca_airfoils import generate_naca_4digit, parse_naca_4digit
                
                try:
                    m, p, t = parse_naca_4digit(designation)
                    # Validate parameters to prevent NaN
                    if p < 0 or p >= 1:
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid camber position p={p} for 4-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    if t <= 0 or t > 0.5:
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid thickness t={t} for 4-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    
                    # Special handling for symmetric airfoils (p=0)
                    if p == 0.0:
                        # For symmetric airfoils, upper and lower surfaces are just thickness distribution
                        x_norm = np.linspace(0, 1, 100)
                        
                        # Thickness distribution (same as in naca_airfoils.py)
                        yt = 5 * t * (0.2969 * np.sqrt(np.abs(x_norm)) - 0.1260 * x_norm - 
                                      0.3516 * x_norm**2 + 0.2843 * x_norm**3 - 0.1015 * x_norm**4)
                        
                        # For symmetric airfoil: yc = 0, theta = 0
                        xu = x_norm
                        yu = yt  # Upper surface
                        xl = x_norm  
                        yl = -yt  # Lower surface (negative)
                    else:
                        # Use regular NACA generation for cambered airfoils
                        x_norm = np.linspace(0, 1, 100)
                        xu, yu, xl, yl = generate_naca_4digit(jnp.array(x_norm), m, p, t)
                        
                        # Convert to numpy
                        xu, yu = np.array(xu), np.array(yu)
                        xl, yl = np.array(xl), np.array(yl)
                except Exception as e:
                    print(f"NACA Error: Failed to generate 4-digit airfoil: {e}")
                    return
                
            elif len(digits) == 5:
                # 5-digit airfoil
                from obstacles.naca_airfoils import generate_naca_5digit, parse_naca_5digit
                
                try:
                    cl, p, m, t = parse_naca_5digit(designation)
                    # Validate parameters to prevent NaN
                    # p=0 is valid for reflexed 5-digit airfoils (e.g., NACA 23012)
                    if p < 0 or p > 1:
                        # Only print error once per designation
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid camber position p={p} for 5-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    x_norm = np.linspace(0, 1, 100)
                    xu, yu, xl, yl = generate_naca_5digit(jnp.array(x_norm), cl, p, m, t)
                    
                    # Convert to numpy
                    xu, yu = np.array(xu), np.array(yu)
                    xl, yl = np.array(xl), np.array(yl)
                except Exception as e:
                    print(f"NACA Error: Failed to generate 5-digit airfoil: {e}")
                    return
            
            # Check for NaN values
            if np.any(np.isnan(xu)) or np.any(np.isnan(yu)) or np.any(np.isnan(xl)) or np.any(np.isnan(yl)):
                # Only print error once per designation
                if self._last_naca_error_designation != designation:
                    print(f"NACA Error: NaN values detected in generated coordinates for {designation}")
                    print(f"  Parameters: digits={digits}, len={len(digits)}")
                    if len(digits) == 4:
                        print(f"  4-digit: m={m}, p={p}, t={t}")
                    elif len(digits) == 5:
                        print(f"  5-digit: cl={cl}, p={p}, m={m}, t={t}")
                    self._last_naca_error_designation = designation
                return
            
            # Scale
            xu, yu = xu * chord, yu * chord
            xl, yl = xl * chord, yl * chord
            
            # Rotate (positive angle - flip Y for screen coordinates where Y increases downward)
            angle_rad = np.radians(angle)
            xu_rot = xu * np.cos(angle_rad) + yu * np.sin(angle_rad)  # Flip Y
            yu_rot = -xu * np.sin(angle_rad) + yu * np.cos(angle_rad)  # Flip Y
            xl_rot = xl * np.cos(angle_rad) + yl * np.sin(angle_rad)  # Flip Y
            yl_rot = -xl * np.sin(angle_rad) + yl * np.cos(angle_rad)  # Flip Y
            
            # Translate (position is leading edge)
            xu_final = xu_rot + pos_x
            yu_final = yu_rot + pos_y
            xl_final = xl_rot + pos_x
            yl_final = yl_rot + pos_y
            
            # Combine upper and lower surfaces
            x_outline = np.concatenate([xu_final, xl_final[::-1], [xu_final[0]]])
            y_outline = np.concatenate([yu_final, yl_final[::-1], [yu_final[0]]])
            
            # Final check for NaN values
            if np.any(np.isnan(x_outline)) or np.any(np.isnan(y_outline)):
                print("NACA Error: NaN values in final outline coordinates")
                return
            
            # Check if outline items are still valid before setting data
            try:
                from PyQt6.QtGui import QPolygonF
                from PyQt6.QtCore import QPointF
                
                # Create QPolygonF from points
                polygon_points = [QPointF(x, y) for x, y in zip(x_outline, y_outline)]
                polygon = QPolygonF(polygon_points)
                
                if (self.vel_outline is not None and
                    hasattr(self.vel_outline, 'setPolygon') and
                    not sip.isdeleted(self.vel_outline)):
                    self.vel_outline.setPolygon(polygon)

                if (self.div_outline is not None and
                    hasattr(self.div_outline, 'setPolygon') and
                    not sip.isdeleted(self.div_outline)):
                    self.div_outline.setPolygon(polygon)

                if (self.vort_outline is not None and
                    hasattr(self.vort_outline, 'setPolygon') and
                    not sip.isdeleted(self.vort_outline)):
                    self.vort_outline.setPolygon(polygon)

                if (self.scalar_outline is not None and
                    hasattr(self.scalar_outline, 'setPolygon') and
                    not sip.isdeleted(self.scalar_outline)):
                    self.scalar_outline.setPolygon(polygon)

                if (self.pressure_outline is not None and
                    hasattr(self.pressure_outline, 'setPolygon') and
                    not sip.isdeleted(self.pressure_outline)):
                    self.pressure_outline.setPolygon(polygon)
            except RuntimeError as e:
                if "has been deleted" in str(e):
                    print("Warning: Outline plot items deleted during cleanup")
                else:
                    raise
            except Exception as e:
                print(f"Error drawing NACA outline: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error drawing NACA outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_custom_outline(self, solver):
        """Draw custom obstacle outline using the custom mask with matplotlib contour extraction"""
        try:
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            from matplotlib import pyplot as plt
            import numpy as np
            import jax.numpy as jnp
            
            # Get the custom mask from sim_params
            custom_mask = getattr(solver.sim_params, 'custom_mask', None)
            if custom_mask is None:
                return
            
            # Get obstacle center position from sliders
            center_x = getattr(solver.sim_params, 'custom_x', solver.grid.lx * 0.25)
            center_y = getattr(solver.sim_params, 'custom_y', solver.grid.ly * 0.5)
            
            # Scale the custom obstacle to fit in the domain while preserving aspect ratio
            # Use the smaller dimension to determine scale, so the drawing fits
            mask_height, mask_width = custom_mask.shape
            
            # Calculate scale to fit in domain (use 60% of the smaller dimension)
            domain_min_dim = min(solver.grid.lx, solver.grid.ly)
            scale = domain_min_dim * 0.6
            
            # Use same scale for both dimensions to preserve aspect ratio
            scale_x = scale
            scale_y = scale
            
            # Calculate offset to center the obstacle at the specified position
            # offset is the center position (matches solver)
            offset_x = center_x
            offset_y = center_y
            
            # Create a fine grid to sample the SDF
            nx_fine = 400
            ny_fine = 100
            lx_fine = solver.grid.lx
            ly_fine = solver.grid.ly
            
            x_fine = jnp.linspace(0, lx_fine, nx_fine)
            y_fine = jnp.linspace(0, ly_fine, ny_fine)
            X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine, indexing='ij')
            
            # Compute the custom SDF/mask on the fine grid
            from obstacles.freeform_drawer import create_freeform_mask_smooth
            # Use a sharp transition (same as in freeform_drawer.py) to prevent visualization artifacts
            mask_fine = create_freeform_mask_smooth(X_fine, Y_fine, custom_mask,
                                                   scale_x=scale_x, scale_y=scale_y,
                                                   offset_x=offset_x, offset_y=offset_y,
                                                   smooth_width=0.001)
            
            # Convert to numpy and transpose for matplotlib
            mask_np = np.array(mask_fine).T
            
            # Extract contour at mask = 0.5 (the boundary between solid and fluid)
            contours = plt.contour(x_fine, y_fine, mask_np, levels=[0.5])
            
            # Collect all contour points
            all_points = []
            for seg in contours.allsegs[0]:  # allsegs[0] corresponds to level 0.5
                for x, y in zip(seg[:, 0], seg[:, 1]):
                    all_points.append(QPointF(float(x), float(y)))
            
            plt.close()  # Close the matplotlib figure
            
            # Create polygon from contour points
            if len(all_points) > 0:
                polygon = QPolygonF(all_points)
                
                # Draw using existing outline items
                if (self.vel_outline is not None and
                    hasattr(self.vel_outline, 'setPolygon') and
                    not sip.isdeleted(self.vel_outline)):
                    self.vel_outline.setPolygon(polygon)
                    self.vel_outline.setVisible(True)

                if (self.div_outline is not None and
                    hasattr(self.div_outline, 'setPolygon') and
                    not sip.isdeleted(self.div_outline)):
                    self.div_outline.setPolygon(polygon)

                if (self.vort_outline is not None and
                    hasattr(self.vort_outline, 'setPolygon') and
                    not sip.isdeleted(self.vort_outline)):
                    self.vort_outline.setPolygon(polygon)

                if (self.scalar_outline is not None and
                    hasattr(self.scalar_outline, 'setPolygon') and
                    not sip.isdeleted(self.scalar_outline)):
                    self.scalar_outline.setPolygon(polygon)

                if (self.pressure_outline is not None and
                    hasattr(self.pressure_outline, 'setPolygon') and
                    not sip.isdeleted(self.pressure_outline)):
                    self.pressure_outline.setPolygon(polygon)
            else:
                # No contour found, hide outlines
                from PyQt6.QtGui import QPolygonF
                empty_polygon = QPolygonF()
                if (self.vel_outline is not None and
                    hasattr(self.vel_outline, 'setPolygon') and
                    not sip.isdeleted(self.vel_outline)):
                    self.vel_outline.setPolygon(empty_polygon)
                if (self.div_outline is not None and
                    hasattr(self.div_outline, 'setPolygon') and
                    not sip.isdeleted(self.div_outline)):
                    self.div_outline.setPolygon(empty_polygon)
                if (self.vort_outline is not None and
                    hasattr(self.vort_outline, 'setPolygon') and
                    not sip.isdeleted(self.vort_outline)):
                    self.vort_outline.setPolygon(empty_polygon)
                if (self.scalar_outline is not None and
                    hasattr(self.scalar_outline, 'setPolygon') and
                    not sip.isdeleted(self.scalar_outline)):
                    self.scalar_outline.setPolygon(empty_polygon)
                if (self.pressure_outline is not None and
                    hasattr(self.pressure_outline, 'setPolygon') and
                    not sip.isdeleted(self.pressure_outline)):
                    self.pressure_outline.setPolygon(empty_polygon)
                
        except Exception as e:
            print(f"Error drawing custom obstacle outline: {e}")
            import traceback
            traceback.print_exc()


