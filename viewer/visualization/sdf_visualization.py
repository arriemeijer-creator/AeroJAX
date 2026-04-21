"""
SDF visualization component
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF

class SDFVisualization:
    """Handles Signed Distance Function visualization"""
    
    def __init__(self, vel_sdf, vort_sdf, parent_vis=None):
        self.vel_sdf = vel_sdf
        self.vort_sdf = vort_sdf
        self.parent_vis = parent_vis  # Reference to FlowVisualization for bounds
        self.is_visible = False  # Track visibility state - hidden by default (using filled polygon instead)
    
    def set_visibility(self, is_visible):
        """Set visibility of SDF overlay"""
        self.is_visible = is_visible
        if self.vel_sdf is not None:
            self.vel_sdf.setVisible(is_visible)
        if self.vort_sdf is not None:
            self.vort_sdf.setVisible(is_visible)
    
    def update_sdf_visualization(self, solver):
        """Update SDF visualization overlay"""
        if not hasattr(solver, 'mask') or solver is None:
            return
    
        try:
            # Check if ImageItem objects are still valid before using them
            if (self.vel_sdf is None or self.vort_sdf is None or
                not hasattr(self.vel_sdf, 'setImage') or not hasattr(self.vort_sdf, 'setImage')):
                return
            
            # Get actual mask from solver
            mask_array = solver.mask
            
            # Convert to numpy if it's a JAX array
            if hasattr(mask_array, 'toArray'):
                mask_array = mask_array.toArray()
            elif not isinstance(mask_array, np.ndarray):
                mask_array = np.array(mask_array)
            
            # Properly threshold the mask - values > 0.5 are solid, <= 0.5 are fluid
            binary_mask = (mask_array > 0.5).astype(np.float64)
            
            # Create visualization array (0 for fluid, 1 for solid)
            sdf_viz = binary_mask.copy()
            
            # Create custom colormap for clear mask visualization
            sdf_lut = np.zeros((256, 4), dtype=np.uint8)
            # Fluid regions: completely transparent
            sdf_lut[:128, :] = [0, 0, 0, 0]  # Fully transparent
            # Solid regions: solid grey with full opacity
            sdf_lut[128:, :] = [128, 128, 128, 255]  # Solid grey
            
            # IMPORTANT: Set bounds rectangle so image aligns properly
            # Get grid dimensions from solver
            if self.parent_vis is not None and hasattr(self.parent_vis, 'current_lx'):
                lx = self.parent_vis.current_lx
                y_min = self.parent_vis.current_y_min
                y_max = self.parent_vis.current_y_max
            else:
                # Fallback to default dimensions
                lx = 20.0
                y_min = 0.0
                y_max = 4.5
                
            rect = QRectF(0, y_min, lx, y_max - y_min)
            
            # Update SDF visualization
            try:
                if self.vel_sdf is not None and hasattr(self.vel_sdf, 'setImage'):
                    self.vel_sdf.setImage(sdf_viz, 
                                         lookupTable=sdf_lut, 
                                         autoLevels=False,
                                         rect=rect)  # CRITICAL: Set bounds
                    self.vel_sdf.setLevels([0, 1])  # Binary mask range
                    self.vel_sdf.setOpacity(1.0)  # FULL opacity for solid appearance
                    self.vel_sdf.setVisible(self.is_visible)
                    
            except Exception as e:
                pass
                
            try:
                if self.vort_sdf is not None and hasattr(self.vort_sdf, 'setImage'):
                    self.vort_sdf.setImage(sdf_viz, 
                                          lookupTable=sdf_lut, 
                                          autoLevels=False,
                                          rect=rect)  # CRITICAL: Set bounds
                    self.vort_sdf.setLevels([0, 1])
                    self.vort_sdf.setOpacity(1.0)  # FULL opacity
                    self.vort_sdf.setVisible(self.is_visible)
                    
            except Exception as e:
                pass
            
        except Exception as e:
            if "has been deleted" not in str(e):
                print(f"Error updating SDF visualization: {e}")
    
    def set_visibility(self, visible=True):
        """Set SDF visualization visibility"""
        self.is_visible = visible  # Track the state
        try:
            if self.vel_sdf is not None and hasattr(self.vel_sdf, 'setVisible'):
                self.vel_sdf.setVisible(visible)
            if self.vort_sdf is not None and hasattr(self.vort_sdf, 'setVisible'):
                self.vort_sdf.setVisible(visible)
        except:
            pass  # Silently handle deletion errors
    
    def clear(self):
        """Clear SDF visualization"""
        try:
            if self.vel_sdf is not None and hasattr(self.vel_sdf, 'clear'):
                self.vel_sdf.clear()
            if self.vort_sdf is not None and hasattr(self.vort_sdf, 'clear'):
                self.vort_sdf.clear()
        except:
            pass  # Silently handle cleanup errors
