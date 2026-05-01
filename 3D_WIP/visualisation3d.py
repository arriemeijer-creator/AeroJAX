"""
3D visualization using vedo.
"""

import numpy as np
import vedo
from solver3d import Solver3D


class Visualizer3D:
    """3D flow visualization using vedo"""
    
    def __init__(self, solver):
        """
        Initialize visualizer.
        
        Args:
            solver: Solver3D instance
        """
        self.solver = solver
        self.plotter = vedo.Plotter()
        
        # Create grid for visualization
        x = np.arange(solver.nx) * solver.dx
        y = np.arange(solver.ny) * solver.dy
        z = np.arange(solver.nz) * solver.dz
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize visualization objects
        self.velocity_arrows = None
        self.pressure_volume = None
        self.vorticity_volume = None
        
        # Cache for converted NumPy arrays (avoid repeated JAX→NumPy conversion)
        self._cached_arrays = None
        self._cache_stride = None
        
    def _convert_arrays(self, stride):
        """Batch convert JAX arrays to NumPy and cache them"""
        if self._cached_arrays is None or self._cache_stride != stride:
            # Batch convert all fields at once
            u = np.array(self.solver.u)
            v = np.array(self.solver.v)
            w = np.array(self.solver.w)
            p = np.array(self.solver.p)
            vel_mag = np.array(self.solver.get_velocity_magnitude())
            vort = np.array(self.solver.get_vorticity())
            
            self._cached_arrays = {
                'u': u, 'v': v, 'w': w, 'p': p,
                'vel_mag': vel_mag, 'vort': vort
            }
            self._cache_stride = stride
        
        return self._cached_arrays
    
    def update_velocity_arrows(self, stride=12, scale=0.1):
        """Update velocity field arrows"""
        arrays = self._convert_arrays(stride)
        
        # Downsample for cleaner visualization
        u = arrays['u'][::stride, ::stride, ::stride]
        v = arrays['v'][::stride, ::stride, ::stride]
        w = arrays['w'][::stride, ::stride, ::stride]
        
        x = self.X[::stride, ::stride, ::stride]
        y = self.Y[::stride, ::stride, ::stride]
        z = self.Z[::stride, ::stride, ::stride]
        
        # Scale arrows for better visibility
        u = u * scale
        v = v * scale
        w = w * scale
        
        # Create arrows with thinner lines
        arrows = vedo.Arrows(
            np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1),
            np.stack([u.flatten(), v.flatten(), w.flatten()], axis=1),
            c='blue',
            res=8  # Lower resolution for thinner arrows
        )
        
        if self.velocity_arrows is not None:
            self.plotter.remove(self.velocity_arrows)
        
        self.velocity_arrows = arrows
        self.plotter.add(arrows)
    
    def update_pressure_volume(self):
        """Update pressure field volume rendering"""
        arrays = self._convert_arrays(stride=1)  # No downsampling for volume
        p = arrays['p']
        
        # Normalize pressure for visualization
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-10)
        
        # Create volume
        vol = vedo.Volume(p_norm)
        
        if self.pressure_volume is not None:
            self.plotter.remove(self.pressure_volume)
        
        self.pressure_volume = vol
        self.plotter.add(vol, mode='volume')
    
    def update_vorticity_volume(self):
        """Update vorticity field volume rendering"""
        arrays = self._convert_arrays(stride=1)  # No downsampling for volume
        vort = arrays['vort']
        
        # Normalize vorticity for visualization
        vort_norm = (vort - vort.min()) / (vort.max() - vort.min() + 1e-10)
        
        # Create volume
        vol = vedo.Volume(vort_norm, c='hot')
        
        if self.vorticity_volume is not None:
            self.plotter.remove(self.vorticity_volume)
        
        self.vorticity_volume = vol
        self.plotter.add(vol, mode='volume')
    
    def update_isosurfaces(self, n_surfaces=3):
        """Update isosurfaces of velocity magnitude"""
        arrays = self._convert_arrays(stride=1)  # No downsampling for isosurfaces
        vel_mag = arrays['vel_mag']
        
        # Remove old isosurfaces
        for obj in self.plotter.objects:
            if isinstance(obj, vedo.Isosurface):
                self.plotter.remove(obj)
        
        # Create new isosurfaces at different levels
        levels = np.linspace(vel_mag.min(), vel_mag.max(), n_surfaces + 2)[1:-1]
        for i, level in enumerate(levels):
            iso = vedo.Isosurface(vel_mag, level, c=vedo.color_map(i, n_surfaces))
            self.plotter.add(iso)
    
    def show(self, mode='arrows'):
        """
        Show visualization.
        
        Args:
            mode: Visualization mode ('arrows', 'pressure', 'vorticity', 'isosurface')
        """
        if mode == 'arrows':
            self.update_velocity_arrows()
        elif mode == 'pressure':
            self.update_pressure_volume()
        elif mode == 'vorticity':
            self.update_vorticity_volume()
        elif mode == 'isosurface':
            self.update_isosurfaces()
        
        self.plotter.show()


def animate_flow(solver, n_steps=1000, viz_interval=25, stride=12, scale=0.1):
    """
    Animate flow evolution with decoupled sim and viz.
    
    Args:
        solver: Solver3D instance
        n_steps: Number of time steps
        viz_interval: Update visualization every N steps
        stride: Downsampling stride for arrows
        scale: Arrow scaling factor
    """
    viz = Visualizer3D(solver)
    
    # Initial visualization
    viz.update_velocity_arrows(stride=stride, scale=scale)
    viz.plotter.show(interactive=False)
    
    for step in range(n_steps):
        # Step solver (decoupled from visualization)
        solver.step()
        
        # Update visualization periodically
        if step % viz_interval == 0:
            # Clear cache to force fresh conversion
            viz._cached_arrays = None
            viz.update_velocity_arrows(stride=stride, scale=scale)
            viz.plotter.render()
            print(f"Step {step}/{n_steps}")
    
    # Keep window open
    viz.plotter.show(interactive=True)


if __name__ == "__main__":
    # Example usage
    nx, ny, nz = 32, 32, 32
    dx, dy, dz = 0.1, 0.1, 0.1
    nu = 0.01
    dt = 0.001
    
    solver = Solver3D(nx, ny, nz, dx, dy, dz, nu, dt)
    
    # Initialize with some flow (add some perturbation)
    solver.u = solver.u.at[:, :, :].set(1.0)
    solver.v = solver.v.at[:, :, :].set(0.1)
    solver.w = solver.w.at[:, :, :].set(0.05)
    
    # Run animated simulation with decoupled sim and viz
    print("Starting animated simulation...")
    animate_flow(solver, n_steps=1000, viz_interval=25, stride=12, scale=0.1)
