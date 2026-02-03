#!/usr/bin/env python3
"""
Visualization module for real-time benzene concentration predictions
Generates heatmaps and tracks NN2 correction statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from matplotlib.colors import LogNorm

# Sensor coordinates (9 sensors used in NN2)
SENSOR_COORDS = np.array([
    [13972.62, 19915.57],  # 482010026
    [3017.18, 12334.2],    # 482010057
    [817.42, 9218.92],     # 482010069
    [8836.35, 15717.2],    # 482010803
    [18413.8, 15068.96],   # 482011015
    [1159.98, 12272.52],   # 482011035
    [13661.93, 5193.24],   # 482011039
    [15077.79, 9450.52],   # 482011614
    [1546.9, 6786.33],     # 482016000
])


class PipelineVisualizer:
    """Generate visualizations and track correction statistics"""
    
    def __init__(self, viz_dir, corrections_dir):
        self.viz_dir = Path(viz_dir)
        self.corrections_dir = Path(corrections_dir)
        self.sensor_bounded_dir = self.viz_dir / "sensor_bounded"
        
        # Create directories
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.corrections_dir.mkdir(parents=True, exist_ok=True)
        self.sensor_bounded_dir.mkdir(parents=True, exist_ok=True)
        
        self.correction_csv = self.corrections_dir / "correction_timeseries.csv"
        
        # Sensor coordinates for bounded area
        self.sensor_coords = np.array([
            [13972.62, 19915.57],  # 482010026
            [3017.18, 12334.2],    # 482010057
            [817.42, 9218.92],     # 482010069
            [8836.35, 15717.2],    # 482010803
            [18413.8, 15068.96],   # 482011015
            [1159.98, 12272.52],   # 482011035
            [13661.93, 5193.24],   # 482011039
            [15077.79, 9450.52],   # 482011614
            [1546.9, 6786.33],     # 482016000
        ])
        
        # Calculate sensor-bounded area (with 10% padding)
        x_min, y_min = self.sensor_coords.min(axis=0)
        x_max, y_max = self.sensor_coords.max(axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = x_range * 0.1
        padding_y = y_range * 0.1
        
        self.bounds = {
            'x_min': x_min - padding_x,
            'x_max': x_max + padding_x,
            'y_min': y_min - padding_y,
            'y_max': y_max + padding_y
        }
    
    def generate_heatmaps(
        self,
        x_grid, y_grid,
        pinn_field, nn2_field,
        forecast_time: datetime,
        grid_resolution: int = 30
    ):
        """
        Generate both PINN and NN2 heatmaps
        
        Args:
            x_grid: 1D array of x coordinates
            y_grid: 1D array of y coordinates  
            pinn_field: 1D array of PINN concentrations (ppb)
            nn2_field: 1D array of NN2 concentrations (ppb)
            forecast_time: Forecast timestamp
            grid_resolution: Grid size (e.g., 30x30)
        """
        # Reshape to 2D grids
        pinn_2d = pinn_field.reshape(grid_resolution, grid_resolution)
        nn2_2d = nn2_field.reshape(grid_resolution, grid_resolution)
        
        # Get unique x and y for meshgrid
        x_unique = np.unique(x_grid)
        y_unique = np.unique(y_grid)
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Format timestamp for filename
        timestamp_str = forecast_time.strftime("%Y-%m-%d_%H%M")
        
        # Generate PINN heatmap - FULL DOMAIN (30km x 30km)
        pinn_path = self.viz_dir / f"pinn_full_domain_{timestamp_str}.png"
        self._create_heatmap(
            X, Y, pinn_2d,
            title=f"PINN Benzene Forecast - Full Domain (30km × 30km)\n{forecast_time.strftime('%Y-%m-%d %H:%M')} UTC",
            filename=pinn_path,
            vmin=0, vmax=max(pinn_field.max(), 100)
        )
        
        # Generate NN2 heatmap - FULL DOMAIN (30km x 30km)
        nn2_path = self.viz_dir / f"nn2_full_domain_{timestamp_str}.png"
        self._create_heatmap(
            X, Y, nn2_2d,
            title=f"NN2-Corrected Benzene Forecast - Full Domain (30km × 30km)\n{forecast_time.strftime('%Y-%m-%d %H:%M')} UTC",
            filename=nn2_path,
            vmin=0, vmax=max(nn2_field.max(), 100)
        )
        
        print(f"  ✓ Saved visualizations:")
        print(f"    PINN: {pinn_path.name}")
        print(f"    NN2:  {nn2_path.name}")
        
        # Generate sensor-bounded visualizations
        self.generate_sensor_bounded_heatmaps(
            x_grid, y_grid, pinn_field, nn2_field,
            forecast_time, grid_resolution
        )
    
    def _create_heatmap(self, X, Y, Z, title, filename, vmin=0, vmax=None):
        """Create a single heatmap with sensors and facilities marked"""
        # Import facilities here to get source locations
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from config import FACILITIES
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot concentration field
        if vmax is None:
            vmax = Z.max()
        
        im = ax.pcolormesh(X, Y, Z, cmap='plasma', shading='auto',
                          vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Benzene Concentration (ppb)', fontsize=12)
        
        # Mark sensor locations (white triangles)
        ax.scatter(SENSOR_COORDS[:, 0], SENSOR_COORDS[:, 1],
                  c='white', s=100, marker='^', edgecolors='black',
                  linewidths=2, label='Sensors (n=9)', zorder=5)
        
        # Mark facility sources (yellow stars)
        facility_x = [f['source_x_cartesian'] for f in FACILITIES.values()]
        facility_y = [f['source_y_cartesian'] for f in FACILITIES.values()]
        ax.scatter(facility_x, facility_y,
                  c='yellow', s=150, marker='*', edgecolors='black',
                  linewidths=1.5, label='Benzene Sources (n=20)', zorder=6)
        
        # Formatting
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_sensor_bounded_heatmaps(
        self,
        x_grid, y_grid,
        pinn_field, nn2_field,
        forecast_time: datetime,
        grid_resolution: int = 30
    ):
        """
        Generate heatmaps for sensor-bounded area only
        This eliminates far-field extrapolation and matches training conditions
        """
        # Filter grid points within sensor bounds
        mask = (
            (x_grid >= self.bounds['x_min']) & (x_grid <= self.bounds['x_max']) &
            (y_grid >= self.bounds['y_min']) & (y_grid <= self.bounds['y_max'])
        )
        
        if mask.sum() == 0:
            print("    ⚠ No grid points within sensor bounds, skipping bounded viz")
            return
        
        # Extract bounded data
        x_bounded = x_grid[mask]
        y_bounded = y_grid[mask]
        pinn_bounded = pinn_field[mask]
        nn2_bounded = nn2_field[mask]
        
        # Create meshgrid for bounded area
        x_unique = np.unique(x_bounded)
        y_unique = np.unique(y_bounded)
        
        if len(x_unique) < 2 or len(y_unique) < 2:
            print("    ⚠ Insufficient grid points for bounded viz")
            return
        
        # Interpolate onto regular grid
        from scipy.interpolate import griddata
        X_bounded, Y_bounded = np.meshgrid(x_unique, y_unique)
        
        pinn_2d = griddata(
            (x_bounded, y_bounded), pinn_bounded,
            (X_bounded, Y_bounded), method='cubic', fill_value=0
        )
        nn2_2d = griddata(
            (x_bounded, y_bounded), nn2_bounded,
            (X_bounded, Y_bounded), method='cubic', fill_value=0
        )
        
        # Format timestamp
        timestamp_str = forecast_time.strftime("%Y-%m-%d_%H%M")
        
        # Generate PINN bounded heatmap
        pinn_path = self.sensor_bounded_dir / f"pinn_bounded_{timestamp_str}.png"
        self._create_heatmap(
            X_bounded, Y_bounded, pinn_2d,
            title=f"PINN Benzene (Sensor Area)\\n{forecast_time.strftime('%Y-%m-%d %H:%M')} UTC",
            filename=pinn_path,
            vmin=0, vmax=max(pinn_bounded.max(), 100)
        )
        
        # Generate NN2 bounded heatmap
        nn2_path = self.sensor_bounded_dir / f"nn2_bounded_{timestamp_str}.png"
        self._create_heatmap(
            X_bounded, Y_bounded, nn2_2d,
            title=f"NN2-Corrected Benzene (Sensor Area)\\n{forecast_time.strftime('%Y-%m-%d %H:%M')} UTC",
            filename=nn2_path,
            vmin=0, vmax=max(nn2_bounded.max(), 100)
        )
        
        print(f"    ✓ Sensor-bounded: max PINN={pinn_bounded.max():.1f}, max NN2={nn2_bounded.max():.1f} ppb")

    
    def save_correction_stats(
        self,
        current_time: datetime,
        forecast_time: datetime,
        pinn_field: np.ndarray,
        nn2_field: np.ndarray
    ):
        """
        Calculate and save NN2 correction statistics
        
        Args:
            current_time: Current timestamp (when prediction was made)
            forecast_time: Forecast timestamp (t+3hr)
            pinn_field: PINN predictions
            nn2_field: NN2 corrections
        """
        # Calculate correction
        correction = nn2_field - pinn_field
        
        # Compute statistics
        stats = {
            'current_timestamp': current_time,
            'forecast_timestamp': forecast_time,
            'pinn_mean': pinn_field.mean(),
            'pinn_max': pinn_field.max(),
            'nn2_mean': nn2_field.mean(),
            'nn2_max': nn2_field.max(),
            'correction_mean': correction.mean(),
            'correction_min': correction.min(),
            'correction_max': correction.max(),
            'correction_std': correction.std(),
            'correction_range': correction.max() - correction.min()
        }
        
        # Append to CSV
        df = pd.DataFrame([stats])
        
        if self.correction_csv.exists():
            df.to_csv(self.correction_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(self.correction_csv, index=False)
        
        print(f"  ✓ Correction stats:")
        print(f"    Mean correction: {stats['correction_mean']:.2f} ppb")
        print(f"    Correction range: [{stats['correction_min']:.2f}, {stats['correction_max']:.2f}] ppb")
