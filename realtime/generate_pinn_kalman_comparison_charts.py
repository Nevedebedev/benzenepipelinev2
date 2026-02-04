#!/usr/bin/env python3
"""
Generate Comparison Charts: PINN vs PINN + Kalman Filter
Creates at least 10 visualizations showing total concentration fields
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import sys
import torch
from scipy.interpolate import Rbf

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append("/Users/neevpratap/simpletesting")

from concentration_predictor import ConcentrationPredictor
from config import FACILITIES
from kalman_filter import BenzeneKalmanFilter
import json

# Output directory
OUTPUT_DIR = Path("realtime/data/pinn_kalman_comparison_charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensor IDs in order
SENSOR_IDS_ORDERED = [
    '482010026', '482010057', '482010069',
    '482010617', '482010803', '482011015',
    '482011035', '482011039', '482016000'
]

# Sensor coordinates (matching order of SENSOR_IDS_ORDERED)
SENSOR_COORDS_ARRAY = np.array([
    [13972.62, 19915.57],  # 482010026
    [3017.18, 12334.2],    # 482010057
    [817.42, 9218.92],     # 482010069
    [27049.57, 22045.66],  # 482010617
    [8836.35, 15717.2],    # 482010803
    [18413.8, 15068.96],   # 482011015
    [1159.98, 12272.52],   # 482011035
    [13661.93, 5193.24],   # 482011039
    [1546.9, 6786.33],     # 482016000
])

# 2021 data paths
MADIS_DIR = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated')
SENSOR_DATA_PATHS = {
    'january': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_jan.csv",
    'february': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_feb.csv",
    'march': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_march.csv",
}

FACILITY_DATA_PATHS = {
    'january': MADIS_DIR / 'training_data_2021_full_jan_REPAIRED',
    'february': MADIS_DIR / 'training_data_2021_feb_REPAIRED',
    'march': MADIS_DIR / 'training_data_2021_march_REPAIRED',
}

UNIT_CONVERSION_FACTOR = 313210039.9
FORECAST_T_HOURS = 3.0


def load_facility_data(month_key):
    """Load facility data for a month."""
    facility_dir = FACILITY_DATA_PATHS[month_key]
    
    if not facility_dir.exists():
        print(f"  Warning: Facility directory not found: {facility_dir}")
        return {}
    
    facility_files = sorted(facility_dir.glob('*_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    facility_data_dict = {}
    for facility_file in facility_files:
        facility_name = facility_file.stem.replace('_training_data', '')
        try:
            df = pd.read_csv(facility_file)
            
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp'})
            elif 'timestamp' not in df.columns:
                continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            required_cols = ['source_x_cartesian', 'source_y_cartesian', 'source_diameter', 'Q_total', 'wind_u', 'wind_v', 'D']
            if not all(col in df.columns for col in required_cols):
                continue
            
            facility_data_dict[facility_name] = df
        except Exception as e:
            print(f"  Error loading {facility_file.name}: {e}")
            continue
    
    return facility_data_dict


def compute_pinn_for_facility(pinn, x, y, t_hours, source_x, source_y, source_d, Q, wind_u, wind_v, D):
    """Compute PINN predictions for a facility across full domain grid."""
    # Batch process all grid points
    n_points = len(x)
    concentrations = np.zeros(n_points)
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch_x = x[i:end_idx]
        batch_y = y[i:end_idx]
        
        with torch.no_grad():
            phi_raw = pinn(
                torch.tensor(batch_x, dtype=torch.float32).reshape(-1, 1),
                torch.tensor(batch_y, dtype=torch.float32).reshape(-1, 1),
                torch.full((len(batch_x), 1), t_hours, dtype=torch.float32),
                torch.full((len(batch_x), 1), source_x, dtype=torch.float32),
                torch.full((len(batch_x), 1), source_y, dtype=torch.float32),
                torch.full((len(batch_x), 1), wind_u, dtype=torch.float32),
                torch.full((len(batch_x), 1), wind_v, dtype=torch.float32),
                torch.full((len(batch_x), 1), source_d, dtype=torch.float32),
                torch.full((len(batch_x), 1), D, dtype=torch.float32),
                torch.full((len(batch_x), 1), Q, dtype=torch.float32),
                normalize=True
            )
            batch_concentrations = phi_raw.cpu().numpy().flatten() * UNIT_CONVERSION_FACTOR
            concentrations[i:end_idx] = batch_concentrations
    
    return np.maximum(concentrations, 0.0)


def predict_pinn_full_domain(pinn, facility_data_dict, timestamp, grid_resolution=100):
    """Predict PINN across full domain grid."""
    # Create spatial grid
    x_min, x_max = 0, 30000
    y_min, y_max = 0, 30000
    
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Flatten for computation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Accumulate plumes from all facilities
    total_pinn_field = np.zeros(len(x_flat))
    
    for facility_name, facility_df in facility_data_dict.items():
        facility_data = facility_df[facility_df['timestamp'] == timestamp]
        
        if len(facility_data) == 0:
            # Try closest within 30 minutes
            time_diffs = (facility_df['timestamp'] - timestamp).abs()
            if len(time_diffs) > 0 and time_diffs.min() <= pd.Timedelta(minutes=30):
                closest_idx = time_diffs.idxmin()
                facility_data = facility_df.iloc[[closest_idx]]
        
        if len(facility_data) == 0:
            continue
        
        for _, row in facility_data.iterrows():
            cx = row['source_x_cartesian']
            cy = row['source_y_cartesian']
            d = row['source_diameter']
            Q = row['Q_total']
            u = row['wind_u']
            v = row['wind_v']
            kappa = row['D']
            
            # Compute PINN for this facility across all grid points
            facility_field = compute_pinn_for_facility(
                pinn, x_flat, y_flat, FORECAST_T_HOURS,
                cx, cy, d, Q, u, v, kappa
            )
            
            # Accumulate (superposition)
            total_pinn_field += facility_field
    
    # Reshape back to 2D
    pinn_field_2d = total_pinn_field.reshape(grid_resolution, grid_resolution)
    
    return X, Y, pinn_field_2d, total_pinn_field


def predict_pinn_at_sensors(pinn, facility_data_dict, timestamp):
    """Predict PINN at sensor locations only."""
    sensor_pinn_values = np.zeros(len(SENSOR_COORDS_ARRAY))
    
    for facility_name, facility_df in facility_data_dict.items():
        facility_data = facility_df[facility_df['timestamp'] == timestamp]
        
        if len(facility_data) == 0:
            time_diffs = (facility_df['timestamp'] - timestamp).abs()
            if len(time_diffs) > 0 and time_diffs.min() <= pd.Timedelta(minutes=30):
                closest_idx = time_diffs.idxmin()
                facility_data = facility_df.iloc[[closest_idx]]
        
        if len(facility_data) == 0:
            continue
        
        for _, row in facility_data.iterrows():
            cx = row['source_x_cartesian']
            cy = row['source_y_cartesian']
            d = row['source_diameter']
            Q = row['Q_total']
            u = row['wind_u']
            v = row['wind_v']
            kappa = row['D']
            
            for i, (sx, sy) in enumerate(SENSOR_COORDS_ARRAY):
                with torch.no_grad():
                    phi_raw = pinn(
                        torch.tensor([[sx]], dtype=torch.float32),
                        torch.tensor([[sy]], dtype=torch.float32),
                        torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32),
                        torch.tensor([[cx]], dtype=torch.float32),
                        torch.tensor([[cy]], dtype=torch.float32),
                        torch.tensor([[u]], dtype=torch.float32),
                        torch.tensor([[v]], dtype=torch.float32),
                        torch.tensor([[d]], dtype=torch.float32),
                        torch.tensor([[kappa]], dtype=torch.float32),
                        torch.tensor([[Q]], dtype=torch.float32),
                        normalize=True
                    )
                    concentration_ppb = phi_raw.item() * UNIT_CONVERSION_FACTOR
                    sensor_pinn_values[i] += concentration_ppb
    
    return np.maximum(sensor_pinn_values, 0.0)


def apply_kalman_correction_to_domain(X, Y, pinn_field_flat, x_flat, y_flat, 
                                      sensor_pinn, kalman_forecast, grid_resolution=100):
    """Apply Kalman corrections to full domain using RBF interpolation."""
    # Calculate correction field at sensors (Kalman - PINN)
    sensor_corrections = kalman_forecast - sensor_pinn
    
    # Interpolate correction field across domain using RBF
    correction_rbf = Rbf(
        SENSOR_COORDS_ARRAY[:, 0], 
        SENSOR_COORDS_ARRAY[:, 1], 
        sensor_corrections,
        function='multiquadric',
        smooth=0.1
    )
    
    # Evaluate at all grid points
    correction_field = correction_rbf(x_flat, y_flat)
    
    # Handle any NaN values from RBF interpolation (replace with 0)
    correction_field = np.nan_to_num(correction_field, nan=0.0)
    
    # Distance-based confidence weighting
    distances_to_sensors = np.zeros(len(x_flat))
    for i in range(len(x_flat)):
        dists = np.sqrt((SENSOR_COORDS_ARRAY[:, 0] - x_flat[i])**2 + 
                       (SENSOR_COORDS_ARRAY[:, 1] - y_flat[i])**2)
        distances_to_sensors[i] = dists.min()
    
    # Define confidence decay: full trust within 2km, linear decay 2-5km, zero trust beyond 5km
    confidence = np.ones(len(x_flat))
    confidence[distances_to_sensors > 2000] = 1.0 - (distances_to_sensors[distances_to_sensors > 2000] - 2000) / 3000
    confidence[distances_to_sensors > 5000] = 0.0
    confidence = np.clip(confidence, 0.0, 1.0)
    
    # Apply confidence-weighted correction
    weighted_correction = correction_field * confidence
    
    # Ensure no NaN values
    weighted_correction = np.nan_to_num(weighted_correction, nan=0.0)
    pinn_field_flat = np.nan_to_num(pinn_field_flat, nan=0.0)
    
    # Add to PINN field
    kalman_field_flat = pinn_field_flat + weighted_correction
    kalman_field_flat = np.maximum(kalman_field_flat, 0.0)
    
    # Final NaN check
    kalman_field_flat = np.nan_to_num(kalman_field_flat, nan=0.0)
    
    # Reshape to 2D
    kalman_field_2d = kalman_field_flat.reshape(grid_resolution, grid_resolution)
    correction_field_2d = correction_field.reshape(grid_resolution, grid_resolution)
    
    return kalman_field_2d, correction_field_2d


def create_comparison_chart(X, Y, pinn_field, kalman_field, timestamp, chart_num, sensor_readings=None):
    """Create a single comparison chart."""
    fig = plt.figure(figsize=(20, 8))
    
    # Ensure no NaN values in input fields
    pinn_field = np.nan_to_num(pinn_field, nan=0.0)
    kalman_field = np.nan_to_num(kalman_field, nan=0.0)
    
    # Calculate difference
    diff_field = kalman_field - pinn_field
    diff_field = np.nan_to_num(diff_field, nan=0.0)
    
    # Determine color scale (handle NaN values)
    pinn_max = np.nanmax(pinn_field)
    kalman_max = np.nanmax(kalman_field)
    vmax = max(pinn_max, kalman_max) if np.isfinite(max(pinn_max, kalman_max)) else pinn_max
    
    diff_min = np.nanmin(diff_field)
    diff_max = np.nanmax(diff_field)
    vmax_diff = max(abs(diff_min), abs(diff_max)) if np.isfinite(max(abs(diff_min), abs(diff_max))) else 1.0
    
    # Panel 1: Raw PINN
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.contourf(X, Y, pinn_field, levels=30, cmap='viridis', vmin=0, vmax=vmax)
    ax1.scatter(SENSOR_COORDS_ARRAY[:, 0], SENSOR_COORDS_ARRAY[:, 1],
               c='white', s=200, marker='^', edgecolors='black', linewidths=2, 
               label='Sensors', zorder=5)
    
    # Mark facilities
    facility_x = [f['source_x_cartesian'] for f in FACILITIES.values()]
    facility_y = [f['source_y_cartesian'] for f in FACILITIES.values()]
    ax1.scatter(facility_x, facility_y, c='yellow', s=150, marker='*', 
               edgecolors='black', linewidths=1.5, label='Sources', zorder=6)
    
    ax1.set_xlabel('X Coordinate (m)', fontsize=12)
    ax1.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax1.set_title('Raw PINN Predictions', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Concentration (ppb)')
    
    # Panel 2: PINN + Kalman Filter
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.contourf(X, Y, kalman_field, levels=30, cmap='viridis', vmin=0, vmax=vmax)
    ax2.scatter(SENSOR_COORDS_ARRAY[:, 0], SENSOR_COORDS_ARRAY[:, 1],
               c='white', s=200, marker='^', edgecolors='black', linewidths=2, 
               label='Sensors', zorder=5)
    ax2.scatter(facility_x, facility_y, c='yellow', s=150, marker='*', 
               edgecolors='black', linewidths=1.5, label='Sources', zorder=6)
    ax2.set_xlabel('X Coordinate (m)', fontsize=12)
    ax2.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax2.set_title('PINN + Kalman Filter', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='Concentration (ppb)')
    
    # Panel 3: Difference (Kalman - PINN)
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.contourf(X, Y, diff_field, levels=30, cmap='RdBu_r', 
                       vmin=-vmax_diff, vmax=vmax_diff)
    ax3.scatter(SENSOR_COORDS_ARRAY[:, 0], SENSOR_COORDS_ARRAY[:, 1],
               c='black', s=200, marker='^', edgecolors='white', linewidths=2, 
               label='Sensors', zorder=5)
    ax3.set_xlabel('X Coordinate (m)', fontsize=12)
    ax3.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax3.set_title('Kalman Correction (Kalman - PINN)', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(im3, ax=ax3, label='Correction (ppb)')
    
    # Add statistics text (handle NaN values)
    pinn_min = np.nanmin(pinn_field)
    pinn_max = np.nanmax(pinn_field)
    pinn_mean = np.nanmean(pinn_field)
    
    kalman_min = np.nanmin(kalman_field)
    kalman_max = np.nanmax(kalman_field)
    kalman_mean = np.nanmean(kalman_field)
    
    diff_min = np.nanmin(diff_field)
    diff_max = np.nanmax(diff_field)
    
    stats_text = (
        f"PINN: min={pinn_min:.3f}, max={pinn_max:.3f}, mean={pinn_mean:.3f} ppb\n"
        f"Kalman: min={kalman_min:.3f}, max={kalman_max:.3f}, mean={kalman_mean:.3f} ppb\n"
        f"Max correction: {diff_max:.3f} ppb, Min correction: {diff_min:.3f} ppb"
    )
    if sensor_readings is not None:
        stats_text += f"\nSensor readings: min={sensor_readings.min():.2f}, max={sensor_readings.max():.2f} ppb"
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle(
        f'Chart {chart_num}: PINN vs PINN + Kalman Filter Comparison\n'
        f'Forecast Time: {timestamp.strftime("%Y-%m-%d %H:%M")}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save
    filename = OUTPUT_DIR / f"comparison_chart_{chart_num:02d}_{timestamp.strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved chart {chart_num}: {filename.name}")
    return filename


def main():
    """Generate comparison charts."""
    print("="*80)
    print("GENERATING PINN vs PINN + KALMAN COMPARISON CHARTS")
    print("="*80)
    
    # Load Kalman parameters
    param_file = Path("realtime/data/kalman_parameters.json")
    if param_file.exists():
        with open(param_file, 'r') as f:
            kf_params = json.load(f)
            kf_params = {k: v for k, v in kf_params.items() 
                        if k in ['process_noise', 'measurement_noise', 'decay_rate', 'pinn_weight']}
    else:
        kf_params = {
            'process_noise': 0.1,
            'measurement_noise': 0.5,
            'decay_rate': 0.7,
            'pinn_weight': 0.1
        }
    
    print(f"\nKalman parameters: {kf_params}")
    
    # Load PINN model
    print("\n[1/4] Loading PINN model...")
    from pinn import ParametricADEPINN
    pinn = ParametricADEPINN()
    checkpoint = torch.load("/Users/neevpratap/Downloads/pinn_combined_final2.pth", 
                           map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    pinn.x_min = torch.tensor(0.0)
    pinn.x_max = torch.tensor(30000.0)
    pinn.y_min = torch.tensor(0.0)
    pinn.y_max = torch.tensor(30000.0)
    pinn.t_min = torch.tensor(0.0)
    pinn.t_max = torch.tensor(10.0)
    pinn.d_min = torch.tensor(0.0)
    pinn.d_max = torch.tensor(200.0)
    pinn.kappa_min = torch.tensor(0.0)
    pinn.kappa_max = torch.tensor(200.0)
    pinn.Q_min = torch.tensor(0.0)
    pinn.Q_max = torch.tensor(0.01)
    pinn.eval()
    print("  ✓ PINN loaded")
    
    # Initialize Kalman filter
    print("\n[2/4] Initializing Kalman filter...")
    kf = BenzeneKalmanFilter(**kf_params)
    print("  ✓ Kalman filter initialized")
    
    # Load data and select timestamps
    print("\n[3/4] Loading data and selecting timestamps...")
    
    selected_timestamps = []
    sensor_data_dict = {}
    
    # Load sensor data and select diverse timestamps
    for month_key in ['january', 'february', 'march']:
        sensor_file = SENSOR_DATA_PATHS[month_key]
        if not sensor_file.exists():
            continue
        
        df = pd.read_csv(sensor_file)
        if 't' in df.columns:
            df = df.rename(columns={'t': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Select timestamps spaced throughout the month
        n_samples = min(4, len(df) // 6)  # 4 timestamps per month
        if n_samples > 0:
            step = len(df) // (n_samples + 1)
            for i in range(1, n_samples + 1):
                idx = i * step
                if idx < len(df):
                    ts = df.iloc[idx]['timestamp']
                    selected_timestamps.append((ts, month_key))
                    
                    # Store sensor readings
                    sensor_cols = [f'sensor_{sid}' for sid in SENSOR_IDS_ORDERED]
                    readings = df.iloc[idx][sensor_cols].values.astype(float)
                    readings = np.nan_to_num(readings, nan=0.0)
                    sensor_data_dict[ts] = readings
    
    # Ensure we have at least 10 timestamps
    if len(selected_timestamps) < 10:
        # Add more from first month
        month_key = selected_timestamps[0][1] if selected_timestamps else 'january'
        sensor_file = SENSOR_DATA_PATHS[month_key]
        if sensor_file.exists():
            df = pd.read_csv(sensor_file)
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            additional_needed = 10 - len(selected_timestamps)
            step = len(df) // (additional_needed + 1)
            for i in range(1, additional_needed + 1):
                idx = i * step
                if idx < len(df):
                    ts = df.iloc[idx]['timestamp']
                    if ts not in [t[0] for t in selected_timestamps]:
                        selected_timestamps.append((ts, month_key))
                        sensor_cols = [f'sensor_{sid}' for sid in SENSOR_IDS_ORDERED]
                        readings = df.iloc[idx][sensor_cols].values.astype(float)
                        readings = np.nan_to_num(readings, nan=0.0)
                        sensor_data_dict[ts] = readings
    
    selected_timestamps = selected_timestamps[:12]  # Take up to 12
    print(f"  ✓ Selected {len(selected_timestamps)} timestamps")
    
    # Load facility data
    facility_data_dicts = {}
    for month_key in ['january', 'february', 'march']:
        facility_data_dicts[month_key] = load_facility_data(month_key)
        print(f"  Loaded {len(facility_data_dicts[month_key])} facilities for {month_key}")
    
    # Generate charts
    print("\n[4/4] Generating comparison charts...")
    
    chart_num = 1
    for timestamp, month_key in selected_timestamps:
        try:
            # Get facility data for this timestamp
            facility_data_dict = facility_data_dicts[month_key]
            if len(facility_data_dict) == 0:
                continue
            
            # Get current sensor readings (3 hours before forecast time)
            current_time = timestamp - pd.Timedelta(hours=3)
            current_sensors = sensor_data_dict.get(timestamp, np.zeros(9))
            
            # Predict PINN across full domain
            print(f"    Computing full domain PINN predictions...")
            X, Y, pinn_field_2d, pinn_field_flat = predict_pinn_full_domain(
                pinn, facility_data_dict, current_time, grid_resolution=100
            )
            
            if np.all(pinn_field_flat == 0):
                print(f"    ⚠ No PINN predictions, skipping...")
                continue
            
            # Get PINN at sensors for Kalman filter
            pinn_at_sensors = predict_pinn_at_sensors(pinn, facility_data_dict, current_time)
            
            # Kalman forecast at sensors
            kalman_at_sensors, _ = kf.forecast(
                current_sensors=current_sensors,
                pinn_predictions=pinn_at_sensors,
                hours_ahead=3,
                return_uncertainty=False
            )
            
            # Apply Kalman corrections to full domain
            print(f"    Applying Kalman corrections to full domain...")
            x_flat = X.flatten()
            y_flat = Y.flatten()
            kalman_field_2d, correction_field_2d = apply_kalman_correction_to_domain(
                X, Y, pinn_field_flat, x_flat, y_flat,
                pinn_at_sensors, kalman_at_sensors, grid_resolution=100
            )
            
            pinn_field = pinn_field_2d
            kalman_field = kalman_field_2d
            
            # Create chart
            create_comparison_chart(
                X, Y, pinn_field, kalman_field, 
                timestamp, chart_num, 
                sensor_readings=current_sensors
            )
            
            chart_num += 1
            
            if chart_num > 12:
                break
                
        except Exception as e:
            print(f"  ⚠ Error processing {timestamp}: {e}")
            continue
    
    print(f"\n✓ Generated {chart_num - 1} comparison charts")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

