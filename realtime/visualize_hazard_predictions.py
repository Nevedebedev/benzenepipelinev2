#!/usr/bin/env python3
"""
Visualize Hazard Predictions - PINN vs EDF

Runs PINN predictions for identified hazard events and generates full-domain visualizations
comparing PINN predictions with actual EDF sensor readings.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime
from tqdm import tqdm

# Add paths
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

from pinn import ParametricADEPINN
from config import FACILITIES

# Paths
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
HAZARDS_PATH = Path(__file__).parent / 'data/hazards_2019_with_conditions.pkl'
OUTPUT_DIR = Path(__file__).parent / 'data/visualizations/hazards_2019'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb
FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast
GRID_RESOLUTION = 100

# Sensor coordinates (Cartesian, meters)
SENSORS = {
    '482010026': (13972.62, 19915.57),
    '482010057': (3017.18, 12334.2),
    '482010069': (817.42, 9218.92),
    '482010617': (27049.57, 22045.66),
    '482010803': (8836.35, 15717.2),
    '482011015': (18413.8, 15068.96),
    '482011035': (1159.98, 12272.52),
    '482011039': (13661.93, 5193.24),
    '482016000': (1546.9, 6786.33),
}

SENSOR_IDS = ['482010026', '482010057', '482010069', '482010617', '482010803',
              '482011015', '482011035', '482011039', '482016000']

def load_pinn_model():
    """Load PINN model"""
    print("Loading PINN model...")
    
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges
    pinn.x_min = torch.tensor(0.0)
    pinn.x_max = torch.tensor(30000.0)
    pinn.y_min = torch.tensor(0.0)
    pinn.y_max = torch.tensor(30000.0)
    pinn.t_min = torch.tensor(0.0)
    pinn.t_max = torch.tensor(8760.0)
    pinn.cx_min = torch.tensor(0.0)
    pinn.cx_max = torch.tensor(30000.0)
    pinn.cy_min = torch.tensor(0.0)
    pinn.cy_max = torch.tensor(30000.0)
    pinn.u_min = torch.tensor(-15.0)
    pinn.u_max = torch.tensor(15.0)
    pinn.v_min = torch.tensor(-15.0)
    pinn.v_max = torch.tensor(15.0)
    pinn.d_min = torch.tensor(0.0)
    pinn.d_max = torch.tensor(200.0)
    pinn.kappa_min = torch.tensor(0.0)
    pinn.kappa_max = torch.tensor(200.0)
    pinn.Q_min = torch.tensor(0.0)
    pinn.Q_max = torch.tensor(0.01)
    
    pinn.eval()
    print("  ✓ PINN model loaded")
    return pinn

def create_grid():
    """Create spatial grid for full domain"""
    x_grid = np.linspace(0, 30000, GRID_RESOLUTION)
    y_grid = np.linspace(0, 30000, GRID_RESOLUTION)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    return x_flat, y_flat, X, Y

def compute_pinn_for_facility(pinn, x, y, t, source_x, source_y, source_d, Q, wind_u, wind_v, D):
    """Compute PINN predictions for single facility across grid"""
    n_points = len(x)
    concentrations = np.zeros(n_points)
    
    # Process in batches
    batch_size = 1000
    
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch_x = x[i:end_idx]
        batch_y = y[i:end_idx]
        batch_size_actual = len(batch_x)
        
        # Create tensors
        x_t = torch.tensor(batch_x.reshape(-1, 1), dtype=torch.float32)
        y_t = torch.tensor(batch_y.reshape(-1, 1), dtype=torch.float32)
        t_t = torch.full((batch_size_actual, 1), t, dtype=torch.float32)
        cx_t = torch.full((batch_size_actual, 1), source_x, dtype=torch.float32)
        cy_t = torch.full((batch_size_actual, 1), source_y, dtype=torch.float32)
        u_t = torch.full((batch_size_actual, 1), wind_u, dtype=torch.float32)
        v_t = torch.full((batch_size_actual, 1), wind_v, dtype=torch.float32)
        d_t = torch.full((batch_size_actual, 1), source_d, dtype=torch.float32)
        kappa_t = torch.full((batch_size_actual, 1), D, dtype=torch.float32)
        Q_t = torch.full((batch_size_actual, 1), Q, dtype=torch.float32)
        
        # Run PINN
        with torch.no_grad():
            phi = pinn(x_t, y_t, t_t, cx_t, cy_t, u_t, v_t, d_t, kappa_t, Q_t, normalize=True)
        
        # Convert to ppb
        concentrations[i:end_idx] = np.maximum(phi.numpy().flatten() * UNIT_CONVERSION_FACTOR, 0.0)
    
    return concentrations

def compute_pinn_at_sensors(pinn, facility_params):
    """Compute PINN predictions at sensor locations"""
    sensor_pinn = np.zeros(len(SENSOR_IDS))
    
    for facility_name, params in facility_params.items():
        source_x = params['source_x_cartesian']
        source_y = params['source_y_cartesian']
        source_d = params['source_diameter']
        Q = params['Q']
        wind_u = params['wind_u']
        wind_v = params['wind_v']
        D = params['D']
        
        # Compute PINN at each sensor location
        for i, sensor_id in enumerate(SENSOR_IDS):
            sx, sy = SENSORS[sensor_id]
            
            with torch.no_grad():
                phi_raw = pinn(
                    torch.tensor([[sx]], dtype=torch.float32),
                    torch.tensor([[sy]], dtype=torch.float32),
                    torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32),
                    torch.tensor([[source_x]], dtype=torch.float32),
                    torch.tensor([[source_y]], dtype=torch.float32),
                    torch.tensor([[wind_u]], dtype=torch.float32),
                    torch.tensor([[wind_v]], dtype=torch.float32),
                    torch.tensor([[source_d]], dtype=torch.float32),
                    torch.tensor([[D]], dtype=torch.float32),
                    torch.tensor([[Q]], dtype=torch.float32),
                    normalize=True
                )
                
                concentration_ppb = phi_raw.item() * UNIT_CONVERSION_FACTOR
                sensor_pinn[i] += concentration_ppb
    
    return sensor_pinn

def predict_full_domain(pinn, facility_params):
    """Predict concentrations across full domain"""
    x_flat, y_flat, X, Y = create_grid()
    
    # Accumulate from all facilities
    total_pinn_field = np.zeros(len(x_flat))
    
    for facility_name, params in facility_params.items():
        facility_field = compute_pinn_for_facility(
            pinn, x_flat, y_flat, FORECAST_T_HOURS,
            params['source_x_cartesian'],
            params['source_y_cartesian'],
            params['source_diameter'],
            params['Q'],
            params['wind_u'],
            params['wind_v'],
            params['D']
        )
        total_pinn_field += facility_field
    
    return total_pinn_field, x_flat, y_flat, X, Y

def create_hazard_visualization(hazard_data, pinn_field, x_flat, y_flat, X, Y, sensor_pinn):
    """Create full-domain visualization for hazard event"""
    timestamp = hazard_data['timestamp']
    hazard_rank = hazard_data['hazard_rank']
    max_edf = hazard_data['max_concentration_edf']
    peak_sensor_id = hazard_data['peak_sensor_id'].replace('sensor_', '')
    
    # Reshape to 2D
    pinn_2d = pinn_field.reshape(GRID_RESOLUTION, GRID_RESOLUTION)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Left plot: Full domain heatmap
    ax1 = axes[0]
    im1 = ax1.pcolormesh(X, Y, pinn_2d, cmap='plasma', shading='auto', vmin=0, vmax=max(pinn_field.max(), 100))
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Benzene Concentration (ppb)', fontsize=12)
    
    # Mark sensor locations
    sensor_x = [SENSORS[sid][0] for sid in SENSOR_IDS]
    sensor_y = [SENSORS[sid][1] for sid in SENSOR_IDS]
    ax1.scatter(sensor_x, sensor_y, c='white', s=150, marker='^', edgecolors='black',
                linewidths=2, label='Sensors (n=9)', zorder=5)
    
    # Mark facility sources
    facility_x = [f['source_x_cartesian'] for f in FACILITIES.values()]
    facility_y = [f['source_y_cartesian'] for f in FACILITIES.values()]
    ax1.scatter(facility_x, facility_y, c='yellow', s=200, marker='*', edgecolors='black',
                linewidths=1.5, label='Benzene Sources (n=20)', zorder=6)
    
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title(f'Hazard #{hazard_rank}: PINN Prediction - Full Domain (30km × 30km)\n'
                  f'EDF Peak: {max_edf:.2f} ppb at {peak_sensor_id} | {timestamp.strftime("%Y-%m-%d %H:%M")} UTC',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: EDF vs PINN comparison at sensors
    ax2 = axes[1]
    
    # Get EDF values
    edf_values = []
    pinn_values = []
    sensor_labels = []
    
    for i, sensor_id in enumerate(SENSOR_IDS):
        sensor_key = f'sensor_{sensor_id}'
        if sensor_key in hazard_data:
            edf_val = hazard_data[sensor_key]
            pinn_val = sensor_pinn[i]
            edf_values.append(edf_val)
            pinn_values.append(pinn_val)
            sensor_labels.append(sensor_id)
    
    x_pos = np.arange(len(sensor_labels))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, edf_values, width, label='EDF Actual', color='red', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, pinn_values, width, label='PINN Predicted', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Sensor ID', fontsize=12)
    ax2.set_ylabel('Concentration (ppb)', fontsize=12)
    ax2.set_title(f'EDF vs PINN Comparison at Sensor Locations\n'
                  f'Hazard #{hazard_rank}: {timestamp.strftime("%Y-%m-%d %H:%M")} UTC',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sensor_labels, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save
    timestamp_str = timestamp.strftime("%Y-%m-%d_%H%M")
    filename = OUTPUT_DIR / f"hazard_{timestamp_str}_pinn_prediction.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def generate_summary_report(hazards_with_predictions):
    """Generate summary report"""
    print("\nGenerating summary report...")
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("2019 HAZARD EVENTS - PINN PREDICTION ANALYSIS")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("Top 10 Hazard Events Identified from EDF Sensor Data")
    report_lines.append("")
    report_lines.append(f"{'Rank':<6} {'Timestamp':<20} {'EDF Peak (ppb)':<15} {'PINN Peak (ppb)':<15} "
                       f"{'Error (ppb)':<12} {'% Error':<10} {'Peak Sensor':<15}")
    report_lines.append("-" * 100)
    
    summary_data = []
    
    for hazard in hazards_with_predictions:
        timestamp = hazard['timestamp']
        rank = hazard['hazard_rank']
        edf_peak = hazard['max_concentration_edf']
        peak_sensor_id = hazard['peak_sensor_id'].replace('sensor_', '')
        
        # Find PINN prediction at peak sensor
        sensor_idx = SENSOR_IDS.index(peak_sensor_id) if peak_sensor_id in SENSOR_IDS else 0
        pinn_peak = hazard['sensor_pinn'][sensor_idx]
        
        error = pinn_peak - edf_peak
        pct_error = (error / edf_peak * 100) if edf_peak > 0 else 0
        
        report_lines.append(f"{rank:<6} {str(timestamp):<20} {edf_peak:<15.2f} {pinn_peak:<15.2f} "
                           f"{error:<12.2f} {pct_error:<10.1f} {peak_sensor_id:<15}")
        
        summary_data.append({
            'rank': rank,
            'timestamp': timestamp,
            'edf_peak_ppb': edf_peak,
            'pinn_peak_ppb': pinn_peak,
            'error_ppb': error,
            'pct_error': pct_error,
            'peak_sensor_id': peak_sensor_id
        })
    
    report_lines.append("")
    report_lines.append("=" * 100)
    
    # Save report
    report_path = Path(__file__).parent / 'data/hazards_2019_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved: {report_path}")
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = Path(__file__).parent / 'data/hazards_2019_pinn_comparison.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Saved: {summary_csv_path}")

def main():
    print("=" * 100)
    print("VISUALIZE HAZARD PREDICTIONS - PINN vs EDF")
    print("=" * 100)
    print()
    
    # Load hazards
    print("Loading hazard events...")
    with open(HAZARDS_PATH, 'rb') as f:
        hazards = pickle.load(f)
    print(f"  Loaded {len(hazards)} hazard events")
    
    # Load PINN model
    pinn = load_pinn_model()
    
    # Process each hazard
    print("\nProcessing hazard events...")
    hazards_with_predictions = []
    
    for hazard in tqdm(hazards, desc="Processing hazards"):
        timestamp = hazard['timestamp']
        rank = hazard['hazard_rank']
        facility_params = hazard['facility_params']
        
        print(f"\n  Hazard #{rank}: {timestamp}")
        
        # Predict full domain
        pinn_field, x_flat, y_flat, X, Y = predict_full_domain(pinn, facility_params)
        
        # Predict at sensors
        sensor_pinn = compute_pinn_at_sensors(pinn, facility_params)
        
        # Store predictions
        hazard['pinn_field'] = pinn_field
        hazard['sensor_pinn'] = sensor_pinn
        hazard['x_flat'] = x_flat
        hazard['y_flat'] = y_flat
        hazard['X'] = X
        hazard['Y'] = Y
        
        # Create visualization
        viz_path = create_hazard_visualization(hazard, pinn_field, x_flat, y_flat, X, Y, sensor_pinn)
        print(f"    Saved: {viz_path.name}")
        
        hazards_with_predictions.append(hazard)
    
    # Generate summary report
    generate_summary_report(hazards_with_predictions)
    
    print("\n" + "=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)
    print(f"\nGenerated visualizations for {len(hazards_with_predictions)} hazard events")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

