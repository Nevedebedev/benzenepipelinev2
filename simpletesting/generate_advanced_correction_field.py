"""
Advanced Correction Field Map - Multiple Conditions
Generate 10 visualizations across varying meteorological conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pinn import ParametricADEPINN
from scipy.interpolate import Rbf
import os

# Paths
BASE_DIR = "/Users/neevpratap/simpletesting"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_visualizations_q1_2021/correction_fields")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sensor locations
SENSORS = {
    'sensor_482010026': (13972.62, 19915.57),
    'sensor_482010057': (3017.18, 12334.2),
    'sensor_482010069': (817.42, 9218.92),
    'sensor_482010617': (27049.57, 22045.66),
    'sensor_482010803': (8836.35, 15717.2),
    'sensor_482011015': (18413.8, 15068.96),
    'sensor_482011035': (1159.98, 12272.52),
    'sensor_482011039': (13661.93, 5193.24),
    'sensor_482016000': (1546.9, 6786.33),
}

print("="*70)
print("CORRECTION FIELD MAPS - MULTIPLE CONDITIONS")
print("="*70)

# Load PINN
print("\n[1/4] Loading PINN...")
pinn = ParametricADEPINN()
checkpoint = torch.load(os.path.join(BASE_DIR, "pinn_combined_final.pth 2"),  
                        map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']
filtered_state_dict = {k: v for k, v in state_dict.items() 
                       if not k.endswith('_min') and not k.endswith('_max')}
pinn.load_state_dict(filtered_state_dict, strict=False)

# Override with benchmark normalization ranges
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
print("  ✓ PINN loaded")

# Load sensor data
print("\n[2/4] Loading sensor data...")
jan_data = pd.read_csv(os.path.join(BASE_DIR, "results_jan_2021_hybrid.csv"))
jan_data['timestamp'] = pd.to_datetime(jan_data['timestamp'])

# Load meteorology
met_path = "/Users/neevpratap/Desktop/madis_data_desktop_updated/training_data_2021_full_jan_REPAIRED/BASF_Pasadena_training_data.csv"
met_df = pd.read_csv(met_path)
met_df['t'] = pd.to_datetime(met_df['t'])
met_df_valid = met_df.dropna(subset=['wind_u', 'wind_v', 'D'])

# Select 10 timestamps with diverse conditions
print("\n[3/4] Selecting diverse timestamps...")
# Get different timestamps throughout the month
timestamp_indices = np.linspace(50, len(met_df_valid)-50, 10, dtype=int)
selected_timestamps = [met_df_valid['t'].iloc[idx] for idx in timestamp_indices]

print(f"  Selected {len(selected_timestamps)} timestamps")

# Create spatial grid
print("\n[4/4] Creating spatial grid...")
grid_resolution = 100
x_grid = np.linspace(0, 30000, grid_resolution)
y_grid = np.linspace(0, 30000, grid_resolution)
X, Y = np.meshgrid(x_grid, y_grid)

# Generate visualizations for each timestamp
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

for viz_idx, representative_ts in enumerate(selected_timestamps, 1):
    print(f"\n[{viz_idx}/10] Processing timestamp: {representative_ts}")
    
    # Get meteorology
    met_row = met_df[met_df['t'] == representative_ts].iloc[0]
    u_val = met_row['wind_u']
    v_val = met_row['wind_v']
    D_val = met_row['D']
    
    # Get sensor data
    ts_data = jan_data[jan_data['timestamp'] == representative_ts]
    
    if len(ts_data) < 5:  # Skip if not enough sensor data
        print(f"  ⚠ Skipping - insufficient sensor data")
        continue
    
    # Extract sensor corrections and PINN values
    sensor_corrections = {}
    sensor_pinn = {}
    for sensor_id, (sx, sy) in SENSORS.items():
        sensor_row = ts_data[ts_data['sensor_id'] == sensor_id]
        if not sensor_row.empty:
            row = sensor_row.iloc[0]
            correction = row['final_pred'] - row['pinn_pred']
            sensor_corrections[(sx, sy)] = correction
            sensor_pinn[(sx, sy)] = row['pinn_pred']
    
    # Extract coordinates and values
    sensor_x = [coords[0] for coords in sensor_corrections.keys()]
    sensor_y = [coords[1] for coords in sensor_corrections.keys()]
    corrections = [corr for corr in sensor_corrections.values()]
    sensor_pinn_values = [pinn_val for pinn_val in sensor_pinn.values()]
    
    # Interpolate PINN field
    rbf_pinn = Rbf(sensor_x, sensor_y, sensor_pinn_values, function='multiquadric', smooth=0.1)
    pinn_field = rbf_pinn(X, Y)
    pinn_field = np.maximum(pinn_field, 0.0)
    
    # Interpolate NN2 corrections
    rbf_interp = Rbf(sensor_x, sensor_y, corrections, function='multiquadric', smooth=0.1)
    nn2_correction_field = rbf_interp(X, Y)
    final_field = pinn_field + nn2_correction_field
    final_field = np.maximum(final_field, 0.0)
    
    print(f"  PINN: {pinn_field.min():.2f}-{pinn_field.max():.2f} ppb")
    print(f"  Corrections: {nn2_correction_field.min():.2f}-{nn2_correction_field.max():.2f} ppb")
    print(f"  Final: {final_field.min():.2f}-{final_field.max():.2f} ppb")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    cmap = plt.cm.viridis
    
    # Panel 1: Raw PINN field
    im1 = axes[0].contourf(X, Y, pinn_field, levels=20, cmap=cmap, alpha=0.8)
    axes[0].scatter(sensor_x, sensor_y, c='black', s=150, edgecolors='white', 
                    linewidth=2.5, marker='o', label='Sensors', zorder=5)
    axes[0].set_xlabel('X Coordinate (m)', fontsize=12)
    axes[0].set_ylabel('Y Coordinate (m)', fontsize=12)
    axes[0].set_title('Raw PINN Predictions', fontsize=14, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.2)
    plt.colorbar(im1, ax=axes[0], label='Concentration (ppb)')
    
    # Panel 2: NN2 Correction field
    vmax_corr = max(abs(nn2_correction_field.min()), abs(nn2_correction_field.max()))
    im2 = axes[1].contourf(X, Y, nn2_correction_field, levels=20, cmap='RdBu_r', 
                            alpha=0.8, vmin=-vmax_corr, vmax=vmax_corr)
    axes[1].scatter(sensor_x, sensor_y, c='black', s=150, edgecolors='white', 
                    linewidth=2.5, marker='o', label='Sensors', zorder=5)
    axes[1].set_xlabel('X Coordinate (m)', fontsize=12)
    axes[1].set_ylabel('Y Coordinate (m)', fontsize=12)
    axes[1].set_title('NN2 Correction Field', fontsize=14, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.2)
    plt.colorbar(im2, ax=axes[1], label='Correction (ppb)')
    
    # Panel 3: Final PINN+NN2 field
    im3 = axes[2].contourf(X, Y, final_field, levels=20, cmap=cmap, alpha=0.8)
    axes[2].scatter(sensor_x, sensor_y, c='black', s=150, edgecolors='white', 
                    linewidth=2.5, marker='o', label='Sensors', zorder=5)
    axes[2].set_xlabel('X Coordinate (m)', fontsize=12)
    axes[2].set_ylabel('Y Coordinate (m)', fontsize=12)
    axes[2].set_title('Final PINN+NN2 Predictions', fontsize=14, fontweight='bold')
    axes[2].set_aspect('equal')
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.2)
    plt.colorbar(im3, ax=axes[2], label='Concentration (ppb)')
    
    # Add meteorology info to title
    fig.suptitle(f'Correction Field Analysis\n{representative_ts.strftime("%Y-%m-%d %H:%M")} | ' +
                 f'Wind: u={u_val:.2f} m/s, v={v_val:.2f} m/s | D={D_val:.2f}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"correction_field_{viz_idx:02d}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: correction_field_{viz_idx:02d}.png")

# Also update the main visualization with black sensors
print("\n" + "="*70)
print("UPDATING MAIN VISUALIZATION")
print("="*70)

representative_ts = selected_timestamps[0]
ts_data = jan_data[jan_data['timestamp'] == representative_ts]
met_row = met_df[met_df['t'] == representative_ts].iloc[0]

sensor_corrections = {}
sensor_pinn = {}
for sensor_id, (sx, sy) in SENSORS.items():
    sensor_row = ts_data[ts_data['sensor_id'] == sensor_id]
    if not sensor_row.empty:
        row = sensor_row.iloc[0]
        correction = row['final_pred'] - row['pinn_pred']
        sensor_corrections[(sx, sy)] = correction
        sensor_pinn[(sx, sy)] = row['pinn_pred']

sensor_x = [coords[0] for coords in sensor_corrections.keys()]
sensor_y = [coords[1] for coords in sensor_corrections.keys()]
corrections = [corr for corr in sensor_corrections.values()]
sensor_pinn_values = [pinn_val for pinn_val in sensor_pinn.values()]

rbf_pinn = Rbf(sensor_x, sensor_y, sensor_pinn_values, function='multiquadric', smooth=0.1)
pinn_field = rbf_pinn(X, Y)
pinn_field = np.maximum(pinn_field, 0.0)

rbf_interp = Rbf(sensor_x, sensor_y, corrections, function='multiquadric', smooth=0.1)
nn2_correction_field = rbf_interp(X, Y)
final_field = pinn_field + nn2_correction_field
final_field = np.maximum(final_field, 0.0)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
cmap = plt.cm.viridis

# All panels with BLACK sensors
im1 = axes[0].contourf(X, Y, pinn_field, levels=20, cmap=cmap, alpha=0.8)
axes[0].scatter(sensor_x, sensor_y, c='black', s=150, edgecolors='white', 
                linewidth=2.5, marker='o', label='Sensors', zorder=5)
axes[0].set_xlabel('X Coordinate (m)', fontsize=12)
axes[0].set_ylabel('Y Coordinate (m)', fontsize=12)
axes[0].set_title('Raw PINN Predictions', fontsize=14, fontweight='bold')
axes[0].set_aspect('equal')
axes[0].legend(loc='upper right', fontsize=10)
axes[0].grid(True, alpha=0.2)
plt.colorbar(im1, ax=axes[0], label='Concentration (ppb)')

vmax_corr = max(abs(nn2_correction_field.min()), abs(nn2_correction_field.max()))
im2 = axes[1].contourf(X, Y, nn2_correction_field, levels=20, cmap='RdBu_r', 
                        alpha=0.8, vmin=-vmax_corr, vmax=vmax_corr)
axes[1].scatter(sensor_x, sensor_y, c='black', s=150, edgecolors='white', 
                linewidth=2.5, marker='o', label='Sensors', zorder=5)
axes[1].set_xlabel('X Coordinate (m)', fontsize=12)
axes[1].set_ylabel('Y Coordinate (m)', fontsize=12)
axes[1].set_title('NN2 Correction Field', fontsize=14, fontweight='bold')
axes[1].set_aspect('equal')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.2)
plt.colorbar(im2, ax=axes[1], label='Correction (ppb)')

im3 = axes[2].contourf(X, Y, final_field, levels=20, cmap=cmap, alpha=0.8)
axes[2].scatter(sensor_x, sensor_y, c='black', s=150, edgecolors='white', 
                linewidth=2.5, marker='o', label='Sensors', zorder=5)
axes[2].set_xlabel('X Coordinate (m)', fontsize=12)
axes[2].set_ylabel('Y Coordinate (m)', fontsize=12)
axes[2].set_title('Final PINN+NN2 Predictions', fontsize=14, fontweight='bold')
axes[2].set_aspect('equal')
axes[2].legend(loc='upper right', fontsize=10)
axes[2].grid(True, alpha=0.2)
plt.colorbar(im3, ax=axes[2], label='Concentration (ppb)')

fig.suptitle(f'Spatial Correction Field Analysis\nTimestamp: {representative_ts}', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
main_output = os.path.join(BASE_DIR, "final_visualizations_q1_2021/06b_correction_field_map_full_domain.png")
plt.savefig(main_output, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Updated main visualization")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"Generated 10 correction field maps in: {OUTPUT_DIR}")
print(f"Updated main visualization: {main_output}")
print("="*70)
