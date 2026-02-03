"""
Run PINN prediction on February 2021 data using EXACT benchmark methodology
"""

import pandas as pd
import numpy as np
import torch
from pinn import ParametricADEPINN
import glob
import os
from tqdm import tqdm

# Paths
FEB_2021_DIR = "/Users/neevpratap/Desktop/madis_data_desktop_updated/training_data_2021_feb_REPAIRED"
OUTPUT_DIR = "/Users/neevpratap/simpletesting"
BASE_DIR = "/Users/neevpratap/simpletesting"

# Sensor coordinates from spec
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

print("="*70)
print("PINN PREDICTION - FEBRUARY 2021")
print("="*70)

# Load PINN model
print("\n[1/5] Loading PINN model...")
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
print("  ✓ PINN loaded with benchmark normalization ranges")

# Load facility data
print("\n[2/5] Loading facility CSVs...")
facilities = {}
for fpath in glob.glob(os.path.join(FEB_2021_DIR, "*_training_data.csv")):
    fname = os.path.basename(fpath).replace("_training_data.csv", "")
    df = pd.read_csv(fpath)
    df['t'] = pd.to_datetime(df['t'])
    facilities[fname] = df
    print(f"  Loaded {fname}: {len(df)} timestamps")

print(f"  ✓ Loaded {len(facilities)} facilities")

# Get all unique timestamps
all_timestamps = set()
for df_fac in facilities.values():
    all_timestamps.update(df_fac['t'].tolist())
all_timestamps = sorted(list(all_timestamps))

print(f"\n[3/5] Processing {len(all_timestamps)} timestamps...")
print(f"  Time range: {all_timestamps[0]} to {all_timestamps[-1]}")

# Reference time for t_hours calculation
t_start = pd.to_datetime('2021-01-01 00:00:00')

# Store all predictions
all_predictions = []

# Process each timestamp
for ts_idx, timestamp in enumerate(tqdm(all_timestamps, desc="  Progress")):
    # Calculate t_hours from 2021-01-01 00:00:00
    t_hours = (timestamp - t_start).total_seconds() / 3600.0
    
    # Superimpose predictions from all facilities
    predictions_per_sensor = {sid: 0.0 for sid in SENSORS.keys()}
    
    for fac_name, df_fac in facilities.items():
        fac_row = df_fac[df_fac['t'] == timestamp]
        if fac_row.empty:
            continue
        
        fac_row = fac_row.iloc[0]
        
        # Facility parameters
        cx = float(fac_row['source_x'])
        cy = float(fac_row['source_y'])
        d = float(fac_row['source_diameter'])
        Q = float(fac_row['Q_total'])
        
        # Meteorology
        u = float(fac_row['wind_u'])
        v = float(fac_row['wind_v'])
        kappa = float(fac_row['D'])  # Use D as kappa
        
        # Predict for all sensors
        for sensor_id, (sensor_x, sensor_y) in SENSORS.items():
            # Prepare tensors
            x_t = torch.tensor([[sensor_x]], dtype=torch.float32)
            y_t = torch.tensor([[sensor_y]], dtype=torch.float32)
            t_t = torch.tensor([[t_hours]], dtype=torch.float32)
            cx_t = torch.tensor([[cx]], dtype=torch.float32)
            cy_t = torch.tensor([[cy]], dtype=torch.float32)
            u_t = torch.tensor([[u]], dtype=torch.float32)
            v_t = torch.tensor([[v]], dtype=torch.float32)
            d_t = torch.tensor([[d]], dtype=torch.float32)
            kappa_t = torch.tensor([[kappa]], dtype=torch.float32)
            Q_t = torch.tensor([[Q]], dtype=torch.float32)
            
            # PINN prediction
            with torch.no_grad():
                phi = pinn(x_t, y_t, t_t, cx_t, cy_t, u_t, v_t, d_t, kappa_t, Q_t, normalize=True)
            
            # Accumulate (superposition)
            predictions_per_sensor[sensor_id] += phi.item()
    
    # Convert to PPB and store
    for sensor_id, raw_phi in predictions_per_sensor.items():
        ppb_val = max(raw_phi * 3.13e8, 0.0)  # Clip negative values
        all_predictions.append({
            'timestamp': timestamp,
            'sensor_id': sensor_id,
            'predicted_concentration': ppb_val
        })

# Create DataFrame
print("\n[4/5] Aggregating predictions...")
df_predictions = pd.DataFrame(all_predictions)

# Pivot to wide format
wide_df = df_predictions.pivot(
    index='timestamp',
    columns='sensor_id',
    values='predicted_concentration'
)
wide_df.columns = [f'sensor_{col}' for col in wide_df.columns]
wide_df = wide_df.reset_index().rename(columns={'timestamp': 't'})

# Apply +3 hour time-shift (forecast alignment)
print("  ⏰ Applying +3 hour forecast time-shift...")
wide_df['t'] = wide_df['t'] + pd.Timedelta(hours=3)

# Sort by timestamp
wide_df = wide_df.sort_values('t')

# Reorder columns
column_order = ['t'] + [f'sensor_{sid}' for sid in sorted(SENSORS.keys())]
wide_df = wide_df[column_order]

# Save
output_path = os.path.join(OUTPUT_DIR, "sensors_pinn_benzene_ppb_2021_feb.csv")
print(f"\n[5/5] Saving results...")
wide_df.to_csv(output_path, index=False)

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Output file: {output_path}")
print(f"Date range: {wide_df['t'].min()} to {wide_df['t'].max()}")
print(f"Total timestamps: {len(wide_df)}")
print(f"Total predictions: {len(all_predictions)}")
print("\nMean concentrations (ppb):")
for col in sorted([c for c in wide_df.columns if c.startswith('sensor_')]):
    print(f"  {col}: {wide_df[col].mean():.4f}")
print("="*70)
