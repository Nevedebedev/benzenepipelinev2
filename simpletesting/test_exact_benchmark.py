"""
Test real-time PINN with EXACT benchmark methodology:
- Per-facility meteorology (each facility has its own wind/D)
- Facility coordinates from CSVs
- NEW sensor coordinates
"""

import pandas as pd
import numpy as np
import torch
from pinn import ParametricADEPINN
import glob
import os

NN2_DATA_DIR = "/Users/neevpratap/simpletesting/nn2trainingdata"
BASE_DIR = "/Users/neevpratap/simpletesting"

# NEW sensor coordinates from benchmark script
SENSORS = [
    {'id': 'sensor_482010026', 'x': 13972.62, 'y': 19915.57},
    {'id': 'sensor_482010057', 'x': 3017.18, 'y': 12334.2},
    {'id': 'sensor_482010069', 'x': 817.42, 'y': 9218.92},
    {'id': 'sensor_482010617', 'x': 27049.57, 'y': 22045.66},
    {'id': 'sensor_482010803', 'x': 8836.35, 'y': 15717.2},
    {'id': 'sensor_482011015', 'x': 18413.8, 'y': 15068.96},
    {'id': 'sensor_482011035', 'x': 1159.98, 'y': 12272.52},
    {'id': 'sensor_482011039', 'x': 13661.93, 'y': 5193.24},
    {'id': 'sensor_482016000', 'x': 1546.9, 'y': 6786.33}
]

# Load PINN model with benchmark ranges
print("Loading PINN model...")
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
print("✓ PINN loaded with benchmark ranges")

# Load benchmark
print("\nLoading benchmark...")
df_bench = pd.read_csv(os.path.join(NN2_DATA_DIR, "total_superimposed_concentrations.csv"))
df_bench['t'] = pd.to_datetime(df_bench['t'])

# Load facility data
print("Loading facility CSVs...")
facilities = {}
for fpath in glob.glob(os.path.join(NN2_DATA_DIR, "*_synced_training_data.csv")):
    fname = os.path.basename(fpath).replace("_synced_training_data.csv", "")
    df = pd.read_csv(fpath)
    df['t'] = pd.to_datetime(df['t'])
    facilities[fname] = df

print(f"Loaded {len(facilities)} facilities")

# Test timestamp
test_ts = pd.Timestamp('2019-01-01 13:00:00')
t_start = pd.to_datetime('2019-01-01 00:00:00')
t_hours = (test_ts - t_start).total_seconds() / 3600.0

print(f"\nTesting timestamp: {test_ts} (t_hours={t_hours})")

# Get benchmark values
bench_row = df_bench[df_bench['t'] == test_ts].iloc[0]

# Superimpose predictions across all facilities (BENCHMARK METHOD)
print("\nRunning PINN superposition (per-facility met)...")
predictions_per_sensor = {s['id']: 0.0 for s in SENSORS}

for fac_name, df_fac in facilities.items():
    fac_row = df_fac[df_fac['t'] == test_ts]
    if fac_row.empty:
        continue
    
    fac_row = fac_row.iloc[0]
    
    # Facility parameters
    cx = float(fac_row['source_x_cartesian'])
    cy = float(fac_row['source_y_cartesian']) 
    d = float(fac_row['source_diameter'])
    Q = float(fac_row['Q_total'])
    
    # Per-facility meteorology
    u = float(fac_row['wind_u'])
    v = float(fac_row['wind_v'])
    kappa = float(fac_row['D'])  # Use D as kappa
    
    # Predict for all sensors with THIS facility's met
    for sensor in SENSORS:
        sensor_x = sensor['x']
        sensor_y = sensor['y']
        
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
        
        # Accumulate (superposition principle)
        phi_val = phi.item()
        predictions_per_sensor[sensor['id']] += phi_val

# Convert to PPB
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

errors = []
for sensor in SENSORS:
    sensor_id = sensor['id']
    raw_phi = predictions_per_sensor[sensor_id]
    pred_ppb = raw_phi * 3.13e8
    bench_ppb = bench_row[sensor_id]
    error = abs(pred_ppb - bench_ppb)
    errors.append(error)
    
    match_str = "✓ MATCH!" if error < 0.5 else ("~ Close" if error < 2.0 else "✗ Error")
    
    print(f"\n{sensor_id}:")
    print(f"  Raw phi:    {raw_phi:.6e}")
    print(f"  Predicted:  {pred_ppb:.4f} ppb")
    print(f"  Benchmark:  {bench_ppb:.4f} ppb")
    print(f"  Error:      {error:.4f} ppb  {match_str}")

print(f"\n{'='*70}")
print(f"Overall Mean Error: {np.mean(errors):.4f} ppb")
print(f"Max Error: {np.max(errors):.4f} ppb")
print(f"{'='*70}")
