"""
Test superposition of ALL 20 facilities to reproduce benchmark
"""

import pandas as pd
import numpy as np
import torch
from pinn import ParametricADEPINN
import glob
import os

NN2_DATA_DIR = "/Users/neevpratap/simpletesting/nn2trainingdata"
BASE_DIR = "/Users/neevpratap/simpletesting"

# Load PINN model
print("Loading PINN model...")
pinn_path = os.path.join(BASE_DIR, "pinn_combined_final.pth 2")
pinn = ParametricADEPINN()
checkpoint = torch.load(pinn_path, map_location='cpu', weights_only=False)
pinn.load_state_dict(checkpoint['model_state_dict'])
pinn.eval()

# Load benchmark
print("Loading benchmark...")
df_bench = pd.read_csv(os.path.join(NN2_DATA_DIR, "total_superimposed_concentrations.csv"))
df_bench['t'] = pd.to_datetime(df_bench['t'])

# Load sensor coordinates (OLD system)
df_coords = pd.read_csv(os.path.join(NN2_DATA_DIR, "sensor_coordinates.csv"))

# Load ALL facility data
print("Loading all facility CSVs...")
facilities = {}
for fpath in glob.glob(os.path.join(NN2_DATA_DIR, "*_synced_training_data.csv")):
    fname = os.path.basename(fpath).replace("_synced_training_data.csv", "")
    df = pd.read_csv(fpath)
    df['t'] = pd.to_datetime(df['t'])
    facilities[fname] = df
    print(f"  {fname}")

# Test timestamp
test_ts = pd.Timestamp('2019-01-01 13:00:00')
print(f"\nTesting timestamp: {test_ts}")

# Get benchmark values
bench_row = df_bench[df_bench['t'] == test_ts].iloc[0]
sensor_names = [
    'sensor_482010026', 'sensor_482010057', 'sensor_482010069',
    'sensor_482010617', 'sensor_482010803', 'sensor_482011015',
    'sensor_482011035', 'sensor_482011039', 'sensor_482016000'
]

print("\nBenchmark values (ppb):")
bench_ppb = []
for s in sensor_names:
    val = bench_row[s]
    bench_ppb.append(val)
    print(f"  {s}: {val:.4f}")

# Get meteorology from first facility
first_fac = list(facilities.values())[0]
met_row = first_fac[first_fac['t'] == test_ts].iloc[0]
u = float(met_row['wind_u'])
v = float(met_row['wind_v'])
D = float(met_row['D'])

print(f"\nMeteorology:")
print(f"  wind_u: {u}")
print(f"  wind_v: {v}")
print(f"  D: {D}")

# Superimpose ALL facilities
print("\n" + "="*70)
print("SUPERIMPOSING ALL 20 FACILITIES")
print("="*70)

# Prepare sensor coordinates as tensors
sensor_coords = df_coords[['x', 'y']].values

# For each sensor
total_phi_per_sensor = np.zeros(len(sensor_names))

for fac_name, df_fac in facilities.items():
    fac_row = df_fac[df_fac['t'] == test_ts]
    if fac_row.empty:
        continue
    
    fac_row = fac_row.iloc[0]
    cx = float(fac_row['source_x_cartesian'])
    cy = float(fac_row['source_y_cartesian'])
    Q = float(fac_row['Q_total'])
    
    print(f"\n{fac_name}:")
    print(f"  coords: ({cx}, {cy}), Q: {Q}")
    
    # Run PINN for this facility at all sensors
    for i, (sensor_x, sensor_y) in enumerate(sensor_coords):
        x_tensor = torch.tensor([[sensor_x]], dtype=torch.float32)
        y_tensor = torch.tensor([[sensor_y]], dtype=torch.float32)
        t_tensor = torch.tensor([[3600.0]], dtype=torch.float32)
        cx_tensor = torch.tensor([[cx]], dtype=torch.float32)
        cy_tensor = torch.tensor([[cy]], dtype=torch.float32)
        u_tensor = torch.tensor([[u]], dtype=torch.float32)
        v_tensor = torch.tensor([[v]], dtype=torch.float32)
        d_tensor = torch.tensor([[D]], dtype=torch.float32)
        kappa_tensor = torch.tensor([[0.05]], dtype=torch.float32)
        Q_tensor = torch.tensor([[Q]], dtype=torch.float32)
        
        with torch.no_grad():
            phi = pinn(x_tensor, y_tensor, t_tensor, cx_tensor, cy_tensor,
                      u_tensor, v_tensor, d_tensor, kappa_tensor, Q_tensor,
                      normalize=True)
        
        total_phi_per_sensor[i] += phi.item()
    
    # Print running total for first sensor
    print(f"  Running total phi (sensor 0): {total_phi_per_sensor[0]:.4e}")

# Convert to PPB
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

for factor_exp in range(6, 11):
    for mantissa in [1, 2, 3, 5]:
        factor = mantissa * (10 ** factor_exp)
        predicted_ppb = total_phi_per_sensor * factor
        
        # Calculate error
        error = np.array([abs(predicted_ppb[i] - bench_ppb[i]) for i in range(len(sensor_names))])
        mean_error = error.mean()
        
        if mean_error < 1.0:
            print(f"\nConversion factor: {factor:.2e}")
            print(f"Mean error: {mean_error:.4f} ppb")
            print("\nSensor-by-sensor comparison:")
            for i, s in enumerate(sensor_names):
                print(f"  {s}:")
                print(f"    Predicted: {predicted_ppb[i]:.4f}, Benchmark: {bench_ppb[i]:.4f}, Error: {error[i]:.4f}")
