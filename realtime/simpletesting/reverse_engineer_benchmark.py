"""
Reverse-engineer the benchmark generation process

This script will try different coordinate systems and emission sources
to see which configuration reproduces the pre-computed PINN values.
"""

import pandas as pd
import numpy as np
import torch
from pinn import ParametricADEPINN
import os

# Paths
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
print("Loading benchmark PINN predictions...")
df_bench = pd.read_csv(os.path.join(NN2_DATA_DIR, "total_superimposed_concentrations.csv"))
df_bench['t'] = pd.to_datetime(df_bench['t'])

# Load sensor coordinates (OLD system)
df_coords = pd.read_csv(os.path.join(NN2_DATA_DIR, "sensor_coordinates.csv"))

# Test timestamp
test_ts = pd.Timestamp('2019-01-01 13:00:00')
print(f"\nTesting timestamp: {test_ts}")

# Get benchmark values
bench_row = df_bench[df_bench['t'] == test_ts]
if bench_row.empty:
    print("Timestamp not found in benchmark!")
    exit()

bench_vals = bench_row.iloc[0]
sensor_names = [
    'sensor_482010026', 'sensor_482010057', 'sensor_482010069',
    'sensor_482010617', 'sensor_482010803', 'sensor_482011015',
    'sensor_482011035', 'sensor_482011039', 'sensor_482016000'
]

print("\nBenchmark PINN values (ppb):")
for s in sensor_names:
    print(f"  {s}: {bench_vals[s]:.4f}")

# Load facility data to get emission rates and met
facility_file = os.path.join(NN2_DATA_DIR, "BASF_Pasadena_synced_training_data.csv")
df_fac = pd.read_csv(facility_file)
df_fac['t'] = pd.to_datetime(df_fac['t'])
fac_row = df_fac[df_fac['t'] == test_ts].iloc[0]

print(f"\nMeteorology:")
print(f"  wind_u: {fac_row['wind_u']}")
print(f"  wind_v: {fac_row['wind_v']}")
print(f"  D: {fac_row['D']}")
print(f"\nFacility info (from CSV):")
print(f"  source_x_cartesian: {fac_row['source_x_cartesian']}")
print(f"  source_y_cartesian: {fac_row['source_y_cartesian']}")
print(f"  Q_total: {fac_row['Q_total']}")
print(f"  diameter: {fac_row['source_diameter']}")

# Try to reproduce PINN output for ONE facility at ONE sensor
sensor_idx = 0  # sensor_482010026
sensor_x = df_coords.iloc[sensor_idx]['x']
sensor_y = df_coords.iloc[sensor_idx]['y']

print(f"\nTest sensor: {sensor_names[sensor_idx]}")
print(f"  coordinates: ({sensor_x}, {sensor_y})")

# Try different configurations
print("\n" + "="*70)
print("TESTING DIFFERENT COORDINATE CONFIGURATIONS")
print("="*70)

# Config 1: Use PINN with OLD facility coordinates from CSV
print("\n[Config 1] Using OLD facility coordinates from CSV")
cx = float(fac_row['source_x_cartesian'])
cy = float(fac_row['source_y_cartesian'])
Q = float(fac_row['Q_total'])
D = float(fac_row['D'])
u = float(fac_row['wind_u'])
v = float(fac_row['wind_v'])
kappa = 0.05
diameter = float(fac_row['source_diameter'])

# Create tensors
x_tensor = torch.tensor([[sensor_x]], dtype=torch.float32)
y_tensor = torch.tensor([[sensor_y]], dtype=torch.float32)
t_tensor = torch.tensor([[3600.0]], dtype=torch.float32)
cx_tensor = torch.tensor([[cx]], dtype=torch.float32)
cy_tensor = torch.tensor([[cy]], dtype=torch.float32)
u_tensor = torch.tensor([[u]], dtype=torch.float32)
v_tensor = torch.tensor([[v]], dtype=torch.float32)
d_tensor = torch.tensor([[D]], dtype=torch.float32)
kappa_tensor = torch.tensor([[kappa]], dtype=torch.float32)
Q_tensor = torch.tensor([[Q]], dtype=torch.float32)

with torch.no_grad():
    phi = pinn(x_tensor, y_tensor, t_tensor, cx_tensor, cy_tensor,
              u_tensor, v_tensor, d_tensor, kappa_tensor, Q_tensor,
              normalize=True)

phi_val = phi.item()
ppb_val = phi_val * 313210039.9

print(f"  Raw phi: {phi_val:.4e}")
print(f"  PPB (×313210039.9): {ppb_val:.4f}")
print(f"  Benchmark PPB: {bench_vals[sensor_names[sensor_idx]]:.4f}")
print(f"  Difference: {abs(ppb_val - bench_vals[sensor_names[sensor_idx]]):.4f}")

# Try different conversion factors
print("\n[Testing different conversion factors]")
for factor_exp in [6, 7, 8, 9, 10]:
    for mantissa in [1, 2, 3, 5]:
        factor = mantissa * (10 ** factor_exp)
        ppb_test = phi_val * factor
        diff = abs(ppb_test - bench_vals[sensor_names[sensor_idx]])
        if diff < 1.0:
            print(f"  Factor {factor:.2e}: PPB={ppb_test:.4f}, diff={diff:.4f} ✓")

# Try with Q scaled
print("\n[Testing different Q scaling]")
for q_scale in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    Q_test = Q * q_scale
    Q_tensor_test = torch.tensor([[Q_test]], dtype=torch.float32)
    
    with torch.no_grad():
        phi = pinn(x_tensor, y_tensor, t_tensor, cx_tensor, cy_tensor,
                  u_tensor, v_tensor, d_tensor, kappa_tensor, Q_tensor_test,
                  normalize=True)
    
    phi_val = phi.item()
    ppb_val = phi_val * 313210039.9
    diff = abs(ppb_val - bench_vals[sensor_names[sensor_idx]])
    
    if diff < 1.0:
        print(f"  Q×{q_scale}: phi={phi_val:.4e}, PPB={ppb_val:.4f}, diff={diff:.4f} ✓")
