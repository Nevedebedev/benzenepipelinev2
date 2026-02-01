"""
Test real-time PINN computation with benchmark-matching parameters
"""

import pandas as pd
import numpy as np
import torch
from benzene_pipeline import BenzenePipeline
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

# Initialize pipeline
print("Initializing Pipeline...")
pipeline = BenzenePipeline(
    os.path.join(BASE_DIR, "pinn_combined_final.pth 2")
)

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

# Test timestamp
test_ts = pd.Timestamp('2019-01-01 13:00:00')
print(f"\nTesting timestamp: {test_ts}")

# Get benchmark values
bench_row = df_bench[df_bench['t'] == test_ts].iloc[0]
print("\nBenchmark values:")
for s in SENSORS:
    sensor_name = s['id']
    print(f"  {sensor_name}: {bench_row[sensor_name]:.4f} ppb")

# Prepare sensor points
sensor_points = np.array([[s['x'], s['y']] for s in SENSORS])

# Get meteorology (use first facility)
first_fac = list(facilities.values())[0]
met_row = first_fac[first_fac['t'] == test_ts].iloc[0]

met_data = {
    'u': float(met_row['wind_u']),
    'v': float(met_row['wind_v']),
    'D': float(met_row['D']),
    't_hours': 13.0,  # 13:00 = 13 hours from 2019-01-01 00:00
    'dt_obj': test_ts
}

print(f"\nMeteorology:")
print(f"  wind_u: {met_data['u']}")
print(f"  wind_v: {met_data['v']}")
print(f"  D: {met_data['D']}")
print(f"  t_hours: {met_data['t_hours']}")

# Get emission rates from all facilities
emissions = {}
for fname, df_fac in facilities.items():
    fac_row = df_fac[df_fac['t'] == test_ts]
    if not fac_row.empty:
        fac_row = fac_row.iloc[0]
        emissions[fname] = float(fac_row['Q_total'])

print(f"\nEmissions for {len(emissions)} facilities")

# Run PINN
print("\nRunning PINN superposition...")
raw_phi = pipeline.superimpose(met_data, sensor_points, emissions)
predicted_ppb = raw_phi * 313210039.9

# Compare
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
for i, s in enumerate(SENSORS):
    sensor_name = s['id']
    bench_val = bench_row[sensor_name]
    pred_val = predicted_ppb[i]
    error = abs(pred_val - bench_val)
    
    print(f"\n{sensor_name}:")
    print(f"  Predicted:  {pred_val:.4f} ppb")
    print(f"  Benchmark:  {bench_val:.4f} ppb")
    print(f"  Error:      {error:.4f} ppb")
    
    if error < 1.0:
        print(f"  ✓ MATCH!")
    elif error < 5.0:
        print(f"  ~ Close")
    else:
        print(f"  ✗ Large error")

# Overall error
overall_error = np.mean([abs(predicted_ppb[i] - bench_row[SENSORS[i]['id']]) for i in range(len(SENSORS))])
print(f"\nOverall Mean Error: {overall_error:.4f} ppb")
