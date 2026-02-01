
import os
import sys
import pandas as pd
import numpy as np
import torch
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benzene_pipeline import BenzenePipeline, load_emissions_for_timestamp

base_dir = "/Users/neevpratap/simpletesting"
data_dir = "/Users/neevpratap/Desktop/madis_data_desktop_updated/synced"

pinn_path = os.path.join(base_dir, "pinn_combined_final.pth 2")
nn2_path = os.path.join(base_dir, "nn2_master_model_spatial-2.pth")
scaler_path = os.path.join(base_dir, "nn2_master_scalers.pkl")

print("Initializing Pipeline...")
pipeline = BenzenePipeline(pinn_path, nn2_path, nn2_scaler_path=scaler_path)

# Test Timestamp
target_time = "2019-01-01 00:00:00"

print(f"\n--- Testing NN2 v2 for Timestamp: {target_time} ---")

# 1. Load Met Data
csv_path = os.path.join(data_dir, "BASF_Pasadena_synced_training_data.csv")
df_met = pd.read_csv(csv_path)
row = df_met[df_met['t'] == target_time]

if row.empty:
    print("Met data row empty!")
    sys.exit(1)

met_data = {
    'u': float(row['wind_u'].values[0]),
    'v': float(row['wind_v'].values[0]),
    'D': float(row['D'].values[0]),
    't': 3600.0,
    'dt_obj': pd.to_datetime(target_time)
}

print(f"Met Inputs: u={met_data['u']}, v={met_data['v']}, D={met_data['D']}")

# 2. Load Emissions
emissions = load_emissions_for_timestamp(target_time, data_dir)
print(f"Total Q: {sum(emissions.values())}")

# 3. Sensor Points (All 9)
SENSORS = [
    {'id': 'sensor_482010026', 'coords': (12809.0, 2135.3)},
    {'id': 'sensor_482010029', 'coords': (13501.5, -2393.7)},
    {'id': 'sensor_482010057', 'coords': (-8551.1, -1698.2)},
    {'id': 'sensor_482010069', 'coords': (9804.1, -11413.7)},
    {'id': 'sensor_482010071', 'coords': (1780.0, -13735.6)},
    {'id': 'sensor_482011034', 'coords': (13350.5, -1320.6)},
    {'id': 'sensor_482011035', 'coords': (4421.3, -3319.4)},
    {'id': 'sensor_482011039', 'coords': (12226.9, 10222.1)},
    {'id': 'sensor_482011015', 'coords': (-2805.0, -2145.0)},
]
sensor_points = np.array([s['coords'] for s in SENSORS])

# 4. Mock Ground Truth (non-zero)
gt = np.zeros(9) + 0.1

try:
    print("Running process_timestep...")
    preds = pipeline.process_timestep(met_data, sensor_points, emissions, ground_truth=gt)
    print(f"Predictions: {preds}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Inspect Scalers
if pipeline.nn2_scalers:
    print("\nScaler Inspections:")
    for k, v in pipeline.nn2_scalers.items():
        print(f"  Scaler '{k}': mean={v.mean_}, scale={v.scale_}")
