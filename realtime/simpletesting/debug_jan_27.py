
import os
import sys
import pandas as pd
import torch
import numpy as np

# Add current path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benzene_pipeline import BenzenePipeline, load_emissions_for_timestamp

# Config
base_dir = "/Users/neevpratap/simpletesting"
data_dir = os.path.join(base_dir, "training_data_2021_full_jan")
pinn_path = os.path.join(base_dir, "pinn_combined_final.pth 2")
nn2_path = os.path.join(base_dir, "nn2_master_model_spatial.pth")

# Initialize Pipeline
print("Initializing Pipeline...")
try:
    pipeline = BenzenePipeline(pinn_path, nn2_path)
    print("Pipeline initialized.")
except Exception as e:
    print(f"Failed to init pipeline: {e}")
    sys.exit(1)

# Target Timestamp (Known valid data from checking csv)
target_time = "2021-01-27 14:00:00"
print(f"\n--- Testing Specific Timestamp: {target_time} ---")

# 1. Manually Load Met Data
csv_path = os.path.join(data_dir, "BASF_Pasadena_training_data.csv")
df_met = pd.read_csv(csv_path)
row = df_met[df_met['t'] == target_time]

met_data = {'u': 0.0, 'v': 0.0, 'D': 1.0, 't': 3600.0}
if not row.empty:
    print("Found Met Data Row:")
    print(row[['wind_u', 'wind_v', 'D']].to_string())
    met_data['u'] = float(row['wind_u'].values[0])
    met_data['v'] = float(row['wind_v'].values[0])
    met_data['D'] = float(row['D'].values[0])
else:
    print("CRITICAL: Met data row not found!")

print(f"Met Data Object: {met_data}")

# 2. Load Emissions
emissions = load_emissions_for_timestamp(target_time, data_dir)
total_Q = sum(emissions.values())
print(f"Total Emissions: {total_Q}")

# 3. Define Sensor Points (Cartesian Coordinates from run_january_loop.py / benzene_pipeline.py)
# BASF Pasadena Source: (4775.0, -2392.0)
# Sensor 482010026: (12809.0, 2135.3)

test_points = {
    "NEAR_BASF": (4775.0, -2392.0),
    "NEAR_BASF_OFFSET_100m": (4875.0, -2392.0), # 100m East
    "SENSOR_482010026": (12809.0, 2135.3)
}

print("\n--- Running Inference ---")
for name, (cx, cy) in test_points.items():
    # Construct numpy array for grid_points
    points_array = np.array([[cx, cy]])
    
    try:
        concs = pipeline.process_timestep(met_data, points_array, emissions)
        print(f"Point {name} ({cx}, {cy}): {concs[0]:.6f} ppb")
    except Exception as e:
        print(f"Error at {name}: {e}")
        import traceback
        traceback.print_exc()
