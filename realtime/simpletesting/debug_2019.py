
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benzene_pipeline import BenzenePipeline, load_emissions_for_timestamp

base_dir = "/Users/neevpratap/simpletesting"
# synced folder
data_dir = "/Users/neevpratap/Desktop/madis_data_desktop_updated/synced" 
pinn_path = os.path.join(base_dir, "pinn_combined_final.pth 2")
nn2_path = os.path.join(base_dir, "nn2_master_model_spatial.pth")

print("Initializing Pipeline...")
pipeline = BenzenePipeline(pinn_path, nn2_path)

# Test Timestamp: 2019-02-14 21:00:00 (Outlier)
target_time = "2019-02-14 21:00:00"

print(f"\n--- Testing Outlier Timestamp: {target_time} ---")

# 1. Load Met Data (BASF Proxy)
csv_path = os.path.join(data_dir, "BASF_Pasadena_synced_training_data.csv")
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
    print("Met Data NOT Found!")

# 2. Load Emissions
emissions = load_emissions_for_timestamp(target_time, data_dir)
total_Q = sum(emissions.values())
print(f"Total Emissions: {total_Q}")
print(f"Emissions breakdown: {emissions}")

# 3. Test Point (Sensor 482010057 - Galena Park)
sensor_point = np.array([[-8551.1, -1698.2]])

try:
    print(f"Running PINN for Sensor at {sensor_point}...")
    # Trace PINN components? 
    # Just run it
    concs = pipeline.process_timestep(met_data, sensor_point, emissions)
    print(f"Prediction at SENSOR_482010057: {concs[0]:.6f} ppb")
except Exception as e:
    print(f"Error: {e}")
