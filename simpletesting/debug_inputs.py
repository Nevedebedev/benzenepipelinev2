
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benzene_pipeline import load_emissions_for_timestamp

data_dir = "/Users/neevpratap/simpletesting/training_data_2021_full_jan"
target_time = "2021-01-01 12:00:00"

print(f"--- Debugging for {target_time} ---")

# 1. Test Emission Loading
print("\n1. Testing Emission Loading:")
try:
    emissions = load_emissions_for_timestamp(target_time, data_dir)
    total_Q = sum(emissions.values())
    print(f"  Total Emissions (Q): {total_Q}")
    print(f"  Non-zero sources: {[k for k, v in emissions.items() if v > 0]}")
except Exception as e:
    print(f"  FAILED: {e}")

# 2. Test Met Data Loading (Manually mimicking pipeline logic)
print("\n2. Testing Met Data Loading:")
found_met = False
FACILITIES_NAMES = ["BASF_Pasadena", "Chevron_Phillips_Chemical_Co"] # Just check a few
for name in FACILITIES_NAMES:
    csv_path = os.path.join(data_dir, f"{name}_training_data.csv")
    if os.path.exists(csv_path):
        print(f"  Found file for {name}")
        try:
            df = pd.read_csv(csv_path)
            # print(df.head())
            row = df[df['t'] == target_time]
            if not row.empty:
                print(f"  Found row for time {target_time}:")
                print(f"    u: {row['wind_u'].values[0]}")
                print(f"    v: {row['wind_v'].values[0]}")
                print(f"    D: {row['D'].values[0]}")
                found_met = True
                break
            else:
                print(f"  DATA MISSING for time {target_time} in {name}")
        except Exception as e:
            print(f"  Error reading {name}: {e}")
    else:
        print(f"  File not found: {csv_path}")

if not found_met:
    print("  CRITICAL: Could not find met data for this timestamp.")
