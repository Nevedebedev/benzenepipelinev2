"""
Test using facility coordinates FROM the CSVs themselves
"""

import pandas as pd
import numpy as np
import torch
from benzene_pipeline import BenzenePipeline, FACILITIES
import glob
import os

DRIVE_DIR = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001"
BASE_DIR = "/Users/neevpratap/simpletesting"

# Sensor coordinates from realtime CSV
SENSORS = [
    {'id': 'sensor_482010026', 'x': 13970.0, 'y': 19920.0},
    {'id': 'sensor_482010057', 'x':3020.0, 'y': 12330.0},
    {'id': 'sensor_482010069', 'x': 820.0, 'y': 9220.0},
    {'id': 'sensor_482010617', 'x': 27050.0, 'y': 22050.0},
    {'id': 'sensor_482010803', 'x': 8840.0, 'y': 15720.0},
    {'id': 'sensor_482011015', 'x': 18410.0, 'y': 15070.0},
    {'id': 'sensor_482011035', 'x': 1160.0, 'y': 12270.0},
    {'id': 'sensor_482011039', 'x': 13660.0, 'y': 5190.0},
    {'id': 'sensor_482016000', 'x': 1550.0, 'y': 6790.0}
]

# Load facilities and extract their coordinates
print("Loading facility CSVs and extracting coordinates...")
facility_files = glob.glob(os.path.join(DRIVE_DIR, "*_synced_training_data.csv"))
facility_data = {}

for fpath in facility_files:
    fname = os.path.basename(fpath).replace("_synced_training_data.csv", "")
    df = pd.read_csv(fpath)
    df['t'] = pd.to_datetime(df['t'])
    
    # Get coordinates from first row
    first_row = df.iloc[0]
    coords = (float(first_row['source_x_cartesian']), float(first_row['source_y_cartesian']))
    diameter = float(first_row['source_diameter'])
    
    facility_data[fname] = {
        'df': df,
        'coords': coords,
        'diameter': diameter
    }
    
    print(f"  {fname}: coords={coords}, diameter={diameter}")

# Compare with FACILITIES in benzene_pipeline
print("\n Comparing with pipeline FACILITIES...")
for fac in FACILITIES:
    if fac['name'] in facility_data:
        csv_coords = facility_data[fac['name']]['coords']
        pipeline_coords = fac['coords']
        diff = np.sqrt((csv_coords[0] - pipeline_coords[0])**2 + (csv_coords[1] - pipeline_coords[1])**2)
        print(f"{fac['name']}:")
        print(f"  CSV coords: {csv_coords}")
        print(f"  Pipeline coords: {pipeline_coords}")
        print(f"  Distance: {diff:.2f} meters")
