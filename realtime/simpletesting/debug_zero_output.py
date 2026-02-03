import pandas as pd
import torch
from benzene_pipeline import BenzenePipeline
import numpy as np
import glob

DRIVE_DIR = '/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001'

# Load facilities
facilities = {}
for f in glob.glob(DRIVE_DIR + '/*_training_data.csv'):
    fname = f.split('/')[-1].replace('_synced_training_data.csv', '')
    df = pd.read_csv(f)
    df['t'] = pd.to_datetime(df['t'])
    facilities[fname] = df

# Initialize pipeline
pipeline = BenzenePipeline(
    '/Users/neevpratap/simpletesting/pinn_combined_final.pth 2',
    '/Users/neevpratap/simpletesting/nn2_master_model_spatial.pth'
)

# Test timestamp
ts = pd.Timestamp('2019-01-01 00:00:00')

# Get emissions
emissions = {}
for fname, df in facilities.items():
    row = df[df['t'] == ts]
    if not row.empty:
        Q = float(row.iloc[0]['Q_total'])
        emissions[fname] = Q

print(f"Number of emissions: {len(emissions)}")
print(f"Emissions sample: {list(emissions.items())[:3]}")
print(f"Sum of Q: {sum(emissions.values())}")

# Sensor points
sensor_points = np.array([[13970.0, 19920.0], [3020.0, 12330.0]])

# Met data
met_data = {
    'u': -0.21,
    'v': 1.32,
    'D': 1.0,
    't': 3600.0
}

print(f"\nCalling superimpose...")
raw = pipeline.superimpose(met_data, sensor_points, emissions)
print(f"Raw phi output: {raw}")
print(f"PPB output: {raw * 313210039.9}")
