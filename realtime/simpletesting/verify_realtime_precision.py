
import pandas as pd
import numpy as np

gt_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/sensors_final_synced.csv"
realtime_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/total_superimposed_realtime_concentrations.csv"

df_gt = pd.read_csv(gt_path)
df_real = pd.read_csv(realtime_path)

# Prepare GT for narrow merge
df_gt['t'] = pd.to_datetime(df_gt['t'])
gt_melted = df_gt.melt(id_vars=['t'], var_name='sensor_id', value_name='gt_ppb')
# Clean sensor_id "sensor_482010026" -> "482010026"
gt_melted['sensor_id'] = gt_melted['sensor_id'].str.replace('sensor_', '').astype(float)

# Prepare Realtime
df_real['t'] = pd.to_datetime(df_real['t'])
df_real['sensor_id'] = df_real['sensor_id'].astype(float)

merged = pd.merge(gt_melted, df_real[['t', 'sensor_id', 'total_phi_ppb']], on=['t', 'sensor_id'])

# Calculate MAE
merged['err'] = (merged['gt_ppb'] - merged['total_phi_ppb']).abs()
overall_mae = merged['err'].mean()

print(f"Overall Realtime Dataset MAE: {overall_mae:.4f} ppb")
print(f"Number of samples: {len(merged)}")

# per sensor
per_sensor = merged.groupby('sensor_id')['err'].mean()
print("\nMAE per sensor:")
print(per_sensor)
