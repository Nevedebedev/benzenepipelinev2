
import pandas as pd
import numpy as np

gt_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/sensors_final_synced.csv"
pinn_drive_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/total_superimposed_concentrations.csv"

df_gt = pd.read_csv(gt_path)
df_pinn = pd.read_csv(pinn_drive_path)

# Merge on time 't'
df_gt['t'] = pd.to_datetime(df_gt['t'])
df_pinn['t'] = pd.to_datetime(df_pinn['t'])

merged = pd.merge(df_gt, df_pinn, on='t', suffixes=('_gt', '_pinn'))

sensors = [col for col in df_gt.columns if col.startswith('sensor_')]

maes = []
for s in sensors:
    gt_col = f"{s}_gt"
    pinn_col = f"{s}_pinn"
    if pinn_col in merged.columns:
        diff = (merged[gt_col] - merged[pinn_col]).abs()
        mae = diff.mean()
        maes.append(mae)
        print(f"Sensor {s}: Drive PINN MAE = {mae:.4f}")

if maes:
    print(f"\nOverall Average Drive PINN MAE: {np.mean(maes):.4f}")
