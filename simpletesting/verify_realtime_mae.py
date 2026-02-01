
import pandas as pd
import numpy as np

gt_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/sensors_final_synced.csv"
pinn_drive_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/total_superimposed_realtime_concentrations.csv"

df_gt = pd.read_csv(gt_path)
df_pinn = pd.read_csv(pinn_drive_path)

# Merge on time 't'
df_gt['t'] = pd.to_datetime(df_gt['t'])
df_pinn['t'] = pd.to_datetime(df_pinn['t'])

# Handle Long Format PINN Data
# pinn should be grouped by t and sensor_id, or we can pivot it
pinn_pivot = df_pinn.pivot(index='t', columns='sensor_id', values='total_phi_ppb')
# Match sensor ID names in gt (sensor_482010026) to numeric IDs in pinn (482010026.0)
pinn_pivot.columns = [f"sensor_{int(c)}" for c in pinn_pivot.columns]

merged = pd.merge(df_gt, pinn_pivot, on='t', suffixes=('_gt', '_pinn'))

sensors = [col for col in df_gt.columns if col.startswith('sensor_')]

maes = []
for s in sensors:
    gt_col = f"{s}_gt"
    pinn_col = f"{s}_pinn"
    if pinn_col in merged.columns:
        valid = merged[[gt_col, pinn_col]].dropna()
        if not valid.empty:
            diff = (valid[gt_col] - valid[pinn_col]).abs()
            mae = diff.mean()
            maes.append(mae)
            print(f"Sensor {s}: Realtime Drive PINN MAE = {mae:.4f}")

if maes:
    print(f"\nOverall Average Realtime Drive PINN MAE: {np.mean(maes):.4f}")
