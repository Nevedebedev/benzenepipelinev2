"""
Apply NN2 correction to February 2021 PINN predictions and evaluate
"""

import pandas as pd
import numpy as np
import torch
import os
from benzene_pipeline import BenzenePipeline

BASE_DIR = "/Users/neevpratap/simpletesting"
MADIS_DIR = "/Users/neevpratap/Desktop/madis_data_desktop_updated"

# Load PINN predictions
print("Loading PINN predictions...")
pinn_path = os.path.join(BASE_DIR, "sensors_pinn_benzene_ppb_2021_feb.csv")
df_pinn = pd.read_csv(pinn_path)
df_pinn['t'] = pd.to_datetime(df_pinn['t'])

# Load ground truth
print("Loading ground truth...")
gt_path = os.path.join(MADIS_DIR, "results_2021/sensors_actual_wide_2021_full_feb.csv")
df_gt = pd.read_csv(gt_path)
df_gt['t'] = pd.to_datetime(df_gt['t'])

# Load meteorology
print("Loading meteorology...")
met_path = os.path.join(MADIS_DIR, "training_data_2021_feb_REPAIRED/BASF_Pasadena_training_data.csv")
df_met = pd.read_csv(met_path)
df_met['t'] = pd.to_datetime(df_met['t'])

# Initialize pipeline with NN2
print("\nInitializing NN2...")
pipeline = BenzenePipeline(
    os.path.join(BASE_DIR, "pinn_combined_final.pth 2"),
    os.path.join(BASE_DIR, "nn2_master_model_spatial.pth"),
    os.path.join(BASE_DIR, "nn2_master_scalers.pkl")
)

sensor_names = [
    'sensor_482010026', 'sensor_482010057', 'sensor_482010069',
    'sensor_482010617', 'sensor_482010803', 'sensor_482011015',
    'sensor_482011035', 'sensor_482011039', 'sensor_482016000'
]

# Merge data
print("\nMerging datasets...")
df_merged = df_pinn.merge(df_gt, on='t', how='inner', suffixes=('_pinn', '_gt'))
df_merged = df_merged.merge(df_met[['t', 'wind_u', 'wind_v', 'D']], on='t', how='inner')

print(f"Processing {len(df_merged)} timestamps...")

results = []

for idx, row in df_merged.iterrows():
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{len(df_merged)}")
    
    # Get PINN predictions
    pinn_ppb = np.array([row[s + '_pinn'] for s in sensor_names])
    
    # Get ground truth
    gt_ppb = np.array([row[s + '_gt'] for s in sensor_names])
    
    # Meteorological data
    met_data = {
        'u': row['wind_u'],
        'v': row['wind_v'],
        'D': row['D'],
        't': 3600.0,
        'dt_obj': pd.to_datetime(row['t'])
    }
    
    # Apply NN2 correction
    final_pred = pipeline.apply_nn2_correction(pinn_ppb, gt_ppb, met_data)
    
    # Store results
    for i, sensor in enumerate(sensor_names):
        results.append({
            'timestamp': row['t'],
            'sensor_id': sensor,
            'pinn_pred': pinn_ppb[i],
            'final_pred': final_pred[i],
            'ground_truth': gt_ppb[i]
        })

# Analysis
print("\n" + "="*70)
print("FEBRUARY 2021 EVALUATION")
print("="*70)

df_results = pd.DataFrame(results)
df_valid = df_results.dropna(subset=['ground_truth'])

if len(df_valid) > 0:
    mae_pinn = (df_valid['pinn_pred'] - df_valid['ground_truth']).abs().mean()
    mae_hybrid = (df_valid['final_pred'] - df_valid['ground_truth']).abs().mean()
    
    print(f"\nValid samples: {len(df_valid)}")
    print(f"\nMAE (PINN only):  {mae_pinn:.4f} ppb")
    print(f"MAE (Hybrid/NN2): {mae_hybrid:.4f} ppb")
    print(f"Improvement:      {((mae_pinn - mae_hybrid) / mae_pinn * 100):.2f}%")
    
    # Per-sensor breakdown
    print("\n" + "-"*70)
    print("PER-SENSOR MAE (NN2 Correction):")
    print("-"*70)
    per_sensor = df_valid.groupby('sensor_id').apply(
        lambda x: (x['final_pred'] - x['ground_truth']).abs().mean(),
        include_groups=False
    )
    for sensor, mae in per_sensor.sort_values().items():
        print(f"  {sensor}: {mae:.4f} ppb")
    
    # Save results
    output_path = os.path.join(BASE_DIR, "results_feb_2021_hybrid.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to: {output_path}")
    
    # Save wide format for user
    wide_df = df_results.pivot(index='timestamp', columns='sensor_id', values='final_pred')
    wide_df.to_csv(os.path.join(BASE_DIR, "feb_test_1ppb.csv"))
    
    # Log to resultslog
    import datetime
    log_path = os.path.join(BASE_DIR, "logs/resultslog.md")
    with open(log_path, 'a') as f:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"| {timestamp} | Feb 2021 (Fixed PINN) | {len(df_valid)//9} hrs | "
               f"MAE PINN: {mae_pinn:.4f} | MAE Hybrid: {mae_hybrid:.4f} | "
               f"Real-time PINN + NN2 |\n")
    
    print(f"✓ Logged to resultslog.md")
    print(f"✓ Saved wide format to: feb_test_1ppb.csv")
    
else:
    print("\n⚠ No valid results to analyze!")

print("\n" + "="*70)
