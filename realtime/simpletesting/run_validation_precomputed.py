"""
Run validation using PRE-COMPUTED PINN predictions

This script uses the pre-computed PINN values from total_superimposed_concentrations.csv
instead of re-computing them, allowing the pipeline to achieve NN2's training performance.
"""

import pandas as pd
import numpy as np
import torch
import os
from benzene_pipeline import BenzenePipeline

# Paths
NN2_DATA_DIR = "/Users/neevpratap/simpletesting/nn2trainingdata"
BASE_DIR = "/Users/neevpratap/simpletesting"

def run_validation_with_precomputed_pinn():
    """Run validation using pre-computed PINN predictions"""
    
    print("="*70)
    print("VALIDATION WITH PRE-COMPUTED PINN VALUES")
    print("="*70)
    
    # Load pre-computed PINN predictions
    print("\n1. Loading pre-computed PINN predictions...")
    pinn_path = os.path.join(NN2_DATA_DIR, "total_superimposed_concentrations.csv")
    df_pinn = pd.read_csv(pinn_path)
    df_pinn['t'] = pd.to_datetime(df_pinn['t'])
    print(f"   Loaded {len(df_pinn)} timestamps")
    
    # Load ground truth
    print("\n2. Loading ground truth...")
    gt_path = os.path.join(NN2_DATA_DIR, "sensors_final_synced.csv")
    df_gt = pd.read_csv(gt_path)
    df_gt['timestamp'] = pd.to_datetime(df_gt['t'])
    
    # Load meteorological data
    print("\n3. Loading meteorological data...")
    met_file = os.path.join(NN2_DATA_DIR, "BASF_Pasadena_synced_training_data.csv")
    df_met = pd.read_csv(met_file)
    df_met['t'] = pd.to_datetime(df_met['t'])
    
    # Initialize pipeline (for NN2 correction only)
    print("\n4. Initializing NN2 correction...")
    pipeline = BenzenePipeline(
        os.path.join(BASE_DIR, "pinn_combined_final.pth 2"),
        os.path.join(BASE_DIR, "nn2_master_model_spatial.pth"),
        os.path.join(BASE_DIR, "nn2_master_scalers.pkl")
    )
    
    # Merge data
    print("\n5. Merging datasets...")
    df_merged = df_pinn.merge(df_gt, left_on='t', right_on='timestamp', how='inner', suffixes=('_pinn', '_gt'))
    df_merged = df_merged.merge(df_met[['t', 'wind_u', 'wind_v', 'D']], left_on='t_pinn', right_on='t', how='inner')
    
    print(f"   Processing {len(df_merged)} timestamps...")
    
    sensor_names = [
        'sensor_482010026', 'sensor_482010057', 'sensor_482010069',
        'sensor_482010617', 'sensor_482010803', 'sensor_482011015',
        'sensor_482011035', 'sensor_482011039', 'sensor_482016000'
    ]
    
    results = []
    
    print("\n6. Running NN2 correction on pre-computed PINN values...")
    for idx, row in df_merged.iterrows():
        if idx % 500 == 0:
            print(f"   Progress: {idx}/{len(df_merged)} ({100*idx/len(df_merged):.1f}%)")
        
        # Get pre-computed PINN predictions (already in ppb)
        pinn_ppb = np.array([row[s + '_pinn'] for s in sensor_names])
        
        # Get ground truth
        gt_ppb = np.array([row[s + '_gt'] for s in sensor_names])
        
        # Meteorological data
        met_data = {
            'u': row['wind_u'],
            'v': row['wind_v'],
            'D': row['D'],
            't': 3600.0,
            'dt_obj': pd.to_datetime(row['t_pinn'])
        }
        
        # Apply NN2 correction
        final_pred = pipeline.apply_nn2_correction(pinn_ppb, gt_ppb, met_data)
        
        # Store results
        for i, sensor in enumerate(sensor_names):
            results.append({
                'timestamp': row['t_pinn'],
                'sensor_id': sensor,
                'pinn_pred': pinn_ppb[i],
                'final_pred': final_pred[i],
                'ground_truth': gt_ppb[i]
            })
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    df_valid = df_results.dropna(subset=['ground_truth'])
    
    if len(df_valid) > 0:
        mae_pinn = (df_valid['pinn_pred'] - df_valid['ground_truth']).abs().mean()
        mae_hybrid = (df_valid['final_pred'] - df_valid['ground_truth']).abs().mean()
        
        print(f"\nValid samples: {len(df_valid)}")
        print(f"\nMAE (Pre-computed PINN): {mae_pinn:.4f} ppb")
        print(f"MAE (Hybrid/NN2):        {mae_hybrid:.4f} ppb")
        print(f"Improvement:             {((mae_pinn - mae_hybrid) / mae_pinn * 100):.2f}%")
        
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
        output_path = os.path.join(BASE_DIR, "validation_precomputed_pinn.csv")
        df_results.to_csv(output_path, index=False)
        print(f"\n✓ Detailed results saved to: {output_path}")
        
        # Log to resultslog
        import datetime
        log_path = os.path.join(BASE_DIR, "logs/resultslog.md")
        with open(log_path, 'a') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"| {timestamp} | Pre-computed PINN + NN2 | {len(df_valid)//9} hrs | "
                   f"MAE PINN: {mae_pinn:.4f} | MAE Hybrid: {mae_hybrid:.4f} | "
                   f"Pre-computed validation |\n")
        
        print(f"✓ Logged to resultslog.md")
        
    else:
        print("\n⚠ No valid results to analyze!")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    run_validation_with_precomputed_pinn()
