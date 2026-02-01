"""
Test NN2 on its EXACT training data

This will load the pre-computed PINN predictions from total_superimposed_concentrations.csv
and apply NN2 correction exactly as it was trained.
"""

import pandas as pd
import numpy as np
import torch
import os
from benzene_pipeline import BenzenePipeline

# Paths
NN2_DATA_DIR = "/Users/neevpratap/simpletesting/nn2trainingdata"
BASE_DIR = "/Users/neevpratap/simpletesting"

def test_nn2_on_exact_training_data():
    """Test NN2 using the exact pre-computed PINN predictions it was trained on"""
    
    # Load pre-computed PINN predictions (in ppb)
    print("Loading pre-computed PINN predictions...")
    pinn_path = os.path.join(NN2_DATA_DIR, "total_superimposed_concentrations.csv")
    df_pinn = pd.read_csv(pinn_path)
    df_pinn['t'] = pd.to_datetime(df_pinn['t'])
    
    # Load ground truth
    print("Loading ground truth...")
    gt_path = os.path.join(NN2_DATA_DIR, "sensors_final_synced.csv")
    df_gt = pd.read_csv(gt_path)
    df_gt['timestamp'] = pd.to_datetime(df_gt['t'])
    
    # Load sensor coordinates (OLD system - what NN2 was trained with)
    print("Loading sensor coordinates...")
    coords_path = os.path.join(NN2_DATA_DIR, "sensor_coordinates.csv")
    df_coords = pd.read_csv(coords_path)
    df_coords = df_coords.sort_values('sensor_id').reset_index(drop=True)
    sensor_coords = df_coords[['x', 'y']].values
    
    sensor_names = [
        'sensor_482010026', 'sensor_482010057', 'sensor_482010069',
        'sensor_482010617', 'sensor_482010803', 'sensor_482011015',
        'sensor_482011035', 'sensor_482011039', 'sensor_482016000'
    ]
    
    # Initialize pipeline (mainly to use its NN2 correction method)
    print("\nInitializing Pipeline...")
    pipeline = BenzenePipeline(
        os.path.join(BASE_DIR, "pinn_combined_final.pth 2"),
        os.path.join(BASE_DIR, "nn2_master_model_spatial.pth"),
        os.path.join(BASE_DIR, "nn2_master_scalers.pkl")
    )
    
    # Load meteorological data from one of the facility files
    print("Loading meteorological data...")
    met_file = os.path.join(NN2_DATA_DIR, "BASF_Pasadena_synced_training_data.csv")
    df_met = pd.read_csv(met_file)
    df_met['t'] = pd.to_datetime(df_met['t'])
    
    # Merge all data
    df_merged = df_pinn.merge(df_gt, left_on='t', right_on='timestamp', how='inner', suffixes=('_pinn', '_gt'))
    df_merged = df_merged.merge(df_met[['t', 'wind_u', 'wind_v', 'D']], left_on='t_pinn', right_on='t', how='inner')
    
    print(f"\nTesting on {len(df_merged)} timestamps...")
    
    results = []
    
    for idx, row in df_merged.iterrows():
        if idx % 500 == 0:
            print(f"Processing {idx}/{len(df_merged)}...")
        
        # Get PINN predictions (already in ppb) - with _pinn suffix
        pinn_ppb = np.array([row[s + '_pinn'] for s in sensor_names])
        
        # Get ground truth - with _gt suffix
        gt_ppb = np.array([row[s + '_gt'] for s in sensor_names])
        
        # Get meteorological data
        met_data = {
            'u': row['wind_u'],
            'v': row['wind_v'],
            'D': row['D'],
            't': 3600.0,
            'dt_obj': pd.to_datetime(row['t_pinn'])
        }
        
        # Apply NN2 correction using the pipeline method
        # This will handle all the scaling internally
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
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    df_valid = df_results.dropna(subset=['ground_truth'])
    
    if len(df_valid) > 0:
        mae_pinn = (df_valid['pinn_pred'] - df_valid['ground_truth']).abs().mean()
        mae_hybrid = (df_valid['final_pred'] - df_valid['ground_truth']).abs().mean()
        
        print(f"\n{'='*70}")
        print(f"RESULTS ON EXACT NN2 TRAINING DATA ({len(df_valid)} samples)")
        print(f"{'='*70}")
        print(f"MAE (Pre-computed PINN): {mae_pinn:.4f} ppb")
        print(f"MAE (NN2 Correction):    {mae_hybrid:.4f} ppb")
        print(f"Improvement:             {((mae_pinn - mae_hybrid) / mae_pinn * 100):.2f}%")
        print(f"{'='*70}")
        
        # Per-sensor breakdown
        print("\nPer-Sensor MAE (NN2 Correction):")
        per_sensor = df_valid.groupby('sensor_id').apply(
            lambda x: (x['final_pred'] - x['ground_truth']).abs().mean(),
            include_groups=False
        )
        print(per_sensor.sort_values())
        
        # Save results
        output_path = os.path.join(BASE_DIR, "nn2_exact_training_test.csv")
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo valid results!")

if __name__ == "__main__":
    test_nn2_on_exact_training_data()
