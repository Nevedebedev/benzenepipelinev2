"""
Test Pipeline on Exact NN2 Training Data

This script:
1. Loads facility-specific training data from drive-download
2. For each timestamp, runs PINN for all facilities at sensor locations
3. Superimposes concentrations and converts to ppb
4. Applies NN2 correction
5. Compares against ground truth
"""

import pandas as pd
import numpy as np
import torch
import os
import glob
from benzene_pipeline import BenzenePipeline

# Paths
DRIVE_DIR = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001"
BASE_DIR = "/Users/neevpratap/simpletesting"

# Sensor Coordinates (from drive data)
SENSORS = [
    {'id': 'sensor_482010026', 'x': 13970.0, 'y': 19920.0},
    {'id': 'sensor_482010057', 'x': 3020.0, 'y': 12330.0},
    {'id': 'sensor_482010069', 'x': 820.0, 'y': 9220.0},
    {'id': 'sensor_482010617', 'x': 27050.0, 'y': 22050.0},
    {'id': 'sensor_482010803', 'x': 8840.0, 'y': 15720.0},
    {'id': 'sensor_482011015', 'x': 18410.0, 'y': 15070.0},
    {'id': 'sensor_482011035', 'x': 1160.0, 'y': 12270.0},
    {'id': 'sensor_482011039', 'x': 13660.0, 'y': 5190.0},
    {'id': 'sensor_482016000', 'x': 1550.0, 'y': 6790.0}
]

def load_facility_data():
    """Load all facility training data from drive-download folder"""
    print("Loading facility training data...")
    facility_files = glob.glob(os.path.join(DRIVE_DIR, "*_synced_training_data.csv"))
    
    facilities = {}
    for fpath in facility_files:
        fname = os.path.basename(fpath).replace("_synced_training_data.csv", "")
        df = pd.read_csv(fpath)
        df['t'] = pd.to_datetime(df['t'])
        facilities[fname] = df
        print(f"  Loaded {fname}: {len(df)} rows")
    
    return facilities

def run_training_data_test():
    """Test pipeline on exact NN2 training data"""
    
    # Initialize Pipeline
    print("\nInitializing Pipeline...")
    pinn_path = os.path.join(BASE_DIR, "pinn_combined_final.pth 2")
    nn2_path = os.path.join(BASE_DIR, "nn2_master_model_spatial.pth")
    scaler_path = os.path.join(BASE_DIR, "nn2_master_scalers.pkl")
    
    pipeline = BenzenePipeline(pinn_path, nn2_path, scaler_path)
    
    # Load Ground Truth
    print("Loading ground truth...")
    gt_path = os.path.join(DRIVE_DIR, "sensors_final_synced.csv")
    df_gt = pd.read_csv(gt_path)
    df_gt['timestamp'] = pd.to_datetime(df_gt['t'])
    
    # Load Facility Data
    facilities = load_facility_data()
    
    # Get common timestamps (intersection of all facility data)
    print("\nFinding common timestamps...")
    all_timestamps = None
    for fname, df in facilities.items():
        if all_timestamps is None:
            all_timestamps = set(df['t'].unique())
        else:
            all_timestamps = all_timestamps.intersection(set(df['t'].unique()))
    
    all_timestamps = sorted(list(all_timestamps))
    print(f"Found {len(all_timestamps)} common timestamps")
    
    # Limit to first 500 for better statistics
    test_timestamps = all_timestamps[:500]
    print(f"Testing on first {len(test_timestamps)} timestamps...")
    
    # Prepare sensor points
    sensor_points = np.array([[s['x'], s['y']] for s in SENSORS])
    
    results = []
    
    for i, ts in enumerate(test_timestamps):
        if i % 10 == 0:
            print(f"Processing timestamp {i+1}/{len(test_timestamps)}: {ts}")
        
        # Get ground truth for this timestamp
        row_gt = df_gt[df_gt['timestamp'] == ts]
        if row_gt.empty:
            continue
        
        # Build ground truth vector
        gt_values = []
        for s in SENSORS:
            if s['id'] in row_gt.columns:
                gt_values.append(float(row_gt[s['id']].values[0]))
            else:
                gt_values.append(np.nan)
        gt_array = np.array(gt_values)
        
        # Get meteorology from first facility (they should all have same met)
        first_facility = list(facilities.values())[0]
        row_met = first_facility[first_facility['t'] == ts]
        if row_met.empty:
            continue
        
        met_data = {
            'u': float(row_met.iloc[0]['wind_u']),
            'v': float(row_met.iloc[0]['wind_v']),
            'D': float(row_met.iloc[0]['D']),
            't': 3600.0,
            'dt_obj': ts
        }
        
        # Get emission rates for each facility
        emissions = {}
        for fname, df in facilities.items():
            row_fac = df[df['t'] == ts]
            if not row_fac.empty:
                Q_val = float(row_fac.iloc[0]['Q_total'])
                emissions[fname] = Q_val
        
        # Run PINN superposition and NN2 correction
        try:
            final_pred_ppb = pipeline.process_timestep(
                met_data, 
                sensor_points, 
                emissions, 
                ground_truth=gt_array
            )
            
            # Get PINN-only prediction (without NN2)
            raw_phi = pipeline.superimpose(met_data, sensor_points, emissions)
            pinn_pred_ppb = raw_phi * 313210039.9
            
            # Store results for each sensor
            for j, s in enumerate(SENSORS):
                results.append({
                    'timestamp': ts,
                    'sensor_id': s['id'],
                    'pinn_pred': pinn_pred_ppb[j],
                    'final_pred': final_pred_ppb[j],
                    'ground_truth': gt_array[j]
                })
        except Exception as e:
            print(f"  Error at {ts}: {e}")
            continue
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    
    # Calculate MAE
    df_valid = df_results.dropna(subset=['ground_truth'])
    
    if len(df_valid) > 0:
        mae_pinn = (df_valid['pinn_pred'] - df_valid['ground_truth']).abs().mean()
        mae_hybrid = (df_valid['final_pred'] - df_valid['ground_truth']).abs().mean()
        
        print(f"\n{'='*60}")
        print(f"RESULTS ON NN2 TRAINING DATA ({len(df_valid)} valid samples)")
        print(f"{'='*60}")
        print(f"MAE (PINN Only):  {mae_pinn:.4f} ppb")
        print(f"MAE (Hybrid/NN2): {mae_hybrid:.4f} ppb")
        print(f"Improvement:      {((mae_pinn - mae_hybrid) / mae_pinn * 100):.2f}%")
        print(f"{'='*60}")
        
        # Save detailed results
        output_path = os.path.join(BASE_DIR, "training_data_validation_results.csv")
        df_results.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
    else:
        print("\nNo valid results to analyze!")

if __name__ == "__main__":
    run_training_data_test()
