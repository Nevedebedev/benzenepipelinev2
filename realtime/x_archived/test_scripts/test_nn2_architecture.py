#!/usr/bin/env python3
"""
Test NN2 Architecture - Diagnose why it's performing poorly

This script will:
1. Load the NN2 model
2. Test it on a small sample of data
3. Analyze what corrections it's making
4. Check if the architecture is fundamentally flawed
"""

import torch
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Add paths
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting')
sys.path.append('/Users/neevpratap/Desktop/benzenepipelinev2/realtime')
sys.path.append('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/drive-download-20260202T042428Z-3-001')

from nn2_model_only import NN2_CorrectionNetwork
from test_pipeline_2019_fixed import SENSORS

# Configuration
NN2_MODEL_PATH = '/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_pinncorrect/nn2_master_model.pth'
NN2_SCALERS_PATH = '/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_pinncorrect/nn2_scalers.pkl'
TRAINING_DATA_PATH = '/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata/total_superimposed_concentrations.csv'
SENSOR_DATA_PATH = '/Users/neevpratap/Downloads/sensors_final_synced.csv'

def load_nn2_model():
    """Load NN2 model and scalers"""
    print("Loading NN2 model...")
    
    checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    
    scaler_mean = checkpoint.get('scaler_mean', None)
    scaler_scale = checkpoint.get('scaler_scale', None)
    output_ppb = checkpoint.get('output_ppb', True)
    
    nn2 = NN2_CorrectionNetwork(
        n_sensors=9,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        output_ppb=output_ppb
    )
    nn2.load_state_dict(checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print(f"  ✓ Model loaded (output_ppb={output_ppb})")
    return nn2, scalers, sensor_coords_spatial

def test_nn2_on_sample(nn2, scalers, sensor_coords_spatial, n_samples=10):
    """Test NN2 on a small sample and analyze its behavior"""
    print("\n" + "="*80)
    print("TESTING NN2 ARCHITECTURE ON SAMPLE DATA")
    print("="*80)
    
    # Load training data
    pinn_df = pd.read_csv(TRAINING_DATA_PATH)
    pinn_df['t'] = pd.to_datetime(pinn_df['t'])
    
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
    elif 'timestamp' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Find common timestamps
    common_times = sorted(set(pinn_df['t']) & set(sensor_df['timestamp']))[:n_samples]
    print(f"\nTesting on {len(common_times)} samples...\n")
    
    results = []
    
    for timestamp in common_times:
        # Get PINN predictions
        pinn_row = pinn_df[pinn_df['t'] == timestamp].iloc[0]
        sensor_ids_sorted = sorted(SENSORS.keys())
        pinn_values = np.array([pinn_row[f'sensor_{sid}'] for sid in sensor_ids_sorted])
        
        # Get actual sensor values
        sensor_row = sensor_df[sensor_df['timestamp'] == timestamp].iloc[0]
        actual_values = np.array([sensor_row[f'sensor_{sid}'] for sid in sensor_ids_sorted])
        
        # Get meteo data (simplified - use averages from training)
        # In real pipeline, this would come from facility CSVs
        # For testing, we'll use placeholder values that match training distribution
        avg_u = 2.0  # Typical wind u
        avg_v = 1.0  # Typical wind v
        avg_D = 50.0  # Typical diffusion
        
        # Temporal features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        
        temporal_vals = np.array([[
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            is_weekend,
            month / 12.0
        ]])
        
        # Scale inputs (matching test_pipeline_2019_fixed.py)
        pinn_nonzero_mask = pinn_values != 0.0
        p_s = np.zeros_like(pinn_values)
        if pinn_nonzero_mask.any():
            p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
                pinn_values[pinn_nonzero_mask].reshape(-1, 1)
            ).flatten()
        p_s = p_s.reshape(1, -1)
        
        w_s = scalers['wind'].transform(np.array([[avg_u, avg_v]]))
        d_s = scalers['diffusion'].transform(np.array([[avg_D]]))
        c_s = scalers['coords'].transform(sensor_coords_spatial)
        
        # Convert to tensors
        p_tensor = torch.tensor(p_s, dtype=torch.float32)
        c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
        w_tensor = torch.tensor(w_s, dtype=torch.float32)
        d_tensor = torch.tensor(d_s, dtype=torch.float32)
        t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
        
        # Run NN2
        with torch.no_grad():
            corrected_ppb, corrections_scaled = nn2(p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
        
        corrected_ppb_np = corrected_ppb.cpu().numpy().flatten()
        corrections_scaled_np = corrections_scaled.cpu().numpy().flatten()
        
        # Analyze for each sensor
        for i, sid in enumerate(sensor_ids_sorted):
            if actual_values[i] > 0:  # Only analyze non-zero actuals
                pinn_val = pinn_values[i]
                actual_val = actual_values[i]
                corrected_val = corrected_ppb_np[i]
                correction_scaled = corrections_scaled_np[i]
                
                # Convert correction to ppb space for analysis
                # correction_ppb = correction_scaled * scalers['sensors'].scale_[0] + scalers['sensors'].mean_[0]
                # Actually, corrections are in scaled space, but we need to see their impact
                # The corrected value is already in ppb, so:
                correction_ppb = corrected_val - pinn_val
                
                results.append({
                    'timestamp': timestamp,
                    'sensor': sid,
                    'pinn_ppb': pinn_val,
                    'actual_ppb': actual_val,
                    'corrected_ppb': corrected_val,
                    'correction_ppb': correction_ppb,
                    'correction_scaled': correction_scaled,
                    'pinn_error': abs(pinn_val - actual_val),
                    'corrected_error': abs(corrected_val - actual_val),
                    'improvement': abs(pinn_val - actual_val) - abs(corrected_val - actual_val)
                })
    
    results_df = pd.DataFrame(results)
    
    # Print analysis
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nTotal samples analyzed: {len(results_df)}")
    print(f"  (Non-zero actual sensor readings)")
    
    print(f"\n1. ERROR METRICS:")
    print(f"   PINN MAE: {results_df['pinn_error'].mean():.4f} ppb")
    print(f"   NN2 MAE:  {results_df['corrected_error'].mean():.4f} ppb")
    print(f"   Improvement: {results_df['improvement'].mean():.4f} ppb")
    print(f"   % Improvement: {(results_df['improvement'].mean() / results_df['pinn_error'].mean() * 100):.1f}%")
    
    print(f"\n2. CORRECTION STATISTICS:")
    print(f"   Mean correction: {results_df['correction_ppb'].mean():.4f} ppb")
    print(f"   Median correction: {results_df['correction_ppb'].median():.4f} ppb")
    print(f"   Std correction: {results_df['correction_ppb'].std():.4f} ppb")
    print(f"   Min correction: {results_df['correction_ppb'].min():.4f} ppb")
    print(f"   Max correction: {results_df['correction_ppb'].max():.4f} ppb")
    
    print(f"\n3. CORRECTION DIRECTION:")
    positive_corrections = (results_df['correction_ppb'] > 0).sum()
    negative_corrections = (results_df['correction_ppb'] < 0).sum()
    print(f"   Positive corrections: {positive_corrections} ({positive_corrections/len(results_df)*100:.1f}%)")
    print(f"   Negative corrections: {negative_corrections} ({negative_corrections/len(results_df)*100:.1f}%)")
    
    print(f"\n4. CORRECTION SIZE vs NEEDED:")
    results_df['needed_correction'] = results_df['actual_ppb'] - results_df['pinn_ppb']
    results_df['correction_ratio'] = results_df['correction_ppb'] / (results_df['needed_correction'] + 1e-10)
    print(f"   Mean correction ratio (actual/needed): {results_df['correction_ratio'].mean():.4f}")
    print(f"   Median correction ratio: {results_df['correction_ratio'].median():.4f}")
    
    # Check if corrections are in the right direction
    correct_direction = ((results_df['correction_ppb'] > 0) == (results_df['needed_correction'] > 0)).sum()
    print(f"   Corrections in correct direction: {correct_direction}/{len(results_df)} ({correct_direction/len(results_df)*100:.1f}%)")
    
    print(f"\n5. CORRELATION ANALYSIS:")
    corr_correction_actual = results_df['correction_ppb'].corr(results_df['actual_ppb'])
    corr_correction_pinn = results_df['correction_ppb'].corr(results_df['pinn_ppb'])
    corr_correction_needed = results_df['correction_ppb'].corr(results_df['needed_correction'])
    print(f"   Correction ↔ Actual: {corr_correction_actual:.4f}")
    print(f"   Correction ↔ PINN: {corr_correction_pinn:.4f}")
    print(f"   Correction ↔ Needed: {corr_correction_needed:.4f}")
    
    print(f"\n6. WORST CASES:")
    worst = results_df.nlargest(5, 'corrected_error')
    print(f"   Top 5 worst corrections:")
    for idx, row in worst.iterrows():
        print(f"     Sensor {row['sensor']}: PINN={row['pinn_ppb']:.3f}, Actual={row['actual_ppb']:.3f}, "
              f"Corrected={row['corrected_ppb']:.3f}, Error={row['corrected_error']:.3f}")
    
    print(f"\n7. BEST CASES:")
    best = results_df.nsmallest(5, 'corrected_error')
    print(f"   Top 5 best corrections:")
    for idx, row in best.iterrows():
        print(f"     Sensor {row['sensor']}: PINN={row['pinn_ppb']:.3f}, Actual={row['actual_ppb']:.3f}, "
              f"Corrected={row['corrected_ppb']:.3f}, Error={row['corrected_error']:.3f}")
    
    return results_df

def main():
    print("="*80)
    print("NN2 ARCHITECTURE DIAGNOSTIC TEST")
    print("="*80)
    
    # Load model
    nn2, scalers, sensor_coords_spatial = load_nn2_model()
    
    # Test on sample
    results_df = test_nn2_on_sample(nn2, scalers, sensor_coords_spatial, n_samples=50)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print("  - If corrections correlate highly with actual (not needed), model is predicting actuals")
    print("  - If corrections are in wrong direction >50%, model is systematically wrong")
    print("  - If correction ratio is >>1 or <<1, model is over/under-correcting")
    print("  - If NN2 MAE > PINN MAE, model is degrading performance")

if __name__ == '__main__':
    main()

