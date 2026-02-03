#!/usr/bin/env python3
"""
Test NN2 with Precomputed PINN Values for 2019

Uses the exact precomputed PINN predictions that trained NN2, ensuring
we're testing the NN2 correction logic, not PINN computation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'simpletesting'))
sys.path.append(str(Path(__file__).parent / 'drive-download-20260202T042428Z-3-001'))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle

# Import the correct model definition
# We'll dynamically import based on the model checkpoint
from nn2_model_only import NN2_CorrectionNetwork as StandardNN2, InverseTransformLayer

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
# Updated to use Phase 1.5 balanced model
NN2_MODEL_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_balanced_model/nn2_master_model_ppb_balanced.pth"
NN2_SCALERS_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_balanced_model/nn2_master_scalers_balanced.pkl"

# Precomputed PINN file
PRECOMPUTED_PINN_PATH = SYNCED_DIR / "total_superimposed_concentrations.csv"

# Sensor IDs in order
SENSOR_IDS = [
    '482010026', '482010057', '482010069', '482010617', 
    '482010803', '482011015', '482011035', '482011039', '482016000'
]

def load_precomputed_pinn():
    """Load precomputed PINN predictions"""
    print("Loading precomputed PINN predictions...")
    df = pd.read_csv(PRECOMPUTED_PINN_PATH)
    df['t'] = pd.to_datetime(df['t'])
    print(f"  ✓ Loaded {len(df)} timestamps")
    return df

def load_sensor_data():
    """Load ground truth sensor data"""
    print("Loading sensor data...")
    df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in df.columns:
        df = df.rename(columns={'t': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter to 2019
    year_mask = (df['timestamp'] >= '2019-01-01') & (df['timestamp'] < '2020-01-01')
    df = df[year_mask].reset_index(drop=True)
    print(f"  ✓ Loaded {len(df)} timestamps for 2019")
    return df

def load_meteorological_data():
    """Load meteorological data from facility files"""
    print("Loading meteorological data...")
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    facility_data = {}
    for f in facility_files:
        df = pd.read_csv(f)
        if 't' in df.columns:
            df['t'] = pd.to_datetime(df['t'])
        facility_data[f.stem.replace('_synced_training_data', '')] = df
    
    print(f"  ✓ Loaded {len(facility_data)} facilities")
    return facility_data

def get_met_data_for_timestamp(facility_data, timestamp):
    """Get average meteorological data for a timestamp"""
    all_u, all_v, all_D = [], [], []
    
    for facility_name, df in facility_data.items():
        # Get met data from 3 hours before (for 3-hour forecast)
        met_timestamp = timestamp - pd.Timedelta(hours=3)
        row = df[df['t'] == met_timestamp]
        if len(row) > 0:
            all_u.append(row['wind_u'].values[0])
            all_v.append(row['wind_v'].values[0])
            all_D.append(row['D'].values[0])
    
    if len(all_u) > 0:
        return {
            'u': np.mean(all_u),
            'v': np.mean(all_v),
            'D': np.mean(all_D),
            'dt_obj': timestamp
        }
    else:
        return {
            'u': 0.0,
            'v': 0.0,
            'D': 1.0,
            'dt_obj': timestamp
        }

def apply_nn2_correction_fixed(nn2, scalers, sensor_coords_spatial, pinn_values, met_data, timestamp):
    """
    Apply NN2 correction using FIXED code (no current_sensors, correct model call)
    """
    # Scale inputs - handle zeros the same way as training (mask before scaling)
    pinn_nonzero_mask = pinn_values != 0.0
    
    # Scale PINN predictions (only non-zero values)
    p_s = np.zeros_like(pinn_values)
    if pinn_nonzero_mask.any():
        p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
            pinn_values[pinn_nonzero_mask].reshape(-1, 1)
        ).flatten()
    p_s = p_s.reshape(1, -1)
    
    # Wind
    w_in = np.array([[met_data['u'], met_data['v']]])
    w_s = scalers['wind'].transform(w_in)
    
    # Diffusion
    d_in = np.array([[met_data['D']]])
    d_s = scalers['diffusion'].transform(d_in)
    
    # Sensor coordinates
    c_s = scalers['coords'].transform(sensor_coords_spatial)
    
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
    
    # Convert to tensors
    p_tensor = torch.tensor(p_s, dtype=torch.float32)
    c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
    w_tensor = torch.tensor(w_s, dtype=torch.float32)
    d_tensor = torch.tensor(d_s, dtype=torch.float32)
    t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
    
    # FIXED: Model call - correct order and number of arguments
    # Model signature: forward(pinn_predictions, sensor_coords, wind, diffusion, temporal)
    # Model outputs directly in ppb space (output_ppb=True)
    with torch.no_grad():
        corrected_ppb, _ = nn2(p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    # Model outputs directly in ppb space, no inverse transform needed
    corrected = corrected_ppb.cpu().numpy().flatten()
    
    # Clip negative values (concentrations cannot be negative)
    corrected = np.maximum(corrected, 0.0)
    
    return corrected

def main():
    print("="*80)
    print("TEST NN2 WITH PRECOMPUTED PINN VALUES - 2019 DATA")
    print("="*80)
    print()
    
    # Load data
    df_pinn = load_precomputed_pinn()
    df_sensors = load_sensor_data()
    facility_data = load_meteorological_data()
    
    # Load NN2 model and scalers directly
    print("\nLoading NN2 model and scalers...")
    try:
        # Load checkpoint
        checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
        
        # Get scaler parameters from checkpoint
        scaler_mean = checkpoint.get('scaler_mean', None)
        scaler_scale = checkpoint.get('scaler_scale', None)
        output_ppb = checkpoint.get('output_ppb', True)
        
        # Check model architecture from checkpoint
        architecture = checkpoint.get('architecture', None)
        if architecture == 'tiny':
            print("  ✓ Detected Phase 1 tiny model (36→32→9)")
            # Create tiny model class inline
            class TinyNN2_CorrectionNetwork(nn.Module):
                """Tiny NN2 model: 36 → 32 → 9"""
                def __init__(self, n_sensors=9, scaler_mean=None, scaler_scale=None, output_ppb=True):
                    super().__init__()
                    self.n_sensors = n_sensors
                    self.output_ppb = output_ppb
                    self.correction_network = nn.Sequential(
                        nn.Linear(36, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(32, n_sensors)
                    )
                    if output_ppb and scaler_mean is not None and scaler_scale is not None:
                        self.inverse_transform = InverseTransformLayer(scaler_mean, scaler_scale)
                    else:
                        self.inverse_transform = None
                
                def forward(self, pinn_predictions, sensor_coords, wind, diffusion, temporal):
                    batch_size = pinn_predictions.shape[0]
                    coords_flat = sensor_coords.reshape(batch_size, -1)
                    features = torch.cat([
                        pinn_predictions, coords_flat, wind, diffusion, temporal
                    ], dim=-1)
                    corrections = self.correction_network(features)
                    corrected_scaled = pinn_predictions + corrections
                    if self.inverse_transform is not None:
                        corrected_ppb = self.inverse_transform(corrected_scaled)
                        return corrected_ppb, corrections
                    else:
                        return corrected_scaled, corrections
            
            nn2 = TinyNN2_CorrectionNetwork(
                n_sensors=9,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                output_ppb=output_ppb
            )
            print("  ✓ Loaded tiny model architecture")
        elif architecture == 'balanced':
            print("  ✓ Detected Phase 1.5 balanced model (36→64→16→9)")
            # Create balanced model class inline
            class BalancedNN2_CorrectionNetwork(nn.Module):
                """Balanced NN2 model: 36 → 64 → 16 → 9"""
                def __init__(self, n_sensors=9, scaler_mean=None, scaler_scale=None, output_ppb=True):
                    super().__init__()
                    self.n_sensors = n_sensors
                    self.output_ppb = output_ppb
                    self.correction_network = nn.Sequential(
                        nn.Linear(36, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 16),
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(16, n_sensors)
                    )
                    if output_ppb and scaler_mean is not None and scaler_scale is not None:
                        self.inverse_transform = InverseTransformLayer(scaler_mean, scaler_scale)
                    else:
                        self.inverse_transform = None
                
                def forward(self, pinn_predictions, sensor_coords, wind, diffusion, temporal):
                    batch_size = pinn_predictions.shape[0]
                    coords_flat = sensor_coords.reshape(batch_size, -1)
                    features = torch.cat([
                        pinn_predictions, coords_flat, wind, diffusion, temporal
                    ], dim=-1)
                    corrections = self.correction_network(features)
                    corrected_scaled = pinn_predictions + corrections
                    if self.inverse_transform is not None:
                        corrected_ppb = self.inverse_transform(corrected_scaled)
                        return corrected_ppb, corrections
                    else:
                        return corrected_scaled, corrections
            
            nn2 = BalancedNN2_CorrectionNetwork(
                n_sensors=9,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                output_ppb=output_ppb
            )
            print("  ✓ Loaded balanced model architecture")
        else:
            # Standard architecture
            print("  ✓ Using standard architecture")
            nn2 = StandardNN2(
                n_sensors=9,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                output_ppb=output_ppb
            )
        
        # Load state dict
        nn2.load_state_dict(checkpoint['model_state_dict'])
        nn2.eval()
        
        # Load scalers
        with open(NN2_SCALERS_PATH, 'rb') as f:
            scalers = pickle.load(f)
        
        # Get sensor coordinates from checkpoint or scalers
        sensor_coords_spatial = checkpoint.get('sensor_coords', None)
        if sensor_coords_spatial is None:
            # Fallback: use sensor coordinates from training data
            sensor_coords_spatial = np.array([
                [13972.62, 19915.57],  # 482010026
                [3017.18, 12334.2],    # 482010057
                [817.42, 9218.92],     # 482010069
                [27049.57, 22045.66],  # 482010617
                [8836.35, 15717.2],    # 482010803
                [18413.8, 15068.96],   # 482011015
                [1159.98, 12272.52],   # 482011035
                [13661.93, 5193.24],   # 482011039
                [1546.9, 6786.33],     # 482016000
            ])
        
        print("  ✓ NN2 model loaded")
        print("  ✓ Scalers loaded")
        
    except Exception as e:
        print(f"  ✗ Failed to load NN2 model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Merge PINN and sensor data
    print("\nMerging datasets...")
    df_merged = df_pinn.merge(
        df_sensors,
        left_on='t',
        right_on='timestamp',
        how='inner',
        suffixes=('_pinn', '_sensor')
    )
    print(f"  ✓ Found {len(df_merged)} matching timestamps")
    
    if len(df_merged) == 0:
        print("  ✗ No matching timestamps found!")
        return
    
    # Process each timestamp
    print("\nProcessing predictions...")
    results = []
    
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Processing"):
        timestamp = row['t']
        
        # Get precomputed PINN values (already in ppb)
        pinn_values = np.array([
            row[f'sensor_{sid}_pinn'] if f'sensor_{sid}_pinn' in row else row.get(f'sensor_{sid}', 0.0)
            for sid in SENSOR_IDS
        ])
        
        # Get ground truth
        actual_values = np.array([
            row.get(f'sensor_{sid}_sensor', row.get(f'sensor_{sid}', np.nan))
            for sid in SENSOR_IDS
        ])
        
        # Get meteorological data
        met_data = get_met_data_for_timestamp(facility_data, timestamp)
        
        # Apply NN2 correction using fixed code
        try:
            nn2_values = apply_nn2_correction_fixed(
                nn2, scalers, sensor_coords_spatial,
                pinn_values, met_data, timestamp
            )
        except Exception as e:
            print(f"\n  ✗ Error at {timestamp}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Store results
        for i, sensor_id in enumerate(SENSOR_IDS):
            if not np.isnan(actual_values[i]) and actual_values[i] > 0:
                results.append({
                    'timestamp': timestamp,
                    'sensor_id': sensor_id,
                    'actual': actual_values[i],
                    'pinn': pinn_values[i],
                    'nn2': nn2_values[i] if isinstance(nn2_values, np.ndarray) else nn2_values
                })
    
    if len(results) == 0:
        print("\n  ✗ No valid results!")
        return
    
    # Calculate metrics
    results_df = pd.DataFrame(results)
    
    print(f"\n✓ Processed {len(results_df)} valid samples")
    print()
    
    # Overall metrics
    pinn_mae = np.abs(results_df['actual'] - results_df['pinn']).mean()
    nn2_mae = np.abs(results_df['actual'] - results_df['nn2']).mean()
    improvement = ((pinn_mae - nn2_mae) / pinn_mae * 100) if pinn_mae > 0 else 0
    
    # Check for catastrophic predictions
    nn2_max = results_df['nn2'].max()
    nn2_min = results_df['nn2'].min()
    nn2_mean = results_df['nn2'].mean()
    negative_count = (results_df['nn2'] < 0).sum()
    extreme_count = (results_df['nn2'].abs() > 1000).sum()
    
    print("="*80)
    print("RESULTS - 2019 DATA WITH PRECOMPUTED PINN")
    print("="*80)
    print(f"\nOverall Performance:")
    print(f"  PINN MAE: {pinn_mae:.6f} ppb")
    print(f"  NN2 MAE:  {nn2_mae:.6f} ppb")
    print(f"  Improvement: {improvement:.2f}%")
    print()
    
    print(f"NN2 Prediction Statistics:")
    print(f"  Range: [{nn2_min:.2f}, {nn2_max:.2f}] ppb")
    print(f"  Mean: {nn2_mean:.2f} ppb")
    print(f"  Negative values: {negative_count} ({100*negative_count/len(results_df):.1f}%)")
    print(f"  Extreme values (>1000 ppb): {extreme_count} ({100*extreme_count/len(results_df):.1f}%)")
    print()
    
    if nn2_max > 1000 or nn2_min < -100:
        print("  ⚠️  WARNING: Predictions still show extreme values!")
        print("     This suggests the model itself may need retraining.")
    elif negative_count > 0:
        print("  ⚠️  WARNING: Some negative predictions (should be clipped)")
    else:
        print("  ✓ Predictions in reasonable range!")
    
    # Per-sensor metrics
    print("\nPer-Sensor Performance:")
    print("-"*80)
    sensor_metrics = []
    for sensor_id in sorted(SENSOR_IDS):
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        if len(sensor_data) == 0:
            continue
        
        sensor_pinn_mae = np.abs(sensor_data['actual'] - sensor_data['pinn']).mean()
        sensor_nn2_mae = np.abs(sensor_data['actual'] - sensor_data['nn2']).mean()
        sensor_improvement = ((sensor_pinn_mae - sensor_nn2_mae) / sensor_pinn_mae * 100) if sensor_pinn_mae > 0 else 0
        
        sensor_metrics.append({
            'sensor_id': sensor_id,
            'pinn_mae': sensor_pinn_mae,
            'nn2_mae': sensor_nn2_mae,
            'improvement': sensor_improvement,
            'n_samples': len(sensor_data),
            'nn2_max': sensor_data['nn2'].max(),
            'nn2_min': sensor_data['nn2'].min()
        })
        
        print(f"  {sensor_id}:")
        print(f"    PINN MAE: {sensor_pinn_mae:.6f} ppb")
        print(f"    NN2 MAE:  {sensor_nn2_mae:.6f} ppb")
        print(f"    Improvement: {sensor_improvement:.2f}%")
        print(f"    NN2 Range: [{sensor_data['nn2'].min():.2f}, {sensor_data['nn2'].max():.2f}] ppb")
        print(f"    Samples: {len(sensor_data)}")
        print()
    
    # Save results
    output_file = Path(__file__).parent / 'nn2_precomputed_pinn_2019_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to: {output_file}")
    
    # Save summary
    summary_file = Path(__file__).parent / 'nn2_precomputed_pinn_2019_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("NN2 TEST WITH PRECOMPUTED PINN VALUES - 2019 VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Performance:\n")
        f.write(f"  PINN MAE: {pinn_mae:.6f} ppb\n")
        f.write(f"  NN2 MAE:  {nn2_mae:.6f} ppb\n")
        f.write(f"  Improvement: {improvement:.2f}%\n")
        f.write(f"  Total samples: {len(results_df)}\n\n")
        f.write(f"NN2 Prediction Statistics:\n")
        f.write(f"  Range: [{nn2_min:.2f}, {nn2_max:.2f}] ppb\n")
        f.write(f"  Mean: {nn2_mean:.2f} ppb\n")
        f.write(f"  Negative values: {negative_count} ({100*negative_count/len(results_df):.1f}%)\n")
        f.write(f"  Extreme values (>1000 ppb): {extreme_count} ({100*extreme_count/len(results_df):.1f}%)\n\n")
        f.write("Per-Sensor Performance:\n")
        f.write("-"*80 + "\n")
        for m in sensor_metrics:
            f.write(f"{m['sensor_id']}: PINN={m['pinn_mae']:.6f}, NN2={m['nn2_mae']:.6f}, "
                   f"Imp={m['improvement']:.2f}%, Range=[{m['nn2_min']:.2f}, {m['nn2_max']:.2f}], n={m['n_samples']}\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    print()
    print("="*80)

if __name__ == '__main__':
    main()

