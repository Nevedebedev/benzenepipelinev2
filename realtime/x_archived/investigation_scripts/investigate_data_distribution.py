#!/usr/bin/env python3
"""
Investigate Data Distribution Differences and Overfitting

Compare:
1. Training data distribution (from total_concentrations.csv and sensors_final_synced.csv)
2. Validation data distribution (from January 2019 validation)
3. Check for overfitting indicators
"""

import sys
sys.path.append('/Users/neevpratap/simpletesting')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Paths
TRAINING_PINN_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata/total_concentrations.csv"
TRAINING_SENSOR_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
VALIDATION_RESULTS_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/validation_results/nn2_validation_2019.csv"
LEAVE_ONE_OUT_RESULTS = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/leave_one_out_results_spatial-3.json"

SENSORS = {
    '482010026': (13972.62, 19915.57),
    '482010057': (3017.18, 12334.2),
    '482010069': (817.42, 9218.92),
    '482010617': (27049.57, 22045.66),
    '482010803': (8836.35, 15717.2),
    '482011015': (18413.8, 15068.96),
    '482011035': (1159.98, 12272.52),
    '482011039': (13661.93, 5193.24),
    '482016000': (1546.9, 6786.33),
}

def load_training_data():
    """Load training data"""
    print("Loading training data...")
    
    # Load PINN predictions
    pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    pinn_df['timestamp'] = pd.to_datetime(pinn_df['timestamp'])
    
    # Load sensor data
    sensor_df = pd.read_csv(TRAINING_SENSOR_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Find common timestamps
    common_times = sorted(set(pinn_df['timestamp']) & set(sensor_df['timestamp']))
    print(f"  Common timestamps: {len(common_times)}")
    
    # Filter to common timestamps
    pinn_df = pinn_df[pinn_df['timestamp'].isin(common_times)].sort_values('timestamp').reset_index(drop=True)
    sensor_df = sensor_df[sensor_df['timestamp'].isin(common_times)].sort_values('timestamp').reset_index(drop=True)
    
    # Extract sensor columns
    sensor_cols = [f'sensor_{sid}' for sid in sorted(SENSORS.keys())]
    
    # Get PINN predictions
    pinn_values = pinn_df[sensor_cols].values
    
    # Get actual sensor readings
    sensor_values = sensor_df[sensor_cols].values
    
    return pinn_values, sensor_values, common_times

def compute_validation_distribution():
    """Run validation and collect distribution data"""
    print("\nRunning validation to collect distribution data...")
    
    import torch
    from pinn import ParametricADEPINN
    from nn2 import NN2_CorrectionNetwork
    import pickle
    
    # Load models
    PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
    NN2_MODEL_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/nn2_master_model_spatial-3.pth"
    NN2_SCALERS_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/nn2_master_scalers-2.pkl"
    SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
    SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
    UNIT_CONVERSION = 313210039.9
    
    # Load PINN
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    pinn.eval()
    
    # Load sensor data
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Filter to 2019
    year_mask = (sensor_df['timestamp'] >= '2019-01-01') & (sensor_df['timestamp'] < '2020-01-01')
    sensor_df = sensor_df[year_mask].reset_index(drop=True)
    
    # Load facility data
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_dfs = []
    for f in facility_files:
        if 'summary' in f.name:
            continue
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['t'])
        facility_dfs.append(df)
    merged_facilities = pd.concat(facility_dfs, ignore_index=True)
    
    # Collect predictions
    pinn_predictions = []
    sensor_readings = []
    
    FORECAST_T_HOURS = 3.0
    
    for idx, row in sensor_df.iterrows():
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(sensor_df)}")
        
        forecast_timestamp = row['timestamp']
        met_data_timestamp = forecast_timestamp - pd.Timedelta(hours=3)
        
        facility_data = merged_facilities[merged_facilities['timestamp'] == met_data_timestamp]
        if len(facility_data) == 0:
            continue
        
        # Compute PINN predictions
        sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
        
        for _, fac_row in facility_data.iterrows():
            cx = fac_row['source_x_cartesian']
            cy = fac_row['source_y_cartesian']
            d = fac_row['source_diameter']
            Q = fac_row['Q_total']
            u = fac_row['wind_u']
            v = fac_row['wind_v']
            kappa = fac_row['D']
            
            for sensor_id, (sx, sy) in SENSORS.items():
                with torch.no_grad():
                    phi_raw = pinn(
                        torch.tensor([[sx]], dtype=torch.float32),
                        torch.tensor([[sy]], dtype=torch.float32),
                        torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32),
                        torch.tensor([[cx]], dtype=torch.float32),
                        torch.tensor([[cy]], dtype=torch.float32),
                        torch.tensor([[u]], dtype=torch.float32),
                        torch.tensor([[v]], dtype=torch.float32),
                        torch.tensor([[d]], dtype=torch.float32),
                        torch.tensor([[kappa]], dtype=torch.float32),
                        torch.tensor([[Q]], dtype=torch.float32)
                    )
                    concentration_ppb = phi_raw.item() * UNIT_CONVERSION
                    sensor_pinn_ppb[sensor_id] += concentration_ppb
        
        # Store predictions and actuals
        sensor_ids_sorted = sorted(SENSORS.keys())
        pinn_array = np.array([sensor_pinn_ppb[sid] for sid in sensor_ids_sorted])
        sensor_array = np.array([row.get(f'sensor_{sid}', np.nan) if f'sensor_{sid}' in row else np.nan 
                                 for sid in sensor_ids_sorted])
        # Replace NaN with 0 for consistency
        sensor_array = np.nan_to_num(sensor_array, nan=0.0)
        
        pinn_predictions.append(pinn_array)
        sensor_readings.append(sensor_array)
    
    pinn_predictions = np.array(pinn_predictions)
    sensor_readings = np.array(sensor_readings)
    
    return pinn_predictions, sensor_readings

def analyze_distributions(train_pinn, train_sensors, val_pinn, val_sensors):
    """Compare distributions"""
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS")
    print("="*80)
    
    sensor_ids = sorted(SENSORS.keys())
    
    results = {
        'training': {},
        'validation': {},
        'differences': {}
    }
    
    for i, sensor_id in enumerate(sensor_ids):
        train_p = train_pinn[:, i]
        train_s = train_sensors[:, i]
        val_p = val_pinn[:, i]
        val_s = val_sensors[:, i]
        
        # Filter out zeros and NaNs for statistics
        train_p_nonzero = train_p[(train_p != 0) & ~np.isnan(train_p)]
        train_s_nonzero = train_s[(train_s != 0) & ~np.isnan(train_s)]
        val_p_nonzero = val_p[(val_p != 0) & ~np.isnan(val_p)]
        val_s_nonzero = val_s[(val_s != 0) & ~np.isnan(val_s)]
        
        results['training'][sensor_id] = {
            'pinn': {
                'mean': np.mean(train_p_nonzero) if len(train_p_nonzero) > 0 else 0,
                'std': np.std(train_p_nonzero) if len(train_p_nonzero) > 0 else 0,
                'min': np.min(train_p_nonzero) if len(train_p_nonzero) > 0 else 0,
                'max': np.max(train_p_nonzero) if len(train_p_nonzero) > 0 else 0,
                'n_nonzero': len(train_p_nonzero),
                'n_total': len(train_p),
                'zero_ratio': 1.0 - len(train_p_nonzero) / len(train_p) if len(train_p) > 0 else 0
            },
            'sensors': {
                'mean': np.mean(train_s_nonzero) if len(train_s_nonzero) > 0 else 0,
                'std': np.std(train_s_nonzero) if len(train_s_nonzero) > 0 else 0,
                'min': np.min(train_s_nonzero) if len(train_s_nonzero) > 0 else 0,
                'max': np.max(train_s_nonzero) if len(train_s_nonzero) > 0 else 0,
                'n_nonzero': len(train_s_nonzero),
                'n_total': len(train_s),
                'zero_ratio': 1.0 - len(train_s_nonzero) / len(train_s) if len(train_s) > 0 else 0
            }
        }
        
        results['validation'][sensor_id] = {
            'pinn': {
                'mean': np.mean(val_p_nonzero) if len(val_p_nonzero) > 0 else 0,
                'std': np.std(val_p_nonzero) if len(val_p_nonzero) > 0 else 0,
                'min': np.min(val_p_nonzero) if len(val_p_nonzero) > 0 else 0,
                'max': np.max(val_p_nonzero) if len(val_p_nonzero) > 0 else 0,
                'n_nonzero': len(val_p_nonzero),
                'n_total': len(val_p),
                'zero_ratio': 1.0 - len(val_p_nonzero) / len(val_p) if len(val_p) > 0 else 0
            },
            'sensors': {
                'mean': np.mean(val_s_nonzero) if len(val_s_nonzero) > 0 else 0,
                'std': np.std(val_s_nonzero) if len(val_s_nonzero) > 0 else 0,
                'min': np.min(val_s_nonzero) if len(val_s_nonzero) > 0 else 0,
                'max': np.max(val_s_nonzero) if len(val_s_nonzero) > 0 else 0,
                'n_nonzero': len(val_s_nonzero),
                'n_total': len(val_s),
                'zero_ratio': 1.0 - len(val_s_nonzero) / len(val_s) if len(val_s) > 0 else 0
            }
        }
        
        # Calculate differences
        results['differences'][sensor_id] = {
            'pinn_mean_diff': results['validation'][sensor_id]['pinn']['mean'] - results['training'][sensor_id]['pinn']['mean'],
            'pinn_std_diff': results['validation'][sensor_id]['pinn']['std'] - results['training'][sensor_id]['pinn']['std'],
            'sensors_mean_diff': results['validation'][sensor_id]['sensors']['mean'] - results['training'][sensor_id]['sensors']['mean'],
            'sensors_std_diff': results['validation'][sensor_id]['sensors']['std'] - results['training'][sensor_id]['sensors']['std'],
        }
    
    return results

def print_distribution_summary(results):
    """Print distribution summary"""
    print("\n" + "="*80)
    print("TRAINING vs VALIDATION DISTRIBUTION COMPARISON")
    print("="*80)
    
    sensor_ids = sorted(SENSORS.keys())
    
    for sensor_id in sensor_ids:
        print(f"\n{'='*80}")
        print(f"Sensor {sensor_id}")
        print("="*80)
        
        train = results['training'][sensor_id]
        val = results['validation'][sensor_id]
        diff = results['differences'][sensor_id]
        
        print("\nPINN Predictions:")
        print(f"  Training:  mean={train['pinn']['mean']:.4f}, std={train['pinn']['std']:.4f}, "
              f"range=[{train['pinn']['min']:.4f}, {train['pinn']['max']:.4f}], "
              f"non-zero={train['pinn']['n_nonzero']}/{train['pinn']['n_total']} ({100*(1-train['pinn']['zero_ratio']):.1f}%)")
        print(f"  Validation: mean={val['pinn']['mean']:.4f}, std={val['pinn']['std']:.4f}, "
              f"range=[{val['pinn']['min']:.4f}, {val['pinn']['max']:.4f}], "
              f"non-zero={val['pinn']['n_nonzero']}/{val['pinn']['n_total']} ({100*(1-val['pinn']['zero_ratio']):.1f}%)")
        print(f"  Difference: mean_diff={diff['pinn_mean_diff']:.4f}, std_diff={diff['pinn_std_diff']:.4f}")
        
        print("\nSensor Readings:")
        print(f"  Training:  mean={train['sensors']['mean']:.4f}, std={train['sensors']['std']:.4f}, "
              f"range=[{train['sensors']['min']:.4f}, {train['sensors']['max']:.4f}], "
              f"non-zero={train['sensors']['n_nonzero']}/{train['sensors']['n_total']} ({100*(1-train['sensors']['zero_ratio']):.1f}%)")
        print(f"  Validation: mean={val['sensors']['mean']:.4f}, std={val['sensors']['std']:.4f}, "
              f"range=[{val['sensors']['min']:.4f}, {val['sensors']['max']:.4f}], "
              f"non-zero={val['sensors']['n_nonzero']}/{val['sensors']['n_total']} ({100*(1-val['sensors']['zero_ratio']):.1f}%)")
        print(f"  Difference: mean_diff={diff['sensors_mean_diff']:.4f}, std_diff={diff['sensors_std_diff']:.4f}")

def check_overfitting():
    """Check for overfitting indicators"""
    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS")
    print("="*80)
    
    # Load leave-one-out results
    with open(LEAVE_ONE_OUT_RESULTS, 'r') as f:
        loo_results = json.load(f)
    
    print("\nLeave-One-Out Cross-Validation Results:")
    print("-" * 80)
    
    held_out_improvements = []
    training_improvements = []
    
    for fold, data in loo_results.items():
        held_out = data['held_out']
        training = data['training_sensors_avg']
        
        held_out_improvements.append(held_out['improvement'])
        training_improvements.append(training['improvement'])
        
        print(f"\nFold {fold}:")
        print(f"  Held-out sensor: PINN MAE={held_out['pinn_mae']:.4f}, NN2 MAE={held_out['nn2_mae']:.4f}, "
              f"Improvement={held_out['improvement']:.1f}%")
        print(f"  Training sensors: NN2 MAE={training['nn2_mae']:.4f}, Improvement={training['improvement']:.1f}%")
    
    avg_held_out = np.mean(held_out_improvements)
    avg_training = np.mean(training_improvements)
    
    print("\n" + "-" * 80)
    print(f"Average held-out improvement: {avg_held_out:.1f}%")
    print(f"Average training improvement: {avg_training:.1f}%")
    print(f"Gap (training - held-out): {avg_training - avg_held_out:.1f}%")
    
    if avg_training - avg_held_out > 20:
        print("\n⚠️  WARNING: Large gap suggests potential overfitting!")
        print("   Training performance is much better than held-out performance.")
    elif avg_training - avg_held_out < 5:
        print("\n✓ Good generalization: Training and held-out performance are similar.")
    else:
        print("\n⚠️  Moderate gap: Some overfitting may be present.")

def main():
    print("="*80)
    print("DATA DISTRIBUTION AND OVERFITTING INVESTIGATION")
    print("="*80)
    
    # Load training data
    train_pinn, train_sensors, train_times = load_training_data()
    
    # Compute validation distribution
    val_pinn, val_sensors = compute_validation_distribution()
    
    # Analyze distributions
    results = analyze_distributions(train_pinn, train_sensors, val_pinn, val_sensors)
    
    # Print summary
    print_distribution_summary(results)
    
    # Check for overfitting
    check_overfitting()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

