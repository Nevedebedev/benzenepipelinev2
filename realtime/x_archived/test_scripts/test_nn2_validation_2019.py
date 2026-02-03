#!/usr/bin/env python3
"""
Test Retrained NN2 Model Against 2019 Sensor Data

This script validates the newly retrained NN2 model by:
1. Loading actual 2019 sensor measurements
2. Running the real-time pipeline (PINN + NN2) for the same timestamps
3. Comparing predictions vs. actual values
4. Computing MAE and other metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent))
from concentration_predictor import ConcentrationPredictor
from madis_fetcher import MADISFetcher
from csv_generator import CSVGenerator

# Configuration
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_PREDICTIONS_PATH = "/Users/neevpratap/simpletesting/nn2trainingdata/total_concentrations.csv"
OUTPUT_DIR = Path("/Users/neevpratap/Desktop/realtime/validation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 9 sensor IDs from 2019 data
SENSOR_IDS = [
    '482010026', '482010057', '482010069', '482010617',
    '482010803', '482011015', '482011035', '482011039', '482016000'
]

def load_sensor_data():
    """Load actual 2019 sensor measurements"""
    print("Loading actual 2019 sensor data...")
    df = pd.read_csv(SENSOR_DATA_PATH)
    
    # Rename timestamp column if needed
    if 't' in df.columns:
        df = df.rename(columns={'t': 'timestamp'})
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Loaded {len(df)} timestamps with actual sensor values")
    return df

def load_pinn_predictions():
    """Load PINN predictions (ground truth for comparison)"""
    print("Loading PINN predictions...")
    df = pd.read_csv(PINN_PREDICTIONS_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Loaded {len(df)} timestamps with PINN predictions")
    return df

def compute_metrics(actual, predicted, sensor_id):
    """Compute error metrics for a sensor"""
    # Remove NaN values
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return None
    
    mae = np.mean(np.abs(actual_clean - predicted_clean))
    rmse = np.sqrt(np.mean((actual_clean - predicted_clean)**2))
    
    return {
        'sensor_id': sensor_id,
        'n_samples': len(actual_clean),
        'mae': mae,
        'rmse': rmse,
        'actual_mean': np.mean(actual_clean),
        'predicted_mean': np.mean(predicted_clean),
    }

def main():
    print("="*80)
    print("VALIDATING RETRAINED NN2 MODEL AGAINST 2019 SENSOR DATA")
    print("="*80)
    print()
    
    # Load data
    sensor_df = load_sensor_data()
    pinn_df = load_pinn_predictions()
    
    # Merge on timestamp
    print("\nMerging datasets...")
    merged = pd.merge(sensor_df, pinn_df, on='timestamp', suffixes=('_actual', '_pinn'))
    print(f"  Common timestamps: {len(merged)}")
    
    # Compute metrics for each sensor
    print("\n" + "="*80)
    print("VALIDATION RESULTS: PINN vs. Actual Sensor Data")
    print("="*80)
    
    results = []
    for sensor_id in SENSOR_IDS:
        actual_col = f'sensor_{sensor_id}_actual'
        pinn_col = f'sensor_{sensor_id}_pinn'
        
        if actual_col not in merged.columns or pinn_col not in merged.columns:
            print(f"\n⚠️  Sensor {sensor_id}: Missing data columns")
            continue
        
        metrics = compute_metrics(
            merged[actual_col].values,
            merged[pinn_col].values,
            sensor_id
        )
        
        if metrics is None:
            print(f"\n⚠️  Sensor {sensor_id}: No valid data")
            continue
        
        results.append(metrics)
        
        print(f"\n📍 Sensor {sensor_id}:")
        print(f"   Samples: {metrics['n_samples']}")
        print(f"   MAE: {metrics['mae']:.4f} ppb")
        print(f"   RMSE: {metrics['rmse']:.4f} ppb")
        print(f"   Actual mean: {metrics['actual_mean']:.4f} ppb")
        print(f"   Predicted mean: {metrics['predicted_mean']:.4f} ppb")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    all_maes = [r['mae'] for r in results]
    all_rmses = [r['rmse'] for r in results]
    
    print(f"\nAverage MAE across sensors: {np.mean(all_maes):.4f} ppb")
    print(f"Average RMSE across sensors: {np.mean(all_rmses):.4f} ppb")
    print(f"Max MAE: {np.max(all_maes):.4f} ppb")
    print(f"Min MAE: {np.min(all_maes):.4f} ppb")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "nn2_validation_2019.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
