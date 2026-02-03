#!/usr/bin/env python3
"""
Investigate Training Data Adequacy

Analyzes:
1. Data quantity (samples, data-to-parameter ratio)
2. Sample distribution per sensor
3. Data quality issues
4. Whether more data is needed
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting')

# Paths
TRAINING_DATA_PATH = '/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata/total_superimposed_concentrations.csv'
SENSOR_DATA_PATH = '/Users/neevpratap/Downloads/sensors_final_synced.csv'

def calculate_model_parameters():
    """Calculate number of parameters in NN2 model"""
    # Architecture: 36 → 512 → 512 → 256 → 128 → 9
    params = (
        36 * 512 + 512 +           # Layer 1: Linear + bias
        512 * 512 + 512 +          # Layer 2: Linear + bias
        512 * 256 + 256 +          # Layer 3: Linear + bias
        256 * 128 + 128 +          # Layer 4: Linear + bias
        128 * 9 + 9                # Layer 5: Linear + bias
    )
    # BatchNorm parameters (mean, var, weight, bias for each layer)
    batchnorm_params = (
        512 * 4 +                  # BN after layer 1
        512 * 4 +                  # BN after layer 2
        256 * 4 +                  # BN after layer 3
        128 * 4                    # BN after layer 4
    )
    total = params + batchnorm_params
    return total, params, batchnorm_params

def analyze_training_data():
    """Analyze training data quantity and quality"""
    print("="*80)
    print("TRAINING DATA ADEQUACY ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    pinn_df = pd.read_csv(TRAINING_DATA_PATH)
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    
    # Handle timestamps
    if 't' in pinn_df.columns:
        pinn_df['timestamp'] = pd.to_datetime(pinn_df['t'])
    elif 'timestamp' in pinn_df.columns:
        pinn_df['timestamp'] = pd.to_datetime(pinn_df['timestamp'])
    
    if 't' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
    elif 'timestamp' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Find overlapping timestamps
    pinn_times = set(pinn_df['timestamp'])
    sensor_times = set(sensor_df['timestamp'])
    common_times = sorted(list(pinn_times & sensor_times))
    
    print("1. DATA QUANTITY")
    print("-"*80)
    print(f"   PINN data rows: {len(pinn_df)}")
    print(f"   Sensor data rows: {len(sensor_df)}")
    print(f"   Overlapping timestamps: {len(common_times)}")
    print(f"   Unique timestamps: {len(common_times)}")
    print()
    
    # Calculate model parameters
    total_params, linear_params, bn_params = calculate_model_parameters()
    print("2. MODEL COMPLEXITY")
    print("-"*80)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Linear layer params: {linear_params:,}")
    print(f"   BatchNorm params: {bn_params:,}")
    print()
    
    # Data-to-parameter ratio
    samples = len(common_times)
    data_ratio = samples / total_params
    print("3. DATA-TO-PARAMETER RATIO")
    print("-"*80)
    print(f"   Training samples: {samples:,}")
    print(f"   Model parameters: {total_params:,}")
    print(f"   Ratio: {data_ratio:.6f}")
    print()
    
    if data_ratio < 0.1:
        print("   ⚠️  VERY LOW RATIO - High risk of overfitting!")
        print("   Recommendation: Need at least 10x more data or simpler model")
    elif data_ratio < 1.0:
        print("   ⚠️  LOW RATIO - Risk of overfitting")
        print("   Recommendation: More data or stronger regularization")
    elif data_ratio < 10.0:
        print("   ✓ MODERATE RATIO - May be sufficient with regularization")
    else:
        print("   ✓ GOOD RATIO - Sufficient data")
    print()
    
    # Per-sensor analysis
    sensor_cols = [c for c in pinn_df.columns if c.startswith('sensor_')]
    n_sensors = len(sensor_cols)
    samples_per_sensor = samples / n_sensors
    
    print("4. PER-SENSOR ANALYSIS")
    print("-"*80)
    print(f"   Number of sensors: {n_sensors}")
    print(f"   Samples per sensor: {samples_per_sensor:.1f}")
    print()
    
    # Check data distribution per sensor
    print("   Sensor-specific sample counts (non-zero PINN values):")
    for col in sensor_cols:
        non_zero = (pinn_df[col] > 0).sum()
        print(f"     {col}: {non_zero} non-zero samples ({non_zero/samples*100:.1f}%)")
    print()
    
    # Data quality checks
    print("5. DATA QUALITY CHECKS")
    print("-"*80)
    
    # Check for missing values
    pinn_missing = pinn_df[sensor_cols].isna().sum().sum()
    print(f"   Missing PINN values: {pinn_missing}")
    
    # Check for zeros
    all_pinn = pinn_df[sensor_cols].values.flatten()
    zero_count = (all_pinn == 0).sum()
    zero_pct = zero_count / len(all_pinn) * 100
    print(f"   Zero PINN values: {zero_count} ({zero_pct:.1f}%)")
    
    # Check for outliers
    all_pinn_nonzero = all_pinn[all_pinn > 0]
    q99 = np.percentile(all_pinn_nonzero, 99)
    outliers = (all_pinn_nonzero > q99).sum()
    print(f"   Outliers (>99th percentile): {outliers} ({outliers/len(all_pinn_nonzero)*100:.1f}%)")
    print()
    
    # Distribution statistics
    print("6. DATA DISTRIBUTION")
    print("-"*80)
    print(f"   PINN values (non-zero):")
    print(f"     Mean: {np.mean(all_pinn_nonzero):.4f} ppb")
    print(f"     Median: {np.median(all_pinn_nonzero):.4f} ppb")
    print(f"     Std: {np.std(all_pinn_nonzero):.4f} ppb")
    print(f"     Min: {np.min(all_pinn_nonzero):.4f} ppb")
    print(f"     Max: {np.max(all_pinn_nonzero):.4f} ppb")
    print(f"     99th percentile: {q99:.4f} ppb")
    print()
    
    # Recommendations
    print("7. RECOMMENDATIONS")
    print("-"*80)
    
    if data_ratio < 0.1:
        print("   🔴 CRITICAL: Data-to-parameter ratio is extremely low")
        print("      Options:")
        print("      1. Collect 10-50x more training data")
        print("      2. Simplify model architecture (fewer parameters)")
        print("      3. Use stronger regularization (dropout, weight decay)")
        print("      4. Use data augmentation")
    elif data_ratio < 1.0:
        print("   🟡 WARNING: Data-to-parameter ratio is low")
        print("      Options:")
        print("      1. Collect more training data (2-5x more)")
        print("      2. Simplify model architecture")
        print("      3. Increase regularization")
    else:
        print("   ✓ Data quantity appears adequate")
        print("      Focus investigation on other aspects (architecture, learnability)")
    
    print()
    print("="*80)
    
    return {
        'samples': samples,
        'total_params': total_params,
        'data_ratio': data_ratio,
        'samples_per_sensor': samples_per_sensor,
        'n_sensors': n_sensors
    }

if __name__ == '__main__':
    results = analyze_training_data()

