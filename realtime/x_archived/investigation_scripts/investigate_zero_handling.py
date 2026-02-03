#!/usr/bin/env python3
"""
CRITICAL INVESTIGATION: Zero-value handling in NN2

The training code fits scalers ONLY on non-zero values (lines 280-290 in nn2.py).
This script checks:
1. If there are zeros in the PINN predictions
2. How the training code vs validation code handles zeros
3. If scaler transform/inverse_transform behaves differently with zeros
"""

import sys
sys.path.append('/Users/neevpratap/simpletesting')

import pandas as pd
import numpy as np
import pickle

# Load data
training_pinn = pd.read_csv("/Users/neevpratap/simpletesting/nn2trainingdata/total_concentrations.csv")
training_sensor = pd.read_csv("/Users/neevpratap/Downloads/sensors_final_synced.csv")

# Load scalers
with open("/Users/neevpratap/Desktop/nn2_updated/nn2_master_scalers-2.pkl", 'rb') as f:
    scalers = pickle.load(f)

print("="*80)
print("ZERO-VALUE INVESTIGATION")
print("="*80)

# Check for zeros in training PINN data
print("\n1. Checking for zeros in training PINN data...")
sensor_cols = [c for c in training_pinn.columns if c.startswith('sensor_')]

zero_counts = {}
for col in sensor_cols:
    zeros = (training_pinn[col] == 0.0).sum()
    total = len(training_pinn)
    zero_counts[col] = (zeros, total, zeros/total*100)
    print(f"  {col}: {zeros}/{total} zeros ({zeros/total*100:.1f}%)")

print("\n2. Testing scaler behavior on zeros...")
print("\nSCALER: pinn")
print(f"  Mean: {scalers['pinn'].mean_[0]:.6f}")
print(f"  Var:  {scalers['pinn'].var_[0]:.6f}")

# Test transformation of zeros
test_values = np.array([0.0, 0.1, 1.0, 10.0]).reshape(-1, 1)
transformed = scalers['pinn'].transform(test_values)
print("\nTransforming values:")
for orig, trans in zip(test_values.flatten(), transformed.flatten()):
    print(f"  {orig:.4f} → {trans:.6f}")

print("\n3. Checking training vs validation scaling...")

# Simulate how training code scales (only non-zero)
jan_pinn = training_pinn[training_pinn['timestamp'].str.startswith('2019-01')]
sample_sensor = jan_pinn['sensor_482010026'].values

nonzero_values = sample_sensor[sample_sensor != 0]
print(f"\nsensor_482010026 in January:")
print(f"  Total samples: {len(sample_sensor)}")
print(f"  Non-zero samples: {len(nonzero_values)}")
print(f"  Zero samples: {len(sample_sensor) - len(nonzero_values)}")
print(f"  Mean (all): {sample_sensor.mean():.6f}")
print(f"  Mean (non-zero only): {nonzero_values.mean():.6f}")

# Test transforming with zeros included vs excluded
print("\nScaler transform:")
print(f"  Scaler was fit on non-zero values only")
print(f"  Scaler mean: {scalers['pinn'].mean_[0]:.6f}")

# What happens when we transform a zero?
zero_transformed = scalers['pinn'].transform(np.array([[0.0]]))
print(f"\n  0.0 transforms to: {zero_transformed[0][0]:.6f}")
print(f"  This is {abs(zero_transformed[0][0])} standard deviations from mean")

print("\n4. Critical finding:")
if abs(zero_transformed[0][0]) > 2.0:
    print("  ⚠️  ZEROS TRANSFORM TO EXTREME VALUES!")
    print("  This will cause NN2 to make incorrect corrections on zero predictions")
else:
    print("  ✓ Zeros transform reasonably")

print("\n5. Checking validation script behavior...") 
print("  Validation script does NOT filter zeros before transforming")
print("  This means zeros get transformed to extreme scaled values")
print("  NN2 then makes corrections based on these extreme values")
print("  Result: Nonsensical corrections for zero-valued predictions")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nThe training code fits scalers on NON-ZERO values only.")
print("The validation code transforms ALL values (including zeros).")
print("\nThis mismatch causes:")
print("1. Zeros → extreme scaled values")
print("2. NN2 makes corrections based on these extreme values")
print("3. After inverse transform, predictions are corrupted")
print("\n**FIX**: Validation must handle zeros the same way as training:")
print("  - Don't transform zeros")
print("  - Or mask them out before NN2")
print("  - Or retrain with a better scaler (RobustScaler)")

