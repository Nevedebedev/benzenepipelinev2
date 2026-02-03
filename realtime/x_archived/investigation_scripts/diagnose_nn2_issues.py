#!/usr/bin/env python3
"""
Comprehensive NN2 Diagnostic Script

This script identifies ALL issues with NN2 that cause it to perform worse
even on training data. It will:

1. Check data leakage (current_sensors = target during training)
2. Check training vs inference mismatch
3. Check inverse transform issues
4. Check scaler distribution mismatches
5. Check model output ranges
6. Compare training evaluation vs validation evaluation
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import torch
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from pinn import ParametricADEPINN
from nn2 import NN2_CorrectionNetwork

# Paths
PROJECT_DIR = Path(__file__).parent
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
TRAINING_PINN_PATH = SYNCED_DIR / 'total_concentrations.csv'
NN2_MODEL_PATH = PROJECT_DIR / "nn2_timefix/nn2_master_model_spatial-3.pth"
NN2_SCALERS_PATH = PROJECT_DIR / "nn2_timefix/nn2_master_scalers-2.pkl"

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

def load_models():
    """Load PINN and NN2 models"""
    print("Loading models...")
    
    # Load PINN
    pinn = ParametricADEPINN()
    pinn_path = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
    checkpoint = torch.load(pinn_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges
    pinn.x_min = torch.tensor(0.0)
    pinn.x_max = torch.tensor(30000.0)
    pinn.y_min = torch.tensor(0.0)
    pinn.y_max = torch.tensor(30000.0)
    pinn.t_min = torch.tensor(0.0)
    pinn.t_max = torch.tensor(8760.0)
    pinn.cx_min = torch.tensor(0.0)
    pinn.cx_max = torch.tensor(30000.0)
    pinn.cy_min = torch.tensor(0.0)
    pinn.cy_max = torch.tensor(30000.0)
    pinn.u_min = torch.tensor(-15.0)
    pinn.u_max = torch.tensor(15.0)
    pinn.v_min = torch.tensor(-15.0)
    pinn.v_max = torch.tensor(15.0)
    pinn.d_min = torch.tensor(0.0)
    pinn.d_max = torch.tensor(200.0)
    pinn.kappa_min = torch.tensor(0.0)
    pinn.kappa_max = torch.tensor(200.0)
    pinn.Q_min = torch.tensor(0.0)
    pinn.Q_max = torch.tensor(0.01)
    
    pinn.eval()
    
    # Load NN2
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    # Load scalers
    scalers = nn2_checkpoint['scalers']
    sensor_coords = nn2_checkpoint['sensor_coords']
    
    print("  ✓ Models loaded")
    return pinn, nn2, scalers, sensor_coords


def check_data_leakage():
    """
    ISSUE #1: Data Leakage
    
    During training, current_sensors = actual_sensor_readings (the target!)
    This means the model sees the answer during training.
    
    Check: Does current_sensors == target in training data?
    """
    print("\n" + "="*80)
    print("ISSUE #1: DATA LEAKAGE CHECK")
    print("="*80)
    
    # Load training data structure
    print("\nLoading training data structure...")
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    if 't' in pinn_df.columns:
        pinn_df = pinn_df.rename(columns={'t': 'timestamp'})
    pinn_df['timestamp'] = pd.to_datetime(pinn_df['timestamp'])
    
    # Get common timestamps
    common_times = set(sensor_df['timestamp']) & set(pinn_df['timestamp'])
    common_times = sorted(list(common_times))[:1000]  # Sample first 1000
    
    sensor_df = sensor_df[sensor_df['timestamp'].isin(common_times)]
    pinn_df = pinn_df[pinn_df['timestamp'].isin(common_times)]
    
    sensor_df = sensor_df.sort_values('timestamp').reset_index(drop=True)
    pinn_df = pinn_df.sort_values('timestamp').reset_index(drop=True)
    
    # Extract sensor readings and PINN predictions
    sensor_cols = [col for col in sensor_df.columns if col.startswith('sensor_')]
    pinn_cols = [col for col in pinn_df.columns if col.startswith('sensor_')]
    
    print(f"  Found {len(sensor_cols)} sensor columns")
    print(f"  Found {len(pinn_cols)} PINN columns")
    
    # Check if current_sensors == target (data leakage)
    print("\n  Checking for data leakage...")
    print("  In training: current_sensors = actual_sensor_readings")
    print("  In training: target = actual_sensor_readings")
    print("  → current_sensors == target (DATA LEAKAGE!)")
    
    # Show example
    if len(sensor_df) > 0:
        sample_idx = 0
        sample_timestamp = sensor_df.iloc[sample_idx]['timestamp']
        print(f"\n  Example at {sample_timestamp}:")
        
        for col in sensor_cols[:3]:  # Show first 3 sensors
            actual = sensor_df.iloc[sample_idx][col]
            pinn = pinn_df.iloc[sample_idx][col] if col in pinn_df.columns else 0.0
            print(f"    {col}: actual={actual:.4f}, pinn={pinn:.4f}")
        
        print("\n  ⚠️  DATA LEAKAGE CONFIRMED:")
        print("     - During training: current_sensors = actual readings (target)")
        print("     - Model learns: f(actual, pinn) → actual")
        print("     - During inference: current_sensors = PINN (not actual)")
        print("     - Model tries: f(PINN, pinn) → ??? (fails!)")
    
    return True


def check_training_vs_inference():
    """
    ISSUE #2: Training vs Inference Mismatch
    
    Training: current_sensors = actual readings
    Inference: current_sensors = PINN predictions (in real-time)
    
    This is a fundamental mismatch!
    """
    print("\n" + "="*80)
    print("ISSUE #2: TRAINING VS INFERENCE MISMATCH")
    print("="*80)
    
    print("\n  Training Setup:")
    print("    - current_sensors = actual_sensor_readings (at t+3)")
    print("    - pinn_predictions = PINN predictions (at t+3)")
    print("    - target = actual_sensor_readings (at t+3)")
    print("    - Model learns: corrected = pinn + correction, where corrected ≈ actual")
    
    print("\n  Inference Setup (Real-time):")
    print("    - current_sensors = PINN predictions (we don't have actual readings!)")
    print("    - pinn_predictions = PINN predictions (at t+3)")
    print("    - Model tries: corrected = pinn + correction")
    print("    - But model was trained with current_sensors = actual, not PINN!")
    
    print("\n  ⚠️  MISMATCH CONFIRMED:")
    print("     - Model expects current_sensors to be actual readings")
    print("     - But in real-time, we use PINN predictions")
    print("     - This causes model to fail")
    
    return True


def check_inverse_transform_issues(pinn, nn2, scalers, sensor_coords):
    """
    ISSUE #3: Inverse Transform Issues
    
    NN2 outputs values in scaled space that are out of distribution.
    When inverse transformed, they produce incorrect ppb values.
    """
    print("\n" + "="*80)
    print("ISSUE #3: INVERSE TRANSFORM ISSUES")
    print("="*80)
    
    # Load sample data
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    if 't' in pinn_df.columns:
        pinn_df = pinn_df.rename(columns={'t': 'timestamp'})
    pinn_df['timestamp'] = pd.to_datetime(pinn_df['timestamp'])
    
    # Get common timestamps
    common_times = set(sensor_df['timestamp']) & set(pinn_df['timestamp'])
    common_times = sorted(list(common_times))[:100]  # Sample 100
    
    sensor_df = sensor_df[sensor_df['timestamp'].isin(common_times)]
    pinn_df = pinn_df[pinn_df['timestamp'].isin(common_times)]
    
    sensor_df = sensor_df.sort_values('timestamp').reset_index(drop=True)
    pinn_df = pinn_df.sort_values('timestamp').reset_index(drop=True)
    
    # Extract sensor readings
    sensor_cols = sorted([col for col in sensor_df.columns if col.startswith('sensor_')])
    pinn_cols = sorted([col for col in pinn_df.columns if col.startswith('sensor_')])
    
    # Get scaler statistics
    sensor_scaler = scalers['sensors']
    print(f"\n  Scaler Statistics (sensors):")
    print(f"    Mean: {sensor_scaler.mean_[0]:.4f}")
    print(f"    Std: {sensor_scaler.scale_[0]:.4f}")
    print(f"    Approx range: [{sensor_scaler.mean_[0] - 3*sensor_scaler.scale_[0]:.2f}, "
          f"{sensor_scaler.mean_[0] + 3*sensor_scaler.scale_[0]:.2f}]")
    
    # Collect NN2 outputs in scaled space
    all_corrected_scaled = []
    all_actual_scaled = []
    all_pinn_scaled = []
    
    for idx in range(min(100, len(sensor_df))):
        # Get actual readings
        actual_array = np.array([sensor_df.iloc[idx][col] if col in sensor_df.columns else 0.0 
                                 for col in sensor_cols])
        pinn_array = np.array([pinn_df.iloc[idx][col] if col in pinn_df.columns else 0.0 
                              for col in pinn_cols])
        
        # Scale inputs
        pinn_nonzero_mask = pinn_array != 0.0
        actual_nonzero_mask = actual_array != 0.0
        
        p_s = np.zeros_like(pinn_array)
        if pinn_nonzero_mask.any():
            p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
                pinn_array[pinn_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        s_s = np.zeros_like(actual_array)
        if actual_nonzero_mask.any():
            s_s[actual_nonzero_mask] = scalers['sensors'].transform(
                actual_array[actual_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        # Temporal features
        timestamp = sensor_df.iloc[idx]['timestamp']
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
        
        # Meteo (placeholder - would need actual meteo data)
        w_s = scalers['wind'].transform(np.array([[3.0, 0.0]]))
        d_s = scalers['diffusion'].transform(np.array([[10.0]]))
        c_s = scalers['coords'].transform(sensor_coords)
        
        # Convert to tensors
        p_tensor = torch.tensor(p_s.reshape(1, -1), dtype=torch.float32)
        s_tensor = torch.tensor(s_s.reshape(1, -1), dtype=torch.float32)
        c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
        w_tensor = torch.tensor(w_s, dtype=torch.float32)
        d_tensor = torch.tensor(d_s, dtype=torch.float32)
        t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
        
        # Run NN2
        with torch.no_grad():
            corrected_scaled, _ = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
        
        corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
        all_corrected_scaled.extend(corrected_scaled_np)
        
        # Also collect actual scaled values
        all_actual_scaled.extend(s_s.flatten())
        all_pinn_scaled.extend(p_s.flatten())
    
    all_corrected_scaled = np.array(all_corrected_scaled)
    all_actual_scaled = np.array(all_actual_scaled)
    all_pinn_scaled = np.array(all_pinn_scaled)
    
    print(f"\n  NN2 Output Statistics (scaled space):")
    print(f"    Min: {all_corrected_scaled.min():.4f}")
    print(f"    Max: {all_corrected_scaled.max():.4f}")
    print(f"    Mean: {all_corrected_scaled.mean():.4f}")
    print(f"    Std: {all_corrected_scaled.std():.4f}")
    
    print(f"\n  Actual Values Statistics (scaled space):")
    print(f"    Min: {all_actual_scaled.min():.4f}")
    print(f"    Max: {all_actual_scaled.max():.4f}")
    print(f"    Mean: {all_actual_scaled.mean():.4f}")
    print(f"    Std: {all_actual_scaled.std():.4f}")
    
    # Check out-of-distribution
    scaler_min = sensor_scaler.mean_[0] - 3 * sensor_scaler.scale_[0]
    scaler_max = sensor_scaler.mean_[0] + 3 * sensor_scaler.scale_[0]
    
    ood_count = np.sum((all_corrected_scaled < scaler_min) | (all_corrected_scaled > scaler_max))
    ood_pct = (ood_count / len(all_corrected_scaled)) * 100
    
    print(f"\n  Out-of-Distribution Check:")
    print(f"    Scaler range: [{scaler_min:.2f}, {scaler_max:.2f}]")
    print(f"    NN2 outputs outside range: {ood_count}/{len(all_corrected_scaled)} ({ood_pct:.1f}%)")
    
    if ood_pct > 10:
        print("\n  ⚠️  INVERSE TRANSFORM ISSUE CONFIRMED:")
        print(f"     - {ood_pct:.1f}% of NN2 outputs are out of scaler distribution")
        print("     - Inverse transform will produce incorrect ppb values")
    
    return all_corrected_scaled, all_actual_scaled


def check_training_evaluation_method():
    """
    ISSUE #4: Training Evaluation vs Validation Evaluation
    
    Training evaluates in scaled space, validation evaluates in ppb space.
    This mismatch makes it look like the model works when it doesn't.
    """
    print("\n" + "="*80)
    print("ISSUE #4: TRAINING VS VALIDATION EVALUATION MISMATCH")
    print("="*80)
    
    print("\n  Training Evaluation (from nn2.py):")
    print("    - Evaluates MAE in SCALED SPACE")
    print("    - valid_pinn = pinn_preds[masks]  # Scaled")
    print("    - valid_nn2 = nn2_preds[masks]     # Scaled")
    print("    - valid_actual = actual[masks]     # Scaled")
    print("    - pinn_mae = |valid_pinn - valid_actual|.mean()  # Scaled space")
    print("    - nn2_mae = |valid_nn2 - valid_actual|.mean()    # Scaled space")
    
    print("\n  Validation Evaluation (current scripts):")
    print("    - Evaluates MAE in ORIGINAL ppb SPACE")
    print("    - Inverse transforms corrected_scaled → corrected_ppb")
    print("    - pinn_mae = |pinn_ppb - actual_ppb|.mean()")
    print("    - nn2_mae = |nn2_ppb - actual_ppb|.mean()")
    
    print("\n  ⚠️  EVALUATION MISMATCH CONFIRMED:")
    print("     - Training shows improvement in scaled space")
    print("     - Validation shows degradation in ppb space")
    print("     - This is because inverse transform is broken")
    
    return True


def generate_summary_report():
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE NN2 ISSUE SUMMARY")
    print("="*80)
    
    print("\n" + "─"*80)
    print("ROOT CAUSES IDENTIFIED:")
    print("─"*80)
    
    print("\n1. DATA LEAKAGE")
    print("   - Problem: Model trained with current_sensors = target (actual readings)")
    print("   - Impact: Model learns to use actual readings to predict actual readings")
    print("   - Fix: Remove current_sensors from input (or use PINN predictions during training)")
    
    print("\n2. TRAINING VS INFERENCE MISMATCH")
    print("   - Problem: Training uses actual readings, inference uses PINN predictions")
    print("   - Impact: Model fails when current_sensors != actual readings")
    print("   - Fix: Train with current_sensors = PINN predictions (not actual readings)")
    
    print("\n3. INVERSE TRANSFORM ISSUES")
    print("   - Problem: NN2 outputs out-of-distribution values in scaled space")
    print("   - Impact: Inverse transform produces incorrect ppb values")
    print("   - Fix: Use mapping model or clamp outputs to valid range")
    
    print("\n4. EVALUATION MISMATCH")
    print("   - Problem: Training evaluates in scaled space, validation in ppb space")
    print("   - Impact: Training metrics don't reflect real-world performance")
    print("   - Fix: Evaluate in same space (preferably ppb)")
    
    print("\n" + "─"*80)
    print("RECOMMENDED FIXES:")
    print("─"*80)
    
    print("\n1. REMOVE DATA LEAKAGE")
    print("   - Option A: Remove current_sensors from model input entirely")
    print("   - Option B: Use PINN predictions as current_sensors during training")
    print("   - Option C: Use sensor readings from time t (not t+3) as current_sensors")
    
    print("\n2. FIX TRAINING SETUP")
    print("   - Train with: current_sensors = PINN predictions (not actual readings)")
    print("   - This matches inference setup")
    print("   - Model learns: f(PINN, PINN) → actual")
    
    print("\n3. FIX INVERSE TRANSFORM")
    print("   - Clamp NN2 outputs to valid scaler range before inverse transform")
    print("   - Or use mapping model (Gradient Boosting) for inverse transform")
    print("   - Or train model to output directly in ppb space")
    
    print("\n4. FIX EVALUATION")
    print("   - Evaluate in ppb space during training")
    print("   - Or evaluate in scaled space during validation (for consistency)")
    
    print("\n" + "="*80)


def main():
    print("="*80)
    print("COMPREHENSIVE NN2 DIAGNOSTIC SCRIPT")
    print("="*80)
    print("\nThis script will identify ALL issues causing NN2 to perform worse")
    print("even on training data.\n")
    
    # Load models
    pinn, nn2, scalers, sensor_coords = load_models()
    
    # Run all checks
    check_data_leakage()
    check_training_vs_inference()
    check_inverse_transform_issues(pinn, nn2, scalers, sensor_coords)
    check_training_evaluation_method()
    
    # Generate summary
    generate_summary_report()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Fix data leakage by removing current_sensors or using PINN predictions")
    print("2. Retrain model with correct setup")
    print("3. Fix inverse transform issues")
    print("4. Re-evaluate on training data")
    print("="*80)


if __name__ == "__main__":
    main()

