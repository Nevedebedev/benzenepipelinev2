#!/usr/bin/env python3
"""
Investigate Learnability of PINN Corrections

Tests if the problem of correcting PINN predictions is learnable:
1. Correlation between PINN errors and conditions
2. Baseline model tests (linear, simple ML)
3. Pattern analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting')

# Paths
TRAINING_DATA_PATH = '/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata/total_superimposed_concentrations.csv'
SENSOR_DATA_PATH = '/Users/neevpratap/Downloads/sensors_final_synced.csv'

# Sensor coordinates
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

def prepare_features_and_targets():
    """Prepare features and targets for learnability analysis"""
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
    
    # Find common timestamps
    common_times = sorted(list(set(pinn_df['timestamp']) & set(sensor_df['timestamp'])))
    
    # Prepare data
    features_list = []
    targets_list = []
    pinn_values_list = []
    
    sensor_ids_sorted = sorted(SENSORS.keys())
    
    for timestamp in common_times:
        pinn_row = pinn_df[pinn_df['timestamp'] == timestamp]
        sensor_row = sensor_df[sensor_df['timestamp'] == timestamp]
        
        if len(pinn_row) == 0 or len(sensor_row) == 0:
            continue
        
        pinn_row = pinn_row.iloc[0]
        sensor_row = sensor_row.iloc[0]
        
        # For each sensor
        for i, sensor_id in enumerate(sensor_ids_sorted):
            pinn_val = pinn_row[f'sensor_{sensor_id}']
            actual_val = sensor_row[f'sensor_{sensor_id}']
            
            # Only use non-zero actual values
            if actual_val <= 0 or pinn_val < 0:
                continue
            
            # Calculate needed correction
            needed_correction = actual_val - pinn_val
            
            # Features: PINN value, sensor coordinates, temporal features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            is_weekend = 1.0 if day_of_week >= 5 else 0.0
            
            sensor_coords = SENSORS[sensor_id]
            
            features = [
                pinn_val,                           # PINN prediction
                sensor_coords[0] / 30000.0,         # Normalized x
                sensor_coords[1] / 30000.0,         # Normalized y
                np.sin(2 * np.pi * hour / 24),      # Hour (sin)
                np.cos(2 * np.pi * hour / 24),      # Hour (cos)
                np.sin(2 * np.pi * day_of_week / 7), # Day (sin)
                np.cos(2 * np.pi * day_of_week / 7), # Day (cos)
                is_weekend,                         # Weekend
                month / 12.0,                       # Month
            ]
            
            features_list.append(features)
            targets_list.append(needed_correction)
            pinn_values_list.append(pinn_val)
    
    X = np.array(features_list)
    y = np.array(targets_list)
    pinn_vals = np.array(pinn_values_list)
    
    return X, y, pinn_vals

def analyze_correlations(X, y, pinn_vals):
    """Analyze correlations between features and needed corrections"""
    print("="*80)
    print("LEARNABILITY ANALYSIS - CORRELATIONS")
    print("="*80)
    print()
    
    feature_names = [
        'PINN_value', 'Sensor_X', 'Sensor_Y', 'Hour_sin', 'Hour_cos',
        'Day_sin', 'Day_cos', 'Is_weekend', 'Month'
    ]
    
    print("1. FEATURE-CORRECTION CORRELATIONS")
    print("-"*80)
    
    correlations = []
    for i, name in enumerate(feature_names):
        # Remove NaN/inf values for correlation
        x_vals = X[:, i]
        y_vals = y
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if valid_mask.sum() > 1:
            corr = np.corrcoef(x_vals[valid_mask], y_vals[valid_mask])[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        correlations.append((name, corr))
        print(f"   {name:15s}: {corr:7.4f}", end="")
        if abs(corr) > 0.3:
            print(" ⭐ (strong)")
        elif abs(corr) > 0.1:
            print(" ✓ (moderate)")
        else:
            print("   (weak)")
    
    print()
    
    # PINN error correlation
    pinn_error = y  # needed_correction = actual - pinn = error
    valid_mask = np.isfinite(pinn_vals) & np.isfinite(y)
    if valid_mask.sum() > 1:
        corr_pinn_error = np.corrcoef(pinn_vals[valid_mask], y[valid_mask])[0, 1]
        if np.isnan(corr_pinn_error):
            corr_pinn_error = 0.0
    else:
        corr_pinn_error = 0.0
    print(f"   PINN_value vs Error: {corr_pinn_error:7.4f}")
    print()
    
    # Overall learnability assessment
    max_corr = max([abs(c) for _, c in correlations])
    strong_features = [name for name, c in correlations if abs(c) > 0.3]
    
    print("2. LEARNABILITY ASSESSMENT")
    print("-"*80)
    print(f"   Maximum correlation: {max_corr:.4f}")
    print(f"   Strong features (|corr| > 0.3): {len(strong_features)}")
    print(f"   Features: {', '.join(strong_features) if strong_features else 'None'}")
    print()
    
    if max_corr > 0.5:
        print("   ✅ STRONG LEARNABILITY - Problem should be learnable")
        print("      High correlation suggests systematic patterns")
    elif max_corr > 0.3:
        print("   ⚠️  MODERATE LEARNABILITY - Problem may be learnable")
        print("      Moderate correlation, may need complex models")
    else:
        print("   ❌ WEAK LEARNABILITY - Problem may not be learnable")
        print("      Low correlations suggest errors are mostly random")
    
    print()
    return correlations, max_corr

def test_baseline_models(X, y):
    """Test simple baseline models"""
    print("3. BASELINE MODEL TESTS")
    print("-"*80)
    
    # Remove NaN/inf values
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    if len(X_clean) == 0:
        print("   ❌ No valid data after cleaning NaN/inf values")
        return {'Zero Correction': np.nan}
    
    print(f"   Using {len(X_clean)} valid samples (removed {len(X) - len(X_clean)} invalid)")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    
    results = {}
    
    # Baseline 1: Always predict zero correction
    zero_pred = np.zeros_like(y_test)
    zero_mae = mean_absolute_error(y_test, zero_pred)
    results['Zero Correction'] = zero_mae
    print(f"   Zero Correction (baseline):")
    print(f"     MAE: {zero_mae:.6f} ppb")
    print()
    
    # Baseline 2: Linear Regression
    try:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        results['Linear Regression'] = lr_mae
        print(f"   Linear Regression:")
        print(f"     MAE: {lr_mae:.6f} ppb")
        print(f"     Improvement: {(zero_mae - lr_mae) / zero_mae * 100:.1f}%")
        print(f"     R²: {lr.score(X_test, y_test):.4f}")
        print()
    except Exception as e:
        print(f"   Linear Regression: FAILED - {e}")
        print()
    
    # Baseline 3: Random Forest (small)
    try:
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        results['Random Forest'] = rf_mae
        print(f"   Random Forest (50 trees, max_depth=10):")
        print(f"     MAE: {rf_mae:.6f} ppb")
        print(f"     Improvement: {(zero_mae - rf_mae) / zero_mae * 100:.1f}%")
        print(f"     R²: {rf.score(X_test, y_test):.4f}")
        print()
    except Exception as e:
        print(f"   Random Forest: FAILED - {e}")
        print()
    
    # Baseline 4: Gradient Boosting (small)
    try:
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        results['Gradient Boosting'] = gb_mae
        print(f"   Gradient Boosting (50 trees, max_depth=5):")
        print(f"     MAE: {gb_mae:.6f} ppb")
        print(f"     Improvement: {(zero_mae - gb_mae) / zero_mae * 100:.1f}%")
        print(f"     R²: {gb.score(X_test, y_test):.4f}")
        print()
    except Exception as e:
        print(f"   Gradient Boosting: FAILED - {e}")
        print()
    
    # Assessment
    print("4. BASELINE MODEL ASSESSMENT")
    print("-"*80)
    best_model = min(results.items(), key=lambda x: x[1])
    print(f"   Best baseline model: {best_model[0]} (MAE: {best_model[1]:.6f} ppb)")
    
    if best_model[1] < zero_mae * 0.9:
        print("   ✅ Baseline models show improvement - problem is learnable")
        print("      Neural network should be able to learn this")
    elif best_model[1] < zero_mae * 0.95:
        print("   ⚠️  Baseline models show slight improvement - problem may be learnable")
        print("      Neural network might help but may struggle")
    else:
        print("   ❌ Baseline models show no improvement - problem may not be learnable")
        print("      Neural network unlikely to help")
    
    print()
    return results

def analyze_error_patterns(X, y, pinn_vals):
    """Analyze patterns in PINN errors"""
    print("5. ERROR PATTERN ANALYSIS")
    print("-"*80)
    
    # Remove NaN/inf values
    valid_mask = np.isfinite(y) & np.isfinite(pinn_vals)
    y_clean = y[valid_mask]
    pinn_clean = pinn_vals[valid_mask]
    
    if len(y_clean) == 0:
        print("   ❌ No valid data for error pattern analysis")
        return
    
    # Error statistics
    print(f"   Error (needed correction) statistics:")
    print(f"     Mean: {np.mean(y_clean):.6f} ppb")
    print(f"     Median: {np.median(y_clean):.6f} ppb")
    print(f"     Std: {np.std(y_clean):.6f} ppb")
    print(f"     Min: {np.min(y_clean):.6f} ppb")
    print(f"     Max: {np.max(y_clean):.6f} ppb")
    print()
    
    # Error vs PINN magnitude
    print(f"   Error vs PINN magnitude:")
    small_pinn = y_clean[pinn_clean < 0.3]
    medium_pinn = y_clean[(pinn_clean >= 0.3) & (pinn_clean < 1.0)]
    large_pinn = y_clean[pinn_clean >= 1.0]
    
    if len(small_pinn) > 0:
        print(f"     Small PINN (<0.3 ppb): mean error = {np.mean(small_pinn):.6f} ppb (n={len(small_pinn)})")
    if len(medium_pinn) > 0:
        print(f"     Medium PINN (0.3-1.0 ppb): mean error = {np.mean(medium_pinn):.6f} ppb (n={len(medium_pinn)})")
    if len(large_pinn) > 0:
        print(f"     Large PINN (>1.0 ppb): mean error = {np.mean(large_pinn):.6f} ppb (n={len(large_pinn)})")
    print()
    
    # Systematic vs random
    error_std = np.std(y_clean)
    error_mean_abs = np.mean(np.abs(y_clean))
    systematic_ratio = error_mean_abs / (error_std + 1e-10)
    
    print(f"   Systematic vs Random:")
    print(f"     Mean absolute error: {error_mean_abs:.6f} ppb")
    print(f"     Error std: {error_std:.6f} ppb")
    print(f"     Ratio (mean_abs/std): {systematic_ratio:.4f}")
    
    if systematic_ratio > 1.0:
        print("     ✅ Errors appear systematic (learnable)")
    else:
        print("     ❌ Errors appear random (not learnable)")
    print()

def main():
    print("Loading and preparing data...")
    X, y, pinn_vals = prepare_features_and_targets()
    
    print(f"Prepared {len(X)} samples with {X.shape[1]} features")
    print()
    
    # Analyze correlations
    correlations, max_corr = analyze_correlations(X, y, pinn_vals)
    
    # Test baseline models
    baseline_results = test_baseline_models(X, y)
    
    # Analyze error patterns
    analyze_error_patterns(X, y, pinn_vals)
    
    # Final assessment
    print("="*80)
    print("FINAL LEARNABILITY ASSESSMENT")
    print("="*80)
    print()
    
    if max_corr > 0.3 and min(baseline_results.values()) < baseline_results['Zero Correction'] * 0.9:
        print("✅ PROBLEM IS LEARNABLE")
        print("   - Strong feature correlations found")
        print("   - Baseline models show improvement")
        print("   - Neural network should be able to learn corrections")
        print()
        print("   Recommendation: Focus on architecture and training, not data collection")
    elif max_corr > 0.1:
        print("⚠️  PROBLEM MAY BE LEARNABLE")
        print("   - Moderate feature correlations")
        print("   - Baseline models show some improvement")
        print("   - Neural network might work with proper architecture")
        print()
        print("   Recommendation: Try simpler architectures first, then complex")
    else:
        print("❌ PROBLEM MAY NOT BE LEARNABLE")
        print("   - Weak feature correlations")
        print("   - Baseline models show little/no improvement")
        print("   - Neural network unlikely to help")
        print()
        print("   Recommendation: Consider different approach (physics-based, ensemble)")
    
    print("="*80)

if __name__ == '__main__':
    main()

