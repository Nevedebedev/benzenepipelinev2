#!/usr/bin/env python3
"""
Test Alternative Approaches

Tests simpler models to see if they can learn corrections:
1. Linear regression
2. Random Forest
3. Gradient Boosting
4. Compare with neural network performance
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

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

def prepare_data():
    """Prepare features and targets"""
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
    
    # Remove NaN/inf
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(pinn_vals)
    X = X[valid_mask]
    y = y[valid_mask]
    pinn_vals = pinn_vals[valid_mask]
    
    return X, y, pinn_vals

def test_models():
    """Test various models"""
    print("="*80)
    print("ALTERNATIVE APPROACHES TEST")
    print("="*80)
    print()
    
    print("Loading and preparing data...")
    X, y, pinn_vals = prepare_data()
    print(f"✓ Prepared {len(X)} samples with {X.shape[1]} features")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate baseline (zero correction)
    zero_pred = np.zeros_like(y_test)
    zero_mae = mean_absolute_error(y_test, zero_pred)
    pinn_mae = mean_absolute_error(y_test, pinn_vals[:len(y_test)] - y_test)  # PINN error
    
    print("1. BASELINE PERFORMANCE")
    print("-"*80)
    print(f"   Zero Correction MAE: {zero_mae:.6f} ppb")
    print(f"   PINN Error MAE: {pinn_mae:.6f} ppb")
    print()
    
    results = {}
    
    # Test Linear Regression
    print("2. LINEAR REGRESSION")
    print("-"*80)
    try:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        
        results['Linear Regression'] = {
            'mae': lr_mae,
            'r2': lr_r2,
            'improvement': (zero_mae - lr_mae) / zero_mae * 100
        }
        
        print(f"   MAE: {lr_mae:.6f} ppb")
        print(f"   R²: {lr_r2:.4f}")
        print(f"   Improvement: {results['Linear Regression']['improvement']:.1f}%")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print()
    
    # Test Ridge Regression
    print("3. RIDGE REGRESSION (L2 Regularization)")
    print("-"*80)
    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        ridge_mae = mean_absolute_error(y_test, ridge_pred)
        ridge_r2 = r2_score(y_test, ridge_pred)
        
        results['Ridge Regression'] = {
            'mae': ridge_mae,
            'r2': ridge_r2,
            'improvement': (zero_mae - ridge_mae) / zero_mae * 100
        }
        
        print(f"   MAE: {ridge_mae:.6f} ppb")
        print(f"   R²: {ridge_r2:.4f}")
        print(f"   Improvement: {results['Ridge Regression']['improvement']:.1f}%")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print()
    
    # Test Random Forest
    print("4. RANDOM FOREST")
    print("-"*80)
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        results['Random Forest'] = {
            'mae': rf_mae,
            'r2': rf_r2,
            'improvement': (zero_mae - rf_mae) / zero_mae * 100
        }
        
        print(f"   MAE: {rf_mae:.6f} ppb")
        print(f"   R²: {rf_r2:.4f}")
        print(f"   Improvement: {results['Random Forest']['improvement']:.1f}%")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print()
    
    # Test Gradient Boosting
    print("5. GRADIENT BOOSTING")
    print("-"*80)
    try:
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        
        results['Gradient Boosting'] = {
            'mae': gb_mae,
            'r2': gb_r2,
            'improvement': (zero_mae - gb_mae) / zero_mae * 100
        }
        
        print(f"   MAE: {gb_mae:.6f} ppb")
        print(f"   R²: {gb_r2:.4f}")
        print(f"   Improvement: {results['Gradient Boosting']['improvement']:.1f}%")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print()
    
    # Summary
    print("6. SUMMARY")
    print("-"*80)
    
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['mae'])
        print(f"   Best model: {best_model[0]}")
        print(f"     MAE: {best_model[1]['mae']:.6f} ppb")
        print(f"     R²: {best_model[1]['r2']:.4f}")
        print(f"     Improvement: {best_model[1]['improvement']:.1f}%")
        print()
        
        print("   All models:")
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['mae']):
            print(f"     {name:20s}: MAE={metrics['mae']:.6f}, R²={metrics['r2']:.4f}, Imp={metrics['improvement']:.1f}%")
        print()
        
        # Compare with neural network
        print("7. COMPARISON WITH NEURAL NETWORK")
        print("-"*80)
        print("   Neural Network (from training logs):")
        print("     LOOCV Improvement: 12-62%")
        print("     Master Model Improvement: 39-84%")
        print()
        
        if best_model[1]['improvement'] > 20:
            print("   ✅ Simple models show significant improvement")
            print("      Neural network should work, but may be overkill")
            print("      Consider using simpler model for deployment")
        elif best_model[1]['improvement'] > 10:
            print("   ⚠️  Simple models show moderate improvement")
            print("      Neural network may help, but architecture matters")
        else:
            print("   ❌ Simple models show little improvement")
            print("      Neural network unlikely to help significantly")
            print("      Problem may not be learnable with current features")
    
    print()
    print("="*80)

if __name__ == '__main__':
    test_models()

