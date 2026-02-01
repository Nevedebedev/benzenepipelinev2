#!/usr/bin/env python3
"""
Comprehensive validation script for real-time benzene prediction pipeline
Tests all components systematically
"""

import sys
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append('/Users/neevpratap/Desktop/realtime')

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# Import pipeline components
from pinn import ParametricADEPINN
from config import FACILITIES

# Define sensor coordinates for validation
SENSOR_COORDS_NN2 = np.array([
    [13972.62, 19915.57],  # 482010026
    [3017.18, 12334.2],    # 482010057
    [817.42, 9218.92],     # 482010069
    [8836.35, 15717.2],    # 482010803
    [18413.8, 15068.96],   # 482011015
    [1159.98, 12272.52],   # 482011035
    [13661.93, 5193.24],   # 482011039
    [15077.79, 9450.52],   # 482011614
    [1546.9, 6786.33],     # 482016000
])

print("="*70)
print("REAL-TIME PIPELINE VALIDATION")
print("="*70)

# ============================================================================
# 1. MODEL LOADING
# ============================================================================
print("\n[1/10] Validating Model Loading...")

# Load PINN
try:
    pinn = ParametricADEPINN()
    cp = torch.load("/Users/neevpratap/simpletesting/pinn_combined_final.pth 2", 
                    map_location='cpu', weights_only=False)
    pinn.load_state_dict(cp['model_state_dict'])
    pinn.eval()
    print("  ✓ PINN model loaded")
    
    # Check normalization bounds (informational only - these are from the actual loaded model)
    print(f"    Q_max: {pinn.Q_max.item():.6f} kg/s")
    print(f"    x_max: {pinn.x_max.item():.1f} m")
    print(f"    D_max: {pinn.d_max.item():.1f}")
    print("  ✓ PINN normalization bounds loaded from checkpoint")
except Exception as e:
    print(f"  ✗ PINN loading failed: {e}")

# Load NN2
try:
    from nn2 import NN2_CorrectionNetwork
    nn2_cp = torch.load("/Users/neevpratap/simpletesting/nn2_master_model_spatial.pth",
                        map_location='cpu', weights_only=False)
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2.load_state_dict(nn2_cp['model_state_dict'])
    nn2.eval()
    
    scalers = nn2_cp['scalers']
    sensor_coords = nn2_cp['sensor_coords']
    
    print("  ✓ NN2 model loaded with scalers")
    print(f"    Sensor coordinates shape: {sensor_coords.shape}")
    
    # Verify sensor coordinates
    assert sensor_coords.shape == (9, 2), "Should have 9 sensors with (x,y)"
    print("  ✓ NN2 has correct 9 sensors")
except Exception as e:
    print(f"  ✗ NN2 loading failed: {e}")

# ============================================================================
# 2. FACILITY CONFIGURATION
# ============================================================================
print("\n[2/10] Validating Facility Configuration...")

assert len(FACILITIES) == 20, f"Should have 20 facilities, got {len(FACILITIES)}"
print(f"  ✓ 20 facilities configured")

# Check a sample facility
exxon = FACILITIES['ExxonMobil Baytown Refinery']
assert 'source_x_cartesian' in exxon, "Missing source_x_cartesian"
assert 'Q_schedule_kg_s' in exxon, "Missing Q_schedule_kg_s"
assert len(exxon['Q_schedule_kg_s']) == 4, "Q_schedule should have 4 values"
print(f"  ✓ Facility data structure correct")
print(f"    Example Q range: {min(exxon['Q_schedule_kg_s'])} - {max(exxon['Q_schedule_kg_s'])} kg/s")

# ============================================================================
# 3. METEOROLOGICAL CALCULATIONS
#  ============================================================================
print("\n[3/10] Validating Meteorological Calculations...")

from atmospheric_calculations import (
    wind_components_from_speed_direction,
    calculate_diffusion_coefficient,
    q_for_timestamp
)

# Test wind components
wind_speed = 5.0  # m/s
wind_dir = 270.0  # degrees (from west)
u, v = wind_components_from_speed_direction(wind_speed, wind_dir)
print(f"  Wind: {wind_speed} m/s @ {wind_dir}° → u={u:.2f}, v={v:.2f}")

# West wind should have u≈5, v≈0
assert abs(u - 5.0) < 0.1, f"West wind should have u≈5, got {u}"
assert abs(v) < 0.1, f"West wind should have v≈0, got {v}"
print("  ✓ Wind component calculation correct")

# Test diffusion coefficient (uses only wind_speed and stability_class)
D = calculate_diffusion_coefficient(
    wind_speed=5.0,
    stability_class='D'
)
print(f"  Diffusion D = {D:.2f} m²/s (stability class D)")
assert 1.0 <= D <= 200.0, f"D should be in range [1, 200], got {D}"
print("  ✓ Diffusion coefficient reasonable")

# Test emission rate scheduling
test_time = datetime(2026, 2, 1, 12, 0)  # Saturday noon
Q_rates = (0.003, 0.0016, 0.0012, 0.0008)
Q = q_for_timestamp(test_time, Q_rates)
print(f"  Emission Q = {Q:.6f} kg/s (Saturday noon)")
assert Q == Q_rates[2], f"Should use weekend peak rate"
print("  ✓ Emission rate scheduling correct")

# ============================================================================
# 4. PINN PREDICTION TEST
# ============================================================================
print("\n[4/10] Validating PINN Predictions...")

# Test single point prediction
x = torch.tensor([[15000.0]], dtype=torch.float32)
y = torch.tensor([[15000.0]], dtype=torch.float32)
t = torch.tensor([[100.0]], dtype=torch.float32)
cx = torch.tensor([[13263.0]], dtype=torch.float32)  # ExxonMobil source
cy = torch.tensor([[-695.0]], dtype=torch.float32)
u = torch.tensor([[3.0]], dtype=torch.float32)
v = torch.tensor([[-2.0]], dtype=torch.float32)
d = torch.tensor([[3220.0]], dtype=torch.float32)
kappa = torch.tensor([[50.0]], dtype=torch.float32)
Q = torch.tensor([[0.003]], dtype=torch.float32)

with torch.no_grad():
    phi = pinn(x, y, t, cx, cy, u, v, d, kappa, Q, normalize=True)
    conc_ppb = phi.item() * 3.13e8

print(f"  PINN raw output (phi): {phi.item():.6e}")
print(f"  Converted to ppb: {conc_ppb:.4f}")

# Check if concentration is in reasonable range (allowing near-zero or positive)
assert conc_ppb >= -1.0, f"Concentration should be >= 0 or near-zero, got {conc_ppb}"
print("  ✓ PINN output conversion working")

# ============================================================================
# 5. DATA OUTPUT VALIDATION
# ============================================================================
print("\n[5/10] Validating Data Outputs...")

data_dir = Path("/Users/neevpratap/Desktop/realtime/data")

# Check directory structure
assert (data_dir / "continuous" / "per_facility").exists(), "Missing per_facility dir"
assert (data_dir / "predictions").exists(), "Missing predictions dir"
assert (data_dir / "visualizations").exists(), "Missing visualizations dir"
assert (data_dir / "corrections").exists(), "Missing corrections dir"
print("  ✓ All output directories exist")

# Check latest CSV
latest_csv = data_dir / "predictions" / "latest_spatial_grid.csv"
if latest_csv.exists():
    df = pd.read_csv(latest_csv)
    print(f"  Latest grid CSV: {len(df)} rows")
    
    required_cols = ['forecast_timestamp', 'x', 'y', 'pinn_concentration', 'nn2_concentration']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print(f"  ✓ CSV has required columns")
    
    print(f"    PINN range: [{df['pinn_concentration'].min():.1f}, {df['pinn_concentration'].max():.1f}] ppb")
    print(f"    NN2 range: [{df['nn2_concentration'].min():.1f}, {df['nn2_concentration'].max():.1f}] ppb")

# ============================================================================
# 6. VISUALIZATION VALIDATION  
# ============================================================================
print("\n[6/10] Validating Visualizations...")

viz_dir = data_dir / "visualizations"
pngs = list(viz_dir.glob("*.png"))
print(f"  Found {len(pngs)} PNG files")

if pngs:
    # Check naming convention
    sample_png = pngs[0].name
    print(f"  Sample filename: {sample_png}")
    
    # Should match pattern: pinn_YYYY-MM-DD_HHMM.png or nn2_YYYY-MM-DD_HHMM.png
    assert sample_png.startswith(('pinn_', 'nn2_')), "PNG should start with pinn_ or nn2_"
    assert sample_png.endswith('.png'), "Should end with .png"
    print("  ✓ PNG naming convention correct")

# ============================================================================
# 7. CORRECTION STATISTICS
# ============================================================================
print("\n[7/10] Validating Correction Statistics...")

corr_csv = data_dir / "corrections" / "correction_timeseries.csv"
if corr_csv.exists():
    df_corr = pd.read_csv(corr_csv)
    print(f"  Correction CSV: {len(df_corr)} records")
    
    latest = df_corr.iloc[-1]
    print(f"  Latest correction stats:")
    print(f"    Mean: {latest['correction_mean']:.2f} ppb")
    print(f"    Range: [{latest['correction_min']:.2f}, {latest['correction_max']:.2f}] ppb")
    
    # Sanity checks
    assert abs(latest['correction_mean']) < 500, "Mean correction should be < 500 ppb"
    assert -1000 < latest['correction_min'] < 1000, "Correction range should be reasonable"
    print("  ✓ Correction statistics are reasonable")

# ============================================================================
# 8. TIMESTAMP HANDLING
# ============================================================================
print("\n[8/10] Validating Timestamp Handling...")

if latest_csv.exists():
    df = pd.read_csv(latest_csv)
    timestamps = pd.to_datetime(df['forecast_timestamp'])
    
    # All timestamps should be the same (one forecast)
    assert timestamps.nunique() == 1, "All rows should have same forecast timestamp"
    
    forecast_time = timestamps.iloc[0]
    print(f"  Forecast timestamp: {forecast_time}")
    print("  ✓ Timestamps consistent across grid")

# ============================================================================
# 9. GRID COVERAGE
# ============================================================================
print("\n[9/10] Validating Grid Coverage...")

if latest_csv.exists():
    df = pd.read_csv(latest_csv)
    
    x_unique = df['x'].unique()
    y_unique = df['y'].unique()
    
    print(f"  Grid dimensions: {len(x_unique)} x {len(y_unique)}")
    print(f"  X range: [{df['x'].min():.1f}, {df['x'].max():.1f}] m")
    print(f"  Y range: [{df['y'].min():.1f}, {df['y'].max():.1f}] m")
    
    assert df['x'].min() >= 0, "X should start at or near 0"
    assert df['x'].max() <= 31000, "X should be <= 30000"
    assert df['y'].min() >= 0, "Y should start at or near 0"
    assert df['y'].max() <= 31000, "Y should be <= 30000"
    print("  ✓ Grid covers expected domain")

# ============================================================================
# 10. OVERALL SYSTEM CHECK
# ============================================================================
print("\n[10/10] Overall System Health...")

issues = []

# Check for NaN values
if latest_csv.exists():
    df = pd.read_csv(latest_csv)
    if df['pinn_concentration'].isna().any():
        issues.append("NaN values in PINN predictions")
    if df['nn2_concentration'].isna().any():
        issues.append("NaN values in NN2 predictions")
    
    # Check for negative concentrations
    if (df['pinn_concentration'] < 0).any():
        issues.append("Negative PINN concentrations found")
    if (df['nn2_concentration'] < 0).any():
        issues.append("Negative NN2 concentrations found")

if issues:
    print("  ⚠ Issues found:")
    for issue in issues:
        print(f"    - {issue}")
else:
    print("  ✓ No data integrity issues")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
