# Historical Validation Fixes Applied

## Summary

Fixed two critical bugs that were causing NN2 degradation and validation failures:

1. **PINN Time Dependency Bug** - Fixed
2. **Zero-Value Handling Mismatch** - Fixed

---

## Issue 1: PINN Time Dependency (ROOT CAUSE)

### Problem
The PINN model was incorrectly using **absolute calendar time** (hours since 2019-01-01) as an input, causing:
- **12.13x variation** for identical conditions just by changing timestamp
- January predictions: ~1.4 ppb average
- October predictions: ~67.2 ppb average (max 1,922 ppb)
- This is physically incorrect for steady-state plume modeling

**Why this is wrong**: For physics simulations, each scenario should start at t=0 (initial condition), then predict forward in time. The absolute calendar date should NOT affect predictions - only meteorological conditions should matter.

### Impact on NN2
- Training data had extreme range: 0.02 - 1,922 ppb (dominated by October spikes)
- NN2 scalers fit to distribution with mean ~32 ppb, std ~80 ppb
- January data (mean 1.4 ppb) appeared as extreme outliers
- NN2 couldn't make meaningful corrections → -21.8% degradation

### Fix Applied

**File**: `realtime/simpletesting/nn2trainingdata/regenerate_training_data_correct_pinn.py`

Changed from:
```python
# OLD (BUGGY): Absolute calendar time
t_hours = (timestamp - t_start).total_seconds() / 3600.0
t_hours_forecast = t_hours + 3.0
```

To:
```python
# NEW (FIXED): Simulation time - each scenario resets to t=0
FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast
t_hours_forecast = FORECAST_T_HOURS

# CRITICAL: Shift timestamps forward by 3 hours
# Predictions made at time t (using met data from t) are labeled as t+3
forecast_timestamp = input_timestamp + pd.Timedelta(hours=3)
```

**File**: `realtime/validate_nn2_january_2019.py`

Applied same fix:
- Use simulation time `t=3.0 hours` instead of absolute calendar time
- For sensor reading at time `t+3`, load met data from time `t` (3 hours before)
- PINN prediction made at `t` forecasts for `t+3`, aligned with sensor reading at `t+3`

### Timestamp Staggering for NN2 Training

The training data is **staggered** to align predictions with actual sensor readings:

1. **At timestamp `t`**:
   - Load meteorological data (wind, diffusion, emissions) from time `t`
   - Run PINN with simulation time `t=3.0 hours` (to predict 3 hours ahead)
   - PINN output represents concentration at `t+3` hours

2. **Store prediction with timestamp `t+3`**:
   - Shift the timestamp: `output_timestamp = input_timestamp + 3 hours`
   - This prediction will be paired with actual sensor readings at `t+3` for NN2 training

3. **NN2 Training Alignment**:
   - PINN predictions labeled as `t+3` (forecast made at `t`)
   - Actual sensor readings at `t+3` (ground truth)
   - NN2 learns: "Given PINN prediction for `t+3` made at `t`, correct it to match actual at `t+3`"

### Expected Results
- PINN predictions will be consistent across months (~7-8 ppb MAE)
- Only wind, diffusion, and emissions will affect predictions (correct physics)
- NN2 scalers will fit to stationary distribution
- NN2 should achieve 40-60% improvements uniformly
- Timestamp alignment ensures predictions match actual sensor readings correctly

---

## Issue 2: Zero-Value Handling Mismatch

### Problem
- **Training**: Scalers fit on non-zero values only (lines 280-290 in nn2.py)
- **Validation**: All values (including zeros) were transformed
- Zeros transformed to extreme scaled values (many std devs from mean)
- NN2 made corrections based on these extreme values
- After inverse transform, predictions were corrupted

### Fix Applied
**File**: `realtime/validate_nn2_january_2019.py`

Added zero masking before scaling:
```python
# FIX: Handle zeros the same way as training (mask before scaling)
pinn_nonzero_mask = pinn_array != 0.0
sensors_nonzero_mask = current_sensors != 0.0

# Scale PINN predictions (only non-zero values)
p_s = np.zeros_like(pinn_array)
if pinn_nonzero_mask.any():
    p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
        pinn_array[pinn_nonzero_mask].reshape(-1, 1)
    ).flatten()

# Same for current sensors
s_s = np.zeros_like(current_sensors)
if sensors_nonzero_mask.any():
    s_s[sensors_nonzero_mask] = scalers['sensors'].transform(
        current_sensors[sensors_nonzero_mask].reshape(-1, 1)
    ).flatten()

# Inverse transform also respects zero mask
nn2_corrected = np.zeros_like(corrected_scaled_np)
if sensors_nonzero_mask.any():
    nn2_corrected[sensors_nonzero_mask] = scalers['sensors'].inverse_transform(
        corrected_scaled_np[sensors_nonzero_mask].reshape(-1, 1)
    ).flatten()
```

### Expected Results
- Zeros remain zeros (not transformed to extreme values)
- NN2 corrections only applied to non-zero predictions
- Predictions match training behavior

---

## Next Steps

1. **Regenerate Training Data** (REQUIRED):
   ```bash
   cd realtime/simpletesting/nn2trainingdata
   python regenerate_training_data_correct_pinn.py
   ```
   This will create new `total_concentrations.csv` with time-normalized PINN predictions.

2. **Retrain NN2** (REQUIRED):
   - Use the regenerated training data
   - Train new NN2 model with corrected PINN predictions
   - Expected: Consistent performance across all months

3. **Validate** (VERIFICATION):
   ```bash
   python validate_nn2_january_2019.py
   ```
   Should now show:
   - PINN MAE: ~7-8 ppb (consistent, not 2.10 ppb with time bias)
   - NN2 MAE: ~3-4 ppb (40-60% improvement)
   - No degradation

---

## Files Modified

1. `realtime/simpletesting/nn2trainingdata/regenerate_training_data_correct_pinn.py`
   - Changed time calculation to use simulation time `t=3.0 hours` (not absolute calendar time)
   - Added 3-hour timestamp shift: predictions made at `t` are labeled as `t+3` in output
   - This aligns PINN predictions with actual sensor readings at `t+3` for NN2 training

2. `realtime/validate_nn2_january_2019.py`
   - Changed time calculation to use simulation time `t=3.0 hours` (not absolute calendar time)
   - Fixed timestamp alignment: for sensor reading at `t+3`, load met data from `t` (3 hours before)
   - Added zero-value masking before scaling
   - Added zero-value masking before inverse transform

---

## Notes

- The PINN model itself is not retrained - we just use it with simulation time input
- **Simulation time vs Calendar time**:
  - **Simulation time**: Each scenario starts at t=0, predicts at t=3 hours (correct physics)
  - **Calendar time**: Absolute hours since 2019-01-01 (incorrect - causes time dependency bug)
- This makes PINN truly steady-state (only physics affects predictions, not calendar date)
- NN2 will need retraining on the corrected data
- All validation scripts should use the same simulation time approach
- **Timestamp staggering**: Predictions made at time `t` are labeled as `t+3` to align with actual sensor readings at `t+3`

---

## Validation Checklist

After regenerating training data and retraining NN2:

- [ ] PINN predictions consistent across months (no 50x variation)
- [ ] January 2019 PINN MAE: ~7-8 ppb (not 2.10 ppb)
- [ ] NN2 MAE: ~3-4 ppb (40-60% improvement over PINN)
- [ ] No degradation on January validation
- [ ] Zero values handled correctly (no extreme transformations)
- [ ] Cross-validation and real-world validation align

