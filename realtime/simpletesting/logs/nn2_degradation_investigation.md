# NN2 Degradation Investigation Report

## Summary

I've investigated the two critical issues you mentioned:

### 1. **NN2 Degradation Issue** (-21.8% performance drop)

The NN2 model, when retrained with the corrected PINN, shows **worse** performance than the PINN alone on January 2019 validation data:
- **PINN MAE**: 2.10 ppb
- **NN2 Hybrid MAE**: 2.55 ppb
- **Degradation**: -21.8%

This is completely unexpected because:
- Leave-one-out cross-validation showed 19-63% improvements
- January 2019 IS part of the training data
- The model should achieve near-perfect performance on training data

### 2. **PINN Time Dependency Issue** (The Root Cause)

The PINN model has a **critical bug**: it treats absolute time (hours since 2019-01-01) as a variable that affects concentration predictions, even when all other conditions are identical.

#### Diagnostic Test Results

When testing the PINN with **identical conditions** but different timestamps:

**Test Setup:**
- Location: (500, 500) from source
- Wind: u=0.63, v=0.63 (speed ~0.89 m/s)
- Diffusion: D=14.88
- Emissions: Q=0.0008
- Source diameter: 1300 m

**Results:**
- **October time** (t_hours = 6943): High concentration
- **January time** (t_hours = 7): Low concentration
- **Ratio: 12.13x difference**

This is physically incorrect! For steady-state plume modeling, absolute timestamp should NOT affect physics.

---

## Why This Causes NN2 Degradation

### The Problem Chain:

1. **PINN produces non-stationary distribution** due to time dependency
   - January average: 1.4 ppb
   - October average: 67.2 ppb  
   - **October max: 1,921 ppb** (sensor actual: 0.4 ppb)

2. **NN2 training data has extreme range** (0.02 - 1,922 ppb)
   - StandardScalers fit to this distribution
   - Mean ~32 ppb, Std ~80 ppb

3. **January data appears as outlier** to the trained scalers
   - January mean (1.4 ppb) is far below training distribution mean (32 ppb)
   - NN2 cannot make meaningful corrections on out-of-distribution inputs

4. **Result: Degradation instead of improvement**
   - NN2 adds noise instead of corrections
   - Performance worse than PINN alone

---

## Why Cross-Validation Looked Good

Leave-one-sensor-out CV showed good results (19-63% improvements) because:
- ✓ Used same 7,419 timestamps as training
- ✓ Same temporal distribution 
- ✓ Only tested **spatial** generalization (holding out sensors)
- ✗ Never tested **temporal** generalization

Real-world validation failed because:
- ✗ Different subset of timestamps (January only)
- ✗ Completely different PINN output distribution
- ✗ NN2 scalers out-of-distribution

---

## Seasonal Variation Analysis

### PINN Predictions vs. Reality

| Month | PINN Avg (ppb) | Sensor Avg (ppb) | Overprediction |
|-------|----------------|------------------|----------------|
| Jan   | 1.4            | 0.39             | 3.6x           |
| Feb   | 4.7            | 0.29             | 16.2x          |
| Mar   | 14.0           | 0.92             | 15.2x          |
| Jul   | 45.6           | 0.25             | 182x           |
| Aug   | 38.5           | 0.54             | 71x            |
| Sep   | 55.3           | 0.47             | 118x           |
| **Oct**   | **67.2**       | **0.40**         | **168x** ⚠️    |
| Nov   | 43.9           | 0.31             | 142x           |
| Dec   | 51.4           | 0.41             | 125x           |

**Key Finding**: PINN predictions vary **50x** across the year (1.4 → 67 ppb), but actual sensor readings are stable (~0.4 ppb).

### October Extreme Values

- Mean: 67.2 ppb
- Median: 16.1 ppb
- 95th percentile: 288.6 ppb
- **99th percentile: 803.8 ppb**
- **Max: 1,921.7 ppb** (Oct 17, 07:00)

**16.5%** of October timestamps exceed 100 ppb (vs. 0.1% in January)

---

## Meteorological Analysis

### January vs. October Comparison

| Variable | January | October | Difference |
|----------|---------|---------|------------|
| Mean Wind Speed | 1.81 m/s | 1.67 m/s | -7.7% |
| Median Wind Speed | 1.78 m/s | 1.34 m/s | -25% |
| Calm (<1 m/s) | 36.7% | 39.6% | +8% |
| Mean Diffusion (D) | 25.4 | 21.0 | -17% |
| Median Diffusion (D) | 8.1 | 7.1 | -12% |
| **Mean Q_total** | **0.000563** | **0.000557** | **-1%** |

**Conclusion**: Meteorological conditions are remarkably similar! A 25% wind speed difference and 17% diffusion difference **CANNOT** explain a **5000% concentration increase**.

---

## What You Don't Understand: The Time Dependency

### Why PINN is Time-Dependent

The PINN model was trained with `t` (hours since 2019-01-01) as an input feature. This means:

1. **During training**, the model learned patterns like:
   - "When t is small (January), predict low concentrations"
   - "When t is large (October), predict high concentrations"

2. **The model treats time as a predictor**, not as a physical simulation parameter

3. **This means the PINN is NOT solving the physics equations** in a time-independent way - it's learning spurious correlations with absolute time

### What This Should Be Instead

For steady-state plume dispersion, only these should matter:
- Wind speed and direction
- Atmospheric diffusion coefficient
- Emission rate
- Source-receptor geometry

**Absolute time should not affect the result** if all these conditions are the same!

---

## Proposed Solutions

### Option 1: Time-Normalize PINN ⭐ **Recommended**

Regenerate training data with **fixed t value** (e.g., t=100 hours) to remove temporal drift.

**Pros:**
- Fixes root cause in physics model
- Makes PINN truly steady-state
- NN2 can focus on real physics corrections (wind/diffusion)

**Cons:**
- Requires regenerating all training data
- Requires retraining NN2

**Implementation:**
```python
# In PINN computation
t_hours = 100.0  # Fixed reference time
# Instead of: t_hours = (timestamp - pd.Timestamp('2019-01-01')).total_seconds() / 3600
```

---

### Option 2: Monthly Detrending

Calculate monthly PINN baseline bias and subtract before NN2 input.

**Pros:**
- Quick implementation
- Handles seasonal bias

**Cons:**
- Band-aid solution
- Doesn't fix underlying physics bug
- Still leaves PINN physically incorrect

---

### Option 3: Month-Aware Scalers

Use separate StandardScaler for each month.

**Pros:**
- Prevents out-of-distribution issues

**Cons:**
- 12 different scalers to maintain
- Doesn't fix PINN physics
- Complex deployment

---

## Recommended Next Steps

1. **Immediate Decision**: Choose solution approach (I recommend Option 1)

2. **Short-term** (Option 1):
   - Update PINN computation to use fixed `t=100` hours
   - Regenerate all training data with time-normalized PINN
   - This will make PINN predictions consistent across months

3. **Medium-term**:
   - Retrain NN2 on corrected data
   - Expect PINN MAE to become consistent (~7-8 ppb across all months)
   - NN2 should achieve 40-60% improvements uniformly

4. **Long-term**:
   - Validate on full 2019 dataset (all months)
   - Investigate why PINN still overpredicts during stagnant conditions
   - Consider retraining PINN from scratch without time as input

---

## Files for Reference

- `/Users/neevpratap/simpletesting/logs/devlog.md` - Detailed investigation log
- `/Users/neevpratap/simpletesting/investigate_pinn_seasonal_bias.py` - Meteorological analysis
- `/Users/neevpratap/simpletesting/analyze_seasonal_bias.py` - PINN time dependency diagnostic
- `/Users/neevpratap/Desktop/nn2_updated/nn2_master_model_spatial-3.pth` - Latest NN2 model
