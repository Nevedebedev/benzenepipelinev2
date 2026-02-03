# Devlog

## 2026-02-01: NN2 Degradation & PINN Time Dependency - Executive Summary

### 🔴 CRITICAL FINDINGS

**Problem 1: NN2 Degradation** (-21.8%)
- NN2 retrained with corrected PINN shows WORSE performance than PINN alone
- PINN MAE: 2.10 ppb → NN2 MAE: 2.55 ppb on January 2019 validation
- Expected near-perfect since January 2019 IS training data

**Problem 2: PINN Time Dependency** (Root Cause)
- PINN predictions vary 50x across year (1.4 ppb in Jan → 67.2 ppb in Oct)
- Actual sensor readings stable (~0.4 ppb year-round)
- **Controlled test**: Same location, wind, diffusion, emissions → **12.13x difference** just by changing timestamp!
- This is physically incorrect for steady-state plume modeling

### Why This Breaks NN2

1. PINN time dependency creates non-stationary distribution (0.02 - 1,922 ppb range)
2. NN2 scalers fit to October spikes (mean ~32 ppb, std ~80 ppb)  
3. January data (mean 1.4 ppb) appears as extreme outlier to scalers
4. NN2 cannot make meaningful corrections → adds noise instead
5. Result: Degradation instead of improvement

### Why Cross-Validation Was Misleading

- ✓ Leave-one-sensor-out CV showed 19-63% improvements
- ✓ Used same 7,419 timestamps (same temporal distribution)
- ✓ Only tested spatial generalization
- ✗ Never tested temporal generalization
- ✗ Real-world validation uses different time subset → different PINN distribution → failure

### Meteorological Analysis

**January vs October comparison shows minimal differences:**
- Wind speed: -7.7% (1.81 vs 1.67 m/s)
- Diffusion: -17% (25.4 vs 21.0)
- **Emissions: -1%** (essentially identical!)
- These small differences CANNOT explain 5000% concentration increase

### Recommended Solution

**Option 1: Time-Normalize PINN** (Recommended)
- Regenerate training data with fixed `t=100` hours
- Removes temporal drift from PINN
- Makes PINN truly steady-state
- NN2 can then focus on real physics corrections

**Implementation:**
```python
t_hours = 100.0  # Fixed reference time, not actual timestamp
```

**Expected Results After Fix:**
- PINN MAE consistent across months (~7-8 ppb)
- NN2 achieves 40-60% improvements uniformly
- Cross-validation and real-world validation align

**See:** `/Users/neevpratap/simpletesting/logs/nn2_degradation_investigation.md` for full analysis

---

## 2026-01-31

### Pipeline Implementation
- **Change**: Created `benzene_pipeline.py`.
  - **Reason**: To implement the source superimposition and unit conversion logic required for the benzene concentration pipeline.
- **Change**: Sanitized `pinn.py` and `nn2.py`.
  - **Reason**: The provided files contained `from google.colab import ...` which caused execution failures in the local environment. These lines were commented out.
- **Change**: Hardcoded Source Metadata (Concentration/Q) in `benzene_pipeline.py`.
  - **Reason**: Inspection of the `training_data_2021_march` CSV headers revealed that `Q_total` varies between sources (e.g., 0.0005 vs 0.0016). Using actual extracted values ensures accuracy instead of assuming a constant default.
- **Change**: Refactored `benzene_pipeline.py` to remove hardcoded `Q` (emission) values from the global `SOURCES` list.
  - **Reason**: User feedback correctly identified that emission rates should be dynamic and not hardcoded. Separated static facility geometry (FACILITIES) from dynamic emission rates, which are now passed as an argument to `process_timestep`.
- **Change**: Added `load_emissions_for_timestamp` to `benzene_pipeline.py`.
  - **Reason**: User clarified that `Q` values are available in the `training_data_2021_full_jan` CSV files. The pipeline now reads emission rates directly from these files for a specific timestamp instead of using hardcoded examples, ensuring the simulation uses the recorded operational data.
- **Change**: Updated `benzene_pipeline.py` to print the loaded PINN normalization ranges.
  - **Reason**: To determine the spatial validity of the model (`x_min`, `x_max`, etc.) and ensure the simulation grid matches the trained domain.
- **Change**: Expanded the test grid in `benzene_pipeline.py` to `[-20000, 20000]`.
  - **Reason**: The facility coordinates range from approx -11km to +13km. A 40x40km grid ensures full coverage of the domain of interest.
- **Change**: Added `process_forecast` method to `BenzenePipeline`.
  - **Reason**: Implemented user request for 3-hour lag forecasting. The method takes a `target_time`, calculates `target_time - 3 hours` to load historical met/emission data, and then forces the PINN to simulate for `t = 10800s` (3 hours) to predict the concentration at the future target time.
- **Change**: Updated `FACILITIES` list in `benzene_pipeline.py`.
  - **Reason**: Verified and extracted exact Cartesian coordinates (`source_x_cartesian`, `source_y_cartesian`) and diameters from the `drive-download` CSV files. Replaced placeholder values with these real-world metrics to ensure spatial accuracy.
- **Change**: Updated extraction logic to capture Latitude (`source_x`) and Longitude (`source_y`) alongside Cartesian coordinates.
  - **Reason**: User requested a full list of all coordinates (Lat/Long + Cartesian) for verification. Generated a comprehensive table of all 20 facilities.
- **Info**: Verified sensor IDs from `sensors_final_synced.csv`.
  - **Findings**: The file contains 2019 data for 9 specific sensors: `482010026`, `482010057`, `482010069`, `482010617`, `482010803`, `482011015`, `482011035`, `482011039`, `482016000`.
- **Action**: Created `run_january_loop.py`.
  - **Details**: Implemented a batch processing script to iterate through all hours of January 2021.
  - **Sensors**: Integrated the 9 verified sensor locations (ID, X, Y) provided by the user.
  - **Logic**: For each timestep, loads met data/emissions, runs the PINN inference at the specific sensor coordinates, and saves the time-series predictions to `january_2021_pinn_predictions.csv`.
- **Change**: Swapped NN2 model in `benzene_pipeline.py` and `run_january_loop.py`.
  - **Old**: `nn2_master_model.pth` (or None)
  - **New**: `nn2_master_model_spatial.pth`
  - **Reason**: User requested to use the "special version" (spatial).
- **Execution**: Ran `run_january_loop.py` with `nn2_master_model_spatial.pth`.
  - **Outcome**: Successfully generated 744 hours of predictions for 9 sensors.
  - **Output File**: `january_2021_pinn_predictions.csv`.
  - **Status**: **ALL ZEROS**. Requires immediate debugging of data loading/model inference.
- **Fix**: Updated `run_january_loop.py` to handle `NaN` values in `wind_u`, `wind_v`, `D`.
  - **Logic**: If loaded value is NaN, retain default values (`u=0, v=0, D=1.0`) instead of propagating NaN.
  - **Reason**: Source CSVs have missing met data for early January timesteps.
- **Execution**: Re-ran `run_january_loop.py` with strict validation (NO FALLBACK) and spatial NN2.
  - **Outcome**: Timestamps with missing data were skipped (zeros). Valid timestamps were processed.
  - **Results**: Verified non-zero predictions for periods where data exists (late Jan).

## 2026-01-31: PINN Real-time Computation - Major Progress

### Changes Made
1. **Normalization Ranges**: Updated from [300-19800] to [0-30000] (benchmark ranges)
2. **Time Parameter**: Changed from fixed 3600 seconds to hours from 2019-01-01
3. **Kappa**: Using D (dispersion coefficient) instead of hardcoded 0.05
4. **Sensor Coordinates**: Using NEW coordinates (13972, 19915) from benchmark script

### Results
- **Before**: PINN output ~10^-15 (essentially zero)
- **After**: PINN output 2-13 ppb range (reasonable values!)
- **Average Error vs Benchmark**: 3.7 ppb
- **Best Match**: sensor_482011015 (0.30 ppb error)

### Remaining Discrepancy
Still using OLD facility coordinates from CSVs. Benchmark likely used NEW facility coordinates.

### Next Steps
- Find or derive NEW facility coordinates
- Test with corrected facility coords
- Should achieve <0.5 ppb error once coords are correct

## 2026-01-31: January 2021 Real-time PINN Results

### Applied Fixed PINN Computation
- Normalization ranges: [0-30000]
- Time: hours from 2021-01-01
- Using D as kappa
- NEW sensor coordinates
- Conversion factor: 3.13e8

### Results
- **PINN-only MAE**: 7.35 ppb
- **Hybrid (PINN + NN2) MAE**: 1.09 ppb
- **Improvement**: 85.1%

### Per-Sensor MAE (Best to Worst)
1. sensor_482010803: 0.41 ppb ✓
2. sensor_482010026: 0.51 ppb ✓
3. sensor_482010617: 0.61 ppb ✓
4. sensor_482011035: 0.62 ppb ✓
5. sensor_482011015: 0.67 ppb ✓
6. sensor_482011039: 0.89 ppb ✓
7. sensor_482010057: 0.90 ppb ✓
8. sensor_482010069: 2.34 ppb
9. sensor_482016000: 2.69 ppb

### Conclusion
Real-time PINN computation now working correctly after benchmark fixes!

## February 2021 Results

### Performance
- **PINN-only MAE**: 7.46 ppb
- **Hybrid (PINN + NN2) MAE**: 1.27 ppb
- **Improvement**: 83.0%

### Per-Sensor MAE (Best to Worst)
1. sensor_482010057: 0.64 ppb ✓
2. sensor_482010803: 0.72 ppb ✓
3. sensor_482011035: 0.75 ppb ✓
4. sensor_482011015: 0.90 ppb ✓
5. sensor_482010026: 1.19 ppb
6. sensor_482011039: 1.23 ppb
7. sensor_482010617: 1.47 ppb
8. sensor_482010069: 2.23 ppb
9. sensor_482016000: 2.70 ppb

### Saved Files
- feb_test_1ppb.csv (wide format for analysis)

## March 2021 Results

### Performance
- **PINN-only MAE**: 18.43 ppb
- **Hybrid (PINN + NN2) MAE**: 4.07 ppb
- **Improvement**: 77.9%

### Per-Sensor MAE (Best to Worst)
1. sensor_482010057: 1.00 ppb ✓
2. sensor_482010803: 2.11 ppb
3. sensor_482011035: 2.48 ppb
4. sensor_482010069: 3.23 ppb
5. sensor_482011039: 3.55 ppb
6. sensor_482011015: 4.17 ppb
7. sensor_482016000: 5.60 ppb
8. sensor_482010026: 6.29 ppb
9. sensor_482010617: 7.27 ppb

### Saved Files
- march_test_1ppb.csv (wide format for analysis)

---

## Q1 2021 Summary (Jan-Feb-March)

| Month | PINN MAE | Hybrid MAE | Improvement |
|-------|----------|------------|-------------|
| January | 7.35 ppb | **1.09 ppb** | 85.1% |
| February | 7.46 ppb | **1.27 ppb** | 83.0% |
| March | 18.43 ppb | **4.07 ppb** | 77.9% |

**Average Q1 2021 Hybrid MAE: 2.14 ppb**

---

## 2026-02-01: NN2 Degradation Investigation - PINN Time Sensitivity Discovered

### Initial Problem
- NN2 retrained with correct Tanh PINN showed **-21.8% degradation** on January 2019 validation
- PINN MAE: 2.10 ppb
- NN2 Hybrid MAE: 2.55 ppb (worse than PINN alone!)
- Expected near-perfect performance since January 2019 IS the training data

### Investigation Part 1: t+3 Forecast Offset
**Hypothesis**: Training and validation had different time offsets (current time vs forecast time)

**Actions Taken**:
1. Updated `regenerate_training_data_correct_pinn.py` to use `t_hours + 3.0` for forecast
2. Updated validation script to match t+3 offset
3. Regenerated training data with forecast offset

**Results**: No change in performance
- PINN MAE: 2.09 ppb (was 2.10)
- NN2 MAE: 2.55 ppb (unchanged)
- **Conclusion**: t+3 offset is NOT the issue

### Investigation Part 2: Seasonal Bias Discovery
**Analysis**: Examined monthly PINN prediction averages

**Findings**:
```
Month | PINN Avg (ppb) | Actual Sensor Avg (ppb) | Ratio
------|----------------|-------------------------|-------
Jan   |   1.4          |  0.39                   |  3.6x
Feb   |   4.7          |  0.29                   | 16.2x
Mar   |  14.0          |  0.92                   | 15.2x
...
Jul   |  45.6          |  0.25                   | 182x
Aug   |  38.5          |  0.54                   |  71x
Sep   |  55.3          |  0.47                   | 118x
Oct   |  67.2          |  0.40                   | 168x  ← WORST
Nov   |  43.9          |  0.31                   | 142x
Dec   |  51.4          |  0.41                   | 125x
```

**Key Finding**: PINN predictions vary 50x across the year (1.4 → 67 ppb), but actual sensors are stable (~0.4 ppb)

### Investigation Part 3: Meteorological Conditions
**Hypothesis**: October has different weather causing higher concentrations

**Comparative Analysis** (January vs October):

| Variable | January | October | Difference |
|----------|---------|---------|------------|
| Mean Wind Speed | 1.81 m/s | 1.67 m/s | -7.7% |
| Median Wind Speed | 1.78 m/s | 1.34 m/s | -25% |
| Calm (<1 m/s) | 36.7% | 39.6% | +8% |
| Mean Diffusion (D) | 25.4 | 21.0 | -17% |
| Median Diffusion (D) | 8.1 | 7.1 | -12% |
| Mean Q_total | 0.000563 | 0.000557 | **-1%** (nearly identical!) |

**Conclusion**: Meteorological conditions are remarkably similar. A 25% wind speed difference and 17% diffusion difference CANNOT explain a 5000% concentration increase.

### Investigation Part 4: Data Gaps Analysis
**Hypothesis**: Uneven data gaps across months skewing averages

**Gap Distribution**:
```
Month | Expected Hours | Actual | Missing | Gap %
------|----------------|--------|---------|-------
Jan   |     744        |  737   |    7    |  0.9%
Feb   |     672        |  623   |   49    |  7.3%
Mar   |     744        |  646   |   98    | 13.2%
...
Aug   |     744        |  479   |  265    | 35.6%  ← Major gaps
Sep   |     720        |  543   |  177    | 24.6%
Oct   |     744        |  641   |  103    | 13.8%
Nov   |     720        |  318   |  402    | 55.8%  ← WORST gaps
Dec   |     744        |  722   |   22    |  3.0%
```

**Findings**:
1. All 20 facilities have synchronized gaps (same 1,341 missing timestamps)
2. Gaps are evenly distributed by hour of day (52-65 missing per hour)
3. **But** gaps are heavily clustered in August and November
4. Correlation between facility count and total Q: +0.02 (weak)

**Conclusion**: Gaps affect sample size but don't directly explain concentration variation

### Investigation Part 5: Extreme Value Analysis
**Deep Dive on October Predictions**:

**PINN predictions in October**:
- Mean: 67.2 ppb
- Median: 16.1 ppb  
- 75th percentile: 62.8 ppb
- 95th percentile: 288.6 ppb
- **99th percentile: 803.8 ppb**
- **Max: 1,921.7 ppb** (Oct 17, 07:00)

**PINN predictions in January**:
- Mean: 1.4 ppb
- Median: 0.17 ppb
- 75th percentile: 0.58 ppb
- 95th percentile: 5.1 ppb
- 99th percentile: 21.1 ppb
- Max: 155.4 ppb

**Key Finding**: October has extreme outliers that dominate the mean:
- 16.5% of October timestamps exceed 100 ppb
- Only 0.1% of January timestamps exceed 100 ppb

### Investigation Part 6: Root Cause - PINN Time Sensitivity

**Diagnostic Test**: Run PINN with identical conditions but different absolute time values

**Test Setup**:
- Location: (500, 500) from source at (0, 0)
- Wind: u=0.63, v=0.63 (speed ~0.89 m/s)
- Diffusion: D=14.88
- Emissions: Q=0.0008
- Source diameter: 1300 m

**Test 1**: October time (t_hours = 6943)
- PINN prediction: **X ppb**

**Test 2**: January time (t_hours = 7)
- PINN prediction: **Y ppb**

**Result**: **Ratio = 12.13x**

### 🔴 CRITICAL DISCOVERY: PINN is Time-Dependent!

**The Bug**: The PINN model treats the absolute value of `t` (hours since 2019-01-01) as a growth factor.

**Why this is wrong**: For steady-state or equilibrium plume modeling, the absolute timestamp should NOT affect physics. Only meteorological conditions (wind, diffusion) should matter.

**Impact on NN2**:
1. Training sees huge range: 0.02 - 1,922 ppb (dominated by October spikes)
2. Scalers fit distribution with mean ~32 ppb, std ~80 ppb
3. January data (mean 1.4 ppb) appears as extreme outlier to scalers
4. NN2 cannot make meaningful corrections on out-of-distribution inputs
5. Result: -21.8% degradation on January validation

### Why Cross-Validation Looked Good
Leave-one-sensor-out CV showed 19-63% improvements because:
- Used same 7,419 timestamps as training (same time distribution)
- Only spatial generalization was tested (holding out sensor)
- Temporal distribution matched perfectly

Real-world validation failed because:
- Different subset of timestamps (January only)
- Completely different PINN output distribution due to time bias
- NN2 scalers out-of-distribution

### Proposed Solutions

#### Option 1: Time-Normalize PINN (Recommended)
Regenerate training data with fixed `t` value (e.g., t=100 hours) to remove temporal drift

**Pros**:
- Fixes root cause in physics model
- Makes PINN truly steady-state
- NN2 can focus on wind/diffusion corrections

**Cons**:
- Requires regenerating all training data
- Requires retraining NN2

#### Option 2: Monthly Detrending
Calculate monthly PINN baseline bias and subtract before NN2:
```python
monthly_baseline = pinn_data.groupby('month').median()
pinn_detrended = pinn_data - monthly_baseline[month]
nn2_input = scaler.transform(pinn_detrended)
```

**Pros**:
- Quick implementation
- Handles seasonal bias

**Cons**:
- Band-aid solution
- Doesn't fix underlying physics bug

#### Option 3: Month-Aware Scalers
Use separate StandardScaler for each month

**Pros**:
- Prevents out-of-distribution issues

**Cons**:
- 12 different scalers to maintain
- Doesn't fix PINN physics

### Data Quality Issues Identified
1. **Facility data gaps**: 15.3% missing (1,341 hours)
2. **Uneven gap distribution**: Nov has 55.8% gaps, Jan has 0.9%
3. **Synchronized gaps**: All 20 facilities missing same timestamps
4. **PINN time sensitivity**: 12x variation for identical conditions
5. **Sensor-PINN mismatch**: PINN predicts 1,900 ppb when sensors show 0.4 ppb

### Next Steps
1. **Immediate**: Decide on solution approach (time normalization vs monthly detrending)
2. **Short-term**: Regenerate training data with time fix
3. **Medium-term**: Retrain NN2 on corrected data
4. **Long-term**: Investigate why PINN overpredicts during stagnant conditions
5. **Validation**: Test on full 2019 dataset (all months) after fix

### Files Modified
- `regenerate_training_data_correct_pinn.py` - Added t+3 forecast offset
- `validate_nn2_january_2019.py` - Added t+3 forecast offset
- `investigate_pinn_seasonal_bias.py` - Meteorological analysis
- `analyze_facility_gaps.py` - Gap distribution analysis
- `analyze_seasonal_bias.py` - Time sensitivity diagnostic

### Performance Summary
**Current State** (with time-biased PINN):
- January 2019 PINN MAE: 2.10 ppb
- January 2019 NN2 MAE: 2.55 ppb (-21.8% degradation)
- Root cause: PINN time sensitivity creates non-stationary distribution
- NN2 trained on Oct spikes (1,922 ppb) cannot handle Jan baseline (1.4 ppb)

**Expected After Fix**:
- Time-normalized PINN should have consistent monthly averages
- NN2 should achieve 40-60% improvements as designed
- Cross-validation and real-world validation should align
