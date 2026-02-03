# Important Guide - Benzene Dispersion Pipeline

This folder contains the complete pipeline documentation and validation results.

## Files

### Documentation
- **COMPLETE_PIPELINE_DOCUMENTATION.md** - Complete technical specification of the entire benzene dispersion prediction pipeline, from training data generation to real-time prediction. Includes all constants, methods, fixes, and implementation details.

### Validation Results

#### January-March 2021 Validation
- **validation_jan_mar_2021_detailed.csv** - Detailed per-timestamp, per-sensor predictions and actuals for January-March 2021
  - Columns: timestamp, month, met_data_timestamp, sensor_id, actual (EDF), pinn (prediction), nn2 (corrected)
  - 15,356 samples total
  
- **validation_jan_mar_2021_summary.csv** - Summary statistics including:
  - Overall PINN MAE: 0.4529 ppb
  - Overall Hybrid (NN2) MAE: 1.2814 ppb
  - Per-sensor MAE breakdown
  - Per-month MAE breakdown

- **PINN_MAE_CONFIRMATION.md** - Confirmation and documentation of the exceptional PINN-only performance
  - PINN MAE: 0.4529 ppb (15,356 samples)
  - Validation process details
  - Per-sensor and per-month breakdowns

#### 2019 Validation
- **nn2_validation_2019.csv** - Validation results for 2019 data

## Key Results

### PINN Performance (Jan-Mar 2021)
- **Overall MAE: 0.4529 ppb** - Exceptional performance for physics-based model
- **Best Month**: February 2021 (0.3185 ppb)
- **Best Sensor**: 482011039 (0.2089 ppb)
- **Total Samples**: 15,356

### Validation Process
The PINN MAE is calculated by:
1. Running PINN with simulation time t=3.0 hours
2. Processing each of 20 facilities separately
3. Superimposing concentrations at 9 sensor locations
4. Converting to ppb
5. Comparing directly to EDF ground truth sensor readings

**No NN2 correction is applied for PINN MAE calculation.**

## Notes

- The NN2-corrected predictions show higher MAE than PINN alone (1.2814 ppb vs 0.4529 ppb). This discrepancy will be investigated separately.
- All validation uses the exact same method as training data generation (simulation time t=3.0 hours, direct PINN computation, superposition).

---
**Last Updated**: 2024
**Validation Period**: January-March 2021

