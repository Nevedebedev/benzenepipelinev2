# NN2 Discrepancy Investigation Summary - 2019 Data

## Executive Summary

**Problem**: NN2 performs **worse** than PINN alone on 2019 data (the training data), despite leave-one-out cross-validation showing 60-77% improvements.

**Key Finding**: Even when using the **exact same training data** (same PINN predictions, same sensor readings, same timestamps), NN2 MAE is **2.5x worse** than PINN MAE.

## Results

### Exact Training Data Test
- **PINN MAE**: 0.5019 ppb (46,557 samples)
- **NN2 MAE**: 1.2322 ppb
- **Degradation**: -145.5% (worse than PINN)

### Comparison with Leave-One-Out Results

**Leave-One-Out Results** (from `leave_one_out_results_spatial-3.json`):
- Showed 60-77% improvements
- PINN MAE: ~0.4-1.2 ppb
- NN2 MAE: ~0.12-0.33 ppb

**Our Validation** (on exact training data):
- PINN MAE: 0.5019 ppb
- NN2 MAE: 1.2322 ppb
- **Complete opposite result!**

## Investigation Findings

### 1. Zero-Handling
- ✅ **Matches training**: Both use zero-masking (only transform non-zero values)
- ✅ **No mismatch here**

### 2. Model Output Structure
- ✅ **Correct**: `corrected = pinn + corrections` (verified)
- ✅ **Inverse transform**: Using correct scaler

### 3. Data Alignment
- ✅ **Correct**: Using exact training PINN predictions from `total_concentrations.csv`
- ✅ **Correct**: Using actual sensor readings at same timestamps
- ✅ **Correct**: Timestamp alignment matches training

### 4. Input Structure
- ✅ **Correct**: `current_sensors = actual_sensors` at timestamp t+3 (matches training)
- ✅ **Correct**: `pinn_predictions` at timestamp t+3 (matches training)
- ✅ **Correct**: Temporal features match training

### 5. Leave-One-Out vs Full Validation
- Tested both approaches (LOO: held-out sensor = 0, Validation: all sensors = actual)
- **Both perform similarly poorly**
- LOO approach: 1.2345 ppb MAE
- Validation approach: 1.1472 ppb MAE
- Both worse than PINN (0.3229 ppb)

## Critical Observations

### 1. Large Corrections When PINN is Accurate
- When PINN error < 0.1 ppb: NN2 still adds average **0.88 ppb correction**
- NN2 makes it worse in **80% of cases** when PINN is accurate
- When PINN error > 0.5 ppb: NN2 improves in **60% of cases**

### 2. Model Behavior
- Model outputs large corrections even when PINN is already close to actual
- Corrections are **not proportional** to PINN error (correlation: 0.45)
- Model seems to be over-correcting

### 3. Training vs Inference Mismatch?

**Possible Issues**:
1. **Meteorology**: We're using placeholder meteo (u=3.0, v=0.0, D=10.0) instead of actual training meteo
2. **Scaler fitting**: The scalers were fit on training sensors only (in LOO), but we're using them on all sensors
3. **Model interpretation**: Maybe the model was trained/used differently than we think

## Next Steps

1. **Load exact training meteo**: Use the exact meteorology data that was used during training
2. **Check training logs**: Verify how the model was actually trained
3. **Compare with leave-one-out evaluation code**: See exactly how LOO results were calculated
4. **Check model checkpoint**: Verify the model state matches what was used in LOO

## Files Generated

- `nn2_inverse_transform_investigation.csv` - Inverse transform strategies
- `nn2_loo_vs_validation.csv` - LOO vs validation approach comparison
- `nn2_zero_handling_investigation.csv` - Zero-handling strategies
- `nn2_exact_training_data_test.csv` - Test on exact training data
- `NN2_DISCREPANCY_INVESTIGATION.md` - Detailed investigation notes

---
**Date**: 2024
**Investigation**: 2019 data (training data)
**Conclusion**: NN2 performs worse than PINN even on exact training data, suggesting a fundamental issue with model training, usage, or interpretation.

