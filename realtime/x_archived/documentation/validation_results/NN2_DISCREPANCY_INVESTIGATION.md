# NN2 Discrepancy Investigation - 2019 Data

## Problem Statement

NN2 is performing **worse** than PINN alone on 2019 data (the training data), despite:
- Leave-one-out cross-validation showing 60-77% improvements
- Using the exact same PINN predictions as training
- Using the exact same data structure

## Investigation Results

### Key Findings

1. **All NN2 strategies perform worse than PINN**
   - PINN MAE: 0.3245 ppb
   - NN2 Zero-Masking: 1.1718 ppb
   - NN2 Transform-All: 1.1270 ppb
   - NN2 LOO approach: 1.2345 ppb
   - NN2 Validation approach: 1.1472 ppb

2. **NN2 adds large corrections even when PINN is accurate**
   - When PINN error < 0.1 ppb: NN2 still adds average 0.88 ppb correction
   - NN2 makes it worse in 171/213 cases (80%) when PINN is accurate
   - When PINN error > 0.5 ppb: NN2 improves in 73/122 cases (60%)

3. **Zero-handling matches training**
   - Training code (nn2.py): Only transforms non-zero values (zero-masking)
   - Validation code: Also uses zero-masking
   - **No mismatch here**

4. **Leave-one-out vs validation approach**
   - Both approaches perform similarly poorly
   - LOO approach (as trained): 1.2345 ppb MAE
   - Validation approach: 1.1472 ppb MAE
   - Both worse than PINN (0.3229 ppb)

### Training Data Structure

From `nn2.py`:
- `current_sensors`: Actual sensor readings at timestamp t+3 (ALL sensors, including held-out)
- `target`: Same as `current_sensors` (actual sensor readings at t+3)
- `pinn_predictions`: PINN predictions at timestamp t+3
- Model learns: `pinn_predictions + corrections ≈ target`

### Critical Observation

**The model was trained with `current_sensors = target = actual_sensors`**

This means:
- During training, the model sees the actual reading for ALL sensors (including held-out)
- The model learns to predict: `corrected = pinn + correction`, where `corrected ≈ actual`
- So: `correction ≈ actual - pinn`

**But if PINN is already accurate, corrections should be small!**

### Why Large Corrections?

Possible explanations:

1. **Model overfitting**: The model learned spurious patterns in training data
2. **Scaler distribution mismatch**: The scalers were fit on a different distribution than what we're seeing
3. **Model output interpretation**: Maybe the model outputs something different than we think
4. **Training vs inference mismatch**: Some subtle difference in how inputs are prepared

### Next Steps

1. Check if the model was actually trained correctly
2. Verify the exact training loss and what the model learned
3. Check if there's a bug in how we're interpreting the model output
4. Compare training-time predictions vs validation predictions on the same data

---
**Date**: 2024
**Investigation Period**: 2019 data (training data)
**Key Finding**: NN2 performs worse than PINN even on training data, suggesting a fundamental issue with model training or usage

