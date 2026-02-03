# NN2 Scaled Model Investigation Report

**Date:** 2025-02-02  
**Model:** `nn2_scaled/nn2_master_model_ppb-2.pth`  
**Test:** Full 2019 validation (5,920 timestamps, 34,213 samples)

---

## Executive Summary

The NN2 model from `nn2_scaled` is producing **catastrophically large predictions** (MAE: 5,878 ppb) compared to PINN alone (MAE: 0.528 ppb), representing a **-1,112,795% degradation**. The model outputs are consistently in the thousands of ppb range, with 76.2% of predictions having absolute values > 1,000 ppb.

---

## Test Results

### Overall Performance
- **PINN MAE:** 0.528 ppb ✓
- **NN2 MAE:** 5,878 ppb ✗
- **Improvement:** -1,112,795% (severe degradation)

### Per-Sensor Performance
All sensors show extreme degradation:
- **482010026:** PINN 0.585 ppb → NN2 3,200 ppb (-547,024% degradation)
- **482010057:** PINN 0.888 ppb → NN2 7,557 ppb (-850,887% degradation)
- **482010069:** PINN 0.322 ppb → NN2 9,603 ppb (-2,980,970% degradation)
- **482010617:** PINN 0.462 ppb → NN2 603 ppb (-130,367% degradation)
- **482010803:** PINN 0.367 ppb → NN2 381 ppb (-103,895% degradation)
- **482011015:** PINN 1.052 ppb → NN2 1,483 ppb (-140,963% degradation)
- **482011035:** PINN 0.283 ppb → NN2 4,975 ppb (-1,756,296% degradation)
- **482011039:** PINN 0.463 ppb → NN2 1,998 ppb (-431,260% degradation)
- **482016000:** PINN 0.447 ppb → NN2 20,436 ppb (-4,569,206% degradation)

### Prediction Statistics
- **Actual range:** [0.0015, 196.94] ppb
- **PINN range:** [0.0068, 30.60] ppb
- **NN2 range:** [-20,443, 7,562] ppb ⚠️

- **Actual mean:** 0.47 ppb
- **PINN mean:** 0.36 ppb
- **NN2 mean:** -2,546 ppb ⚠️

- **76.2% of predictions** have |NN2| > 1,000 ppb

---

## Model Architecture Analysis

### Model Checkpoint Details
- **Epoch:** 7 (very early stopping)
- **Validation Loss:** 16.35 (extremely high)
- **Output PPB:** True ✓
- **Architecture:** 36 → 256 → 128 → 64 → 9 (simplified) ✓
- **Input Features:** 36 (no `current_sensors`) ✓

### Model Weights
- **First layer:** 256 × 36
- **Weight range:** [-0.180, 0.183] (reasonable)
- **Weight mean:** -0.0007 (near zero, good)
- **Weight std:** 0.096 (reasonable)

### Inverse Transform Layer
- **Mean:** 0.4689 ppb ✓
- **Scale:** 2.9754 ✓
- **Matches scaler:** Yes ✓

---

## Root Cause Analysis

### Problem 1: Extremely Large Corrections in Scaled Space ⚠️ **CRITICAL**

The model is outputting **corrections in scaled space that are 1,000-7,000x larger than the PINN predictions themselves**.

**Actual example from first sample:**
- PINN scaled: [0.204, 0.040, 0.053, 0.375, 0.090, 0.244, 0.033, 0.106, 0.102]
- **Corrections: [-1,075, 2,540, -3,230, 203, 127, 499, 1,671, -672, -6,869]** ⚠️
- Corrected scaled: [-1,075, 2,540, -3,229, 204, 127, 499, 1,671, -672, -6,869]

**The corrections are 1,000-7,000x larger than the PINN predictions!**

When these massive corrections are inverse transformed:
```
PPB = corrected_scaled * scale + mean
PPB = -1,075 * 2.975 + 0.469 = -3,197 ppb ❌
PPB = -6,869 * 2.975 + 0.469 = -20,437 ppb ❌
```

**This is the root cause:** The model is outputting corrections that are completely out of scale. A correction of -1,075 in scaled space is equivalent to correcting by -3,197 ppb, which is nonsensical when actual values are < 1 ppb.

### Problem 2: Training Stopped Too Early

The model was trained for only **7 epochs** with a validation loss of **16.35**, which is extremely high. This suggests:
1. The model never converged
2. Training was stopped prematurely
3. The model is essentially untrained

### Problem 3: Loss Function May Be Wrong

The validation loss of 16.35 suggests the loss function may not be appropriate for the problem, or the model is fundamentally learning the wrong thing.

---

## Sample Prediction Breakdown

**First sample (2019-01-01 16:00:00):**

| Sensor | Actual | PINN | NN2 | Error |
|--------|--------|------|-----|-------|
| 482010026 | 0.199 | 0.204 | -3,197 | -3,197 |
| 482010057 | 0.140 | 0.040 | 7,558 | 7,558 |
| 482010069 | 0.219 | 0.053 | -9,608 | -9,608 |
| 482010617 | 0.484 | 0.375 | 607 | 607 |
| 482010803 | 0.233 | 0.090 | 379 | 379 |
| 482011015 | 0.461 | 0.244 | 1,486 | 1,486 |
| 482011035 | 0.236 | 0.033 | 4,972 | 4,972 |
| 482011039 | 0.606 | 0.106 | -1,998 | -1,998 |
| 482016000 | 0.212 | 0.102 | -20,438 | -20,438 |

**Pattern:** The NN2 predictions are consistently 3-4 orders of magnitude larger than actual values, with many negative predictions.

---

## Potential Causes

### 1. Model Not Properly Trained
- **Evidence:** Only 7 epochs, validation loss 16.35
- **Impact:** Model weights are essentially random/untrained
- **Solution:** Retrain with more epochs, check training logs

### 2. Loss Function Issue
- **Evidence:** Extremely high validation loss
- **Impact:** Model is optimizing for wrong objective
- **Solution:** Review loss function, check if direction/size penalties are too strong

### 3. Data Scaling Mismatch
- **Evidence:** Corrections are 10-100x larger than PINN predictions
- **Impact:** Model outputs in wrong scale
- **Solution:** Check if PINN predictions are correctly scaled before input

### 4. Inverse Transform Applied Incorrectly
- **Evidence:** Predictions are in thousands of ppb
- **Impact:** Wrong conversion from scaled to ppb
- **Solution:** Verify inverse transform calculation

### 5. Model Architecture Too Small
- **Evidence:** Simplified architecture may not have enough capacity
- **Impact:** Model cannot learn complex corrections
- **Solution:** Consider slightly larger architecture

---

## Recommendations

### Immediate Actions

1. **Check Training Logs**
   - Verify training loss progression
   - Check if model was actually learning
   - Confirm early stopping criteria

2. **Retrain Model**
   - Use more epochs (50+)
   - Monitor training/validation loss carefully
   - Ensure model converges before stopping

3. **Verify Data Pipeline**
   - Confirm PINN predictions are correctly scaled
   - Check that all inputs match training data format
   - Verify temporal features are correct

4. **Review Loss Function**
   - Check if direction/size penalties are appropriate
   - Verify loss is being calculated correctly
   - Consider removing penalties temporarily to test

5. **Test on Training Data**
   - Run model on exact training data
   - If it fails on training data, the model is broken
   - If it works on training data, there's a data mismatch

### Long-Term Fixes

1. **Increase Model Capacity** (if needed)
   - Current: 36 → 256 → 128 → 64 → 9 (~53K params)
   - Consider: 36 → 512 → 256 → 128 → 9 (~150K params)

2. **Improve Training Strategy**
   - Use learning rate scheduling
   - Implement gradient clipping
   - Add more regularization if overfitting

3. **Better Validation**
   - Test on training data first
   - Use smaller validation set during training
   - Monitor per-sensor performance

---

## Conclusion

The NN2 model from `nn2_scaled` is **fundamentally broken**. The model:
- Was trained for only 7 epochs (essentially untrained)
- Has extremely high validation loss (16.35)
- **Produces corrections 1,000-7,000x larger than PINN predictions** (the critical bug)
- Outputs predictions in thousands of ppb (should be < 1 ppb)

### Root Cause

The model is outputting corrections in scaled space that are **1,000-7,000x too large**. For example:
- PINN prediction: 0.204 (scaled)
- Model correction: -1,075 (scaled) ← **5,000x larger than PINN!**
- Result: -3,197 ppb (should be ~0.2 ppb)

This suggests the model:
1. Never learned to output small corrections
2. May have a loss function that doesn't penalize large corrections enough
3. May have been trained with incorrect data scaling

### The model needs to be completely retrained** with:
- More epochs (50+ minimum)
- Proper convergence monitoring (loss should drop to < 1.0)
- Loss function that heavily penalizes large corrections
- Verification on training data first
- Check that corrections are in reasonable range (±0.5 scaled space max)

**Do not use this model in production.**

---

## Files Referenced

- Model: `realtime/nn2_scaled/nn2_master_model_ppb-2.pth`
- Scalers: `realtime/nn2_scaled/nn2_master_scalers-2.pkl`
- Test Results: `realtime/nn2_smaller_2019_results.csv`
- Test Summary: `realtime/nn2_smaller_2019_summary.txt`
- Training Script: `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`

