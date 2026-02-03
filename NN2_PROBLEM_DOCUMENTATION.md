# NN2 Model Problem - Complete Documentation

**Date Created:** 2025-02-02  
**Status:** CRITICAL - Model Not Production Ready  
**Model Location:** `realtime/nn2_scaled/nn2_master_model_ppb-2.pth`

---

## Executive Summary

The NN2 (Neural Network Correction) model is **fundamentally broken** and producing catastrophic predictions. The model shows a **-1,112,795% degradation** compared to PINN alone, outputting predictions in the thousands of ppb range when actual values are < 1 ppb.

### Key Findings

- **PINN Performance:** 0.528 ppb MAE ✓ (Good baseline)
- **NN2 Performance:** 5,878 ppb MAE ✗ (Catastrophic failure)
- **Degradation:** -1,112,795%
- **76.2% of predictions** have absolute values > 1,000 ppb
- **Model trained for only 7 epochs** (essentially untrained)
- **Validation loss:** 16.35 (extremely high)

---

## Table of Contents

1. [Current Performance Metrics](#current-performance-metrics)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Model Architecture Details](#model-architecture-details)
4. [Training History and Issues](#training-history-and-issues)
5. [Data Pipeline Problems](#data-pipeline-problems)
6. [Historical Fixes Applied](#historical-fixes-applied)
7. [Technical Deep Dive](#technical-deep-dive)
8. [Recommendations](#recommendations)
9. [Next Steps](#next-steps)
10. [Appendix: Example Predictions](#appendix-example-predictions)

---

## Current Performance Metrics

### Overall Performance (2019 Validation - 34,213 samples)

| Metric | PINN | NN2 | Status |
|--------|------|-----|--------|
| **MAE** | 0.528 ppb | 5,878 ppb | ✗ Catastrophic |
| **Mean Prediction** | 0.36 ppb | -2,546 ppb | ✗ Negative values |
| **Range** | [0.0068, 30.60] ppb | [-20,443, 7,562] ppb | ✗ Out of bounds |
| **Improvement** | Baseline | -1,112,795% | ✗ Severe degradation |

### Per-Sensor Performance Breakdown

| Sensor ID | PINN MAE | NN2 MAE | Degradation | Samples |
|-----------|----------|---------|------------|---------|
| 482010026 | 0.585 ppb | 3,200 ppb | -547,024% | 4,370 |
| 482010057 | 0.888 ppb | 7,557 ppb | -850,887% | 3,920 |
| 482010069 | 0.322 ppb | 9,603 ppb | -2,980,970% | 4,003 |
| 482010617 | 0.462 ppb | 603 ppb | -130,367% | 3,992 |
| 482010803 | 0.367 ppb | 381 ppb | -103,895% | 4,146 |
| 482011015 | 1.052 ppb | 1,483 ppb | -140,963% | 2,972 |
| 482011035 | 0.283 ppb | 4,975 ppb | -1,756,296% | 3,813 |
| 482011039 | 0.463 ppb | 1,998 ppb | -431,260% | 2,782 |
| 482016000 | 0.447 ppb | 20,436 ppb | -4,569,206% | 4,215 |

**Pattern:** All sensors show extreme degradation, with worst-case sensor (482016000) showing predictions 45,000x larger than actual values.

### Prediction Distribution

- **Actual values:** Range [0.0015, 196.94] ppb, Mean 0.47 ppb
- **PINN predictions:** Range [0.0068, 30.60] ppb, Mean 0.36 ppb ✓
- **NN2 predictions:** Range [-20,443, 7,562] ppb, Mean -2,546 ppb ✗
- **76.2% of NN2 predictions** have |value| > 1,000 ppb
- **Many negative predictions** (physically impossible for concentration)

---

## Root Cause Analysis

### Problem 1: Extremely Large Corrections in Scaled Space ⚠️ **CRITICAL**

**The Core Issue:** The model outputs corrections in scaled space that are **1,000-7,000x larger than the PINN predictions themselves**.

#### Example from First Sample (2019-01-01 16:00:00)

| Sensor | PINN (scaled) | Correction (scaled) | Ratio | Result (ppb) |
|--------|---------------|---------------------|-------|--------------|
| 482010026 | 0.204 | **-1,075** | 5,270x | -3,197 ppb |
| 482010057 | 0.040 | **2,540** | 63,500x | 7,558 ppb |
| 482010069 | 0.053 | **-3,230** | 60,943x | -9,608 ppb |
| 482010617 | 0.375 | 203 | 541x | 607 ppb |
| 482010803 | 0.090 | 127 | 1,411x | 379 ppb |
| 482011015 | 0.244 | 499 | 2,045x | 1,486 ppb |
| 482011035 | 0.033 | **1,671** | 50,636x | 4,972 ppb |
| 482011039 | 0.106 | -672 | 6,340x | -1,998 ppb |
| 482016000 | 0.102 | **-6,869** | 67,343x | -20,438 ppb |

**The Math:**
```
Corrected Scaled = PINN Scaled + Correction
PPB = Corrected Scaled × Scale + Mean
PPB = (-1,075) × 2.975 + 0.469 = -3,197 ppb ❌
```

**Expected Behavior:**
- Corrections should be small relative to PINN predictions (typically ±0.1 to ±0.5 in scaled space)
- For a PINN prediction of 0.204 (scaled), a reasonable correction might be -0.05 to +0.05
- The model is outputting corrections 5,000x larger than the PINN prediction itself

### Problem 2: Training Stopped Too Early

**Evidence:**
- Model trained for only **7 epochs**
- Validation loss: **16.35** (extremely high)
- Expected validation loss after proper training: < 1.0

**Impact:**
- Model weights are essentially random/untrained
- Model never learned the correction task
- Early stopping criteria may have been too aggressive or incorrectly configured

### Problem 3: Loss Function May Be Inadequate

**Current Loss Function:**
```python
def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.001):
    valid_pred = pred[valid_mask]
    valid_target = target[valid_mask]
    
    mse_loss = nn.functional.mse_loss(valid_pred, valid_target)
    correction_reg = lambda_correction * (corrections ** 2).mean()
    
    total_loss = mse_loss + correction_reg
    return total_loss
```

**Issues:**
- `lambda_correction = 0.001` is very small - may not penalize large corrections enough
- No explicit penalty for corrections that are orders of magnitude larger than PINN predictions
- Loss function doesn't enforce that corrections should be small relative to PINN predictions

**What Should Happen:**
- Corrections should be bounded (e.g., ±0.5 in scaled space max)
- Loss should heavily penalize corrections > 1.0 in absolute value
- Model should learn that corrections are small adjustments, not complete replacements

### Problem 4: Data Scaling Mismatch (Potential)

**Hypothesis:** The model may have been trained with incorrectly scaled data, or the scaling during inference doesn't match training.

**Evidence:**
- Corrections are 1,000-7,000x larger than PINN predictions
- This suggests the model learned to output in a different scale than expected

**Investigation Needed:**
- Verify PINN predictions are correctly scaled before input to NN2
- Check that scalers match between training and inference
- Verify temporal features are scaled correctly

---

## Model Architecture Details

### Current Architecture

**Model:** `nn2_scaled/nn2_master_model_ppb-2.pth`

**Architecture:** Simplified (36 → 256 → 128 → 64 → 9)
- **Input Features:** 36
  - PINN predictions (scaled): 9
  - Sensor coordinates (flattened): 18
  - Wind (u, v): 2
  - Diffusion: 1
  - Temporal features: 6
- **Hidden Layers:** 256 → 128 → 64
- **Output:** 9 (corrections in scaled space)
- **Parameters:** ~53,000
- **Output PPB:** True (has inverse transform layer)

**Previous Architecture (Old):** 45 → 512 → 512 → 256 → 128 → 9 (~452K params)
- Included `current_sensors` input (removed to prevent data leakage)

### Model Weights Analysis

**First Layer (256 × 36):**
- Weight range: [-0.180, 0.183] ✓ (reasonable)
- Weight mean: -0.0007 ✓ (near zero, good)
- Weight std: 0.096 ✓ (reasonable)

**Inverse Transform Layer:**
- Mean: 0.4689 ppb ✓
- Scale: 2.9754 ✓
- Matches scaler: Yes ✓

**Conclusion:** Model weights appear reasonable, suggesting the problem is not in weight initialization but in what the model learned (or failed to learn).

### Model Checkpoint Details

```python
{
    'epoch': 7,                    # Very early stopping
    'validation_loss': 16.35,     # Extremely high
    'output_ppb': True,           # ✓ Correct
    'scaler_mean': 0.4689,         # ✓ Matches scaler
    'scaler_scale': 2.9754         # ✓ Matches scaler
}
```

---

## Training History and Issues

### Training Configuration

**From:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`

```python
CONFIG = {
    'n_sensors': 9,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,                  # Configured for 50, but stopped at 7
    'lambda_correction': 0.001,    # Very small regularization
    'device': 'cuda',
    'optimizer': 'AdamW',
    'weight_decay': 1e-5
}
```

### Training Process Issues

1. **Early Stopping:** Model stopped at epoch 7 (out of 50 configured)
   - Need to check early stopping criteria
   - Validation loss 16.35 suggests model never converged
   - Training logs needed to understand why training stopped

2. **Loss Progression:** Unknown
   - No training logs available
   - Cannot determine if loss was decreasing or stuck
   - Cannot verify if model was learning at all

3. **Data Quality:** Unknown
   - Need to verify training data was correctly generated
   - Need to check if PINN predictions in training data used correct time normalization
   - Need to verify scalers were fit correctly

### What Should Have Happened

1. **Training Loss:** Should decrease from ~10-20 to < 1.0 over 50 epochs
2. **Validation Loss:** Should track training loss and converge to < 1.0
3. **Correction Magnitude:** Should learn corrections in range ±0.5 (scaled space)
4. **Convergence:** Model should train for full 50 epochs or until validation loss plateaus

---

## Data Pipeline Problems

### Issue 1: PINN Time Dependency (FIXED)

**Problem:** PINN was using absolute calendar time instead of simulation time, causing predictions to vary 12.13x for identical conditions just by changing timestamp.

**Fix Applied:** Changed to use simulation time `t=3.0 hours` for all predictions.

**Status:** ✓ Fixed in training data generation script  
**Impact:** Training data should now be time-normalized

### Issue 2: Zero-Value Handling (FIXED)

**Problem:** Training masked zeros before scaling, but validation didn't, causing zeros to be transformed to extreme values.

**Fix Applied:** Added zero masking in validation script.

**Status:** ✓ Fixed in validation script  
**Impact:** Zeros now handled consistently

### Issue 3: Training Data May Not Be Regenerated

**Problem:** While fixes were applied to the scripts, the training data may not have been regenerated with the corrected PINN approach.

**Status:** ⚠️ Needs Verification  
**Action Required:** Regenerate training data with corrected PINN

### Issue 4: Scaler Mismatch (Potential)

**Problem:** Scalers may have been fit on data with incorrect PINN predictions (before time normalization fix).

**Status:** ⚠️ Needs Investigation  
**Action Required:** Verify scalers match current data distribution

---

## Historical Fixes Applied

### Fix 1: Data Leakage Removal ✓

**Problem:** Model was receiving `current_sensors` (actual sensor readings) as input, which is data leakage.

**Fix:** Removed `current_sensors` from input features (reduced from 45 to 36 features).

**Status:** ✓ Applied  
**Impact:** Model architecture corrected

### Fix 2: PINN Time Dependency ✓

**Problem:** PINN predictions varied with absolute calendar time, not just physics.

**Fix:** Changed to simulation time `t=3.0 hours` for all predictions.

**Files Modified:**
- `realtime/simpletesting/nn2trainingdata/regenerate_training_data_correct_pinn.py`
- `realtime/validate_nn2_january_2019.py`

**Status:** ✓ Code fixed  
**Action Required:** Regenerate training data

### Fix 3: Zero-Value Handling ✓

**Problem:** Zeros transformed to extreme values during scaling.

**Fix:** Added zero masking before scaling and inverse transform.

**File Modified:** `realtime/validate_nn2_january_2019.py`

**Status:** ✓ Applied

### Fix 4: Simplified Architecture ✓

**Problem:** Original architecture (452K params) too large for training data (5,173 samples).

**Fix:** Reduced to 36 → 256 → 128 → 64 → 9 (~53K params).

**Status:** ✓ Applied

---

## Technical Deep Dive

### How NN2 Should Work

1. **Input:** PINN predictions (scaled), sensor coordinates, wind, diffusion, temporal features
2. **Process:** Neural network outputs corrections in scaled space
3. **Correction:** `corrected_scaled = pinn_scaled + correction`
4. **Output:** Inverse transform to ppb: `ppb = corrected_scaled × scale + mean`

### What's Actually Happening

1. **Input:** PINN predictions (scaled) - appears correct
2. **Process:** Neural network outputs **massive corrections** (1,000-7,000x too large)
3. **Correction:** `corrected_scaled = 0.204 + (-1,075) = -1,075` (completely wrong)
4. **Output:** `ppb = -1,075 × 2.975 + 0.469 = -3,197 ppb` (catastrophic)

### Why This Happens

**Hypothesis 1: Model Never Learned**
- Only 7 epochs of training
- Validation loss 16.35 (extremely high)
- Model weights are essentially random
- **Most Likely Cause**

**Hypothesis 2: Loss Function Inadequate**
- `lambda_correction = 0.001` too small
- No penalty for corrections >> PINN predictions
- Model can output any correction size without penalty
- **Contributing Factor**

**Hypothesis 3: Data Mismatch**
- Training data may have incorrect scaling
- Inference scaling doesn't match training
- **Needs Investigation**

**Hypothesis 4: Architecture Issue**
- Model may not have enough capacity (unlikely - 53K params should be sufficient)
- Or model has too much capacity and overfits to noise (unlikely - only 7 epochs)

### Correction Magnitude Analysis

**Expected Corrections:**
- PINN predictions: ~0.1-1.0 ppb (actual values)
- Scaled PINN: ~0.03-0.3 (scaled space)
- Expected corrections: ±0.01 to ±0.1 (scaled space)
- Correction ratio: 10-30% of PINN prediction

**Actual Corrections:**
- PINN predictions: 0.204 (scaled)
- Actual corrections: -1,075 to 2,540 (scaled)
- Correction ratio: 5,000-12,500x of PINN prediction
- **10,000x too large**

---

## Recommendations

### Immediate Actions (Critical)

1. **DO NOT USE THIS MODEL IN PRODUCTION**
   - Current model is fundamentally broken
   - Predictions are 3-4 orders of magnitude wrong
   - Will cause severe errors in any downstream system

2. **Regenerate Training Data**
   ```bash
   cd realtime/simpletesting/nn2trainingdata
   python regenerate_training_data_correct_pinn.py
   ```
   - Verify PINN uses simulation time `t=3.0 hours`
   - Verify timestamp staggering (predictions at `t` labeled as `t+3`)
   - Verify all 20 facilities are processed correctly

3. **Retrain NN2 Model**
   - Use regenerated training data
   - Train for **minimum 50 epochs** (or until convergence)
   - Monitor training/validation loss carefully
   - Loss should decrease to < 1.0
   - Early stopping should only trigger if loss plateaus

4. **Fix Loss Function**
   ```python
   # Add penalty for corrections >> PINN predictions
   correction_ratio = torch.abs(corrections) / (torch.abs(pinn_predictions) + 1e-6)
   large_correction_penalty = torch.relu(correction_ratio - 1.0).mean()  # Penalize if > 100% of PINN
   total_loss = mse_loss + lambda_correction * correction_reg + lambda_large * large_correction_penalty
   ```

5. **Verify on Training Data First**
   - Run model on exact training data
   - If it fails on training data → model is broken
   - If it works on training data but fails on validation → data mismatch

### Medium-Term Fixes

1. **Improve Training Monitoring**
   - Log training loss every epoch
   - Log validation loss every epoch
   - Log correction magnitude statistics
   - Save best model based on validation loss

2. **Add Correction Bounds**
   - Clip corrections to ±0.5 (scaled space) during training
   - Or add strong penalty for corrections outside this range
   - Enforce that corrections are small relative to PINN predictions

3. **Learning Rate Scheduling**
   - Use ReduceLROnPlateau scheduler
   - Reduce learning rate if validation loss plateaus
   - Helps model converge better

4. **Gradient Clipping**
   - Already implemented: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
   - Keep this to prevent gradient explosion

### Long-Term Improvements

1. **Better Validation Strategy**
   - Test on training data first (should be near-perfect)
   - Use temporal validation (train on 2019, validate on 2020)
   - Monitor per-sensor performance during training

2. **Architecture Tuning**
   - Current architecture (53K params) may be appropriate
   - Consider slightly larger if underfitting
   - Consider smaller if overfitting

3. **Data Augmentation**
   - Add noise to PINN predictions during training
   - Helps model generalize better
   - Makes model more robust to PINN errors

---

## Next Steps

### Step 1: Verify Training Data (1-2 hours)

1. Check if training data exists with corrected PINN
2. If not, regenerate using `regenerate_training_data_correct_pinn.py`
3. Verify data statistics:
   - PINN predictions should be consistent across months
   - No extreme spikes (max should be < 100 ppb)
   - Mean should be ~0.5-1.0 ppb

### Step 2: Retrain Model (2-4 hours)

1. Use corrected training data
2. Update loss function to penalize large corrections
3. Train for 50+ epochs
4. Monitor training/validation loss
5. Save best model based on validation loss

### Step 3: Validate on Training Data (30 minutes)

1. Run model on exact training data
2. Should achieve MAE < 0.5 ppb (near-perfect)
3. If not, investigate further

### Step 4: Validate on 2019 Data (30 minutes)

1. Run `test_nn2_smaller_2019.py`
2. Expected results:
   - PINN MAE: ~0.5-1.0 ppb
   - NN2 MAE: ~0.3-0.5 ppb (40-60% improvement)
   - No degradation

### Step 5: Validate on 2021 Data (1 hour)

1. Run validation on January-March 2021
2. Expected results:
   - PINN MAE: ~0.5-1.0 ppb
   - NN2 MAE: ~0.3-0.5 ppb
   - Consistent improvement across all months

---

## Appendix: Example Predictions

### Sample 1: 2019-01-01 16:00:00

| Sensor | Actual | PINN | NN2 | PINN Error | NN2 Error |
|--------|--------|------|-----|------------|-----------|
| 482010026 | 0.199 | 0.204 | -3,197 | 0.005 | -3,197 |
| 482010057 | 0.140 | 0.040 | 7,558 | -0.100 | 7,558 |
| 482010069 | 0.219 | 0.053 | -9,608 | -0.166 | -9,608 |
| 482010617 | 0.484 | 0.375 | 607 | -0.109 | 607 |
| 482010803 | 0.233 | 0.090 | 379 | -0.143 | 379 |
| 482011015 | 0.461 | 0.244 | 1,486 | -0.217 | 1,486 |
| 482011035 | 0.236 | 0.033 | 4,972 | -0.203 | 4,972 |
| 482011039 | 0.606 | 0.106 | -1,998 | -0.500 | -1,998 |
| 482016000 | 0.212 | 0.102 | -20,438 | -0.110 | -20,438 |

**Analysis:**
- PINN errors: -0.5 to +0.01 ppb (reasonable)
- NN2 errors: -20,438 to +7,558 ppb (catastrophic)
- NN2 makes predictions 3-4 orders of magnitude wrong
- Many negative predictions (physically impossible)

### Correction Magnitude Analysis

For sensor 482010026:
- PINN (scaled): 0.204
- Correction (scaled): -1,075
- Correction ratio: 5,270x
- **Expected ratio:** 0.1-0.3x (10-30% of PINN)

**Conclusion:** Corrections are 10,000x too large.

---

## Files Referenced

### Model Files
- **Model:** `realtime/nn2_scaled/nn2_master_model_ppb-2.pth`
- **Scalers:** `realtime/nn2_scaled/nn2_master_scalers-2.pkl`

### Test Results
- **Detailed Results:** `realtime/x_archived/documentation/nn2_smaller_2019_results.csv`
- **Summary:** `realtime/x_archived/documentation/nn2_smaller_2019_summary.txt`

### Training Scripts
- **Training Script:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`
- **Model Definition:** `realtime/drive-download-20260202T042428Z-3-001/nn2_model_only.py`

### Validation Scripts
- **2019 Validation:** `realtime/test_nn2_smaller_2019.py`
- **January 2019 Validation:** `realtime/validate_nn2_january_2019.py`

### Training Data Generation
- **Corrected PINN:** `realtime/simpletesting/nn2trainingdata/regenerate_training_data_correct_pinn.py`

### Documentation
- **Investigation Report:** `realtime/NN2_SCALED_INVESTIGATION.md`
- **Fixes Applied:** `FIXES_APPLIED.md`
- **Pipeline Documentation:** `COMPLETE_PIPELINE_DOCUMENTATION.md`

---

## Conclusion

The NN2 model is **fundamentally broken** and must be completely retrained. The root cause is that the model outputs corrections that are **1,000-7,000x too large**, likely because:

1. **Model was only trained for 7 epochs** (essentially untrained)
2. **Loss function doesn't penalize large corrections enough**
3. **Training data may not have been regenerated** with corrected PINN

**Action Required:** Regenerate training data, fix loss function, and retrain model with proper convergence monitoring.

**DO NOT USE THIS MODEL IN PRODUCTION.**

---

**Document Version:** 1.0  
**Last Updated:** 2025-02-02  
**Author:** System Analysis  
**Status:** Active Investigation

