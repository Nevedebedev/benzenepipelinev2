# NN2 Model Retraining Plan

**Date:** 2025-02-02  
**Status:** Ready for Implementation  
**Current Model:** `nn2_scaled/nn2_master_model_ppb-2.pth` (7 epochs, validation loss 16.35)

---

## Executive Summary

The current NN2 model was only trained for 7 epochs with extremely high validation loss (16.35), resulting in poor performance (-69.47% degradation). This plan outlines the steps to retrain the model properly.

---

## Current Model Status

### Training History
- **Epochs:** 7 (very early stopping)
- **Validation Loss:** 16.35 (extremely high)
- **Expected Loss:** < 1.0 (after proper training)
- **Performance:** -69.47% improvement (worse than PINN)

### Issues
1. Model essentially untrained (only 7 epochs)
2. Validation loss never converged
3. Model outputs corrections that are too large
4. Performance worse than PINN alone

---

## Retraining Strategy

### Step 1: Verify Training Data ✅

**Action:** Ensure training data uses corrected PINN (time-normalized)

**File:** `realtime/simpletesting/nn2trainingdata/total_superimposed_concentrations.csv`

**Check:**
- PINN predictions use simulation time `t=3.0 hours` (not absolute calendar time)
- Predictions are consistent across months (no 50x variation)
- Mean should be ~0.5-1.0 ppb (not 32 ppb with time bias)

**If data needs regeneration:**
```bash
cd realtime/simpletesting/nn2trainingdata
python regenerate_training_data_correct_pinn.py
```

---

### Step 2: Update Training Configuration

**File:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`

#### Current Configuration (PROBLEMATIC)
```python
CONFIG = {
    'epochs': 50,                  # Configured for 50, but stopped at 7
    'learning_rate': 1e-4,
    'lambda_correction': 0.001,    # Very small - may not penalize large corrections enough
    'batch_size': 32,
    'weight_decay': 1e-5
}
```

#### Recommended Configuration
```python
CONFIG = {
    'epochs': 100,                 # Increased from 50
    'learning_rate': 1e-4,         # Keep same
    'lambda_correction': 0.01,     # INCREASED from 0.001 (penalize large corrections more)
    'batch_size': 32,              # Keep same
    'weight_decay': 1e-5,          # Keep same
    'early_stopping_patience': 15, # Add early stopping (stop if no improvement for 15 epochs)
    'min_delta': 0.001             # Minimum change to qualify as improvement
}
```

---

### Step 3: Improve Loss Function

**Current Loss Function:**
```python
def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.001):
    mse_loss = nn.functional.mse_loss(valid_pred, valid_target)
    correction_reg = lambda_correction * (corrections ** 2).mean()
    total_loss = mse_loss + correction_reg
    return total_loss
```

**Problem:** `lambda_correction = 0.001` is too small - doesn't penalize large corrections enough.

**Recommended Loss Function:**
```python
def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.01, 
                   lambda_large=0.1, pinn_ppb=None):
    """
    Enhanced loss function that:
    1. Penalizes prediction errors (MSE)
    2. Regularizes correction magnitude
    3. Heavily penalizes corrections >> PINN predictions
    """
    mse_loss = nn.functional.mse_loss(valid_pred, valid_target)
    
    # Regularization: penalize large corrections
    correction_reg = lambda_correction * (corrections ** 2).mean()
    
    # NEW: Penalize corrections that are much larger than PINN predictions
    if pinn_ppb is not None:
        # Calculate correction ratio: |correction| / (|PINN| + epsilon)
        correction_ratio = torch.abs(corrections) / (torch.abs(pinn_ppb) + 1e-6)
        # Penalize if correction > 100% of PINN prediction
        large_correction_penalty = lambda_large * torch.relu(correction_ratio - 1.0).mean()
    else:
        large_correction_penalty = torch.tensor(0.0, device=corrections.device)
    
    total_loss = mse_loss + correction_reg + large_correction_penalty
    
    return total_loss, {
        'mse': mse_loss,
        'reg': correction_reg,
        'large_penalty': large_correction_penalty,
        'n_valid': valid_pred.numel()
    }
```

**Key Changes:**
- Increased `lambda_correction` from 0.001 to 0.01 (10x stronger)
- Added `large_correction_penalty` to heavily penalize corrections > 100% of PINN
- Ensures corrections are small relative to PINN predictions

---

### Step 4: Training Monitoring

**Add comprehensive logging:**

```python
# Log every epoch
print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']}: "
      f"Train Loss={train_loss:.4f}, "
      f"MSE={loss_dict['mse']:.4f}, "
      f"Reg={loss_dict['reg']:.4f}, "
      f"Large Penalty={loss_dict['large_penalty']:.4f}")

# Log correction statistics
correction_mean = corrections.abs().mean().item()
correction_max = corrections.abs().max().item()
print(f"  Corrections: mean={correction_mean:.4f}, max={correction_max:.4f} (scaled space)")

# Expected: Corrections should be < 0.5 in scaled space
if correction_max > 1.0:
    print(f"  ⚠️  WARNING: Large corrections detected (max={correction_max:.2f})")
```

**Save best model:**
```python
best_val_loss = float('inf')
best_model_state = None

for epoch in range(CONFIG['epochs']):
    train_loss = train_epoch(...)
    val_loss = validate_epoch(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        print(f"  ✓ New best model (val_loss={val_loss:.4f})")
    
    # Early stopping
    if epoch > CONFIG['early_stopping_patience']:
        if val_loss >= best_val_loss - CONFIG['min_delta']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

# Save best model
if best_model_state is not None:
    torch.save({
        'model_state_dict': best_model_state,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'output_ppb': True,
        'epoch': epoch,
        'validation_loss': best_val_loss,
        'config': CONFIG
    }, model_path)
```

---

### Step 5: Training Convergence Criteria

**Target Metrics:**
- **Training Loss:** Should decrease from ~10-20 to < 1.0
- **Validation Loss:** Should track training loss and converge to < 1.0
- **Correction Magnitude:** Mean < 0.1, Max < 0.5 (in scaled space)
- **Epochs:** Train for 50+ epochs or until validation loss plateaus

**Success Criteria:**
- Validation loss < 1.0
- Training loss < 1.0
- Corrections in range ±0.5 (scaled space)
- No extreme corrections (> 1.0 scaled space)

---

### Step 6: Validation on Training Data

**Before deploying, test on training data:**

```python
# Load model
model = NN2_CorrectionNetwork(...)
model.load_state_dict(checkpoint['model_state_dict'])

# Test on training data
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
train_mae = evaluate_model(model, train_loader, device)

print(f"Training Data MAE: {train_mae:.6f} ppb")
print(f"Expected: < 0.5 ppb (near-perfect on training data)")

if train_mae > 1.0:
    print("  ⚠️  WARNING: Model performs poorly on training data!")
    print("     Model may be broken or data mismatch exists.")
```

**Expected:** MAE < 0.5 ppb on training data (near-perfect).

---

### Step 7: Validation on 2019 Data

**After training, validate on 2019 data:**

```bash
cd realtime
python test_nn2_precomputed_pinn_2019.py
```

**Expected Results:**
- PINN MAE: ~0.5-1.0 ppb
- NN2 MAE: ~0.3-0.5 ppb (40-60% improvement)
- Range: [0, 10] ppb
- No negative values
- Consistent improvement across all sensors

---

## Implementation Checklist

### Pre-Training
- [ ] Verify training data uses corrected PINN (time-normalized)
- [ ] Regenerate training data if needed
- [ ] Check training data statistics (mean ~0.5-1.0 ppb, no extreme spikes)

### Training Configuration
- [ ] Update `epochs` to 100
- [ ] Increase `lambda_correction` to 0.01
- [ ] Add `lambda_large` penalty (0.1)
- [ ] Add early stopping configuration
- [ ] Update loss function with large correction penalty

### Training Process
- [ ] Add comprehensive logging (loss components, correction stats)
- [ ] Implement best model saving
- [ ] Implement early stopping
- [ ] Monitor training/validation loss curves
- [ ] Check correction magnitude during training

### Post-Training
- [ ] Test on training data (should be < 0.5 ppb MAE)
- [ ] Validate on 2019 data (should show 40-60% improvement)
- [ ] Check prediction range (should be 0-10 ppb)
- [ ] Verify no negative values
- [ ] Compare per-sensor performance

---

## Expected Training Progression

### Loss Progression (Expected)

| Epoch | Training Loss | Validation Loss | Correction Mean | Status |
|-------|---------------|-----------------|-----------------|--------|
| 1 | 15.0 | 16.0 | 2.5 | Initial |
| 10 | 5.0 | 5.5 | 1.0 | Learning |
| 20 | 2.0 | 2.2 | 0.5 | Improving |
| 30 | 1.0 | 1.1 | 0.3 | Converging |
| 40 | 0.8 | 0.9 | 0.2 | Good |
| 50 | 0.7 | 0.8 | 0.15 | Excellent |
| 60+ | 0.6-0.7 | 0.7-0.8 | 0.1 | Converged |

**Target:** Validation loss < 1.0, correction mean < 0.2

---

## Files to Modify

1. **Training Script:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`
   - Update CONFIG (epochs, lambda_correction)
   - Update loss function (add large correction penalty)
   - Add training monitoring
   - Add early stopping

2. **Training Data:** `realtime/simpletesting/nn2trainingdata/total_superimposed_concentrations.csv`
   - Verify uses corrected PINN (time-normalized)
   - Regenerate if needed

---

## Training Command

```bash
# In Colab or local environment
cd realtime/drive-download-20260202T042428Z-3-001
python nn2colab_clean_master_only.py
```

**Expected Duration:** 2-4 hours (depending on hardware)

---

## Success Metrics

After retraining, the model should achieve:

- ✅ **Validation Loss:** < 1.0 (currently 16.35)
- ✅ **Training Data MAE:** < 0.5 ppb (near-perfect)
- ✅ **2019 Validation MAE:** ~0.3-0.5 ppb (40-60% improvement over PINN)
- ✅ **Prediction Range:** [0, 10] ppb (not thousands)
- ✅ **No Negative Values:** All predictions ≥ 0
- ✅ **Correction Magnitude:** Mean < 0.2, Max < 0.5 (scaled space)
- ✅ **Consistent Improvement:** All sensors show improvement

---

## Troubleshooting

### If validation loss doesn't decrease:
- Check training data quality
- Verify normalization is correct
- Check learning rate (may need adjustment)
- Verify model architecture matches data

### If corrections are still too large:
- Increase `lambda_correction` further (0.01 → 0.1)
- Increase `lambda_large` penalty (0.1 → 1.0)
- Check if PINN predictions are correctly scaled

### If model overfits:
- Increase dropout rates
- Add more regularization (weight decay)
- Reduce model capacity if needed

### If model underfits:
- Train for more epochs
- Reduce regularization
- Check if learning rate is too low

---

## Notes

- The current model (`nn2_scaled`) is essentially untrained (7 epochs)
- Pipeline fixes are correct - the issue is model quality
- Retraining should achieve 40-60% improvement over PINN
- Use precomputed PINN values for consistent testing
- Monitor correction magnitude during training (should be < 0.5 scaled space)

---

**Status:** Ready for implementation  
**Priority:** High (model needs retraining for production use)

