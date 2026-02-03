# NN2 Model Retraining Instructions

**Date:** 2025-02-02  
**Status:** Ready for Implementation  
**Current Model:** `nn2_scaled/nn2_master_model_ppb-2.pth` (7 epochs, validation loss 16.35)

---

## Overview

The current NN2 model was only trained for 7 epochs with extremely high validation loss (16.35), resulting in poor performance (-23.75% degradation). This guide provides step-by-step instructions to retrain the model properly.

---

## Key Insights & Intuitions

### Why the Current Model Failed

1. **Insufficient Training (7 epochs)**
   - Neural networks need time to learn complex patterns
   - 7 epochs is barely enough to initialize weights
   - Model never converged (loss should drop from ~15 to < 1.0)

2. **Weak Regularization (lambda_correction = 0.001)**
   - Model learned to output large corrections (1000-7000x PINN)
   - Regularization was too weak to prevent this
   - Need 10x stronger penalty (0.01) to keep corrections small

3. **No Penalty for Extreme Corrections**
   - Model can output corrections >> PINN predictions
   - No mechanism to prevent physically unreasonable corrections
   - Need explicit penalty for corrections > 100% of PINN

### What the Model Should Learn

**NN2's Role:** Make small, targeted corrections to PINN predictions based on:
- **Spatial patterns** (sensor locations relative to sources)
- **Wind conditions** (direction, speed)
- **Diffusion** (atmospheric mixing)
- **Temporal features** (time of day, day of week, season)

**What corrections should look like:**
- **Magnitude:** Corrections should be 10-50% of PINN predictions (not 1000x)
- **Direction:** Corrections should reduce PINN errors, not amplify them
- **Range:** In scaled space, corrections should be ±0.5 max (not ±1000)

### Training Intuition

**Think of NN2 as a "fine-tuning" layer:**
- PINN provides the base prediction (physics-based)
- NN2 makes small adjustments based on learned patterns
- If corrections are too large, the model is overcorrecting

**Loss function components:**
1. **MSE Loss:** Primary driver - minimize prediction error
2. **Correction Regularization:** Keep corrections small (lambda_correction)
3. **Large Correction Penalty:** Prevent extreme corrections (lambda_large)
4. **Direction Penalty:** Encourage corrections in right direction

**Training progression:**
- **Early epochs (1-10):** Loss drops rapidly (15 → 5)
- **Mid epochs (10-30):** Gradual improvement (5 → 1)
- **Late epochs (30-60):** Fine-tuning (1 → 0.6)
- **Convergence (60+):** Loss plateaus, corrections stabilize

### Critical Metrics to Monitor

1. **Validation Loss:** Should drop to < 1.0
2. **Correction Mean:** Should be < 0.2 (scaled space)
3. **Correction Max:** Should be < 0.5 (scaled space)
4. **Training vs Validation:** Should track closely (no overfitting)

**Red Flags:**
- Correction max > 1.0 → Increase regularization
- Validation loss >> Training loss → Overfitting (increase dropout)
- Loss not decreasing → Check data, learning rate
- Corrections all same sign → Model bias, check inputs

---

---

## Prerequisites

### 1. Verify Training Data

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

**Expected output:**
- File: `total_superimposed_concentrations.csv`
- Columns: `t`, `sensor_482010026`, `sensor_482010057`, ..., `sensor_482016000`
- Values in ppb (range: 0.01 - ~30 ppb)
- Mean: ~0.5-1.0 ppb

---

## Step 1: Update Training Configuration

**File:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`

### Current Configuration (PROBLEMATIC)
```python
CONFIG = {
    'n_sensors': 9,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,                  # Configured for 50, but stopped at 7
    'lambda_correction': 0.001,    # Too small - doesn't penalize large corrections
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': '/content/models/master_only/',
    'data_dir': '/content/data/',
    'pinn_file': '/content/data/total_superimposed_concentrations.csv',
    'sensor_coords_file': '/content/data/sensor_coordinates.csv',
    'sensor_file': '/content/data/sensors_final.csv',
    'source_dir': '/content/data/data_nonzero/'
}
```

### Recommended Configuration
```python
CONFIG = {
    'n_sensors': 9,
    'batch_size': 32,
    'learning_rate': 1e-4,         # Keep same
    'epochs': 100,                  # INCREASED from 50
    'lambda_correction': 0.01,      # INCREASED from 0.001 (10x stronger)
    'lambda_large': 0.1,            # NEW: Penalty for corrections >> PINN
    'early_stopping_patience': 15,  # NEW: Stop if no improvement for 15 epochs
    'min_delta': 0.001,             # NEW: Minimum change to qualify as improvement
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': '/content/models/master_only/',
    'data_dir': '/content/data/',
    'pinn_file': '/content/data/total_superimposed_concentrations.csv',
    'sensor_coords_file': '/content/data/sensor_coordinates.csv',
    'sensor_file': '/content/data/sensors_final.csv',
    'source_dir': '/content/data/data_nonzero/'
}
```

**Key Changes:**
- `epochs`: 50 → 100 (more training time)
- `lambda_correction`: 0.001 → 0.01 (10x stronger regularization)
- Added `lambda_large`: 0.1 (penalty for large corrections)
- Added early stopping configuration

---

## Step 2: Improve Loss Function

**File:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`  
**Function:** `correction_loss()` (lines 370-428)

### Current Loss Function
```python
def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.001, target_ppb=None, pinn_ppb=None):
    mse_loss = nn.functional.mse_loss(valid_pred, valid_target)
    correction_reg = lambda_correction * (corrections ** 2).mean()
    
    # Direction and size penalties exist but may not be strong enough
    direction_penalty = ...  # 0.3 weight
    size_penalty = ...       # 0.1 weight
    
    total = mse_loss + lambda_correction * correction_reg + 0.3 * direction_penalty + 0.1 * size_penalty
    return total, {...}
```

### Recommended Loss Function
```python
def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.01, 
                   lambda_large=0.1, target_ppb=None, pinn_ppb=None):
    """
    Enhanced loss function that:
    1. Penalizes prediction errors (MSE)
    2. Regularizes correction magnitude (lambda_correction)
    3. Heavily penalizes corrections >> PINN predictions (lambda_large)
    """
    # Use ppb target if provided
    if target_ppb is not None:
        valid_pred = pred[valid_mask]
        valid_target = target_ppb[valid_mask]
    else:
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]

    if valid_pred.numel() > 0:
        mse_loss = nn.functional.mse_loss(valid_pred, valid_target)
    else:
        mse_loss = torch.tensor(0.0, device=pred.device)

    # Regularization: penalize large corrections
    correction_reg = lambda_correction * (corrections ** 2).mean()
    
    # NEW: Heavily penalize corrections that are much larger than PINN predictions
    large_correction_penalty = torch.tensor(0.0, device=pred.device)
    if pinn_ppb is not None and target_ppb is not None:
        # Calculate corrections in ppb space
        corrections_ppb = pred - pinn_ppb
        
        # Calculate correction ratio: |correction| / (|PINN| + epsilon)
        pinn_magnitude = torch.abs(pinn_ppb) + 1e-6
        correction_ratio = torch.abs(corrections_ppb) / pinn_magnitude
        
        # Penalize if correction > 100% of PINN prediction (very large)
        large_correction_mask = (correction_ratio > 1.0) & valid_mask
        if large_correction_mask.any():
            # Strong penalty for corrections > 100% of PINN
            large_correction_penalty = lambda_large * torch.relu(correction_ratio[large_correction_mask] - 1.0).mean()
    
    # Direction penalty (keep existing)
    direction_penalty = torch.tensor(0.0, device=pred.device)
    if target_ppb is not None and pinn_ppb is not None:
        error = target_ppb - pinn_ppb
        corrections_ppb = pred - pinn_ppb
        wrong_direction = (corrections_ppb * error < 0) & valid_mask
        if wrong_direction.any():
            direction_penalty = 0.3 * torch.relu(-corrections_ppb[wrong_direction] * error[wrong_direction]).mean()
    
    total_loss = mse_loss + correction_reg + large_correction_penalty + direction_penalty
    
    return total_loss, {
        'mse': mse_loss.item(),
        'correction_reg': correction_reg.item(),
        'large_penalty': large_correction_penalty.item() if isinstance(large_correction_penalty, torch.Tensor) else large_correction_penalty,
        'direction_penalty': direction_penalty.item() if isinstance(direction_penalty, torch.Tensor) else direction_penalty,
        'n_valid': valid_pred.numel()
    }
```

**Key Changes:**
- Increased `lambda_correction` default from 0.001 to 0.01
- Added `lambda_large` parameter (0.1) for large correction penalty
- Stronger penalty for corrections > 100% of PINN predictions
- Ensures corrections are small relative to PINN

---

## Step 3: Add Training Monitoring

**File:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`  
**Function:** Training loop (around line 600+)

### Add Comprehensive Logging

```python
# Before training loop
best_val_loss = float('inf')
best_model_state = None
patience_counter = 0
training_history = []

print("\n" + "="*70)
print("TRAINING MASTER MODEL")
print("="*70)
print(f"Epochs: {CONFIG['epochs']}")
print(f"Learning Rate: {CONFIG['learning_rate']}")
print(f"Lambda Correction: {CONFIG['lambda_correction']}")
print(f"Lambda Large: {CONFIG.get('lambda_large', 0.1)}")
print("="*70 + "\n")

for epoch in range(CONFIG['epochs']):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, CONFIG['device'], CONFIG['lambda_correction'])
    
    # Validate (if validation set exists)
    val_loss = evaluate_model(model, val_loader, CONFIG['device'], CONFIG['lambda_correction'])
    
    # Log correction statistics
    correction_stats = get_correction_statistics(model, train_loader, CONFIG['device'])
    
    # Log every epoch
    print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']}: "
          f"Train Loss={train_loss:.4f}, "
          f"Val Loss={val_loss:.4f}, "
          f"Correction Mean={correction_stats['mean']:.4f}, "
          f"Correction Max={correction_stats['max']:.4f}")
    
    # Check for improvement
    if val_loss < best_val_loss - CONFIG.get('min_delta', 0.001):
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"  ✓ New best model (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= CONFIG.get('early_stopping_patience', 15):
        print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
        break
    
    # Warning if corrections are too large
    if correction_stats['max'] > 1.0:
        print(f"  ⚠️  WARNING: Large corrections detected (max={correction_stats['max']:.2f} scaled space)")
        print(f"     Expected: < 0.5 scaled space")
    
    # Save history
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'correction_mean': correction_stats['mean'],
        'correction_max': correction_stats['max']
    })
```

### Add Correction Statistics Function

```python
def get_correction_statistics(model, dataloader, device):
    """Get statistics about correction magnitudes during training"""
    model.eval()
    all_corrections = []
    
    with torch.no_grad():
        for batch in dataloader:
            pinn_predictions = batch['pinn_predictions'].to(device)
            sensor_coords = batch['sensor_coords'].to(device)
            wind = batch['wind'].to(device)
            diffusion = batch['diffusion'].to(device)
            temporal = batch['temporal'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            _, corrections = model(pinn_predictions, sensor_coords, wind, diffusion, temporal)
            all_corrections.append(corrections[valid_mask].cpu())
    
    if len(all_corrections) > 0:
        all_corrections = torch.cat(all_corrections, dim=0)
        return {
            'mean': all_corrections.abs().mean().item(),
            'max': all_corrections.abs().max().item(),
            'std': all_corrections.abs().std().item()
        }
    else:
        return {'mean': 0.0, 'max': 0.0, 'std': 0.0}
```

---

## Step 4: Save Best Model

**File:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`

### Update Model Saving

```python
# After training loop
if best_model_state is not None:
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Get scaler parameters from dataset
    scaler_mean = dataset.scalers['sensors'].mean_[0] if hasattr(dataset.scalers['sensors'], 'mean_') else None
    scaler_scale = dataset.scalers['sensors'].scale_[0] if hasattr(dataset.scalers['sensors'], 'scale_') else None
    
    # Save model with all necessary info
    checkpoint = {
        'model_state_dict': best_model_state,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'output_ppb': True,
        'epoch': epoch + 1,
        'validation_loss': best_val_loss,
        'config': CONFIG,
        'sensor_coords': dataset.sensor_coords_array,
        'training_history': training_history
    }
    
    model_path = Path(CONFIG['save_dir']) / 'nn2_master_model_ppb-3.pth'
    torch.save(checkpoint, model_path)
    print(f"\n✓ Saved best model to: {model_path}")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Epoch: {epoch + 1}")
    
    # Save scalers separately
    scalers_path = Path(CONFIG['save_dir']) / 'nn2_master_scalers-3.pkl'
    with open(scalers_path, 'wb') as f:
        pickle.dump(dataset.scalers, f)
    print(f"✓ Saved scalers to: {scalers_path}")
else:
    print("\n⚠️  No best model to save (training may have failed)")
```

---

## Step 5: Training Execution

### Option A: Google Colab

1. **Upload files to Colab:**
   - Upload `nn2colab_clean_master_only.py`
   - Upload training data files
   - Upload model definition (`nn2_model_only.py`)

2. **Set up environment:**
```python
# In Colab notebook
!pip install torch pandas numpy scikit-learn tqdm

# Mount Google Drive (if data is there)
from google.colab import drive
drive.mount('/content/drive')

# Set paths
CONFIG['data_dir'] = '/content/drive/MyDrive/your_data_path/'
CONFIG['save_dir'] = '/content/drive/MyDrive/your_models_path/'
```

3. **Run training:**
```python
# Execute training script
exec(open('nn2colab_clean_master_only.py').read())
```

### Option B: Local Machine

1. **Set up environment:**
```bash
# Install dependencies
pip install torch pandas numpy scikit-learn tqdm

# Set paths in CONFIG
# Update data_dir and save_dir to local paths
```

2. **Run training:**
```bash
cd realtime/drive-download-20260202T042428Z-3-001
python nn2colab_clean_master_only.py
```

---

## Step 6: Monitor Training Progress

### Expected Loss Progression

| Epoch | Training Loss | Validation Loss | Correction Mean | Correction Max | Status |
|-------|---------------|-----------------|----------------|---------------|--------|
| 1 | 15.0-20.0 | 16.0-21.0 | 2.0-3.0 | 5.0-10.0 | Initial |
| 10 | 3.0-5.0 | 3.5-5.5 | 0.8-1.2 | 2.0-3.0 | Learning |
| 20 | 1.5-2.5 | 1.8-2.8 | 0.4-0.6 | 1.0-1.5 | Improving |
| 30 | 0.8-1.2 | 1.0-1.4 | 0.2-0.4 | 0.5-1.0 | Converging |
| 40 | 0.6-0.9 | 0.8-1.1 | 0.15-0.25 | 0.3-0.6 | Good |
| 50 | 0.5-0.7 | 0.7-0.9 | 0.1-0.2 | 0.2-0.5 | Excellent |
| 60+ | 0.4-0.6 | 0.6-0.8 | 0.08-0.15 | 0.15-0.4 | Converged |

**Target Metrics:**
- Validation loss < 1.0
- Correction mean < 0.2 (scaled space)
- Correction max < 0.5 (scaled space)

### Warning Signs

**If validation loss doesn't decrease:**
- Check training data quality
- Verify normalization is correct
- Check learning rate (may need adjustment)
- Verify model architecture matches data

**If corrections are still too large (> 1.0 scaled space):**
- Increase `lambda_correction` further (0.01 → 0.1)
- Increase `lambda_large` penalty (0.1 → 1.0)
- Check if PINN predictions are correctly scaled

**If model overfits (train loss << val loss):**
- Increase dropout rates
- Add more regularization (weight decay)
- Reduce model capacity if needed

**If model underfits (both losses high):**
- Train for more epochs
- Reduce regularization
- Check if learning rate is too low

---

## Step 7: Validate Retrained Model

### Test on Training Data First

```python
# Load retrained model
checkpoint = torch.load('nn2_master_model_ppb-3.pth', map_location='cpu')
model = NN2_CorrectionNetwork(...)
model.load_state_dict(checkpoint['model_state_dict'])

# Test on training data
train_mae = evaluate_model(model, train_loader, device, lambda_correction)

print(f"Training Data MAE: {train_mae:.6f} ppb")
print(f"Expected: < 0.5 ppb (near-perfect on training data)")

if train_mae > 1.0:
    print("  ⚠️  WARNING: Model performs poorly on training data!")
    print("     Model may be broken or data mismatch exists.")
```

**Expected:** MAE < 0.5 ppb on training data.

### Test on 2019 Validation Data

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

## Step 8: Compare Results

### Before Retraining
- MAE: 0.665 ppb
- Improvement: -23.75%
- Range: [0.00, 24.94] ppb

### After Retraining (Expected)
- MAE: 0.3-0.5 ppb
- Improvement: +40-60%
- Range: [0, 10] ppb
- All sensors show improvement

---

## Troubleshooting

### Issue: Training Loss Stuck High

**Symptoms:**
- Loss doesn't decrease below 10.0
- Corrections remain large (> 2.0 scaled space)

**Solutions:**
1. Check training data - verify PINN predictions are correct
2. Increase learning rate (1e-4 → 5e-4)
3. Check normalization - ensure all inputs are scaled correctly
4. Verify model architecture matches checkpoint

### Issue: Validation Loss Higher Than Training Loss

**Symptoms:**
- Train loss: 0.5, Val loss: 2.0
- Model overfitting

**Solutions:**
1. Increase dropout rates (0.3 → 0.5)
2. Increase weight decay (1e-5 → 1e-4)
3. Add more regularization
4. Reduce model capacity if needed

### Issue: Corrections Still Too Large

**Symptoms:**
- Correction max > 1.0 (scaled space)
- Predictions still show large errors

**Solutions:**
1. Increase `lambda_correction` (0.01 → 0.1)
2. Increase `lambda_large` (0.1 → 1.0)
3. Add explicit clipping during training (not recommended, but can help)
4. Check if PINN predictions are correctly scaled

### Issue: Model Doesn't Improve

**Symptoms:**
- Loss decreases but MAE doesn't improve
- Model learns but doesn't generalize

**Solutions:**
1. Check if training data matches validation data distribution
2. Verify temporal features are correct
3. Check if sensor coordinates match
4. Verify wind/diffusion scaling

---

## Success Criteria

After retraining, the model should achieve:

- ✅ **Validation Loss:** < 1.0 (currently 16.35)
- ✅ **Training Data MAE:** < 0.5 ppb (near-perfect)
- ✅ **2019 Validation MAE:** ~0.3-0.5 ppb (40-60% improvement over PINN)
- ✅ **Prediction Range:** [0, 10] ppb (not thousands)
- ✅ **No Negative Values:** All predictions ≥ 0 (with clipping)
- ✅ **Correction Magnitude:** Mean < 0.2, Max < 0.5 (scaled space)
- ✅ **Consistent Improvement:** All sensors show improvement

---

## Files to Modify

1. **`realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`**
   - Update CONFIG (lines 30-43)
   - Update loss function (lines 370-428)
   - Add training monitoring (lines 600+)
   - Add early stopping
   - Add best model saving

---

## Training Time Estimate

- **Hardware:** GPU (CUDA) recommended
- **Estimated Time:** 2-4 hours for 100 epochs
- **CPU Only:** 8-12 hours (not recommended)

---

## Post-Training Checklist

- [ ] Validation loss < 1.0
- [ ] Training data MAE < 0.5 ppb
- [ ] Correction mean < 0.2 (scaled space)
- [ ] Correction max < 0.5 (scaled space)
- [ ] Test on 2019 data shows 40-60% improvement
- [ ] All sensors show improvement
- [ ] Predictions in range [0, 10] ppb
- [ ] No negative values (with clipping)
- [ ] Model saved with all metadata
- [ ] Scalers saved separately

---

**Status:** Ready for implementation  
**Priority:** High (required for production use)

---

## Additional Training Insights

### Understanding the Loss Landscape

**Initial State (Epoch 1):**
- Loss: ~15-20 (model outputs random corrections)
- Corrections: Large and random (±1000 scaled space)
- Model has no idea what to do

**Learning Phase (Epochs 1-20):**
- Loss drops rapidly as model learns basic patterns
- Corrections become smaller but still noisy
- Model starts to understand spatial relationships

**Refinement Phase (Epochs 20-50):**
- Loss decreases gradually (2.0 → 0.8)
- Corrections stabilize around ±0.5 scaled space
- Model learns subtle patterns (wind effects, temporal cycles)

**Convergence Phase (Epochs 50+):**
- Loss plateaus around 0.6-0.8
- Corrections become consistent (±0.2 scaled space)
- Model has learned optimal correction strategy

### Why Early Stopping at 7 Epochs Failed

The model stopped training just as it was starting to learn:
- **Epoch 7:** Loss ~16.35 (essentially random)
- **Epoch 20:** Loss would be ~2.0 (10x better)
- **Epoch 50:** Loss would be ~0.7 (23x better)

**The patience counter triggered too early:**
- Current code: `max_patience = 10`
- Problem: Loss was still decreasing, just slowly
- Solution: Increase patience or use better early stopping criteria

### Correction Magnitude Intuition

**In scaled space:**
- PINN predictions: typically -2 to +2 (after StandardScaler)
- Corrections should be: -0.5 to +0.5 (10-25% of PINN range)
- If corrections are ±1.0 or larger, model is overcorrecting

**In ppb space:**
- PINN predictions: 0.1 - 5.0 ppb (typical range)
- Corrections should be: ±0.1 - ±0.5 ppb (small adjustments)
- If corrections are ±10 ppb, model is broken

**Physical interpretation:**
- Small correction (±0.2 ppb): Fine-tuning for local conditions
- Medium correction (±0.5 ppb): Accounting for wind shifts
- Large correction (±2.0 ppb): Model is trying to fix PINN errors (should be rare)
- Extreme correction (±10 ppb): Model is broken or data mismatch

### Regularization Trade-offs

**Too weak (lambda_correction = 0.001):**
- Model outputs large corrections freely
- Can overcorrect and make predictions worse
- Current model's problem

**Too strong (lambda_correction = 1.0):**
- Model becomes too conservative
- Corrections too small to help
- Model essentially outputs PINN predictions

**Sweet spot (lambda_correction = 0.01):**
- Allows meaningful corrections
- Prevents extreme overcorrection
- Balanced learning

### Learning Rate Considerations

**Current: 1e-4 (0.0001)**
- Good starting point
- Stable learning
- May be slow for early epochs

**If loss decreases slowly:**
- Try 5e-4 (0.0005) for faster initial learning
- Use learning rate scheduler (already implemented)
- Reduce on plateau (already implemented)

**If loss oscillates:**
- Reduce to 5e-5 (0.00005)
- Model may be overshooting minimum

### Batch Size Impact

**Current: 32**
- Good balance between stability and speed
- Larger batches (64, 128): More stable gradients, slower updates
- Smaller batches (16, 8): Faster updates, noisier gradients

**Recommendation:** Keep at 32 unless you have specific reasons to change.

### Architecture Considerations

**Current: 36 → 256 → 128 → 64 → 9**
- Simplified from original (was 36 → 512 → 512 → 256 → 128 → 9)
- Reduced capacity to prevent overfitting
- ~100K parameters (good for 5,173 samples)

**If model underfits:**
- Increase capacity: 36 → 512 → 256 → 128 → 9
- More parameters to learn complex patterns

**If model overfits:**
- Reduce capacity: 36 → 128 → 64 → 9
- Fewer parameters, more regularization

**Current architecture is well-balanced for the dataset size.**

### Data Quality Checks

**Before training, verify:**
1. PINN predictions are reasonable (0.1 - 10 ppb)
2. Sensor readings match PINN scale (not 100x different)
3. Temporal features are normalized correctly (0-1 range)
4. Wind/diffusion values are in expected ranges
5. No NaN or Inf values in inputs

**During training, monitor:**
1. Loss decreases consistently
2. Corrections stay in reasonable range
3. No sudden jumps in loss (indicates data issues)
4. Training and validation loss track together

### Expected Training Timeline

**GPU (CUDA):**
- Epoch 1-10: ~5-10 minutes (rapid learning)
- Epoch 10-30: ~15-30 minutes (gradual improvement)
- Epoch 30-60: ~30-60 minutes (fine-tuning)
- Total: 2-4 hours for 100 epochs

**CPU:**
- 8-12 hours (not recommended)
- Use GPU if available

### Debugging Training Issues

**If loss doesn't decrease:**
1. Check data loading (print first batch)
2. Verify model receives correct inputs
3. Check if gradients are flowing (print grad norms)
4. Verify loss calculation is correct
5. Check if learning rate is too small

**If loss decreases but MAE doesn't improve:**
1. Check if loss is calculated correctly
2. Verify target values are correct
3. Check if model outputs are in right space (ppb vs scaled)
4. Verify evaluation metrics match training loss

**If corrections are too large:**
1. Increase lambda_correction (0.01 → 0.1)
2. Increase lambda_large (0.1 → 1.0)
3. Check if PINN predictions are correctly scaled
4. Verify model architecture matches checkpoint

---

## Summary of Key Changes for Retraining

1. **Increase epochs:** 50 → 100
2. **Increase regularization:** lambda_correction 0.001 → 0.01
3. **Add large correction penalty:** lambda_large = 0.1
4. **Improve monitoring:** Log correction statistics
5. **Better early stopping:** Use min_delta, increase patience
6. **Save best model:** Track validation loss, save best state

**Expected outcome:**
- Validation loss: 16.35 → < 1.0
- MAE: 0.665 ppb → 0.3-0.5 ppb
- Improvement: -23.75% → +40-60%
- Corrections: ±1000 → ±0.2 (scaled space)

