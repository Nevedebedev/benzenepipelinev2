# Deeper Analysis: NN2 Preprocessing Mismatch Investigation

**Date:** 2025-02-02  
**Status:** Multiple Critical Issues Found

---

## 🔍 Key Discovery: Multiple Model Versions in Use

### Model File Mismatch

| Script | Model Path | Status |
|--------|-----------|--------|
| `concentration_predictor.py` (deployment) | `nn2_timefix/nn2_master_model_spatial-3.pth` | ✅ Code appears fixed |
| `test_nn2_smaller_2019.py` (validation) | `nn2_scaled/nn2_master_model_ppb-2.pth` | ❌ **BROKEN MODEL** |
| Problem Documentation | `nn2_scaled/nn2_master_model_ppb-2.pth` | ❌ **BROKEN MODEL** |

**CRITICAL:** The validation script and problem documentation reference a DIFFERENT model than the deployment pipeline!

---

## 🚨 Issue #1: Model Loading Doesn't Check `output_ppb` Flag

### Problem in `concentration_predictor.py` (Lines 97-122)

```python
def _load_nn2(self):
    # Load checkpoint
    checkpoint = torch.load(self.nn2_path, map_location='cpu', weights_only=False)
    
    # Load model
    from nn2 import NN2_CorrectionNetwork
    nn2 = NN2_CorrectionNetwork(n_sensors=9)  # ❌ NO output_ppb parameter!
    nn2.load_state_dict(checkpoint['model_state_dict'])
    nn2.eval()
```

**Issues:**
1. ❌ Model is created with **default parameters** - doesn't check checkpoint for `output_ppb`
2. ❌ Doesn't extract `scaler_mean` and `scaler_scale` from checkpoint
3. ❌ If checkpoint has `output_ppb=True`, model won't have inverse transform layer initialized
4. ❌ Model might output in scaled space but code expects ppb space (or vice versa)

### What Should Happen

```python
def _load_nn2(self):
    checkpoint = torch.load(self.nn2_path, map_location='cpu', weights_only=False)
    
    # Extract metadata from checkpoint
    output_ppb = checkpoint.get('output_ppb', False)  # Default to False for safety
    scaler_mean = checkpoint.get('scaler_mean', None)
    scaler_scale = checkpoint.get('scaler_scale', None)
    
    # Create model with correct parameters
    from nn2 import NN2_CorrectionNetwork
    nn2 = NN2_CorrectionNetwork(
        n_sensors=9,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        output_ppb=output_ppb  # ✅ Use checkpoint value
    )
    nn2.load_state_dict(checkpoint['model_state_dict'])
    nn2.eval()
    
    # Verify model configuration
    if output_ppb and nn2.inverse_transform is None:
        raise ValueError("Checkpoint says output_ppb=True but model has no inverse transform!")
    
    print(f"  ✓ Model loaded: output_ppb={output_ppb}")
```

---

## 🚨 Issue #2: Validation Scripts Still Use Wrong Model Call

### Files with Wrong Pattern

| File | Line | Current (WRONG) | Should Be |
|------|------|-----------------|-----------|
| `test_pipeline_2019.py` | 232 | `nn2(s_tensor, p_tensor, ...)` | `nn2(p_tensor, c_tensor, ...)` |
| `validate_nn2_january_2019.py` | 228 | `nn2(s_tensor, p_tensor, ...)` | `nn2(p_tensor, c_tensor, ...)` |
| `x_archived/investigation_scripts/diagnose_nn2_issues.py` | 311 | `nn2(s_tensor, p_tensor, ...)` | `nn2(p_tensor, c_tensor, ...)` |
| `x_archived/investigation_scripts/investigate_nn2_2019_degradation.py` | 310 | `nn2(s_tensor, p_tensor, ...)` | `nn2(p_tensor, c_tensor, ...)` |
| `x_archived/test_scripts/test_pipeline_2019_ppb.py` | 230 | `nn2(s_tensor, p_tensor, ...)` | `nn2(p_tensor, c_tensor, ...)` |
| `x_archived/investigation_scripts/analyze_negative_predictions.py` | 164 | `nn2(s_tensor, p_tensor, ...)` | `nn2(p_tensor, c_tensor, ...)` |
| `x_archived/investigation_scripts/investigate_2019_vs_2021_distribution.py` | 206 | `nn2(s_tensor, p_tensor, ...)` | `nn2(p_tensor, c_tensor, ...)` |
| `x_archived/investigation_scripts/analyze_scaler_refit_solution.py` | 166 | `nn2(s_tensor, p_tensor, ...)` | `nn2(p_tensor, c_tensor, ...)` |

**All these scripts are using the OLD model signature with `current_sensors`!**

---

## 🚨 Issue #3: Model Checkpoint Metadata Mismatch

### Checkpoint Structure Analysis

**Training Script Saves:**
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_loss': val_loss,
    'scaler_mean': scaler_mean,      # ✅ Saved
    'scaler_scale': scaler_scale,   # ✅ Saved
    'output_ppb': True              # ✅ Saved
}
```

**Deployment Script Loads:**
```python
checkpoint = torch.load(self.nn2_path, ...)
nn2 = NN2_CorrectionNetwork(n_sensors=9)  # ❌ Ignores checkpoint metadata!
# ❌ Doesn't check for output_ppb
# ❌ Doesn't extract scaler_mean/scale
```

**Result:** Model might be created without inverse transform layer even if checkpoint says `output_ppb=True`!

---

## 🚨 Issue #4: Output Handling Inconsistency

### Different Scripts Handle Output Differently

| Script | Model Output Assumption | Actual Handling |
|--------|------------------------|-----------------|
| `concentration_predictor.py` | ppb space | ✅ Uses output directly (line 408) |
| `test_pipeline_2019.py` | scaled space | ❌ Inverse transforms (line 244) |
| `validate_nn2_january_2019.py` | scaled space | ❌ Inverse transforms (line 241) |
| `test_nn2_smaller_2019.py` | ppb space | ✅ Uses output directly (line 351) |

**Problem:** Scripts assume different output formats, causing inconsistent results!

---

## 🔍 Issue #5: Model Architecture Mismatch Detection

### Scripts That Check Architecture (Good Pattern)

**`test_nn2_smaller_2019.py` (Lines 316-348):**
```python
# Check if model is old architecture (45 features) by checking forward signature
import inspect
sig = inspect.signature(nn2.forward)
num_params = len(sig.parameters) - 1  # Exclude 'self'

if num_params == 6:  # Old architecture
    # Use old call pattern
    nn2_ppb, corrections = nn2(pinn_tensor, coords_tensor, current_sensors_scaled, ...)
else:  # New architecture (5 params)
    # Use new call pattern
    nn2_ppb, corrections = nn2(pinn_tensor, coords_tensor, wind_tensor, ...)
```

**This is the CORRECT approach!** But most scripts don't do this check.

---

## 📊 Summary of All Issues Found

| Issue | Severity | Files Affected | Status |
|-------|----------|---------------|--------|
| Model call order wrong | 🔴 CRITICAL | 8+ scripts | ❌ Not fixed |
| Model loading ignores `output_ppb` | 🔴 CRITICAL | `concentration_predictor.py` | ❌ Not fixed |
| Output handling mismatch | 🟡 HIGH | Multiple scripts | ❌ Inconsistent |
| Model file mismatch | 🟡 HIGH | Different models used | ⚠️ Needs verification |
| No architecture detection | 🟡 MEDIUM | Most scripts | ❌ Missing |

---

## 🎯 Root Cause Analysis

### Why Predictions Are 1000x Too Large

**Scenario 1: Wrong Model Call (Most Likely)**
- Script calls: `nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)`
- Model expects: `nn2(pinn_predictions, sensor_coords, wind, diffusion, temporal)`
- Result: All arguments misaligned → model works with wrong data → massive errors

**Scenario 2: Model Loading Without `output_ppb`**
- Checkpoint has `output_ppb=True` but model loaded without it
- Model outputs in scaled space but code expects ppb space
- Or vice versa: model outputs ppb but code tries to inverse transform
- Result: Wrong scale → predictions off by orders of magnitude

**Scenario 3: Using Wrong Model File**
- Validation uses `nn2_scaled` (broken model)
- Deployment uses `nn2_timefix` (might be correct)
- Results can't be compared directly
- Result: Confusion about which model is actually broken

---

## ✅ Recommended Fixes

### Fix 1: Standardize Model Loading

Create a utility function:

```python
def load_nn2_model(model_path, scalers_path=None):
    """
    Load NN2 model with proper metadata handling
    
    Returns:
        nn2: Loaded model
        scalers: Dictionary of scalers
        sensor_coords: Sensor coordinates array
        output_ppb: Boolean indicating output format
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract metadata
    output_ppb = checkpoint.get('output_ppb', False)
    scaler_mean = checkpoint.get('scaler_mean', None)
    scaler_scale = checkpoint.get('scaler_scale', None)
    
    # Load scalers
    if scalers_path:
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
    elif 'scalers' in checkpoint:
        scalers = checkpoint['scalers']
    else:
        raise ValueError("No scalers found in checkpoint or scalers_path")
    
    # Load sensor coordinates
    if 'sensor_coords' in checkpoint:
        sensor_coords = checkpoint['sensor_coords']
    else:
        raise ValueError("No sensor_coords found in checkpoint")
    
    # Create model with correct parameters
    from nn2 import NN2_CorrectionNetwork
    nn2 = NN2_CorrectionNetwork(
        n_sensors=9,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        output_ppb=output_ppb
    )
    nn2.load_state_dict(checkpoint['model_state_dict'])
    nn2.eval()
    
    # Verify configuration
    if output_ppb and nn2.inverse_transform is None:
        raise ValueError(f"Checkpoint says output_ppb=True but model has no inverse transform!")
    
    return nn2, scalers, sensor_coords, output_ppb
```

### Fix 2: Standardize Model Call

Create a utility function that auto-detects architecture:

```python
def call_nn2_model(nn2, pinn_scaled, coords_normalized, wind_normalized, 
                   diffusion_normalized, temporal_raw, current_sensors_scaled=None):
    """
    Call NN2 model with automatic architecture detection
    
    Args:
        nn2: NN2 model instance
        pinn_scaled: PINN predictions in scaled space [batch, 9]
        coords_normalized: Sensor coordinates normalized [batch, 9, 2]
        wind_normalized: Wind components normalized [batch, 2]
        diffusion_normalized: Diffusion coefficient normalized [batch, 1]
        temporal_raw: Temporal features (not normalized) [batch, 6]
        current_sensors_scaled: Current sensor readings (optional, for old architecture)
    
    Returns:
        corrected_predictions: [batch, 9] in ppb or scaled space
        corrections: [batch, 9] in scaled space
    """
    import inspect
    sig = inspect.signature(nn2.forward)
    num_params = len(sig.parameters) - 1  # Exclude 'self'
    
    if num_params == 6:  # Old architecture (with current_sensors)
        if current_sensors_scaled is None:
            current_sensors_scaled = torch.zeros_like(pinn_scaled)
        corrected, corrections = nn2(
            pinn_scaled,
            coords_normalized,
            current_sensors_scaled,
            wind_normalized,
            diffusion_normalized,
            temporal_raw
        )
    elif num_params == 5:  # New architecture (no current_sensors)
        corrected, corrections = nn2(
            pinn_scaled,
            coords_normalized,
            wind_normalized,
            diffusion_normalized,
            temporal_raw
        )
    else:
        raise ValueError(f"Unexpected model architecture with {num_params} parameters")
    
    return corrected, corrections
```

### Fix 3: Verify Model File Consistency

Check which model is actually being used in production:

```python
# Add to concentration_predictor.py
def _verify_model_config(self):
    """Verify model configuration matches expectations"""
    checkpoint = torch.load(self.nn2_path, map_location='cpu', weights_only=False)
    
    output_ppb = checkpoint.get('output_ppb', False)
    has_inverse_transform = hasattr(self.nn2, 'inverse_transform') and self.nn2.inverse_transform is not None
    
    print(f"  Model checkpoint: {self.nn2_path}")
    print(f"  Checkpoint output_ppb: {output_ppb}")
    print(f"  Model has inverse_transform: {has_inverse_transform}")
    
    if output_ppb != has_inverse_transform:
        raise ValueError(
            f"Model configuration mismatch! "
            f"Checkpoint says output_ppb={output_ppb} but model.inverse_transform is "
            f"{'None' if not has_inverse_transform else 'present'}"
        )
```

---

## 🎯 Next Steps

1. **Verify which model file is actually broken:**
   - Check `nn2_scaled/nn2_master_model_ppb-2.pth` checkpoint metadata
   - Check `nn2_timefix/nn2_master_model_spatial-3.pth` checkpoint metadata
   - Compare their `output_ppb` flags

2. **Fix model loading in `concentration_predictor.py`:**
   - Extract `output_ppb`, `scaler_mean`, `scaler_scale` from checkpoint
   - Initialize model with correct parameters

3. **Fix all validation scripts:**
   - Update model call to use correct 5-argument signature
   - Remove `current_sensors` creation and usage
   - Use architecture detection or standardize on new architecture

4. **Standardize output handling:**
   - Check model's `output_ppb` flag
   - Use output directly if `output_ppb=True`
   - Inverse transform only if `output_ppb=False`

5. **Create utility functions:**
   - `load_nn2_model()` - standardized model loading
   - `call_nn2_model()` - architecture-aware model calling
   - Use these in all scripts

---

## 📝 Files That Need Updates

### Critical (Deployment)
- ✅ `realtime/concentration_predictor.py` - Fix model loading (lines 97-122)

### High Priority (Validation)
- ❌ `realtime/test_pipeline_2019.py` - Fix model call (line 232)
- ❌ `realtime/validate_nn2_january_2019.py` - Fix model call (line 228)
- ❌ `realtime/test_nn2_smaller_2019.py` - Verify model path matches deployment

### Medium Priority (Investigation Scripts)
- Multiple scripts in `x_archived/investigation_scripts/`
- Multiple scripts in `x_archived/test_scripts/`

---

**Status:** Analysis complete. Multiple critical issues identified beyond initial preprocessing mismatch.

