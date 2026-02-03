# NN2 Input Preprocessing Comparison: Training vs Deployment

**Date:** 2025-02-02  
**Issue:** Train/test mismatch - predictions 1000x too large in deployment

---

## Executive Summary

**CRITICAL ISSUES FOUND:**
1. ❌ **Model call order is WRONG** - Deployment passes arguments in incorrect order
2. ❌ **Passing `current_sensors`** - This input was removed from model architecture (data leakage fix), but deployment still creates and passes it
3. ✅ Most input normalization matches, but the wrong model call corrupts everything

---

## Detailed Comparison Table

| Input | Training Preprocessing | Deployment Preprocessing | Match? | Notes |
|-------|------------------------|--------------------------|--------|-------|
| **PINN Predictions** | Scaled using `scalers['pinn']`<br/>Zero masking: only non-zero values scaled<br/>Shape: `[batch, 9]` in scaled space | Scaled using `scalers['pinn']`<br/>Zero masking: only non-zero values scaled<br/>Shape: `[1, 9]` in scaled space | ✅ **YES** | Both handle zeros correctly |
| **Sensor Coordinates** | Normalized using `scalers['coords']`<br/>Shape: `[batch, n_sensors, 2]` → flattened to `[batch, 18]` | Normalized using `scalers['coords']`<br/>Shape: `[1, 9, 2]` → flattened to `[1, 18]` | ✅ **YES** | Both normalize before passing to model |
| **Wind (u, v)** | Normalized using `scalers['wind']`<br/>Shape: `[batch, 2]` | Normalized using `scalers['wind']`<br/>Shape: `[1, 2]` | ✅ **YES** | Both normalize correctly |
| **Diffusion (D)** | Normalized using `scalers['diffusion']`<br/>Shape: `[batch, 1]` | Normalized using `scalers['diffusion']`<br/>Shape: `[1, 1]` | ✅ **YES** | Both normalize correctly |
| **Temporal Features** | **NOT normalized** (sin/cos already in [-1,1], is_weekend in [0,1], month/12 in [0,1])<br/>Shape: `[batch, 6]` | **NOT normalized** (passed as-is)<br/>Shape: `[1, 6]` | ✅ **YES** | Both pass temporal features without normalization |
| **Current Sensors** | **REMOVED** from model architecture (data leakage fix)<br/>Model expects 5 inputs, not 6 | **STILL CREATED** and passed to model<br/>`s_tensor` is created and passed as first argument | ❌ **NO** | **CRITICAL: This input doesn't exist in model!** |

---

## Model Forward Signature Comparison

### Training Script (`nn2colab_clean_master_only.py`)

**Model Architecture:** 36 input features (no `current_sensors`)

```python
def forward(self, pinn_predictions, sensor_coords, wind, diffusion, temporal):
    """
    Args:
        pinn_predictions: [batch, n_sensors] - in scaled space
        sensor_coords: [batch, n_sensors, 2] - normalized
        wind: [batch, 2] - normalized
        diffusion: [batch, 1] - normalized
        temporal: [batch, 6] - NOT normalized (sin/cos/0-1 range)
    """
```

**Expected Call Order:**
1. `pinn_predictions` (scaled)
2. `sensor_coords` (normalized)
3. `wind` (normalized)
4. `diffusion` (normalized)
5. `temporal` (raw, not normalized)

### Deployment Code (`concentration_predictor.py`)

**Current (WRONG) Call:**
```python
# Line 413 - WRONG ORDER AND WRONG INPUTS!
corrected_scaled, _ = self.nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
```

**What's Wrong:**
1. ❌ Passing `s_tensor` (current_sensors) as **first argument** - this input doesn't exist in the model!
2. ❌ Wrong argument order - should be `(p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)`
3. ❌ Model expects 5 inputs but receiving 6

**Correct Call Should Be:**
```python
corrected_scaled, _ = self.nn2(p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
```

---

## Code Location References

### Training Script
- **File:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py`
- **Model forward:** Lines 123-160
- **Dataset preprocessing:** Lines 265-344
- **Temporal features:** Lines 314-327 (NOT normalized)

### Deployment Script
- **File:** `realtime/concentration_predictor.py`
- **NN2 correction method:** `_apply_nn2_correction()` (Lines 282-469)
- **Input scaling:** Lines 375-409
- **Model call (WRONG):** Line 413
- **Temporal features:** Lines 356-369 (NOT normalized - correct)

---

## Normalization Details

### 1. PINN Predictions
**Training:**
```python
# Lines 265-272
pinn_predictions_scaled = np.zeros_like(self.pinn_predictions)
for i in range(len(self.pinn_predictions)):
    mask = self.pinn_predictions[i] != 0
    if mask.any():
        self.pinn_predictions_scaled[i, mask] = self.scalers['pinn'].transform(
            self.pinn_predictions[i, mask].reshape(-1, 1)
        ).flatten()
```

**Deployment:**
```python
# Lines 379-385
p_s = np.zeros_like(sensor_pinn)
if pinn_nonzero_mask.any():
    p_s[pinn_nonzero_mask] = self.scalers['pinn'].transform(
        sensor_pinn[pinn_nonzero_mask].reshape(-1, 1)
    ).flatten()
p_s = p_s.reshape(1, -1)
```
✅ **MATCHES**

### 2. Sensor Coordinates
**Training:**
```python
# Lines 333-340
if fit_scalers:
    self.scalers['coords'].fit(self.sensor_coords_array)
self.sensor_coords_normalized = self.scalers['coords'].transform(self.sensor_coords_array)
```

**Deployment:**
```python
# Line 401
c_s = self.scalers['coords'].transform(self.sensor_coords_spatial)
```
✅ **MATCHES**

### 3. Wind
**Training:**
```python
# Lines 333-343
if fit_scalers:
    self.scalers['wind'].fit(self.wind)
self.wind_normalized = self.scalers['wind'].transform(self.wind)
```

**Deployment:**
```python
# Lines 395-396
w_in = np.array([[avg_u, avg_v]])
w_s = self.scalers['wind'].transform(w_in)
```
✅ **MATCHES**

### 4. Diffusion
**Training:**
```python
# Lines 333-344
if fit_scalers:
    self.scalers['diffusion'].fit(self.diffusion)
self.diffusion_normalized = self.scalers['diffusion'].transform(self.diffusion)
```

**Deployment:**
```python
# Lines 398-399
d_in = np.array([[avg_D]])
d_s = self.scalers['diffusion'].transform(d_in)
```
✅ **MATCHES**

### 5. Temporal Features
**Training:**
```python
# Lines 314-327
self.temporal.append([
    np.sin(2 * np.pi * hour / 24),
    np.cos(2 * np.pi * hour / 24),
    np.sin(2 * np.pi * day_of_week / 7),
    np.cos(2 * np.pi * day_of_week / 7),
    is_weekend,
    month / 12.0
])
# NO normalization applied - passed directly to model
```

**Deployment:**
```python
# Lines 356-369
temporal_vals = np.array([[
    np.sin(2 * np.pi * hour / 24),
    np.cos(2 * np.pi * hour / 24),
    np.sin(2 * np.pi * day_of_week / 7),
    np.cos(2 * np.pi * day_of_week / 7),
    is_weekend,
    month / 12.0
]])
# NO normalization applied - passed directly to model
```
✅ **MATCHES**

### 6. Current Sensors (REMOVED FROM MODEL)
**Training:**
```python
# Lines 353-354 (Dataset.__getitem__)
# REMOVED: 'current_sensors' - this was causing data leakage!
# The model should learn corrections based on PINN + conditions, not actual sensor values
```

**Deployment:**
```python
# Lines 387-393 - STILL CREATING THIS!
s_s = np.zeros_like(current_sensors)
if sensors_nonzero_mask.any():
    s_s[sensors_nonzero_mask] = self.scalers['sensors'].transform(
        current_sensors[sensors_nonzero_mask].reshape(-1, 1)
    ).flatten()
s_s = s_s.reshape(1, -1)
# Line 405: s_tensor = torch.tensor(s_s, dtype=torch.float32)
# Line 413: Passed as FIRST argument to model (WRONG!)
```
❌ **DOES NOT MATCH** - This input was removed from the model!

---

## Root Cause Analysis

### Why Predictions Are 1000x Too Large

1. **Wrong Model Call:**
   - Deployment calls: `nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)`
   - Model expects: `nn2(pinn_predictions, sensor_coords, wind, diffusion, temporal)`
   - **Result:** All arguments are shifted! `s_tensor` (current_sensors) is being interpreted as `pinn_predictions`, `p_tensor` (actual PINN) is being interpreted as `sensor_coords`, etc.

2. **Argument Misalignment:**
   - What model receives vs. what it expects:
     - Model gets `s_tensor` → thinks it's `pinn_predictions` ❌
     - Model gets `p_tensor` → thinks it's `sensor_coords` ❌
     - Model gets `c_tensor` → thinks it's `wind` ❌
     - Model gets `w_tensor` → thinks it's `diffusion` ❌
     - Model gets `d_tensor` → thinks it's `temporal` ❌
     - Model gets `t_tensor` → **EXTRA ARGUMENT** (model only expects 5) ❌

3. **Scale Mismatch:**
   - `s_tensor` (current_sensors scaled) has different distribution than `pinn_predictions` scaled
   - When model interprets `s_tensor` as PINN predictions, it's working with wrong input scale
   - This causes the model to output massive corrections in the wrong direction

---

## What the Pipeline SHOULD Do

### Correct Implementation

```python
def _apply_nn2_correction(self, x, y, pinn_pred, facility_params, forecast_time):
    # ... compute sensor_pinn at sensor locations ...
    
    # Scale PINN predictions (only non-zero values)
    p_s = np.zeros_like(sensor_pinn)
    if pinn_nonzero_mask.any():
        p_s[pinn_nonzero_mask] = self.scalers['pinn'].transform(
            sensor_pinn[pinn_nonzero_mask].reshape(-1, 1)
        ).flatten()
    p_tensor = torch.tensor(p_s, dtype=torch.float32)  # Shape: [1, 9]
    
    # Normalize sensor coordinates
    c_s = self.scalers['coords'].transform(self.sensor_coords_spatial)
    c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 9, 2]
    
    # Normalize wind
    w_in = np.array([[avg_u, avg_v]])
    w_s = self.scalers['wind'].transform(w_in)
    w_tensor = torch.tensor(w_s, dtype=torch.float32)  # Shape: [1, 2]
    
    # Normalize diffusion
    d_in = np.array([[avg_D]])
    d_s = self.scalers['diffusion'].transform(d_in)
    d_tensor = torch.tensor(d_s, dtype=torch.float32)  # Shape: [1, 1]
    
    # Temporal features (NOT normalized)
    temporal_vals = np.array([[
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
        is_weekend,
        month / 12.0
    ]])
    t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)  # Shape: [1, 6]
    
    # CORRECT MODEL CALL - 5 arguments in correct order
    with torch.no_grad():
        corrected_ppb, corrections = self.nn2(
            p_tensor,    # pinn_predictions (scaled)
            c_tensor,   # sensor_coords (normalized)
            w_tensor,   # wind (normalized)
            d_tensor,   # diffusion (normalized)
            t_tensor    # temporal (raw, not normalized)
        )
    
    # Model outputs directly in ppb space (if output_ppb=True)
    # No need for inverse transform
    sensor_corrected = corrected_ppb.cpu().numpy().flatten()
```

### Key Changes Needed

1. **Remove `s_tensor` creation** (Lines 387-393, 405)
2. **Fix model call order** (Line 413):
   - Change from: `self.nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)`
   - Change to: `self.nn2(p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)`
3. **Remove inverse transform** (Lines 415-422) - Model outputs in ppb space directly
4. **Update output handling** - Model returns `(corrected_ppb, corrections)` where `corrected_ppb` is already in ppb

---

## Summary

| Issue | Status | Impact |
|-------|--------|--------|
| PINN scaling | ✅ Correct | No issue |
| Sensor coords normalization | ✅ Correct | No issue |
| Wind normalization | ✅ Correct | No issue |
| Diffusion normalization | ✅ Correct | No issue |
| Temporal features (no normalization) | ✅ Correct | No issue |
| **Model call order** | ❌ **WRONG** | **CRITICAL - Causes 1000x error** |
| **Passing current_sensors** | ❌ **WRONG** | **CRITICAL - Input doesn't exist** |
| **Output handling** | ❌ **WRONG** | **Model outputs ppb, but code tries to inverse transform** |

---

**Next Steps:**
1. Fix model call in `concentration_predictor.py` line 413
2. Remove `s_tensor` creation and usage
3. Update output handling to use model's ppb output directly
4. Test with corrected preprocessing

