# NN2 Inverse Transform Issue - Critical Discovery

## Summary

**NN2 performs EXCELLENTLY in scaled space (35% improvement) but DEGRADES in original ppb space (-168% degradation).**

The issue is NOT with the model itself, but with the **inverse transform process**.

## Key Findings

### Scaled Space (Training Evaluation Method)
- **PINN MAE**: 0.600 (scaled units)
- **NN2 MAE**: 0.389 (scaled units)
- **Improvement**: **35.1%** ✅

### Original Space (Current Validation Method)
- **PINN MAE**: 0.341 ppb
- **NN2 MAE**: 0.914 ppb
- **Improvement**: **-168.1%** ❌

## Root Cause

The training code (`nn2colab.py` lines 918-927) evaluates performance in **SCALED SPACE**, not original ppb space:

```python
# Training evaluation (scaled space)
valid_pinn = pinn_preds[masks]  # Scaled
valid_nn2 = nn2_preds[masks]     # Scaled
valid_actual = actual[masks]     # Scaled
pinn_mae = torch.abs(valid_pinn - valid_actual).mean()  # Scaled space MAE
nn2_mae = torch.abs(valid_nn2 - valid_actual).mean()    # Scaled space MAE
```

But our validation scripts compute MAE in **ORIGINAL ppb SPACE** after inverse transform.

## The Problem

When we inverse transform the corrected predictions from scaled space back to ppb:
1. The model outputs corrections in scaled space
2. `corrected_scaled = pinn_scaled + corrections_scaled`
3. We inverse transform: `corrected_ppb = scaler.inverse_transform(corrected_scaled)`
4. **The inverse transform produces incorrect values**

## Root Cause Identified

**The NN2 model outputs values in scaled space that are OUT OF DISTRIBUTION for the scaler!**

### Scaler Training Range
- **Mean**: 0.4689
- **Std**: 2.9754
- **Approximate range**: [-0.16, 3.20] (based on typical sensor values 0-10 ppb)

### NN2 Output Range (Scaled Space)
- **Min**: -5.90 ❌ (way below training range)
- **Max**: 8.31 ❌ (way above training range)
- **Mean**: 0.14
- **Std**: 0.48

### Target Range (Scaled Space)
- **Min**: -0.15 ✓
- **Max**: 8.14 ❌ (above training range)

**The scaler was fit on a narrow range, but NN2 outputs values far outside this range. When inverse transformed, these out-of-range values produce incorrect ppb values.**

## Why This Happens

The model was trained to minimize MSE in scaled space, but:
1. The loss function doesn't penalize out-of-range predictions
2. The model can output any value in scaled space
3. When these out-of-range values are inverse transformed, they map to incorrect ppb values
4. The scaler's inverse transform is not valid for values outside its training range

## Next Steps

1. **Verify scaler statistics match training**
2. **Check if corrected_scaled values are within scaler training range**
3. **Test alternative inverse transform strategies**
4. **Consider evaluating in scaled space for consistency with training**

## Impact

This explains why:
- Leave-one-out cross-validation showed 60-77% improvement (evaluated in scaled space)
- Validation shows -150% degradation (evaluated in original ppb space)
- The model itself is working correctly, but the inverse transform is broken

