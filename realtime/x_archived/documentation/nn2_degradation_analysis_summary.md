# NN2 Performance Degradation Investigation - 2019 Data

## Summary

NN2 shows **-114.1% degradation** (worse than PINN) when tested on 2019 data, despite being trained on the same data.

## Key Findings

### 1. Performance Metrics
- **PINN MAE**: 0.5736 ppb
- **NN2 MAE**: 1.2282 ppb
- **Degradation**: -114.1%
- **86.3% of cases** have NN2 error > PINN error

### 2. Distribution Comparison

**Training vs Validation:**
- **PINN (ppb)**: Training mean=0.3792, Validation mean=0.3139 (similar)
- **Actual (ppb)**: Training mean=0.4689, Validation mean=0.5527 (slightly higher)
- **Scaled PINN**: Training std=1.0, Validation std=1.15 (slightly wider)
- **Scaled Actual**: Training std=1.0, Validation std=1.50 (much wider - distribution shift)

### 3. Correction Analysis

**Corrections in ppb space:**
- Mean: 0.8621 ppb
- Std: 5.2365 ppb
- Range: [-26.67, 227.63] ppb
- **29.6% are negative corrections**

**Overcorrection:**
- **22.1% of cases** have corrections larger than PINN (making result negative)
- Average PINN in overcorrections: 0.28 ppb
- Average correction in overcorrections: -1.29 ppb
- Average actual in overcorrections: 0.24 ppb

### 4. Correlation Analysis (CRITICAL)

- **PINN vs Correction**: -0.0733 (weak negative correlation)
- **Actual vs Correction**: **0.9622** (very strong positive correlation!)

**Interpretation**: The model's corrections are highly correlated with actual values but weakly correlated with PINN. This suggests the model learned to **predict actual values directly** rather than learning **corrections to PINN**.

### 5. When NN2 Performs Worse

**Characteristics of cases where NN2 error > PINN error:**
- Average PINN: 0.24 ppb (small values)
- Average actual: 0.26 ppb (also small)
- Average correction: 0.67 ppb (correction is **2.8x larger than PINN**)
- Average PINN error: 0.22 ppb
- Average NN2 error: 1.29 ppb

**Pattern**: When PINN predictions are small (< 0.3 ppb), NN2 applies corrections that are too large (0.67 ppb), causing overcorrection.

### 6. Inverse Transform Issues

**Scaler Training Range**: [-8.46, 9.40] (scaled space)
**NN2 Output Range**: [-5.92, 76.55] (scaled space)

- Most values are within range (99.7%)
- However, extreme values (up to 76.55) are way outside training distribution
- **29.3% of outputs are negative** in scaled space
- **22.1% are negative** after inverse transform (ppb)

## Root Cause Hypothesis

The NN2 model appears to have learned to **predict actual sensor values directly** rather than learning **corrections to PINN predictions**. Evidence:

1. **High correlation (0.96) between corrections and actual values**
2. **Low correlation (-0.07) between corrections and PINN**
3. **Corrections are too large** relative to PINN (especially for small PINN values)
4. **Model outputs negative values** when it should be correcting small positive PINN predictions

## Potential Solutions

1. **Retrain with better loss function**: Penalize large corrections relative to PINN
2. **Add constraint**: Limit correction magnitude (e.g., correction < 2x PINN)
3. **Different architecture**: Force model to learn corrections, not direct predictions
4. **Regularization**: Add penalty for corrections that are too large
5. **Training data balance**: Ensure model sees more cases where PINN is small and needs small corrections

## Next Steps

1. Check training loss function - is it encouraging direct prediction vs correction?
2. Analyze training data - what's the distribution of (actual - PINN) corrections?
3. Check if model architecture allows it to bypass PINN input
4. Consider retraining with modified loss that explicitly penalizes overcorrection

