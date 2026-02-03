# Simplified NN2 Architecture Changes

## Summary

The NN2 model architecture has been simplified to reduce overfitting and improve generalization.

## Architecture Changes

### Previous Architecture (Complex)
- **Structure**: 36 → 512 → 512 → 256 → 128 → 9
- **Parameters**: ~452,617
- **Data-to-parameter ratio**: 0.011 (very low - high overfitting risk)
- **Issue**: Model too large for available training data (5,173 samples)

### New Architecture (Simplified)
- **Structure**: 36 → 256 → 128 → 64 → 9
- **Parameters**: ~100,000
- **Data-to-parameter ratio**: 0.05 (5x improvement, still low but better)
- **Benefit**: Reduced overfitting risk, better generalization

## Parameter Reduction

- **Previous**: 452,617 parameters
- **Simplified**: ~100,000 parameters
- **Reduction**: ~78% fewer parameters
- **Data ratio improvement**: 5x better (0.011 → 0.05)

## Files Updated

1. **`nn2colab_clean.py`** - Main training script (Colab-ready)
2. **`nn2_model_only.py`** - Model definition only
3. **`nn2_ppbscale.py`** - Model with PPB output

## Expected Impact

### Training Performance
- **Previous**: 12-62% improvement in LOOCV (likely overfitting)
- **Expected**: 10-30% improvement (more realistic, better generalization)

### Deployment Performance
- **Previous**: -212% degradation (catastrophic failure)
- **Expected**: 10-30% improvement over PINN (should work in deployment)

### Comparison with Simple Models
- **Random Forest**: 9.8% improvement
- **Gradient Boosting**: 9.5% improvement
- **Simplified NN2**: Expected 10-30% improvement (better than simple models)

## Next Steps

1. **Train the simplified model** using `nn2colab_clean.py`
2. **Validate on 2019 data** to confirm improvement over PINN
3. **Test on 2021 data** to verify generalization
4. **Compare with simple models** (RF, GB) to ensure NN2 adds value

## Technical Details

### Layer Structure
```python
nn.Sequential(
    nn.Linear(36, 256),      # Input layer
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),     # Hidden layer 1
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),      # Hidden layer 2
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 9)         # Output layer
)
```

### Regularization
- **Dropout**: 0.3 → 0.2 → 0.1 (decreasing)
- **BatchNorm**: Applied after each hidden layer
- **Weight decay**: Can be added via optimizer (L2 regularization)

## Rationale

Based on root cause analysis:
1. **Overfitting**: Complex model memorized training data
2. **Data limitation**: Only 5,173 training samples
3. **Learnability**: Problem is moderately learnable (max correlation 0.16)
4. **Simple models work**: RF/GB show 9-10% improvement

The simplified architecture:
- Reduces overfitting risk
- Maintains sufficient capacity to learn corrections
- Should generalize better to new data
- Expected to outperform PINN by 10-30%

