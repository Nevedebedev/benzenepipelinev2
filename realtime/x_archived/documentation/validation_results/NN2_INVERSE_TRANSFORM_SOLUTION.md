# NN2 Inverse Transform Solution - Most Accurate & Scalable Method

## Problem Summary

NN2 performs excellently in scaled space (35% improvement) but degrades in original ppb space (-168% degradation) due to out-of-distribution values when using the original scaler's inverse transform.

## Solution: Direct Mapping from NN2 Scaled Outputs to PPB

Instead of using `scalers['sensors'].inverse_transform()`, we learn a **direct mapping** from NN2 scaled outputs to actual ppb values using the training data.

## Results

### Performance Comparison

| Method | MAE (ppb) | Improvement | Scalability |
|--------|-----------|-------------|-------------|
| **Original Scaler Inverse** | 1.2256 | Baseline | High |
| **Linear Regression Mapping** | 0.8011 | **34.6%** | **Very High** ⭐ |
| **Gradient Boosting Mapping** | 0.2529 | **79.4%** | Medium |

### Model Details

**Linear Regression:**
- Equation: `ppb = 2.3215 * scaled + -0.0294`
- R²: 0.8601
- Inference: O(1) - extremely fast
- Memory: Minimal (2 parameters)

**Gradient Boosting:**
- R²: 0.9717
- Inference: O(n_estimators) - slower but still fast
- Memory: Moderate (100 trees)

## Recommendation

### For Maximum Scalability: **Linear Regression**
- **34.6% improvement** over baseline
- **Extremely fast inference** (single multiplication + addition)
- **Minimal memory footprint**
- **Easy to implement** in any language/environment
- **Suitable for real-time pipelines** with high throughput requirements

### For Maximum Accuracy: **Gradient Boosting**
- **79.4% improvement** over baseline
- **Still fast enough** for most applications
- **Better handles non-linear relationships**
- **Suitable when accuracy is paramount**

## Implementation

The mapping model has been saved to:
```
realtime/nn2_timefix/nn2_output_to_ppb_mapping.pkl
```

### Usage

```python
import pickle
import numpy as np

# Load mapping model
with open('nn2_timefix/nn2_output_to_ppb_mapping.pkl', 'rb') as f:
    mapping_data = pickle.load(f)

mapping_model = mapping_data['model']
mapping_type = mapping_data['type']  # 'linear' or 'gbr'

# Convert NN2 scaled output to ppb
nn2_output_scaled = 0.5  # Example: NN2 output in scaled space
ppb = mapping_model.predict(np.array([[nn2_output_scaled]]))[0]
```

### Integration into Pipeline

Replace this:
```python
# OLD (incorrect)
nn2_ppb = scalers['sensors'].inverse_transform(corrected_scaled.reshape(-1, 1))
```

With this:
```python
# NEW (correct)
with open('nn2_timefix/nn2_output_to_ppb_mapping.pkl', 'rb') as f:
    mapping_data = pickle.load(f)
mapping_model = mapping_data['model']

nn2_ppb = mapping_model.predict(corrected_scaled.reshape(-1, 1))
```

## Why This Works

1. **Learns actual relationship**: The mapping is learned from the actual NN2 outputs and their corresponding ppb values, not from the input distribution.

2. **Handles out-of-range values**: The model learns how to map values outside the original scaler's training range.

3. **Accounts for model behavior**: The mapping reflects how NN2 actually behaves, not how we expect it to behave based on input statistics.

4. **Scalable**: Linear regression is extremely fast and can be implemented anywhere.

## Next Steps

1. **Update validation scripts** to use the mapping model instead of inverse transform
2. **Update real-time pipeline** (`concentration_predictor.py`) to use the mapping model
3. **Test on validation data** to confirm improvement
4. **Choose Linear vs GBR** based on accuracy vs speed trade-off

## Files Created

- `implement_nn2_output_mapping.py` - Script to generate mapping model
- `nn2_timefix/nn2_output_to_ppb_mapping.pkl` - Saved mapping model

