# NN2 Architecture Analysis - Critical Issues Found

## Diagnostic Results

### Performance Metrics
- **PINN MAE**: 0.20 ppb
- **NN2 MAE**: 0.92 ppb
- **Degradation**: -358.6% (catastrophic failure)

### Key Findings

1. **Wrong Direction Corrections**: 67.5% of corrections are in the wrong direction
   - Only 32.5% of corrections help
   - Model is systematically making things worse

2. **Correction Correlates with PINN (0.66), NOT with Needed Correction (-0.19)**
   - Model is learning: `correction = f(PINN)` 
   - Should learn: `correction = f(PINN_error, conditions)`
   - This is the **core architectural flaw**

3. **Massive Over-Correction**: Correction ratio = 8.7
   - Model makes corrections 8.7x larger than needed
   - Combined with wrong direction = disaster

4. **Sensor-Specific Failure**: All worst cases are sensor 482011039
   - Model outputs negative values when actual is positive
   - Specific sensor causing extreme failures

## Root Cause Analysis

### The Problem

The model architecture is learning a **SYSTEMATIC BIAS**:
```
correction = -k * PINN + noise
```

Instead of learning **TRUE CORRECTIONS**:
```
correction = f(PINN_error, conditions)
```

### Why This Happens

1. **Model sees PINN as input** → learns to output something proportional to PINN
2. **Model doesn't see actual values** → can't directly learn error
3. **Model should learn patterns** like:
   - "When PINN is low and wind is from X, actual is usually higher"
   - "When PINN is high and conditions are Y, actual is usually lower"
4. **But current architecture isn't learning these patterns effectively**

### Evidence

- **Correction ↔ PINN: 0.66** (high correlation - model is learning PINN-dependent corrections)
- **Correction ↔ Needed: -0.19** (negative correlation - doing opposite of needed)
- **Correction ↔ Actual: 0.02** (no correlation - not learning actual patterns)

## Architectural Flaws

### Current Architecture
```python
features = [PINN_predictions, coords, wind, diffusion, temporal]
corrections = network(features)
corrected = PINN + corrections
```

**Problem**: Network learns to output corrections based on PINN magnitude, not PINN error.

### What Should Happen

The model should learn:
- **Relative corrections** (percentage of PINN)
- **Condition-dependent corrections** (based on wind, location, etc.)
- **Small corrections** (not massive over-corrections)

## Solution Options

### Option 1: Relative Correction Architecture ⭐⭐⭐
```python
# Learn percentage corrections instead of absolute
corrections = network(features)  # Output: [-1, 1] range
corrected = PINN * (1 + corrections)  # Relative correction
```

**Pros**: 
- Naturally bounded
- Works for all PINN magnitudes
- Forces model to learn relative patterns

**Cons**: 
- Still needs to learn correct direction

### Option 2: Explicit Error Signal During Training ⭐⭐
```python
# Add PINN error as explicit feature (only during training)
error_signal = actual - PINN  # Only available during training
features = [PINN, error_signal, coords, wind, diffusion, temporal]
```

**Pros**: 
- Directly learns error patterns
- Can learn: "When error is X and conditions are Y, correction should be Z"

**Cons**: 
- Error signal not available at deployment
- Model might overfit to error signal

### Option 3: Residual Architecture with Bounded Corrections ⭐⭐⭐
```python
# Force corrections to be small and bounded
raw_corrections = network(features)
corrections = 0.3 * torch.tanh(raw_corrections)  # Max ±0.3 ppb
corrected = PINN + corrections
```

**Pros**: 
- Prevents massive over-corrections
- Forces model to learn small, meaningful corrections
- More stable

**Cons**: 
- Might limit model's ability to correct large errors

### Option 4: Enhanced Loss Function ⭐⭐
```python
def enhanced_loss(pred, target, corrections, pinn_pred):
    # Prediction error
    mse = (pred - target)²
    
    # Penalize wrong-direction corrections
    error = target - pinn_pred
    wrong_direction = (corrections * error < 0)  # Opposite sign
    direction_penalty = torch.relu(-corrections * error).mean()
    
    # Penalize large corrections
    correction_size_penalty = torch.relu(torch.abs(corrections) - 0.5).mean()
    
    loss = mse + 0.5 * direction_penalty + 0.1 * correction_size_penalty
    return loss
```

**Pros**: 
- Explicitly penalizes wrong-direction corrections
- Can be combined with other options

**Cons**: 
- Doesn't fix architecture, just training

## Recommended Solution

**Combine Option 1 (Relative Corrections) + Option 3 (Bounded Corrections) + Option 4 (Enhanced Loss)**

### New Architecture:
```python
class NN2_CorrectionNetwork_Improved(nn.Module):
    def forward(self, pinn_predictions, sensor_coords, wind, diffusion, temporal):
        # ... feature extraction ...
        
        # Network outputs relative correction factors [-1, 1]
        raw_corrections = self.correction_network(features)
        relative_corrections = torch.tanh(raw_corrections)  # Bound to [-1, 1]
        
        # Apply as percentage correction (bounded)
        max_correction_pct = 0.3  # Max 30% correction
        corrections_ppb = pinn_predictions * relative_corrections * max_correction_pct
        
        corrected_ppb = pinn_predictions + corrections_ppb
        
        return corrected_ppb, corrections_ppb
```

### Enhanced Loss:
```python
def improved_loss(pred, target, corrections, pinn_pred, valid_mask):
    # Prediction error
    mse = nn.functional.mse_loss(pred[valid_mask], target[valid_mask])
    
    # Direction penalty: penalize corrections in wrong direction
    error = target - pinn_pred
    wrong_direction = (corrections * error < 0) & valid_mask
    direction_penalty = torch.relu(-corrections[wrong_direction] * error[wrong_direction]).mean()
    
    # Size penalty: penalize corrections > 50% of PINN
    correction_ratio = torch.abs(corrections) / (torch.abs(pinn_pred) + 1e-6)
    size_penalty = torch.relu(correction_ratio - 0.5).mean()
    
    loss = mse + 0.3 * direction_penalty + 0.1 * size_penalty
    return loss
```

## Next Steps

1. **Implement improved architecture** with relative corrections
2. **Update loss function** to penalize wrong-direction corrections
3. **Retrain model** on corrected training data
4. **Validate** that corrections are now in correct direction >70%
5. **Test** that NN2 MAE < PINN MAE

