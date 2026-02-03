# NN2 Model Test Attempts Log

## Attempt 1: Updated Loss Function Model (45-feature with current_sensors)
**Date**: 2025-02-02  
**Model**: `nn2_updatedlossfunction/nn2_master_model_ppb.pth`  
**Training Code**: `nn2colab_clean.py` with updated loss function (direction + size penalties)  
**Architecture**: 45 input features (includes current_sensors - data leakage still present)

### Training Results (Leave-One-Out Cross-Validation)
- **Held-out sensors**: 12-62% improvement
- **Training sensors**: 3-51% improvement
- **Status**: ✅ Good performance when current_sensors are available

### Deployment Test Results

#### 2019 Full Year Test
- **PINN MAE**: 0.5296 ppb
- **NN2 MAE**: 1.6565 ppb
- **Improvement**: -212.8% ❌
- **Status**: FAILED - Model degrades performance

**Per-Sensor Results:**
- 482010026: -164.7%
- 482010057: -123.9%
- 482010069: -289.8%
- 482010617: -342.0%
- 482010803: -144.3%
- 482011015: -155.5%
- 482011035: -233.4%
- 482011039: -429.2%
- 482016000: -203.5%

#### January-March 2021 Test
- **PINN MAE**: 0.4529 ppb
- **NN2 MAE**: 1.4795 ppb
- **Improvement**: -226.7% ❌
- **Status**: FAILED - Model degrades performance

**Monthly Breakdown:**
- January: -212.0%
- February: -285.0%
- March: -227.3%

**Per-Sensor Results:**
- 482010026: -116.9%
- 482010057: -73.5%
- 482010069: -262.0%
- 482010617: -413.9%
- 482010803: -37.0%
- 482011015: -217.7%
- 482011035: -122.5%
- 482011039: -896.9%
- 482016000: -289.1%

### Root Cause Analysis
1. **Model Architecture Issue**: Model trained with 45 features (includes `current_sensors`)
2. **Deployment Mismatch**: During deployment, zeros provided for `current_sensors` (actual values unavailable)
3. **Model Dependency**: Model learned to depend on actual sensor values as input feature
4. **Result**: Model fails when `current_sensors` are zeros

### Key Findings
- ✅ Updated loss function (direction + size penalties) appears to help during training
- ❌ Model still has data leakage (current_sensors as input)
- ❌ Model cannot work in deployment without actual sensor values
- ❌ Performance degrades significantly when current_sensors are zeros

### Conclusion
**Status**: FAILED  
**Reason**: Model architecture still includes data leakage (current_sensors as input). Model works well during training/validation when actual sensor values are available, but fails in deployment when they are not.

**Next Steps Required**:
1. Retrain model with 36-feature architecture (NO current_sensors)
2. Use updated loss function (direction + size penalties)
3. Test deployment with no actual sensor values

---

## Previous Attempts (Summary)

### Attempt 0: PPB Output Model (Direct PPB Scale)
**Date**: Prior to 2025-02-02  
**Model**: `nn2_ppbscale/nn2_master_model.pth`  
**Architecture**: 36 features (no current_sensors) - CORRECT  
**Training**: Model trained to output directly in ppb space

#### 2019 Full Year Test
- **PINN MAE**: 0.5296 ppb
- **NN2 MAE**: 0.5051 ppb
- **Improvement**: +4.6% ✅
- **Status**: PARTIAL SUCCESS - Small improvement but many negative predictions

**Issues**:
- 60% of predictions were negative (clamped to zero)
- Model systematically overcorrecting (subtracting too much)
- Clamping improved MAE but indicates systematic issue

### Attempt -1: Original NN2 Model (with data leakage)
**Date**: Prior to 2025-02-02  
**Model**: Various versions with data leakage  
**Architecture**: 45 features (includes current_sensors) - DATA LEAKAGE  
**Status**: FAILED  

**Issues**:
- Data leakage - model received `current_sensors` as input during training
- Model learned to predict actual values directly rather than corrections
- Correlation analysis: corrections correlated 0.96 with actual values, -0.07 with PINN
- Performance: -114.1% to -145.5% degradation on 2019 data

**Key Findings**:
- Corrections correlated 0.96 with actual values (predicting actuals, not corrections)
- Corrections correlated -0.07 with PINN (ignoring PINN)
- 86.3% of cases had NN2 error > PINN error
- Model outputs large corrections even when PINN is accurate

### Attempt -2: Gradient Boosting Mapping Solution
**Date**: Prior to 2025-02-02  
**Model**: Original NN2 + Gradient Boosting Regressor mapping  
**Architecture**: 36 features (no current_sensors) - CORRECT  
**Status**: ARCHIVED (not in active use)

**Results**:
- 2019: +50% improvement (0.2529 ppb MAE with Gradient Boosting)
- 2021: Degradation due to distribution shift
- **Issue**: Mapping model overfitted to 2019 distribution

**Solution**: Archived - not scalable, requires retraining for new distributions

### Attempt -3: PINN Time Dependency Bug
**Status**: FIXED  
**Issue**: PINN used absolute calendar time instead of simulation time  
**Fix**: Changed to fixed t=3.0 hours for all scenarios  
**Result**: PINN now behaves as steady-state physics model

### Attempt -4: Zero-Value Handling Mismatch
**Status**: FIXED  
**Issue**: NN2 scalers fitted only on non-zero values, but validation applied to all values  
**Fix**: Ensured zero masking in validation matches training  
**Result**: Proper scaling for zero values

---

## Summary of All Attempts

| Attempt | Model | Architecture | Loss Function | Training Results | Deployment Results | Status |
|---------|-------|--------------|---------------|------------------|-------------------|--------|
| 1 | Updated Loss | 45 features (leakage) | Direction + Size penalties | ✅ 12-62% improvement | ❌ -212.8% (2019), -226.7% (2021) | FAILED |
| 0 | PPB Output | 36 features (correct) | Standard MSE | ✅ Good | ✅ +4.6% (2019) but 60% negatives | PARTIAL |
| -1 | Original | 45 features (leakage) | Standard MSE | ✅ Good (cheating) | ❌ -114.1% to -145.5% | FAILED |
| -2 | Gradient Mapping | 36 features (correct) | Standard MSE + Mapping | ✅ Good | ✅ +50% (2019), ❌ Degrades (2021) | ARCHIVED |

## Key Insights

1. **Data Leakage is Critical**: All models with `current_sensors` as input (45 features) fail in deployment
2. **Architecture Matters**: Models without `current_sensors` (36 features) can work but need proper training
3. **Loss Function Helps**: Updated loss function (direction + size penalties) improves training but doesn't fix architecture issues
4. **Distribution Shift**: Solutions that work on 2019 may fail on 2021 due to distribution differences

## Next Steps Required

1. **Retrain with correct architecture**: 36 features (NO current_sensors)
2. **Use updated loss function**: Direction + size penalties
3. **Test deployment**: Must work without actual sensor values
4. **Validate on multiple years**: 2019 and 2021 to check generalization

