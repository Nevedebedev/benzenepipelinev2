# Data Distribution Analysis and Overfitting Investigation

## Summary

**Key Finding: Distribution Shift, NOT Overfitting**

The model shows good generalization (no overfitting), but there is a **massive distribution shift** between training and validation PINN predictions, especially for sensor 482011039.

---

## Overfitting Analysis

### Leave-One-Out Cross-Validation Results

- **Average held-out improvement**: 71.3%
- **Average training improvement**: 69.7%
- **Gap (training - held-out)**: -1.6%

**Conclusion**: ✓ **Good generalization** - Training and held-out performance are similar. The model is NOT overfitting.

---

## Distribution Shift Analysis

### Critical Issue: Sensor 482011039

**Training PINN Predictions:**
- Mean: 0.26 ppb
- Std: 0.23 ppb
- Range: [0.02, 2.77] ppb
- Max: 2.77 ppb

**Validation PINN Predictions:**
- Mean: 17.61 ppb (67x higher!)
- Std: 174.92 ppb (767x higher!)
- Range: [0.0001, 4638.97] ppb
- Max: 4638.97 ppb (1675x higher!)

**Impact:**
- This sensor has a **massive distribution shift** in PINN predictions
- Validation PINN predictions are completely out of distribution
- NN2 scalers were trained on mean=0.26, std=0.23
- Validation data has mean=17.61, std=174.92
- This explains why NN2 performs poorly - it's seeing completely different input distributions

### Other Sensors

Most other sensors show **lower** PINN predictions in validation:
- Sensor 482010026: Training mean=0.49, Validation mean=0.00 (all zeros - bug?)
- Sensor 482010057: Training mean=0.25, Validation mean=0.06 (4x lower)
- Sensor 482010069: Training mean=0.38, Validation mean=0.00 (all zeros - bug?)
- Sensor 482010617: Training mean=0.56, Validation mean=0.00 (all zeros - bug?)
- Sensor 482010803: Training mean=0.28, Validation mean=0.00 (all zeros - bug?)
- Sensor 482011015: Training mean=0.40, Validation mean=0.00 (all zeros - bug?)
- Sensor 482011035: Training mean=0.25, Validation mean=0.00 (all zeros - bug?)
- Sensor 482016000: Training mean=0.54, Validation mean=0.04 (13x lower)

**Note**: Many sensors showing 0.00 in validation suggests there may be a bug in how validation PINN predictions are computed, OR the validation data truly has very low concentrations.

### Sensor Readings (Ground Truth)

Sensor readings are **identical** between training and validation (as expected, since we use the same sensor data file):
- All sensors show identical means, stds, and ranges
- This confirms the sensor data is consistent

---

## Root Cause Analysis

### Why NN2 Performs Poorly

1. **Distribution Shift in PINN Predictions:**
   - Training: PINN predictions have moderate values (0.2-0.6 ppb mean)
   - Validation: PINN predictions are either very low (near zero) or very high (sensor 482011039: mean=17.61, max=4638.97)
   - NN2 scalers were trained on training distribution
   - Validation data is out of distribution → poor performance

2. **Sensor 482011039 is an Extreme Outlier:**
   - Training: mean=0.26, max=2.77
   - Validation: mean=17.61, max=4638.97
   - This is a **67x difference in mean** and **1675x difference in max**
   - This sensor alone could be causing most of the degradation

3. **Model is NOT Overfitting:**
   - Leave-one-out shows good generalization
   - The problem is distribution shift, not model capacity

### Why Leave-One-Out Works But Full Validation Doesn't

- **Leave-one-out**: Evaluates on the same data distribution as training (same timestamps, same PINN predictions)
- **Full validation**: Uses different timestamps, different meteorological conditions → different PINN predictions → distribution shift

---

## Recommendations

1. **Investigate PINN Prediction Differences:**
   - Why are validation PINN predictions so different from training?
   - Check if there's a bug in validation PINN computation
   - Check if meteorological data differs between training and validation

2. **Investigate Sensor 482011039:**
   - Why does this sensor have such extreme PINN predictions in validation?
   - Check if there's a facility near this sensor causing high emissions
   - Check if wind patterns differ for this location

3. **Consider Distribution-Aware Training:**
   - Retrain NN2 with validation data included, OR
   - Use RobustScaler instead of StandardScaler, OR
   - Add data augmentation to cover wider distribution

4. **Check Validation PINN Computation:**
   - Many sensors show 0.00 in validation - this might be a bug
   - Verify that validation PINN predictions are computed correctly

---

## Next Steps

1. Debug why validation PINN predictions are so different (especially sensor 482011039)
2. Check if there's a bug in validation PINN computation (many zeros)
3. Investigate if meteorological data differs between training and validation
4. Consider retraining with more diverse data or using more robust scaling

