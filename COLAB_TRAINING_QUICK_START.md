# Colab Training Quick Start Guide

**File:** `realtime/drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only_IMPROVED.py`

---

## Quick Setup (Copy-Paste Ready)

### Step 1: Upload Files to Colab

1. Upload the training script:
   - `nn2colab_clean_master_only_IMPROVED.py`

2. Upload your data files to `/content/data/`:
   - `total_superimposed_concentrations.csv` (PINN predictions)
   - `sensor_coordinates.csv` (sensor locations)
   - `sensors_final.csv` (sensor readings)
   - `data_nonzero/` directory (meteorological data)

### Step 2: Install Dependencies

```python
!pip install torch pandas numpy scikit-learn tqdm
```

### Step 3: Mount Google Drive (if data is there)

```python
from google.colab import drive
drive.mount('/content/drive')

# If your data is in Google Drive, update paths:
# CONFIG['data_dir'] = '/content/drive/MyDrive/your_data_path/'
# CONFIG['save_dir'] = '/content/drive/MyDrive/your_models_path/'
```

### Step 4: Run Training

**Option A: Execute the script directly**
```python
exec(open('nn2colab_clean_master_only_IMPROVED.py').read())
```

**Option B: Import and run**
```python
import sys
sys.path.append('/content')
from nn2colab_clean_master_only_IMPROVED import train_master_model, CONFIG

# Update paths if needed
CONFIG['data_dir'] = '/content/data/'
CONFIG['save_dir'] = '/content/models/master_only/'

# Run training
master_model, results, scalers = train_master_model()
```

---

## What's Improved in This Version

### 1. **Better Configuration**
- **Epochs:** 50 → 100 (more training time)
- **Lambda Correction:** 0.001 → 0.01 (10x stronger regularization)
- **Lambda Large:** 0.1 (new penalty for extreme corrections)
- **Early Stopping:** Patience 15, min_delta 0.001

### 2. **Improved Loss Function**
- Added large correction penalty (prevents corrections >> PINN)
- Stronger regularization (keeps corrections small)
- Better balance between MSE and regularization

### 3. **Comprehensive Monitoring**
- Logs correction statistics every epoch
- Tracks training history
- Warns if corrections are too large
- Shows loss components (MSE, reg, large penalty)

### 4. **Better Model Saving**
- Saves best model state (not just last)
- Includes training history in checkpoint
- Saves all metadata (config, sensor coords, etc.)

---

## Expected Output

### Training Progress
```
Epoch   1/100: Train Loss=15.2345, Val Loss=16.1234, MSE=14.5000, Reg=0.0123, Large Penalty=0.0000
  Corrections: mean=2.5000, max=5.2341, std=1.2345 (scaled space)
  ⚠️  WARNING: Large corrections detected (max=5.23)
     Expected: < 0.5 scaled space

Epoch  10/100: Train Loss=4.5678, Val Loss=5.1234, MSE=4.2000, Reg=0.0234, Large Penalty=0.0001
  Corrections: mean=0.8000, max=1.5000, std=0.4000 (scaled space)
  ⚠️  WARNING: Large corrections detected (max=1.50)
     Expected: < 0.5 scaled space

Epoch  30/100: Train Loss=1.2345, Val Loss=1.3456, MSE=1.1000, Reg=0.0123, Large Penalty=0.0000
  Corrections: mean=0.3000, max=0.6000, std=0.1500 (scaled space)
  ✓ New best model (val_loss=1.3456)

Epoch  50/100: Train Loss=0.7890, Val Loss=0.8901, MSE=0.7500, Reg=0.0078, Large Penalty=0.0000
  Corrections: mean=0.1500, max=0.3500, std=0.0800 (scaled space)
  ✓ New best model (val_loss=0.8901)
```

### Final Results
```
================================================================================
MASTER MODEL RESULTS (All Sensors)
================================================================================
sensor_482010026: PINN MAE=0.5849 ppb, NN2 MAE=0.3500 ppb, Improvement=40.1%
sensor_482010057: PINN MAE=0.8899 ppb, NN2 MAE=0.5000 ppb, Improvement=43.8%
...
Average improvement: 45.2%

✓ Saved best model to: /content/models/master_only/nn2_master_model_ppb-3.pth
  Validation Loss: 0.8901
  Epoch: 50
================================================================================
```

---

## Key Metrics to Watch

### Loss Progression
- **Epoch 1-10:** Loss should drop rapidly (15 → 5)
- **Epoch 10-30:** Gradual improvement (5 → 1)
- **Epoch 30-60:** Fine-tuning (1 → 0.6)
- **Target:** Validation loss < 1.0

### Correction Magnitude
- **Epoch 1:** Mean ~2.5, Max ~5.0 (scaled space) - OK for early training
- **Epoch 10:** Mean ~0.8, Max ~1.5 - Should be improving
- **Epoch 30+:** Mean < 0.3, Max < 0.5 - **Target range**
- **Warning:** If max > 1.0 after epoch 20, increase regularization

### Success Criteria
- ✅ Validation loss < 1.0
- ✅ Correction mean < 0.2 (scaled space)
- ✅ Correction max < 0.5 (scaled space)
- ✅ Average improvement > 40%

---

## Troubleshooting

### Issue: Loss Not Decreasing
**Symptoms:** Loss stays high (> 10) after 20 epochs

**Solutions:**
1. Check data files are correct
2. Verify PINN predictions are reasonable (0.1-10 ppb)
3. Check learning rate (try 5e-4)
4. Verify model receives correct inputs

### Issue: Corrections Too Large
**Symptoms:** Correction max > 1.0 after epoch 20

**Solutions:**
1. Increase `lambda_correction` (0.01 → 0.1)
2. Increase `lambda_large` (0.1 → 1.0)
3. Check if PINN predictions are correctly scaled

### Issue: Overfitting
**Symptoms:** Train loss << Val loss (e.g., 0.5 vs 2.0)

**Solutions:**
1. Increase dropout rates (0.3 → 0.5)
2. Increase weight decay
3. Reduce model capacity

### Issue: Early Stopping Too Early
**Symptoms:** Training stops before loss converges

**Solutions:**
1. Increase `early_stopping_patience` (15 → 25)
2. Decrease `min_delta` (0.001 → 0.0001)
3. Check if loss is still decreasing slowly

---

## Files Generated

After training completes, you'll have:

1. **Model Checkpoint:**
   - `/content/models/master_only/nn2_master_model_ppb-3.pth`
   - Contains: model weights, scaler params, training history, config

2. **Scalers:**
   - `/content/models/master_only/nn2_master_scalers-3.pkl`
   - Contains: all StandardScaler objects for deployment

3. **Training History:**
   - Included in checkpoint
   - Can be extracted for plotting:
   ```python
   checkpoint = torch.load('nn2_master_model_ppb-3.pth')
   history = checkpoint['training_history']
   ```

---

## Downloading Results

```python
# Download model
from google.colab import files
files.download('/content/models/master_only/nn2_master_model_ppb-3.pth')
files.download('/content/models/master_only/nn2_master_scalers-3.pkl')
```

Or copy to Google Drive:
```python
import shutil
shutil.copy('/content/models/master_only/nn2_master_model_ppb-3.pth', 
            '/content/drive/MyDrive/models/')
shutil.copy('/content/models/master_only/nn2_master_scalers-3.pkl', 
            '/content/drive/MyDrive/models/')
```

---

## Next Steps After Training

1. **Test on Training Data:**
   - Should achieve MAE < 0.5 ppb (near-perfect)

2. **Test on 2019 Validation:**
   - Use `test_nn2_precomputed_pinn_2019.py`
   - Expected: 40-60% improvement over PINN

3. **Deploy:**
   - Update deployment code to use new model
   - Verify predictions are in range [0, 10] ppb

---

**Ready to train! Just copy-paste the script into Colab and run it.**

