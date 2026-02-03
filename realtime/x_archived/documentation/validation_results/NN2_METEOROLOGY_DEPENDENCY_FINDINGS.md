# NN2 Meteorology Dependency Test Results

## Test Date
2024-02-02

## Question
Does meteorology (wind_u, wind_v, D) actually affect NN2 predictions, or is it ignored by the model?

## Test Methodology

Tested NN2 on 100 samples from exact training data with:
1. **Real meteorology**: Exact values from training (BASF_Pasadena file)
2. **Zero meteorology**: wind_u=0, wind_v=0, D=0
3. **2x wind, 0.5x diffusion**: Scaled meteorology values
4. **-1x wind, 2x diffusion**: Inverted/scaled meteorology values

## Results

### Test 1: Zero Meteorology vs Real Meteorology
- **Mean absolute difference**: 0.1219 ppb
- **Max difference**: 1.0221 ppb
- **Identical predictions**: 189 / 900 (21.0%)

### Test 2: 2x Wind, 0.5x Diffusion vs Real
- **Mean absolute difference**: 0.1168 ppb
- **Max difference**: 1.2159 ppb

### Test 3: -1x Wind, 2x Diffusion vs Real
- **Mean absolute difference**: 0.1971 ppb
- **Max difference**: 1.9768 ppb

## Conclusion

✅ **METEOROLOGY DOES AFFECT NN2 PREDICTIONS**

### Key Findings:
1. **Meteorology is actively used**: Changing meteorology values changes predictions by ~0.12 ppb on average
2. **Model is sensitive**: Even zero meteorology produces different results than real meteorology
3. **Wrong meteorology causes errors**: Using incorrect meteorology will degrade NN2 performance

### Implications:
- **Must use exact training meteorology**: The investigation scripts must use the same single file (BASF_Pasadena) that was used during training
- **Averaging is wrong**: Averaging meteorology across facilities does NOT match training and will cause prediction errors
- **Meteorology mismatch is a real issue**: This could be contributing to the poor NN2 performance

## Next Steps

1. **Fix investigation scripts**: Update to use single-file meteorology (BASF_Pasadena) matching training
2. **Fix validation scripts**: Ensure all validation uses the exact same meteorology loading method
3. **Re-test NN2 performance**: After fixing meteorology, re-run validation to see if performance improves

## Technical Details

- **Training meteorology source**: First file alphabetically sorted from `data_nonzero/` directory
- **File used**: `BASF_Pasadena_training_data.csv` (after removing `_synced` from filename)
- **Training code location**: `nn2colab.py` lines 660-667
- **Investigation scripts**: Currently averaging meteorology (WRONG - needs fixing)

