# Archived Files

This directory contains archived files that are no longer actively used in the current pipeline implementation.

**Date Archived:** 2025-02-02

## Directory Structure

- `old_models/` - Previous NN2 model versions and training attempts
- `investigation_scripts/` - Scripts used for debugging and investigating issues
- `test_scripts/` - Old validation and test scripts
- `documentation/` - Old analysis reports and documentation
- `logs/` - Old log files
- `old_training_scripts/` - Previous versions of training scripts

## What Was Archived

### Old Model Folders
- `nn2_pinncorrect/` - Model trained after fixing PINN computation mismatch
- `nn2_ppbscale/` - Model with direct ppb output (archived approach)
- `nn2_smaller/` - Model with simplified architecture (previous attempt)
- `nn2_timefix/` - Model after fixing time dependency bug
- `nn2_updated/` - Model with updated architecture
- `nn2_updatedlossfunction/` - Model with updated loss function
- `nn2_master_model_ppb.pth` - Old root-level model file

### Investigation Scripts
All scripts used to investigate and debug various issues:
- `investigate_*.py` - Various investigation scripts
- `analyze_*.py` - Analysis scripts
- `diagnose_*.py` - Diagnostic scripts

### Old Test Scripts
Previous versions of validation and test scripts that have been superseded:
- `test_alternative_approaches.py`
- `test_pipeline_2019_*.py` (multiple versions)
- `test_nn2_*.py` (various test scripts)
- `validate_*.py` (old validation scripts)
- `run_pipeline_2019.py`

### Old Documentation
Previous analysis reports and documentation:
- `nn2_architecture_analysis.md`
- `nn2_degradation_analysis_summary.md`
- `nn2_test_attempts_log.md`
- `root_cause_analysis_report.md`
- `SIMPLIFIED_ARCHITECTURE_CHANGES.md`
- `DATA_LEAKAGE_FIX.md`
- `DISTRIBUTION_ANALYSIS.md`
- `validation_results/` folder
- Various result CSV and TXT files

### Old Implementation Scripts
Archived approaches that are no longer used:
- `implement_nn2_output_mapping.py` - Gradient boosting mapping approach (archived)
- `nn2_mapping_utils.py` - Mapping utilities (archived)

### Old Training Scripts
Previous versions of training scripts:
- `nn2colab.py` - Original training script
- `nn2colab_clean.py` - Clean version with LOOCV (superseded by master_only)
- `nn2_ppbscale.py` - Direct ppb output version (archived approach)
- `traininglogs.txt` - Old training logs

### Logs
- `logs/` folder - Old log files
- `pipeline.log` - Old pipeline log

## Current Active Files

The following files remain in the `realtime/` directory as they are actively used:

### Core Pipeline
- `concentration_predictor.py` - Main prediction class
- `realtime_pipeline.py` - Real-time pipeline runner
- `madis_fetcher.py` - Weather data fetching
- `csv_generator.py` - CSV generation
- `atmospheric_calculations.py` - Atmospheric calculations
- `config.py` - Configuration
- `visualizer.py` - Visualization

### Current NN2 Training & Testing
- `drive-download-20260202T042428Z-3-001/nn2colab_clean_master_only.py` - Current training script (simplified architecture, master-only)
- `drive-download-20260202T042428Z-3-001/nn2_model_only.py` - Model definition for deployment
- `test_nn2_smaller_2019.py` - Current test script
- `nn2_scaled/` - Current model folder

### Training Data Generation
- `simpletesting/nn2trainingdata/regenerate_training_data_correct_pinn.py` - Training data generation
- `simpletesting/nn2trainingdata/total_concentrations.csv` - PINN predictions
- `simpletesting/nn2trainingdata/*_synced_training_data.csv` - Facility data files

### Validation Scripts
- `validate_nn2_january_2019.py` - Validation script
- `validate_jan_mar_2021.py` - Validation script
- `test_pipeline_2019.py` - Full pipeline test

### Current Documentation
- `NN2_SCALED_INVESTIGATION.md` - Current investigation report
- `important_guide/` - Complete pipeline documentation

## Notes

- All archived files are preserved for reference and historical purposes
- If you need to access any archived file, it can be found in the appropriate subdirectory
- The current active files represent the latest working implementation

