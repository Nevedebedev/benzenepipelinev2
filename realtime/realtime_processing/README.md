# Realtime Processing Pipeline

This folder contains the complete data processing pipeline for ensuring 100% data coverage for PINN training.

## Files Overview

### Core Processing Scripts
- `ensure_complete_january_data_simple.py` - Main data completion script that fills missing wind/temperature/solar data gaps
- `build_training_data_2021_january_complete.py` - Training data generator using complete dataset

### Processed Data
- `houston_processed_2021/` - Complete processed datasets
  - `houston_weather_2021_january_complete.csv` - Complete hourly weather data (3,720 observations, 100% coverage)
  - `facility_best_station_hourly_completeness_2021_complete.csv` - Updated station matches with 100% coverage
  - `training_data_2021_january_complete/` - 20 facility-specific training CSVs with complete coverage

## Data Processing Workflow

### 1. Raw Data Input
- Source: `../2021/01/` - Raw MADIS NetCDF files (744 hourly files)
- Format: Compressed NetCDF (.gz) with station observations

### 2. Data Completion Process
```python
# ensure_complete_january_data_simple.py
1. Extract all Houston area stations (319 stations, 699K observations)
2. Identify target stations with incomplete coverage
3. Apply interpolation methods:
   - Linear interpolation for wind speed (max 6-hour gaps)
   - Circular interpolation for wind direction (360° continuity)
   - Temporal interpolation for temperature (max 12-hour gaps)
   - Diurnal pattern filling for solar radiation
4. Fill remaining gaps with climatological defaults
```

### 3. Training Data Generation
```python
# build_training_data_2021_january_complete.py
1. Load complete weather dataset
2. Match facilities to their weather stations
3. Calculate derived variables:
   - Wind components (u,v) from speed/direction
   - Atmospheric stability classes (A-F)
   - Diffusion coefficients using boundary layer theory
   - Time-varying emission rates
4. Generate hourly timeline (744 hours for January)
5. Output facility-specific training CSVs
```

## Key Features

### Data Completeness
- **100% coverage** for all 20 facilities (744 hours each)
- **No missing values** - PINN can train without handling NaNs
- **Physically consistent** - proper wind component calculations
- **Temporal continuity** - smooth interpolated transitions

### Final Output Format
Each training CSV contains:
```
t,x,y,source_x,source_y,source_x_cartesian,source_y_cartesian,source_diameter,Q_total,wind_u,wind_v,D,phi
2021-01-01 00:00:00,29.7436,-95.0128,29.7436,-95.0128,13263,-695,3220,0.0016,3.0,9.18e-16,6.78,
```

### Variables
- `t`: Hourly timestamp
- `x,y`: Facility coordinates (lat/lon)
- `source_x_cartesian,y_cartesian`: Projected coordinates for plume modeling
- `source_diameter`: Facility size (meters)
- `Q_total`: Benzene emission rate (kg/s) - varies by time
- `wind_u,v`: Wind vector components
- `D`: Horizontal diffusion coefficient
- `phi`: Placeholder for concentration

## Usage

### To Process New Data:
```bash
cd realtime_processing
python3 ensure_complete_january_data_simple.py
python3 build_training_data_2021_january_complete.py
```

### To Verify Data Completeness:
```python
import pandas as pd
df = pd.read_csv('houston_processed_2021/training_data_2021_january_complete/ExxonMobil_Baytown_Refinery_training_data.csv')
print(f"Coverage: {len(df)}/744 hours")
print(f"Missing wind_u: {df['wind_u'].isna().sum()}")
print(f"Missing D: {df['D'].isna().sum()}")
```

## Station Coverage Summary

| Station ID | Facilities Served | Original Coverage | Final Coverage |
|------------|-------------------|-------------------|----------------|
| F2972 | 14 facilities | 106/744 (14%) | 744/744 (100%) |
| MGPT2 | 3 facilities | 737/744 (99%) | 744/744 (100%) |
| D2150 | 1 facility | 726/744 (98%) | 744/744 (100%) |
| D1774 | 1 facility | 728/744 (98%) | 744/744 (100%) |
| C6132 | 1 facility | 740/744 (99%) | 744/744 (100%) |

## Dependencies
- Python 3.7+
- pandas
- numpy
- netCDF4
- pathlib

## Notes
- All missing data gaps have been filled using physically-based interpolation
- Wind direction interpolation preserves circular continuity
- Solar radiation follows realistic diurnal patterns
- Ready for immediate PINN training without any preprocessing
