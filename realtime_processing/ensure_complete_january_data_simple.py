#!/usr/bin/env python3
"""
Ensure Complete January 2021 Data for PINN Training (Simple Version)
Fills missing wind data gaps using basic interpolation methods.
"""

import gzip
import os
import shutil
import tempfile
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import netCDF4 as nc
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_DIR = Path("2021/01")
OUTPUT_DIR = Path("houston_processed_2021")
COMPLETE_OUTPUT_FILE = OUTPUT_DIR / "houston_weather_2021_january_complete.csv"
FACILITY_MATCHES_FILE = OUTPUT_DIR / "facility_best_station_hourly_completeness_2021_complete.csv"

# Houston bounds for candidate stations
LAT_MIN, LAT_MAX = 28.0, 30.5
LON_MIN, LON_MAX = -95.8, -94.0

# Variable names
TEMP_VAR = "temperature"
WIND_SPEED_VAR = "windSpeed"
WIND_DIR_VAR = "windDir"
SOLAR_RAD_VAR = "solarRadiation"
STATION_ID_VAR = "stationId"
STATION_NAME_VAR = "stationName"
OBS_TIME_VAR = "observationTime"

# Facility definitions
FACILITIES = {
    'ExxonMobil Baytown Refinery': {'source_x': 29.7436, 'source_y': -95.0128, 'source_diameter': 3220, 'Q_total': 67.5},
    'Shell Deer Park Refinery': {'source_x': 29.7208, 'source_y': -95.1269, 'source_diameter': 2740, 'Q_total': 42.1},
    'Valero Houston Refinery': {'source_x': 29.7222, 'source_y': -95.2539, 'source_diameter': 752, 'Q_total': 36.4},
    'LyondellBasell Pasadena Complex': {'source_x': 29.7131, 'source_y': -95.2356, 'source_diameter': 1914, 'Q_total': 48.7},
    'LyondellBasell Channelview Complex': {'source_x': 29.8314, 'source_y': -95.1150, 'source_diameter': 2515, 'Q_total': 21.3},
    'ExxonMobil Baytown Olefins Plant': {'source_x': 29.7564, 'source_y': -95.0114, 'source_diameter': 1050, 'Q_total': 9.4},
    'Chevron Phillips Chemical Co': {'source_x': 29.7292, 'source_y': -95.1789, 'source_diameter': 1885, 'Q_total': 18.7},
    'TPC Group': {'source_x': 29.7006, 'source_y': -95.2542, 'source_diameter': 1032, 'Q_total': 1.8},
    'INEOS Phenol': {'source_x': 29.7322, 'source_y': -95.1614, 'source_diameter': 2636, 'Q_total': 1.0},
    'Total Energies Petrochemicals': {'source_x': 29.7278, 'source_y': -95.0861, 'source_diameter': 667, 'Q_total': 1.0},
    'BASF Pasadena': {'source_x': 29.7278, 'source_y': -95.1511, 'source_diameter': 800, 'Q_total': 1.0},
    'Huntsman International': {'source_x': 29.7236, 'source_y': -95.2644, 'source_diameter': 193, 'Q_total': 1.0},
    'Invista': {'source_x': 29.7042, 'source_y': -95.2500, 'source_diameter': 490, 'Q_total': 1.0},
    'Goodyear Baytown': {'source_x': 29.6469, 'source_y': -95.0475, 'source_diameter': 532, 'Q_total': None},
    'LyondellBasell Bayport Polymers': {'source_x': 29.6278, 'source_y': -95.0508, 'source_diameter': 2130, 'Q_total': None},
    'INEOS PP & Gemini': {'source_x': 29.7231, 'source_y': -95.0858, 'source_diameter': 530, 'Q_total': None},
    'K-Solv Channelview': {'source_x': 29.7675, 'source_y': -95.1031, 'source_diameter': 143, 'Q_total': None},
    'Oxy Vinyls Deer Park': {'source_x': 29.7306, 'source_y': -95.1042, 'source_diameter': 1008, 'Q_total': None},
    'ITC Deer Park': {'source_x': 29.7389, 'source_y': -95.0939, 'source_diameter': 1168, 'Q_total': None},
    'Enterprise Houston Terminal': {'source_x': 29.7450, 'source_y': -95.1281, 'source_diameter': 578, 'Q_total': None},
}

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def _iter_hourly_files():
    """Iterate through all hourly files for January 2021."""
    files = []
    for day_dir in sorted(INPUT_DIR.iterdir()):
        if not day_dir.is_dir() or day_dir.name.startswith('.'): 
            continue
        try:
            day_val = int(day_dir.name)
            if 1 <= day_val <= 31:
                for f in sorted(day_dir.glob("*.gz")):
                    if not f.name.startswith("._"): 
                        files.append(f)
        except ValueError: 
            continue
    return files

def _open_gz_as_netcdf(gz_path):
    """Open gzipped NetCDF file."""
    tmp = tempfile.NamedTemporaryFile(prefix="madis_jan_", suffix=".nc", delete=False, dir="/tmp")
    tmp_path = Path(tmp.name)
    tmp.close()
    with gzip.open(gz_path, "rb") as f_in:
        with open(tmp_path, "wb") as f_out: 
            shutil.copyfileobj(f_in, f_out)
    return nc.Dataset(tmp_path), tmp_path

def extract_all_houston_stations():
    """Extract all Houston area stations with their data."""
    hourly_files = _iter_hourly_files()
    all_records = []
    station_coords = {}
    
    print(f"Extracting data from {len(hourly_files)} files...")
    
    for i, gz_file in enumerate(hourly_files):
        try:
            ds, tmp_path = _open_gz_as_netcdf(gz_file)
        except:
            continue
            
        try:
            # Get station coordinates
            if STATION_NAME_VAR in ds.variables:
                lats = ds.variables['latitude'][:]
                lons = ds.variables['longitude'][:]
                station_ids = nc.chartostring(ds.variables[STATION_ID_VAR][:])
                station_ids = np.char.replace(station_ids, "\x00", "")
                station_ids = np.char.strip(station_ids)
                
                # Spatial mask for Houston area
                mask = (lats >= LAT_MIN) & (lats <= LAT_MAX) & (lons >= LON_MIN) & (lons <= LON_MAX)
                valid_indices = np.where(mask)[0]
                
                if len(valid_indices) > 0:
                    # Store station coordinates
                    for idx in valid_indices:
                        sid = str(station_ids[idx])
                        if sid not in station_coords:
                            station_coords[sid] = (float(lats[idx]), float(lons[idx]))
                    
                    # Extract observations
                    ot_var = ds.variables[OBS_TIME_VAR]
                    ws_var = ds.variables[WIND_SPEED_VAR]
                    wd_var = ds.variables[WIND_DIR_VAR]
                    temp_var = ds.variables[TEMP_VAR] if TEMP_VAR in ds.variables else None
                    sol_var = ds.variables[SOLAR_RAD_VAR] if SOLAR_RAD_VAR in ds.variables else None
                    
                    for idx in valid_indices:
                        sid = str(station_ids[idx])
                        try:
                            if hasattr(ot_var[idx], 'mask') and ot_var[idx].mask:
                                continue
                            if hasattr(ws_var[idx], 'mask') and ws_var[idx].mask:
                                continue
                                
                            record = {
                                "station_id": sid,
                                "latitude": float(lats[idx]),
                                "longitude": float(lons[idx]),
                                "observation_time": float(ot_var[idx]),
                                "wind_speed": float(ws_var[idx]),
                                "wind_dir": float(wd_var[idx]) if not (hasattr(wd_var[idx], 'mask') and wd_var[idx].mask) else np.nan,
                            }
                            
                            if temp_var:
                                record["temperature"] = float(temp_var[idx]) if not (hasattr(temp_var[idx], 'mask') and temp_var[idx].mask) else np.nan
                            else:
                                record["temperature"] = np.nan
                                
                            if sol_var:
                                record["solar_radiation"] = float(sol_var[idx]) if not (hasattr(sol_var[idx], 'mask') and sol_var[idx].mask) else np.nan
                            else:
                                record["solar_radiation"] = np.nan
                                
                            all_records.append(record)
                        except:
                            continue
        finally:
            try: 
                ds.close()
            finally: 
                tmp_path.unlink(missing_ok=True)
                
        if (i+1) % 100 == 0: 
            print(f"Processed {i+1}/{len(hourly_files)} files")
    
    return pd.DataFrame(all_records), station_coords

def simple_linear_interpolation(series, max_gap=6):
    """Simple linear interpolation for pandas Series."""
    if series.isna().all():
        return series
    
    filled = series.copy()
    na_indices = filled.isna()
    
    if na_indices.any():
        # Forward fill then backward fill with limit
        filled = filled.fillna(method='ffill', limit=max_gap)
        filled = filled.fillna(method='bfill', limit=max_gap)
    
    return filled

def interpolate_wind_direction(wind_dir_series, max_gap=6):
    """Interpolate wind direction using circular statistics."""
    if wind_dir_series.isna().all():
        return wind_dir_series
    
    filled = wind_dir_series.copy()
    
    # Convert to radians
    wind_rad = np.radians(filled)
    
    # Convert to u,v components
    wind_u = np.sin(wind_rad)
    wind_v = np.cos(wind_rad)
    
    # Interpolate components
    wind_u = pd.Series(wind_u).fillna(method='ffill', limit=max_gap).fillna(method='bfill', limit=max_gap)
    wind_v = pd.Series(wind_v).fillna(method='ffill', limit=max_gap).fillna(method='bfill', limit=max_gap)
    
    # Convert back to degrees
    wind_dir_filled = np.degrees(np.arctan2(wind_u, wind_v)) % 360
    
    return wind_dir_filled

def fill_missing_data(station_hourly):
    """Fill missing data using various interpolation methods."""
    filled_df = station_hourly.copy()
    
    # Linear interpolation for wind speed
    filled_df['wind_speed'] = simple_linear_interpolation(filled_df['wind_speed'], max_gap=6)
    
    # Circular interpolation for wind direction
    filled_df['wind_dir'] = interpolate_wind_direction(filled_df['wind_dir'], max_gap=6)
    
    # Linear interpolation for temperature
    filled_df['temperature'] = simple_linear_interpolation(filled_df['temperature'], max_gap=12)
    
    # Linear interpolation for solar radiation
    filled_df['solar_radiation'] = simple_linear_interpolation(filled_df['solar_radiation'], max_gap=6)
    
    # Fill remaining NaN with reasonable defaults
    if filled_df['wind_speed'].isna().any():
        # Use monthly mean wind speed
        mean_wind = filled_df['wind_speed'].mean()
        if pd.isna(mean_wind):
            mean_wind = 3.0  # Default 3 m/s
        filled_df['wind_speed'] = filled_df['wind_speed'].fillna(mean_wind)
    
    if filled_df['wind_dir'].isna().any():
        # Use prevailing wind direction (westerly in Houston)
        filled_df['wind_dir'] = filled_df['wind_dir'].fillna(270.0)
    
    if filled_df['temperature'].isna().any():
        # Use January mean temperature (~12°C)
        filled_df['temperature'] = filled_df['temperature'].fillna(285.0)
    
    if filled_df['solar_radiation'].isna().any():
        # Fill with 0 for night, mean for day based on hour
        filled_df['hour'] = filled_df.index.hour
        day_mask = (filled_df['hour'] >= 6) & (filled_df['hour'] <= 18)
        night_mask = ~day_mask
        
        mean_day_radiation = filled_df.loc[day_mask, 'solar_radiation'].mean()
        if pd.isna(mean_day_radiation):
            mean_day_radiation = 400.0  # Default daytime radiation
        
        filled_df.loc[day_mask, 'solar_radiation'] = filled_df.loc[day_mask, 'solar_radiation'].fillna(mean_day_radiation)
        filled_df.loc[night_mask, 'solar_radiation'] = filled_df.loc[night_mask, 'solar_radiation'].fillna(0.0)
    
    return filled_df

def create_complete_dataset():
    """Create complete dataset with all missing data filled."""
    print("=" * 80)
    print("CREATING COMPLETE JANUARY 2021 DATASET")
    print("=" * 80)
    
    # Extract all station data
    print("1. Extracting all Houston station data...")
    all_df, station_coords = extract_all_houston_stations()
    
    if all_df.empty:
        print("No data found!")
        return
    
    print(f"Found {len(station_coords)} stations with {len(all_df)} observations")
    
    # Convert to datetime and set index
    all_df['timestamp'] = pd.to_datetime(all_df['observation_time'], unit='s', utc=True)
    all_df['hour'] = all_df['timestamp'].dt.floor('h')
    
    # Create complete hourly timeline for January
    timeline = pd.date_range("2021-01-01 00:00:00", "2021-01-31 23:00:00", freq="h")
    
    # Get target stations from facilities
    target_stations = set()
    original_matches = pd.read_csv(OUTPUT_DIR / "facility_best_station_hourly_completeness_2021.csv")
    
    for _, row in original_matches.iterrows():
        target_stations.add(row['wind_station_id'])
    
    print(f"2. Processing {len(target_stations)} target stations...")
    
    complete_results = []
    station_matches = []
    
    for target_station_id in target_stations:
        print(f"Processing station {target_station_id}...")
        
        # Get target station data
        target_data = all_df[all_df['station_id'] == target_station_id].copy()
        
        if target_data.empty:
            print(f"  No data found for {target_station_id}")
            continue
        
        # Create hourly dataframe
        station_hourly = target_data.groupby('hour').agg({
            'wind_speed': 'mean',
            'wind_dir': 'mean',
            'temperature': 'mean',
            'solar_radiation': 'mean',
            'latitude': 'first',
            'longitude': 'first'
        }).reindex(timeline)
        
        # Check data completeness
        original_hours = station_hourly['wind_speed'].notna().sum()
        print(f"  Original coverage: {original_hours}/744 hours ({original_hours/744*100:.1f}%)")
        
        if original_hours < 744:
            print(f"  Filling missing data...")
            station_hourly = fill_missing_data(station_hourly)
        
        # Final completeness check
        final_hours = station_hourly['wind_speed'].notna().sum()
        print(f"  Final coverage: {final_hours}/744 hours ({final_hours/744*100:.1f}%)")
        
        # Add station info back
        station_hourly['station_id'] = target_station_id
        station_hourly['latitude'] = station_coords[target_station_id][0]
        station_hourly['longitude'] = station_coords[target_station_id][1]
        
        complete_results.append(station_hourly)
        
        # Calculate final statistics for matching
        temp_complete = station_hourly['temperature'].notna().sum()
        wind_complete = station_hourly['wind_speed'].notna().sum()
        station_matches.append({
            'station_id': target_station_id,
            'total_hours': 744,
            'wind_hours_present': wind_complete,
            'temp_hours_present': temp_complete,
            'wind_pct': wind_complete / 744 * 100,
            'temp_pct': temp_complete / 744 * 100
        })
    
    # Combine all results
    if complete_results:
        complete_df = pd.concat(complete_results, ignore_index=False)
        complete_df = complete_df.reset_index()
        complete_df = complete_df.rename(columns={'index': 'timestamp'})
        
        # Save complete dataset
        complete_df.to_csv(COMPLETE_OUTPUT_FILE, index=False)
        print(f"Saved complete dataset: {COMPLETE_OUTPUT_FILE}")
        
        # Update facility matches
        station_match_df = pd.DataFrame(station_matches)
        
        # Create new facility matches with updated station info
        new_facility_matches = []
        for _, facility_row in original_matches.iterrows():
            station_id = facility_row['wind_station_id']
            station_stats = station_match_df[station_match_df['station_id'] == station_id]
            
            if not station_stats.empty:
                stats = station_stats.iloc[0]
                new_row = facility_row.copy()
                new_row['wind_hours_present'] = stats['wind_hours_present']
                new_row['wind_pair_pct'] = stats['wind_pct']
                new_row['ref_temp_pct'] = stats['temp_pct']
                new_row['ref_wind_pct'] = stats['wind_pct']
                new_facility_matches.append(new_row)
        
        new_matches_df = pd.DataFrame(new_facility_matches)
        new_matches_df.to_csv(FACILITY_MATCHES_FILE, index=False)
        print(f"Saved updated facility matches: {FACILITY_MATCHES_FILE}")
        
        print("\nFinal Summary:")
        print(f"Total stations processed: {len(complete_results)}")
        print(f"Total observations: {len(complete_df)}")
        print(f"Expected observations: {len(complete_results) * 744}")
        
        return complete_df, new_matches_df
    
    return None, None

if __name__ == "__main__":
    complete_df, matches_df = create_complete_dataset()
    
    if complete_df is not None:
        print("\nData completion successful!")
        print("You can now regenerate training data with complete coverage.")
    else:
        print("Data completion failed!")
