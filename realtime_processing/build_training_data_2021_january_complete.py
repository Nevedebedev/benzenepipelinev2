#!/usr/bin/env python3
"""
Build Gaussian Plume Training Data for 2021 January (Complete Data)
Uses the complete January dataset with all missing data filled.
"""

import gzip
import os
import shutil
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

# ============================================================================
# FACILITY DATA
# ============================================================================
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

# Use the complete dataset
COMPLETE_WEATHER_FILE = Path("houston_processed_2021/houston_weather_2021_january_complete.csv")
BEST_STATIONS_FILE = Path("houston_processed_2021/facility_best_station_hourly_completeness_2021_complete.csv")

CARTESIAN_COORDS = {
    'ExxonMobil Baytown Refinery': {'source_x_cartesian': 13263, 'source_y_cartesian': -695},
    'Shell Deer Park Refinery': {'source_x_cartesian': 2232, 'source_y_cartesian': -3210},
    'Valero Houston Refinery': {'source_x_cartesian': -10043, 'source_y_cartesian': -3073},
    'LyondellBasell Pasadena Complex': {'source_x_cartesian': -8275, 'source_y_cartesian': -4088},
    'LyondellBasell Channelview Complex': {'source_x_cartesian': 3380, 'source_y_cartesian': 8995},
    'ExxonMobil Baytown Olefins Plant': {'source_x_cartesian': 13398, 'source_y_cartesian': 717},
    'Chevron Phillips Chemical Co': {'source_x_cartesian': -2795, 'source_y_cartesian': -2134},
    'TPC Group': {'source_x_cartesian': -10072, 'source_y_cartesian': -5458},
    'INEOS Phenol': {'source_x_cartesian': -1107, 'source_y_cartesian': -1848},
    'Total Energies Petrochemicals': {'source_x_cartesian': 6174, 'source_y_cartesian': -2397},
    'BASF Pasadena': {'source_x_cartesian': -106, 'source_y_cartesian': -2396},
    'Huntsman International': {'source_x_cartesian': -11064, 'source_y_cartesian': -2920},
    'Invista': {'source_x_cartesian': -9662, 'source_y_cartesian': -5413},
    'Goodyear Baytown': {'source_x_cartesian': 9910, 'source_y_cartesian': -11400},
    'LyondellBasell Bayport Polymers': {'source_x_cartesian': 9590, 'source_y_cartesian': -13512},
    'INEOS PP & Gemini': {'source_x_cartesian': 6203, 'source_y_cartesian': -2957},
    'K-Solv Channelview': {'source_x_cartesian': 4531, 'source_y_cartesian': 1947},
    'Oxy Vinyls Deer Park': {'source_x_cartesian': 4428, 'source_y_cartesian': -2240},
    'ITC Deer Park': {'source_x_cartesian': 5424, 'source_y_cartesian': -1227},
    'Enterprise Houston Terminal': {'source_x_cartesian': 2114, 'source_y_cartesian': -553},
}

Q_SCHEDULE_KG_S = {
    'ExxonMobil Baytown Refinery': (0.00300, 0.00160, 0.00120, 0.00080),
    'Shell Deer Park Refinery': (0.00240, 0.00140, 0.00100, 0.00070),
    'Valero Houston Refinery': (0.00200, 0.00120, 0.00090, 0.00060),
    'LyondellBasell Pasadena Complex': (0.00160, 0.00100, 0.00070, 0.00050),
    'LyondellBasell Channelview Complex': (0.00140, 0.00090, 0.00060, 0.00040),
    'ExxonMobil Baytown Olefins Plant': (0.00120, 0.00080, 0.00050, 0.00036),
    'Chevron Phillips Chemical Co': (0.00100, 0.00070, 0.00040, 0.00030),
    'TPC Group': (0.00090, 0.00060, 0.00040, 0.00026),
    'INEOS Phenol': (0.00080, 0.00050, 0.00036, 0.00024),
    'Total Energies Petrochemicals': (0.00110, 0.00070, 0.00046, 0.00032),
    'BASF Pasadena': (0.00080, 0.00050, 0.00036, 0.00024),
    'Huntsman International': (0.00090, 0.00060, 0.00040, 0.00028),
    'Invista': (0.00070, 0.00046, 0.00030, 0.00020),
    'Goodyear Baytown': (0.00040, 0.00026, 0.00018, 0.00012),
    'LyondellBasell Bayport Polymers': (0.00100, 0.00064, 0.00044, 0.00030),
    'INEOS PP & Gemini': (0.00080, 0.00052, 0.00036, 0.00024),
    'K-Solv Channelview': (0.00060, 0.00040, 0.00028, 0.00018),
    'Oxy Vinyls Deer Park': (0.00076, 0.00050, 0.00034, 0.00022),
    'ITC Deer Park': (0.00030, 0.00020, 0.00014, 0.00010),
    'Enterprise Houston Terminal': (0.00024, 0.00016, 0.00012, 0.00008),
}

def wind_components_from_speed_direction(wind_speed, wind_dir):
    direction_rad = np.radians(wind_dir + 180)
    wind_u = wind_speed * np.sin(direction_rad)
    wind_v = wind_speed * np.cos(direction_rad)
    return wind_u, wind_v

def calculate_stability_class(wind_speed, temperature, timestamp, solar_radiation=None):
    hour = timestamp.hour
    is_day = 6 <= hour <= 18
    if is_day:
        if (solar_radiation is not None and not np.isnan(solar_radiation) and solar_radiation > 600) or \
           (solar_radiation is None or np.isnan(solar_radiation)) and 10 <= hour <= 14:
            if wind_speed < 2: return 'A'
            elif wind_speed < 3: return 'A-B'
            elif wind_speed < 5: return 'B'
            elif wind_speed < 6: return 'C'
            else: return 'D'
        elif (solar_radiation is not None and not np.isnan(solar_radiation) and solar_radiation > 300) or \
             ((solar_radiation is None or np.isnan(solar_radiation)) and (8 <= hour <= 10 or 14 <= hour <= 16)):
            if wind_speed < 2: return 'B'
            elif wind_speed < 3: return 'B-C'
            elif wind_speed < 5: return 'C'
            else: return 'D'
        else:
            if wind_speed < 5: return 'C'
            else: return 'D'
    else:
        if wind_speed < 2: return 'F'
        elif wind_speed < 3: return 'E'
        else: return 'D'

def calculate_diffusion_coefficient(wind_speed, stability_class):
    z0 = 0.03
    if wind_speed <= 0: wind_speed = 0.1
    u_star = (0.41 * wind_speed) / np.log(10.0 / z0)
    L_values = {'A': -10.0, 'A-B': -17.5, 'B': -25.0, 'B-C': -37.5, 'C': -50.0, 'D': 1e6, 'E': 50.0, 'F': 20.0}
    h_mix_values = {'A': 1500.0, 'A-B': 1350.0, 'B': 1200.0, 'B-C': 1050.0, 'C': 900.0, 'D': 800.0, 'E': 400.0, 'F': 200.0}
    L = L_values.get(stability_class, 1e6)
    h_mix = h_mix_values.get(stability_class, 800.0)
    if L < 0:
        w_star = u_star * (h_mix / abs(L))**(1/3)
        D_horizontal = 0.1 * w_star * h_mix
    else:
        z_char = 0.1 * h_mix
        D_horizontal = 0.4 * u_star * z_char
    return max(D_horizontal, 1.0)

def _q_for_timestamp(t, rates):
    w1, w2, e1, e2 = rates
    dt = pd.to_datetime(t)
    is_weekend = dt.dt.dayofweek >= 5
    is_peak = (dt.dt.hour >= 6) & (dt.dt.hour < 18)
    out = pd.Series(index=dt.index, dtype="float64")
    out[~is_weekend & is_peak] = w1
    out[~is_weekend & ~is_peak] = w2
    out[is_weekend & is_peak] = e1
    out[is_weekend & ~is_peak] = e2
    return out

def main():
    print("=" * 80)
    print("BUILDING GAUSSIAN PLUME TRAINING DATA FOR 2021 JANUARY (COMPLETE)")
    print("=" * 80)
    
    # Load complete weather data
    print("Loading complete weather data...")
    weather_df = pd.read_csv(COMPLETE_WEATHER_FILE)
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    # Load facility-station matches
    matches_df = pd.read_csv(BEST_STATIONS_FILE)
    
    # Create timeline for January
    timeline = pd.date_range("2021-01-01 00:00:00", "2021-01-31 23:00:00", freq="h")
    timeline_df = pd.DataFrame({"t": timeline})
    
    output_dir = Path("houston_processed_2021/training_data_2021_january_complete")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(FACILITIES)} facilities...")
    
    for fac, info in FACILITIES.items():
        print(f"Processing {fac}...")
        
        match = matches_df[matches_df["facility_name"] == fac]
        if match.empty: 
            print(f"  No station match found for {fac}")
            continue
            
        sid = match.iloc[0]["wind_station_id"]
        fac_weather = weather_df[weather_df["station_id"] == sid].copy()
        
        if fac_weather.empty:
            print(f"  No weather data found for station {sid}")
            continue
        
        # Ensure we have data for all hours
        fac_weather = fac_weather.set_index('timestamp').reindex(timeline)
        
        # Merge with timeline
        df = timeline_df.merge(fac_weather, left_on='t', right_index=True, how='left')
        
        # Add facility information
        df["x"] = df["source_x"] = info["source_x"]
        df["y"] = df["source_y"] = info["source_y"]
        df["source_x_cartesian"] = CARTESIAN_COORDS.get(fac, {}).get("source_x_cartesian")
        df["source_y_cartesian"] = CARTESIAN_COORDS.get(fac, {}).get("source_y_cartesian")
        df["source_diameter"] = info["source_diameter"]
        
        # Calculate emission rates
        rates = Q_SCHEDULE_KG_S.get(fac)
        df["Q_total"] = _q_for_timestamp(df["t"], rates) if rates else np.nan
        
        # Calculate wind components
        df["wind_u"] = df["wind_v"] = np.nan
        mask = df["wind_speed"].notna() & df["wind_dir"].notna()
        if mask.any():
            u, v = wind_components_from_speed_direction(df.loc[mask, "wind_speed"], df.loc[mask, "wind_dir"])
            df.loc[mask, "wind_u"], df.loc[mask, "wind_v"] = u, v
        
        # Calculate diffusion coefficients
        df["D"] = np.nan
        mask_ws = df["wind_speed"].notna()
        if mask_ws.any():
            def _row_D(r):
                stab = calculate_stability_class(r["wind_speed"], r["temperature"] if pd.notna(r["temperature"]) else 285.0, r["t"], r["solar_radiation"])
                return calculate_diffusion_coefficient(r["wind_speed"], stab)
            df.loc[mask_ws, "D"] = df.loc[mask_ws].apply(_row_D, axis=1)
        
        # Add phi column (placeholder)
        df["phi"] = np.nan
        
        # Select and save columns
        cols = ['t', 'x', 'y', 'source_x', 'source_y', 'source_x_cartesian', 'source_y_cartesian', 
                'source_diameter', 'Q_total', 'wind_u', 'wind_v', 'D', 'phi']
        
        # Check data completeness
        total_hours = len(df)
        complete_wind = df['wind_speed'].notna().sum()
        complete_temp = df['temperature'].notna().sum()
        complete_d = df['D'].notna().sum()
        
        print(f"  Data completeness: {complete_wind}/{total_hours} wind ({complete_wind/total_hours*100:.1f}%), "
              f"{complete_temp}/{total_hours} temp ({complete_temp/total_hours*100:.1f}%), "
              f"{complete_d}/{total_hours} D ({complete_d/total_hours*100:.1f}%)")
        
        df[cols].to_csv(output_dir / f"{fac.replace(' ', '_').replace('/', '_')}_training_data.csv", index=False)
        print(f"  Saved: {fac}")

    print("\nTraining data generation complete!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
