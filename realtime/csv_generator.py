#!/usr/bin/env python3
"""
CSV Generator for Real-Time Pipeline
Appends facility-specific meteorological data to continuous time series CSVs
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from config import FACILITIES, BASE_DIR, FACILITY_DIR, CONTINUOUS_DIR
from atmospheric_calculations import (
    wind_components_from_speed_direction,
    calculate_stability_class,
    calculate_diffusion_coefficient,
    q_for_timestamp
)


class CSVGenerator:
    """Generate and append facility-specific meteorological data"""
    
    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = FACILITY_DIR
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.facilities = FACILITIES
    
    def append_facility_data(self, weather_data: dict, current_time: datetime) -> dict:
        """
        Append new meteorological data row to each facility's time series CSV
        
        Args:
            weather_data: dict of {station_id: {wind_speed, wind_dir, temp, solar_rad}}
            current_time: Current timestamp (data fetch time)
        
        Returns:
            dict: {facility_name: meteorological_parameters}
        """
        forecast_time = current_time + timedelta(hours=3)
        
        print(f"\n[CSV Generator] Processing {len(self.facilities)} facilities")
        print(f"  Current time: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Forecast time: {forecast_time.strftime('%Y-%m-%d %H:%M')} (t+3hr)")
        
        facility_params = {}
        
        for facility_name, facility_info in self.facilities.items():
            # Get assigned weather station
            station_id = facility_info['station_id']
            
            if station_id not in weather_data:
                print(f"  ⚠ {facility_name}: Station {station_id} not available, skipping")
                continue
            
            met_data = weather_data[station_id]
            
            # Extract meteorology
            wind_speed = met_data.get('wind_speed')
            wind_dir = met_data.get('wind_dir')
            temperature = met_data.get('temperature', 288.0)  # Default 15°C
            solar_radiation = met_data.get('solar_radiation')
            
            if wind_speed is None or wind_dir is None:
                print(f"  ⚠ {facility_name}: Missing wind data, using defaults")
                wind_speed = 3.0
                wind_dir = 180.0
            
            # Calculate wind components
            wind_u, wind_v = wind_components_from_speed_direction(wind_speed, wind_dir)
            
            # Calculate atmospheric stability
            stability_class = calculate_stability_class(
                wind_speed, temperature, forecast_time, solar_radiation
            )
            
            # Calculate diffusion coefficient
            D = calculate_diffusion_coefficient(wind_speed, stability_class)
            
            # Calculate time-varying emission rate
            Q_rates = facility_info['Q_schedule_kg_s']
            Q = q_for_timestamp(forecast_time, Q_rates)
            
            # Prepare row data
            row_data = {
                'forecast_timestamp': forecast_time,
                'current_timestamp': current_time,
                'x': facility_info['source_x'],
                'y': facility_info['source_y'],
                'source_x_cartesian': facility_info['source_x_cartesian'],
                'source_y_cartesian': facility_info['source_y_cartesian'],
                'source_diameter': facility_info['source_diameter'],
                'Q': Q,
                'wind_u': wind_u,
                'wind_v': wind_v,
                'D': D,
                'stability_class': stability_class,
            }
            
            facility_params[facility_name] = row_data
            
            # Append to CSV
            self._append_to_csv(facility_name, row_data)
            
            print(f"  ✓ {facility_name[:30]}: u={wind_u:.2f}, v={wind_v:.2f}, D={D:.2f}, Q={Q:.6f}")
        
        print(f"  Appended data for {len(facility_params)}/{len(self.facilities)} facilities\n")
        
        return facility_params
    
    def _append_to_csv(self, facility_name: str, row_data: dict):
        """
        Append row to facility's time series CSV
        Creates file with header if it doesn't exist
        """
        # Sanitize filename
        safe_name = facility_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
        csv_path = self.output_dir / f"{safe_name}_timeseries.csv"
        
        # Convert to DataFrame
        df_row = pd.DataFrame([row_data])
        
        # Check if file exists
        if csv_path.exists():
            # Append without header
            df_row.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # Create new file with header
            df_row.to_csv(csv_path, mode='w', header=True, index=False)
            print(f"    Created new file: {csv_path.name}")
    
    def get_facility_count(self) -> dict:
        """
        Get row counts for each facility CSV
        
        Returns:
            dict: {facility_name: row_count}
        """
        counts = {}
        
        for facility_name in self.facilities.keys():
            safe_name = facility_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
            csv_path = self.output_dir / f"{safe_name}_timeseries.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                counts[facility_name] = len(df)
            else:
                counts[facility_name] = 0
        
        return counts


def test_csv_generator():
    """Test the CSV generator"""
    from madis_fetcher import MADISFetcher
    
    print("\n" + "="*70)
    print("Testing CSV Generator")
    print("="*70)
    
    # Fetch mock weather data
    fetcher = MADISFetcher()
    weather_data = fetcher.fetch_latest()
    
    # Generate CSVs
    generator = CSVGenerator()
    facility_params = generator.append_facility_data(weather_data, datetime.now())
    
    # Show row counts
    print("\n" + "="*70)
    print("CSV Row Counts")
    print("="*70)
    counts = generator.get_facility_count()
    for facility, count in sorted(counts.items()):
        if count > 0:
            print(f"  {facility[:40]:<40} {count:4d} rows")
    
    print("\n" + "="*70)
    print("Test Complete")
    print(f"Output directory: {generator.output_dir}")
    print("="*70)


if __name__ == "__main__":
    test_csv_generator()
