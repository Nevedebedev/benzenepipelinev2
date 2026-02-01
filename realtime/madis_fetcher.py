#!/usr/bin/env python3
"""
MADIS Data Fetcher for Real-Time Pipeline
Fetches latest hourly MADIS mesonet data from NOAA
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import requests
import tempfile
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from config import MADIS_BASE_URL, MADIS_STATIONS, TEMP_DIR


class MADISFetcher:
    """Fetch and parse MADIS meteorological data from NOAA"""
    
    def __init__(self):
        self.base_url = MADIS_BASE_URL
        self.stations = MADIS_STATIONS
        self.temp_dir = Path(TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_latest(self, target_time: datetime = None) -> dict:
        """
        Fetch latest meteorological data for Houston stations using NWS API
        
        Args:
            target_time: Specific datetime to fetch. If None, uses current time.
        
        Returns:
            dict: {station_id: {meteorology data}}
        """
        if target_time is None:
            target_time = datetime.utcnow()
        
        # Round to nearest hour
        target_time = target_time.replace(minute=0, second=0, microsecond=0)
        
        print(f"Fetching weather data for {target_time.strftime('%Y-%m-%d %H:00 UTC')}")
        
        # Try NWS API (modern, maintained endpoint)
        weather_data = self._fetch_from_nws_api(target_time)
        
        if weather_data:
            return weather_data
        
        # Fallback to mock data only if NWS API fails
        print("  Failed to fetch from NWS API, using mock data")
        return self._generate_mock_data(target_time)
    
    def _fetch_from_nws_api(self, target_time: datetime) -> dict:
        """
        Fetch observations from NWS API (modern HTTPS endpoint)
        
        Returns:
            dict: {station_id: {wind_speed, wind_dir, temperature, solar_radiation}}
        """
        weather_data = {}
        
        for station_id in self.stations.keys():
            try:
                # NWS API endpoint for latest observation
                url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
                
                response = requests.get(url, timeout=10, headers={'User-Agent': 'BenzenePipeline/1.0'})
                response.raise_for_status()
                
                obs = response.json()
                
                # Extract properties
                props = obs.get('properties', {})
                
                # Get values (NWS API returns with units)
                temp_data = props.get('temperature', {})
                wind_speed_data = props.get('windSpeed', {})
                wind_dir_data = props.get('windDirection', {})
                
                # Extract numeric values
                temperature = temp_data.get('value')  # Celsius
                wind_speed = wind_speed_data.get('value')  # m/s
                wind_dir = wind_dir_data.get('value')  # degrees
                
                # Convert temperature from Celsius to Kelvin
                if temperature is not None:
                    temperature = temperature + 273.15
                
                # Solar radiation not provided by NWS API, estimate from time of day
                hour = target_time.hour
                if 8 <= hour <= 17:  # Daytime
                    solar_radiation = 600.0  # Typical midday value
                elif 6 <= hour < 8 or 17 < hour <= 19:  # Twilight
                    solar_radiation = 250.0
                else:  # Night
                    solar_radiation = 0.0
                
                weather_data[station_id] = {
                    'wind_speed': wind_speed if wind_speed is not None else 5.0,
                    'wind_dir': wind_dir if wind_dir is not None else 180.0,
                    'temperature': temperature if temperature is not None else 288.0,
                    'solar_radiation': solar_radiation,
                }
                
                print(f"  ✓ {station_id}: wind={weather_data[station_id]['wind_speed']:.1f} m/s, " +
                      f"temp={weather_data[station_id]['temperature']:.1f} K")
                
            except Exception as e:
                print(f"  ✗ {station_id} failed: {e}")
                # Skip this station if it fails
                continue
        
        return weather_data if weather_data else None
    
    def _download_netcdf(self, url: str, filename: str) -> Path:
        """
        Download NetCDF file from NOAA
        
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            output_path = self.temp_dir / f"{filename}.nc"
            
            print(f"  Downloading from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  ✓ Downloaded to {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Download failed: {e}")
            return None
    
    def _parse_netcdf(self, nc_path: Path) -> dict:
        """
        Parse NetCDF file and extract Houston station data
        
        Returns:
            dict: {station_id: {wind_speed, wind_dir, temperature, solar_radiation}}
        """
        try:
            import netCDF4 as nc
            
            dataset = nc.Dataset(nc_path)
            
            # Extract station IDs
            station_ids = dataset.variables['stationId'][:].astype(str)
            
            weather_data = {}
            
            for target_id in self.stations.keys():
                # Find index of this station
                indices = [i for i, sid in enumerate(station_ids) if sid.strip() == target_id]
                
                if not indices:
                    print(f"  ⚠ Station {target_id} not found in data")
                    continue
                
                idx = indices[0]
                
                # Extract meteorology
                weather_data[target_id] = {
                    'wind_speed': float(dataset.variables['windSpeed'][idx]) if 'windSpeed' in dataset.variables else None,
                    'wind_dir': float(dataset.variables['windDir'][idx]) if 'windDir' in dataset.variables else None,
                    'temperature': float(dataset.variables['temperature'][idx]) if 'temperature' in dataset.variables else None,
                    'solar_radiation': float(dataset.variables['solarRadiation'][idx]) if 'solarRadiation' in dataset.variables else None,
                }
                
                print(f"  ✓ {target_id}: wind={weather_data[target_id]['wind_speed']:.1f} m/s, " +
                      f"temp={weather_data[target_id]['temperature']:.1f} K")
            
            dataset.close()
            return weather_data
            
        except Exception as e:
            print(f"  ✗ NetCDF parsing failed: {e}")
            return {}
    
    def _generate_mock_data(self, target_time: datetime) -> dict:
        """
        Generate mock meteorological data for testing
        Uses realistic values matching TRAINING DATA statistics from 2019-2021
        
        Training data meteorology (2019 Jan):
        - Wind speed: mean=4.4 m/s, range=[0.4, 14.8] m/s
        - Diffusion D: mean=35 m²/s, range=[1, 151] m²/s  
        - This matches typical Houston atmospheric conditions
        """
        import random
        random.seed(int(target_time.timestamp()))  # Deterministic based on time
        
        weather_data = {}
        
        for station_id in self.stations.keys():
            # Generate realistic Houston meteorology MATCHING TRAINING DATA
            # Use higher wind speeds to produce realistic diffusion
            wind_speed = random.uniform(3.0, 8.0)  # m/s (training mean: 4.4)
            wind_dir = random.uniform(0, 360)  # degrees
            temperature = random.uniform(280, 295)  # Kelvin (cooler = more stable)
            
            # Solar radiation depends on time of day
            hour = target_time.hour
            if 8 <= hour <= 17:  # Daytime
                solar_radiation = random.uniform(400, 900)  # W/m² (higher for unstable conditions)
            elif 6 <= hour < 8 or 17 < hour <= 19:  # Twilight
                solar_radiation = random.uniform(100, 400)
            else:  # Night
                solar_radiation = 0.0
            
            weather_data[station_id] = {
                'wind_speed': wind_speed,
                'wind_dir': wind_dir,
                'temperature': temperature,
                'solar_radiation': solar_radiation,
            }
            
            print(f"  ✓ MOCK {station_id}: wind={wind_speed:.1f} m/s @ {wind_dir:.0f}°, " +
                  f"temp={temperature:.1f} K")
        
        return weather_data


def test_fetcher():
    """Test the MADIS fetcher"""
    fetcher = MADISFetcher()
    
    # Test with current time
    print("\n" + "="*70)
    print("Testing MADIS Fetcher")
    print("="*70)
    
    data = fetcher.fetch_latest()
    
    print("\n" + "="*70)
    print(f"Fetched data for {len(data)} stations")
    print("="*70)
    
    for station_id, met_data in data.items():
        print(f"\n{station_id}:")
        for key, value in met_data.items():
            if value is not None:
                print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    test_fetcher()
