#!/usr/bin/env python3
"""
Real-Time Benzene Concentration Pipeline
Main orchestrator for continuous prediction system

Runs every 15 minutes to:
1. Fetch NOAA MADIS data
2. Generate facility-specific CSVs
3. Compute PINN+NN2 predictions across full domain  
4. Append results to continuous time series
5. Generate forecast visualization
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from madis_fetcher import MADISFetcher
from csv_generator import CSVGenerator
from concentration_predictor import ConcentrationPredictor
from config import BASE_DIR, LOGS_DIR, CONTINUOUS_DIR


class RealtimePipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, grid_resolution: int = 100):
        self.grid_resolution = grid_resolution
        
        # Initialize components
        print("="*70)
        print("REAL-TIME BENZENE CONCENTRATION PIPELINE")
        print("="*70)
        print("\nInitializing components...")
        
        self.fetcher = MADISFetcher()
        self.csv_generator = CSVGenerator()
        self.predictor = ConcentrationPredictor(grid_resolution=grid_resolution)
        
        # Logging
        self.log_dir = LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "pipeline.log"
        
        print("✓ Pipeline initialized\n")
    
    def run_once(self, current_time: datetime = None):
        """
        Execute pipeline for single timestamp
        
        Args:
            current_time: Timestamp to process. If None, uses current time.
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Round to nearest 15 minutes
        minutes = (current_time.minute // 15) * 15
        current_time = current_time.replace(minute=minutes, second=0, microsecond=0)
        forecast_time = current_time + timedelta(hours=3)
        
        print("="*70)
        print(f"PIPELINE RUN: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Forecast for: {forecast_time.strftime('%Y-%m-%d %H:%M')} (t+3hr)")
        print("="*70)
        
        try:
            # Step 1: Fetch MADIS data
            print("\n[1/4] Fetching MADIS meteorological data...")
            weather_data = self.fetcher.fetch_latest(current_time)
            
            if not weather_data:
                raise ValueError("No weather data available")
            
            # Step 2: Generate facility CSVs
            print("\n[2/4] Generating facility-specific data...")
            facility_params = self.csv_generator.append_facility_data(
                weather_data, current_time
            )
            
            if not facility_params:
                raise ValueError("No facility parameters generated")
            
            # Step 3: Predict concentrations
            print("\n[3/4] Computing PINN+NN2 predictions...")
            pinn_field, nn2_field, _ = self.predictor.predict_full_domain(
                facility_params, current_time
            )
            
            # Step 4: Log completion
            print("\n[4/4] Pipeline complete")
            print(f"  PINN range: {pinn_field.min():.4f} - {pinn_field.max():.4f} ppb")
            print(f"  NN2 range: {nn2_field.min():.4f} - {nn2_field.max():.4f} ppb")
            
            self._log_success(current_time, forecast_time, pinn_field, nn2_field)
            
            print("\n" + "="*70)
            print("✓ PIPELINE RUN SUCCESSFUL")
            print("="*70 + "\n")
            
            return True
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}\n{traceback.format_exc()}"
            print(f"\n✗ ERROR: {error_msg}")
            self._log_error(current_time, error_msg)
            return False
    
    def run_continuous(self, interval_minutes: int = 15):
        """
        Run pipeline continuously every N minutes
        
        Args:
            interval_minutes: Interval between runs (default 15)
        """
        print("="*70)
        print(f"STARTING CONTINUOUS MODE - Updates every {interval_minutes} minutes")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        while True:
            try:
                # Run pipeline
                current_time = datetime.now()
                self.run_once(current_time)
                
                # Calculate next run time (aligned to interval)
                next_run = current_time.replace(second=0, microsecond=0)
                while next_run <= datetime.now():
                    next_run += timedelta(minutes=interval_minutes)
                
                # Wait until next run
                sleep_seconds = (next_run - datetime.now()).total_seconds()
                
                if sleep_seconds > 0:
                    print(f"Next run scheduled for: {next_run.strftime('%H:%M:%S')}")
                    print(f"Sleeping for {sleep_seconds:.0f} seconds...\n")
                    time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                print("\n\n" + "="*70)
                print("PIPELINE STOPPED BY USER")
                print("="*70)
                break
                
            except Exception as e:
                error_msg = f"Continuous mode error: {str(e)}\n{traceback.format_exc()}"
                print(f"\n✗ ERROR: {error_msg}")
                self._log_error(datetime.now(), error_msg)
                
                # Wait 1 minute before retrying
                print("Waiting 60 seconds before retry...\n")
                time.sleep(60)
    
    def _log_success(self, current_time, forecast_time, pinn_field, nn2_field):
        """Log successful run"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"[{timestamp}] SUCCESS | "
            f"Current: {current_time.strftime('%Y-%m-%d %H:%M')} | "
            f"Forecast: {forecast_time.strftime('%Y-%m-%d %H:%M')} | "
            f"PINN: {pinn_field.min():.3f}-{pinn_field.max():.3f} ppb | "
            f"NN2: {nn2_field.min():.3f}-{nn2_field.max():.3f} ppb\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def _log_error(self, current_time, error_msg):
        """Log pipeline error"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"[{timestamp}] ERROR | "
            f"Current: {current_time.strftime('%Y-%m-%d %H:%M')} | "
            f"Error: {error_msg}\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def get_statistics(self):
        """Get pipeline statistics"""
        stats = {
            'facility_csv_counts': self.csv_generator.get_facility_count(),
        }
        
        # Check continuous CSVs
        superimposed_path = CONTINUOUS_DIR / "superimposed_concentrations_timeseries.csv"
        nn2_corrected_path = CONTINUOUS_DIR / "nn2_corrected_domain_timeseries.csv"
        
        if superimposed_path.exists():
            import pandas as pd
            df = pd.read_csv(superimposed_path)
            stats['superimposed_rows'] = len(df)
            stats['unique_forecasts'] = df['forecast_timestamp'].nunique()
        
        if nn2_corrected_path.exists():
            import pandas as pd
            df = pd.read_csv(nn2_corrected_path)
            stats['nn2_corrected_rows'] = len(df)
        
        return stats


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Benzene Concentration Pipeline')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once',
                       help='Run mode: once or continuous (default: once)')
    parser.add_argument('--interval', type=int, default=15,
                       help='Update interval in minutes for continuous mode (default: 15)')
    parser.add_argument('--grid-resolution', type=int, default=100,
                       help='Spatial grid resolution (default: 100)')
    parser.add_argument('--stats', action='store_true',
                       help='Show pipeline statistics and exit')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = RealtimePipeline(grid_resolution=args.grid_resolution)
    
    if args.stats:
        # Show statistics
        print("\n" + "="*70)
        print("PIPELINE STATISTICS")
        print("="*70)
        
        stats = pipeline.get_statistics()
        
        print("\nFacility CSV Row Counts:")
        for facility, count in sorted(stats['facility_csv_counts'].items()):
            if count > 0:
                print(f"  {facility[:40]:<40} {count:5d} rows")
        
        if 'superimposed_rows' in stats:
            print(f"\nSuperimposed Concentrations: {stats['superimposed_rows']:,} rows")
            print(f"Unique Forecasts: {stats['unique_forecasts']}")
        
        if 'nn2_corrected_rows' in stats:
            print(f"NN2-Corrected Domain: {stats['nn2_corrected_rows']:,} rows")
        
        print("\n" + "="*70)
        
    elif args.mode == 'once':
        # Run once
        pipeline.run_once()
        
    elif args.mode == 'continuous':
        # Run continuously
        pipeline.run_continuous(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
