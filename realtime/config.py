#!/usr/bin/env python3
"""
Configuration for Real-Time MADIS Weather Monitoring System
Contains facility-station mappings and all constants from original pipeline.
"""

from pathlib import Path

# ============================================================================
# DIRECTORIES
# ============================================================================
BASE_DIR = Path("/Users/neevpratap/Desktop/realtime")
DATA_DIR = BASE_DIR / "data"
CONTINUOUS_DIR = DATA_DIR / "continuous"
FACILITY_DIR = CONTINUOUS_DIR / "per_facility"
PREDICTIONS_DIR = DATA_DIR / "predictions"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
CORRECTIONS_DIR = DATA_DIR / "corrections"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = DATA_DIR / "latest_madis"

# ============================================================================
# MADIS DATA SOURCE
# ============================================================================
MADIS_BASE_URL = "https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/mesonet/netCDF"
UPDATE_INTERVAL_MINUTES = 15  # Check for new data every 15 minutes
ROLLING_WINDOW_HOURS = 24     # Keep last 24 hours of data in CSVs
UNIT_CONVERSION_FACTOR = 313210039.9 # Conversion factor for Benzene Concentration to PPB (from training pipeline)

# ============================================================================
# MADIS STATIONS (from facility_best_station_hourly_completeness_2019.csv)
# ============================================================================
MADIS_STATIONS = {
    'MGPT2': {
        'name': 'Morgans Point, TX',
        'lat': 29.681699752807617,
        'lon': -94.98500061035156,
    },
    'F1563': {
        'name': 'FW1563 Houston TX US',
        'lat': 29.679500579833984,
        'lon': -95.20333099365234,
    },
    'C4814': {
        'name': 'CW4814 Crosby TX US',
        'lat': 29.948179244995117,
        'lon': -95.11087036132812,
    },
    'D1774': {
        'name': 'DW1774 Pasadena TX US',
        'lat': 29.598970413208008,
        'lon': -95.10636138916016,
    },
}

# ============================================================================
# FACILITY DATA (20 facilities from original pipeline)
# Corrected coordinates to match training data (benzene_pipeline.py)
# ============================================================================
FACILITIES = {
    'ExxonMobil Baytown Refinery': {
        'source_x': 29.7436,
        'source_y': -95.0128,
        'source_x_cartesian': 24868.31,
        'source_y_cartesian': 13369.85,
        'source_diameter': 3220.0,
        'Q_total': 67.5,
        'station_id': 'MGPT2',
        'Q_schedule_kg_s': (0.00300, 0.00160, 0.00120, 0.00080),
    },
    'Shell Deer Park Refinery': {
        'source_x': 29.7208,
        'source_y': -95.1269,
        'source_x_cartesian': 13817.66,
        'source_y_cartesian': 10836.02,
        'source_diameter': 2740.0,
        'Q_total': 42.1,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00240, 0.00140, 0.00100, 0.00070),
    },
    'Valero Houston Refinery': {
        'source_x': 29.7222,
        'source_y': -95.2539,
        'source_x_cartesian': 1517.65,
        'source_y_cartesian': 10991.6,
        'source_diameter': 752.0,
        'Q_total': 36.4,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00200, 0.00120, 0.00090, 0.00060),
    },
    'LyondellBasell Pasadena Complex': {
        'source_x': 29.7131,
        'source_y': -95.2344,
        'source_x_cartesian': 3290.01,
        'source_y_cartesian': 9980.29,
        'source_diameter': 1914.0,
        'Q_total': 32.8,
        'station_id': 'F1563', # Fallback
        'Q_schedule_kg_s': (0.00180, 0.00100, 0.00080, 0.00050),
    },
    'LyondellBasell Channelview Complex': {
        'source_x': 29.8322,
        'source_y': -95.1114,
        'source_x_cartesian': 14970.18,
        'source_y_cartesian': 23127.32,
        'source_diameter': 2515.0,
        'Q_total': 29.5,
        'station_id': 'C4814',
        'Q_schedule_kg_s': (0.00160, 0.00090, 0.00070, 0.00040),
    },
    'ExxonMobil Baytown Olefins Plant': {
        'source_x': 29.7436,
        'source_y': -95.0128,
        'source_x_cartesian': 25003.9,
        'source_y_cartesian': 14792.35,
        'source_diameter': 1050.0,
        'Q_total': 28.2,
        'station_id': 'MGPT2',
        'Q_schedule_kg_s': (0.00150, 0.00080, 0.00060, 0.00036),
    },
    'Chevron Phillips Chemical Co': {
        'source_x': 29.7303,
        'source_y': -95.1764,
        'source_x_cartesian': 8781.44,
        'source_y_cartesian': 11769.53,
        'source_diameter': 1885.0,
        'Q_total': 24.1,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00130, 0.00070, 0.00050, 0.00030),
    },
    'TPC Group': {
        'source_x': 29.7003,
        'source_y': -95.2539,
        'source_x_cartesian': 1488.59,
        'source_y_cartesian': 8591.13,
        'source_diameter': 1032.0,
        'Q_total': 21.3,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00110, 0.00060, 0.00040, 0.00026),
    },
    'INEOS Phenol': {
        'source_x': 29.7333,
        'source_y': -95.1583,
        'source_x_cartesian': 10476.32,
        'source_y_cartesian': 12102.93,
        'source_diameter': 2636.0,
        'Q_total': 18.9,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00100, 0.00050, 0.00030, 0.00024),
    },
    'Total Energies Petrochemicals': {
        'source_x': 29.7292,
        'source_y': -95.0808,
        'source_x_cartesian': 17769.16,
        'source_y_cartesian': 11613.95,
        'source_diameter': 667.0,
        'Q_total': 16.5,
        'station_id': 'MGPT2',
        'Q_schedule_kg_s': (0.00120, 0.00060, 0.00050, 0.00032),
    },
    'BASF Pasadena': {
        'source_x': 29.7292,
        'source_y': -95.1472,
        'source_x_cartesian': 11473.88,
        'source_y_cartesian': 11613.95,
        'source_diameter': 800.0,
        'Q_total': 14.8,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00090, 0.00050, 0.00036, 0.00024),
    },
    'Huntsman International': {
        'source_x': 29.7247,
        'source_y': -95.2639,
        'source_x_cartesian': 500.72,
        'source_y_cartesian': 11147.19,
        'source_diameter': 193.0,
        'Q_total': 13.2,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00100, 0.00060, 0.00045, 0.00028),
    },
    'Invista': {
        'source_x': 29.7042,
        'source_y': -95.2494,
        'source_x_cartesian': 1895.36,
        'source_y_cartesian': 8991.21,
        'source_diameter': 490.0,
        'Q_total': 11.5,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00080, 0.00045, 0.00035, 0.00020),
    },
    'Goodyear Baytown': {
        'source_x': 29.6458,
        'source_y': -95.0414,
        'source_x_cartesian': 21507.59,
        'source_y_cartesian': 2623.29,
        'source_diameter': 532.0,
        'Q_total': 9.8,
        'station_id': 'MGPT2',
        'Q_schedule_kg_s': (0.00060, 0.00030, 0.00020, 0.00012),
    },
    'LyondellBasell Bayport Polymers': {
        'source_x': 29.6256,
        'source_y': -95.0444,
        'source_x_cartesian': 21187.99,
        'source_y_cartesian': 500.65,
        'source_diameter': 2130.0,
        'Q_total': 8.5,
        'station_id': 'D1774',
        'Q_schedule_kg_s': (0.00120, 0.00060, 0.00040, 0.00030),
    },
    'INEOS PP & Gemini': {
        'source_x': 29.7242,
        'source_y': -95.0803,
        'source_x_cartesian': 17798.22,
        'source_y_cartesian': 11091.62,
        'source_diameter': 530.0,
        'Q_total': 7.2,
        'station_id': 'MGPT2',
        'Q_schedule_kg_s': (0.00090, 0.00050, 0.00036, 0.00024),
    },
    'K-Solv Channelview': {
        'source_x': 29.7697,
        'source_y': -95.0997,
        'source_x_cartesian': 16122.71,
        'source_y_cartesian': 16025.92,
        'source_diameter': 143.0,
        'Q_total': 6.5,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00070, 0.00040, 0.00028, 0.00018),
    },
    'Oxy Vinyls Deer Park': {
        'source_x': 29.7317,
        'source_y': -95.0992,
        'source_x_cartesian': 16016.17,
        'source_y_cartesian': 11925.12,
        'source_diameter': 1008.0,
        'Q_total': 5.8,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00080, 0.00045, 0.00032, 0.00022),
    },
    'ITC Deer Park': {
        'source_x': 29.7403,
        'source_y': -95.0886,
        'source_x_cartesian': 17013.73,
        'source_y_cartesian': 12847.52,
        'source_diameter': 1168.0,
        'Q_total': 4.2,
        'station_id': 'MGPT2',
        'Q_schedule_kg_s': (0.00050, 0.00025, 0.00015, 0.00010),
    },
    'Enterprise Houston Terminal': {
        'source_x': 29.7467,
        'source_y': -95.1239,
        'source_x_cartesian': 13701.44,
        'source_y_cartesian': 13525.43,
        'source_diameter': 578.0,
        'Q_total': 3.5,
        'station_id': 'F1563',
        'Q_schedule_kg_s': (0.00040, 0.00020, 0.00012, 0.00008),
    },
}

# CSV column order (excluding phi, including wind_quality)
CSV_COLUMNS = [
    't', 'x', 'y', 'source_x', 'source_y', 
    'source_x_cartesian', 'source_y_cartesian',
    'source_diameter', 'Q_total', 'wind_u', 'wind_v', 'D', 'wind_quality'
]
