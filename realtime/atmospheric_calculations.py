#!/usr/bin/env python3
"""
Atmospheric Calculations for Gaussian Plume Modeling
Reused from build_training_data.py - exact same formulas.
"""

import numpy as np
import pandas as pd


def wind_components_from_speed_direction(wind_speed, wind_dir):
    """
    Convert wind speed (m/s) and direction (degrees) to u, v components
    
    Meteorological convention:
    - wind_dir: direction wind is coming FROM (0° = North, 90° = East)
    - u: east-west component (positive = eastward wind)
    - v: north-south component (positive = northward wind)
    
    To convert to where wind is going TO, add 180°
    """
    direction_rad = np.radians(wind_dir + 180)
    
    wind_u = wind_speed * np.sin(direction_rad)
    wind_v = wind_speed * np.cos(direction_rad)
    
    return wind_u, wind_v


def calculate_stability_class(wind_speed, temperature, timestamp, solar_radiation=None):
    """
    Determine atmospheric stability class (A, B, C, D, E, F)
    
    A = Very unstable
    B = Unstable
    C = Slightly unstable
    D = Neutral
    E = Slightly stable
    F = Stable
    """
    hour = timestamp.hour
    is_day = 6 <= hour <= 18
    
    if is_day:
        # Strong insolation (strong sun)
        if (solar_radiation is not None and not np.isnan(solar_radiation) and solar_radiation > 600) or \
           (solar_radiation is None or np.isnan(solar_radiation)) and 10 <= hour <= 14:
            if wind_speed < 2: 
                return 'A'
            elif wind_speed < 3: 
                return 'A-B'
            elif wind_speed < 5: 
                return 'B'
            elif wind_speed < 6: 
                return 'C'
            else: 
                return 'D'
        
        # Moderate insolation
        elif (solar_radiation is not None and not np.isnan(solar_radiation) and solar_radiation > 300) or \
             ((solar_radiation is None or np.isnan(solar_radiation)) and (8 <= hour <= 10 or 14 <= hour <= 16)):
            if wind_speed < 2: 
                return 'B'
            elif wind_speed < 3: 
                return 'B-C'
            elif wind_speed < 5: 
                return 'C'
            else: 
                return 'D'
        
        # Slight insolation (cloudy)
        else:
            if wind_speed < 5: 
                return 'C'
            else: 
                return 'D'
    
    else:  # Night time
        if wind_speed < 2: 
            return 'F'
        elif wind_speed < 3: 
            return 'E'
        else: 
            return 'D'


def calculate_diffusion_coefficient(wind_speed, stability_class):
    """
    Calculate horizontal diffusion coefficient D (m²/s)
    
    Formula based on:
    1. Friction velocity (u_star)
    2. Monin-Obukhov length (L)
    3. Mixing height (h_mix)
    4. Stability class
    
    UNCAPPED - allows physically realistic high values during unstable conditions
    """
    
    # Surface roughness length (typical for urban/industrial area)
    z0 = 0.03  # meters
    
    # Step 1: Calculate friction velocity
    # Avoid log(0) by ensuring wind_speed > 0
    if wind_speed <= 0:
        wind_speed = 0.1
    u_star = (0.41 * wind_speed) / np.log(10.0 / z0)
    
    # Step 2: Get stability parameters based on class
    L_values = {
        'A': -10.0,      # Very unstable
        'A-B': -17.5,    # Between A and B
        'B': -25.0,      # Unstable
        'B-C': -37.5,    # Between B and C
        'C': -50.0,      # Slightly unstable
        'D': 1e6,        # Neutral (very large L)
        'E': 50.0,       # Slightly stable
        'F': 20.0        # Stable
    }
    
    h_mix_values = {
        'A': 1500.0,     # Very unstable (high mixing)
        'A-B': 1350.0,
        'B': 1200.0,     # Unstable
        'B-C': 1050.0,
        'C': 900.0,      # Slightly unstable
        'D': 800.0,      # Neutral
        'E': 400.0,      # Slightly stable
        'F': 200.0       # Stable (low mixing)
    }
    
    L = L_values.get(stability_class, 1e6)
    h_mix = h_mix_values.get(stability_class, 800.0)
    
    # Step 3: Calculate diffusion coefficient
    if L < 0:  # Unstable conditions
        w_star = u_star * (h_mix / abs(L))**(1/3)
        D_horizontal = 0.1 * w_star * h_mix
    else:  # Neutral or Stable conditions
        z_char = 0.1 * h_mix
        D_horizontal = 0.4 * u_star * z_char
    
    # Step 4: Apply physical limits (minimum only - no artificial ceiling)
    D_horizontal = max(D_horizontal, 1.0)  # Only minimum bound, no maximum
    
    return D_horizontal


def q_for_timestamp(timestamp, rates):
    """
    Calculate time-varying emission rate based on schedule
    
    Args:
        timestamp: datetime object
        rates: tuple of (weekday_peak, weekday_base, weekend_peak, weekend_base) in kg/s
    
    Returns:
        Emission rate in kg/s
    """
    weekday_peak, weekday_base, weekend_peak, weekend_base = rates
    
    is_weekend = timestamp.weekday() >= 5  # Saturday=5, Sunday=6
    hour = timestamp.hour
    is_peak = (hour >= 6) and (hour < 18)  # 6 AM to 6 PM
    
    if not is_weekend and is_peak:
        return weekday_peak
    elif not is_weekend and not is_peak:
        return weekday_base
    elif is_weekend and is_peak:
        return weekend_peak
    else:  # weekend base
        return weekend_base
