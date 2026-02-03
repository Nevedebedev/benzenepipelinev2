#!/usr/bin/env python3
"""
Utility functions for NN2 output mapping (scaled space → ppb)

This module provides functions to load and use the Gradient Boosting mapping model
that converts NN2 scaled outputs to ppb values, fixing the inverse transform issue.
"""

import pickle
from pathlib import Path
import numpy as np

# Default path to mapping model
DEFAULT_MAPPING_PATH = Path(__file__).parent / 'nn2_timefix' / 'nn2_output_to_ppb_mapping.pkl'

# Global cache for loaded model
_mapping_model_cache = None


def load_nn2_mapping_model(mapping_path=None):
    """
    Load the NN2 output to ppb mapping model.
    
    Args:
        mapping_path: Path to the mapping model pickle file. If None, uses default path.
        
    Returns:
        dict: Mapping data containing:
            - 'model': The GradientBoostingRegressor model
            - 'type': Model type ('gbr' or 'linear')
            - Other metadata
    """
    global _mapping_model_cache
    
    if _mapping_model_cache is not None:
        return _mapping_model_cache
    
    if mapping_path is None:
        mapping_path = DEFAULT_MAPPING_PATH
    
    mapping_path = Path(mapping_path)
    
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping model not found at {mapping_path}. "
            f"Run implement_nn2_output_mapping.py to generate it."
        )
    
    with open(mapping_path, 'rb') as f:
        mapping_data = pickle.load(f)
    
    _mapping_model_cache = mapping_data
    return mapping_data


def nn2_scaled_to_ppb(corrected_scaled, mapping_model=None, mapping_path=None):
    """
    Convert NN2 scaled outputs to ppb values using the mapping model.
    
    This function handles zero values correctly (only transforms non-zero values).
    
    Args:
        corrected_scaled: Array of NN2 outputs in scaled space (can be 1D or 2D)
        mapping_model: Pre-loaded mapping model. If None, will load from mapping_path.
        mapping_path: Path to mapping model pickle file. Only used if mapping_model is None.
        
    Returns:
        np.ndarray: Array of ppb values with same shape as corrected_scaled
    """
    if mapping_model is None:
        mapping_data = load_nn2_mapping_model(mapping_path)
        mapping_model = mapping_data['model']
    
    # Ensure we have a numpy array
    corrected_scaled = np.asarray(corrected_scaled)
    original_shape = corrected_scaled.shape
    
    # Flatten for processing
    corrected_scaled_flat = corrected_scaled.flatten()
    
    # Identify non-zero values (using small threshold for numerical zeros)
    nn2_output_nonzero_mask = np.abs(corrected_scaled_flat) > 1e-6
    
    # Initialize output array with zeros
    nn2_corrected = np.zeros_like(corrected_scaled_flat)
    
    # Apply mapping model only to non-zero values
    if nn2_output_nonzero_mask.any():
        nn2_corrected[nn2_output_nonzero_mask] = mapping_model.predict(
            corrected_scaled_flat[nn2_output_nonzero_mask].reshape(-1, 1)
        ).flatten()
    
    # Reshape to original shape
    return nn2_corrected.reshape(original_shape)

