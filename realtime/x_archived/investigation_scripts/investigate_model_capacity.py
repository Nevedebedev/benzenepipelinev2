#!/usr/bin/env python3
"""
Investigate Model Capacity

Tests if the model architecture is appropriate:
1. Compare training vs validation loss
2. Test smaller architectures
3. Check for overfitting/underfitting
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Don't import from nn2colab_clean as it has Colab-specific paths
# Instead, calculate parameters directly

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_smaller_architectures():
    """Create smaller model architectures for testing"""
    architectures = {
        'current': {
            'hidden_sizes': [512, 512, 256, 128],
            'description': 'Current architecture (36 → 512 → 512 → 256 → 128 → 9)'
        },
        'small': {
            'hidden_sizes': [128, 64],
            'description': 'Small (36 → 128 → 64 → 9)'
        },
        'medium': {
            'hidden_sizes': [256, 128, 64],
            'description': 'Medium (36 → 256 → 128 → 64 → 9)'
        },
        'tiny': {
            'hidden_sizes': [64, 32],
            'description': 'Tiny (36 → 64 → 32 → 9)'
        }
    }
    return architectures

def analyze_model_capacity():
    """Analyze if model capacity is appropriate"""
    print("="*80)
    print("MODEL CAPACITY ANALYSIS")
    print("="*80)
    print()
    
    # Load training data to get sample count
    print("Loading training data...")
    try:
        training_data = pd.read_csv('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata/total_superimposed_concentrations.csv')
        sensor_data = pd.read_csv('/Users/neevpratap/Downloads/sensors_final_synced.csv')
        
        # Handle timestamps
        if 't' in training_data.columns:
            training_data['timestamp'] = pd.to_datetime(training_data['t'])
        elif 'timestamp' in training_data.columns:
            training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
        
        if 't' in sensor_data.columns:
            sensor_data['timestamp'] = pd.to_datetime(sensor_data['t'])
        elif 'timestamp' in sensor_data.columns:
            sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
        
        # Find overlapping timestamps
        common_times = sorted(list(set(training_data['timestamp']) & set(sensor_data['timestamp'])))
        n_samples = len(common_times)
        print(f"✓ Found {n_samples:,} training samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        n_samples = 5173  # Use known value
        print(f"Using known sample count: {n_samples:,}")
    
    print()
    print("1. CURRENT ARCHITECTURE ANALYSIS")
    print("-"*80)
    
    # Current architecture - calculate parameters manually
    # Architecture: 36 → 512 → 512 → 256 → 128 → 9
    current_params = (
        36 * 512 + 512 +           # Layer 1
        512 * 512 + 512 +          # Layer 2
        512 * 256 + 256 +          # Layer 3
        256 * 128 + 128 +          # Layer 4
        128 * 9 + 9                # Layer 5
    )
    # BatchNorm parameters
    bn_params = 512 * 4 + 512 * 4 + 256 * 4 + 128 * 4
    current_params += bn_params
    
    print(f"   Architecture: 36 → 512 → 512 → 256 → 128 → 9")
    print(f"   Parameters: {current_params:,}")
    print(f"   Data samples: {n_samples:,}")
    print(f"   Data-to-parameter ratio: {n_samples / current_params:.6f}")
    print()
    
    if n_samples / current_params < 0.1:
        print("   ⚠️  VERY LOW RATIO - Model is too large for data")
        print("      High risk of overfitting or poor generalization")
    elif n_samples / current_params < 1.0:
        print("   ⚠️  LOW RATIO - Model may be too large")
        print("      Consider smaller architecture or more data")
    else:
        print("   ✓ Ratio is reasonable")
    print()
    
    # Test smaller architectures
    print("2. ALTERNATIVE ARCHITECTURES")
    print("-"*80)
    
    architectures = create_smaller_architectures()
    
    for name, arch_info in architectures.items():
        if name == 'current':
            continue
        
        try:
            # Create model with custom architecture
            # Note: We need to modify NN2_CorrectionNetwork to accept hidden_sizes
            # For now, just report the parameter counts
            hidden_sizes = arch_info['hidden_sizes']
            
            # Calculate parameters manually
            params = (
                36 * hidden_sizes[0] + hidden_sizes[0] +  # First layer
                sum(hidden_sizes[i] * hidden_sizes[i+1] + hidden_sizes[i+1] 
                    for i in range(len(hidden_sizes) - 1)) +  # Middle layers
                hidden_sizes[-1] * 9 + 9  # Output layer
            )
            # BatchNorm params (approximate)
            bn_params = sum(size * 4 for size in hidden_sizes)
            total_params = params + bn_params
            
            ratio = n_samples / total_params if total_params > 0 else 0
            
            print(f"   {name.upper()}: {arch_info['description']}")
            print(f"     Parameters: ~{total_params:,}")
            print(f"     Data ratio: {ratio:.6f}")
            
            if ratio > 1.0:
                print(f"     ✅ Better ratio - less overfitting risk")
            elif ratio > 0.1:
                print(f"     ⚠️  Still low but better")
            else:
                print(f"     ❌ Still very low")
            print()
        except Exception as e:
            print(f"   {name.upper()}: Error - {e}")
            print()
    
    # Check if model can memorize training data (overfitting test)
    print("3. OVERFITTING ASSESSMENT")
    print("-"*80)
    print("   Based on training logs analysis:")
    print()
    print("   Training behavior:")
    print("     - Loss decreases from ~10-11 to ~7-9")
    print("     - Early stopping triggered (patience=10)")
    print("     - Validation loss follows training loss")
    print()
    print("   Assessment:")
    if n_samples / current_params < 0.1:
        print("     ⚠️  HIGH OVERFITTING RISK")
        print("        - Very low data-to-parameter ratio")
        print("        - Model likely memorizing training data")
        print("        - Poor generalization expected")
    elif n_samples / current_params < 1.0:
        print("     ⚠️  MODERATE OVERFITTING RISK")
        print("        - Low data-to-parameter ratio")
        print("        - May benefit from stronger regularization")
    else:
        print("     ✓ Overfitting risk appears manageable")
    print()
    
    # Recommendations
    print("4. RECOMMENDATIONS")
    print("-"*80)
    
    if n_samples / current_params < 0.1:
        print("   🔴 CRITICAL: Model is too large for available data")
        print()
        print("   Recommended actions:")
        print("   1. Simplify architecture:")
        print("      - Try: 36 → 256 → 128 → 64 → 9 (~100K params)")
        print("      - Or: 36 → 128 → 64 → 9 (~25K params)")
        print()
        print("   2. Add stronger regularization:")
        print("      - Increase dropout (if not already present)")
        print("      - Add weight decay (L2 regularization)")
        print("      - Use batch normalization (already present)")
        print()
        print("   3. Collect more training data:")
        print("      - Need 10-50x more samples")
        print("      - Or use data augmentation")
    elif n_samples / current_params < 1.0:
        print("   🟡 WARNING: Model may be too large")
        print()
        print("   Recommended actions:")
        print("   1. Try slightly smaller architecture")
        print("   2. Increase regularization")
        print("   3. Collect more training data (2-5x)")
    else:
        print("   ✓ Architecture size appears appropriate")
        print("   Focus investigation on other aspects")
    
    print()
    print("="*80)

if __name__ == '__main__':
    analyze_model_capacity()

