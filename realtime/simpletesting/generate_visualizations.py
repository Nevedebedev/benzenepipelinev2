"""
Complete Visualization Suite for NN2 Analysis
Using January & February 2021 data only (excluding March)

Generates 9 publication-quality graphs for ISEF presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = "/Users/neevpratap/simpletesting"
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations_q1_2021")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sensor information
SENSORS = {
    'sensor_482010026': {'coords': (13972.62, 19915.57), 'name': 'Sensor 1'},
    'sensor_482010057': {'coords': (3017.18, 12334.2), 'name': 'Sensor 2'},
    'sensor_482010069': {'coords': (817.42, 9218.92), 'name': 'Sensor 3'},
    'sensor_482010617': {'coords': (27049.57, 22045.66), 'name': 'Sensor 4'},
    'sensor_482010803': {'coords': (8836.35, 15717.2), 'name': 'Sensor 5'},
    'sensor_482011015': {'coords': (18413.8, 15068.96), 'name': 'Sensor 6'},
    'sensor_482011035': {'coords': (1159.98, 12272.52), 'name': 'Sensor 7'},
    'sensor_482011039': {'coords': (13661.93, 5193.24), 'name': 'Sensor 8'},
    'sensor_482016000': {'coords': (1546.9, 6786.33), 'name': 'Sensor 9'},
}

print("="*70)
print("NN2 VISUALIZATION SUITE - Q1 2021 (Jan & Feb)")
print("="*70)

# Load data
print("\n[1/10] Loading data...")
jan_detailed = pd.read_csv(os.path.join(BASE_DIR, "results_jan_2021_hybrid.csv"))
feb_detailed = pd.read_csv(os.path.join(BASE_DIR, "results_feb_2021_hybrid.csv"))

# Combine Jan & Feb
df_combined = pd.concat([jan_detailed, feb_detailed], ignore_index=True)
df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
df_valid = df_combined.dropna(subset=['ground_truth'])

print(f"  Loaded {len(df_valid)} valid data points from Jan-Feb 2021")

# Calculate metrics
print("\n[2/10] Calculating performance metrics...")
metrics_per_sensor = {}
for sensor_id in SENSORS.keys():
    sensor_data = df_valid[df_valid['sensor_id'] == sensor_id]
    
    pinn_mae = (sensor_data['pinn_pred'] - sensor_data['ground_truth']).abs().mean()
    nn2_mae = (sensor_data['final_pred'] - sensor_data['ground_truth']).abs().mean()
    
    pinn_rmse = np.sqrt(((sensor_data['pinn_pred'] - sensor_data['ground_truth'])**2).mean())
    nn2_rmse = np.sqrt(((sensor_data['final_pred'] - sensor_data['ground_truth'])**2).mean())
    
    improvement = ((pinn_mae - nn2_mae) / pinn_mae * 100)
    
    metrics_per_sensor[sensor_id] = {
        'pinn_mae': pinn_mae,
        'nn2_mae': nn2_mae,
        'pinn_rmse': pinn_rmse,
        'nn2_rmse': nn2_rmse,
        'improvement': improvement
    }

print("  Metrics calculated for all 9 sensors")

# ============================================================================
# GRAPH 1: Training Dashboard (NOTE: Requires training logs - placeholder)
# ============================================================================
print("\n[3/10] Graph 1: Training Dashboard...")
print("  ⚠️  Skipping - requires training history logs")
print("  💡 Recommendation: Add logging to nn2.py during training")

# ============================================================================
# GRAPH 2: Spatial Generalization Map ⭐ HERO GRAPH
# ============================================================================
print("\n[4/10] Graph 2: Spatial Generalization Map...")

fig, ax = plt.subplots(figsize=(12, 10))

# Plot sensors with color based on performance
for sensor_id, info in SENSORS.items():
    x, y = info['coords']
    improvement = metrics_per_sensor[sensor_id]['improvement']
    
    # Color: green if better, red if worse
    color = plt.cm.RdYlGn((improvement + 10) / 110)  # Scale to 0-1
    size = 500
    
    ax.scatter(x, y, s=size, c=[color], edgecolors='black', linewidth=2, 
               alpha=0.8, zorder=3)
    
    # Label
    ax.text(x, y-1000, f"{improvement:.1f}%", 
            ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel('X Coordinate (meters)', fontsize=14)
ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
ax.set_title('Spatial Map: NN2 Improvement by Sensor Location\n(Jan-Feb 2021)', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                            norm=plt.Normalize(vmin=-10, vmax=100))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Improvement (%)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_spatial_generalization_map.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_spatial_generalization_map.png")

# ============================================================================
# GRAPH 3: Before/After Time Series ⭐⭐⭐ EASIEST TO UNDERSTAND
# ============================================================================
print("\n[5/10] Graph 3: Before/After Time Series...")

# Pick one representative sensor for clarity
sensor_to_plot = 'sensor_482010026'
sensor_subset = df_valid[df_valid['sensor_id'] == sensor_to_plot].copy()
sensor_subset = sensor_subset.sort_values('timestamp')

# Sample every 10th point for cleaner visualization
sensor_subset = sensor_subset.iloc[::10]

fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(sensor_subset['timestamp'], sensor_subset['ground_truth'], 
        'k-', linewidth=2, label='Actual (Ground Truth)', alpha=0.8)
ax.plot(sensor_subset['timestamp'], sensor_subset['pinn_pred'], 
        'r--', linewidth=1.5, label='PINN Prediction', alpha=0.6)
ax.plot(sensor_subset['timestamp'], sensor_subset['final_pred'], 
        'b-', linewidth=2, label='NN2 Hybrid Prediction', alpha=0.8)

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Benzene Concentration (ppb)', fontsize=14)
ax.set_title(f'Time Series Comparison - {SENSORS[sensor_to_plot]["name"]}\n(Jan-Feb 2021)', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_before_after_timeseries.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_before_after_timeseries.png")

# ============================================================================
# GRAPH 4: Performance Heatmap
# ============================================================================
print("\n[6/10] Graph 4: Performance Heatmap...")

# Create metrics table
metrics_df = pd.DataFrame.from_dict(metrics_per_sensor, orient='index')
metrics_df.index = [SENSORS[sid]['name'] for sid in metrics_df.index]

fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap data (normalized for coloring)
heatmap_data = metrics_df[['pinn_mae', 'nn2_mae', 'improvement']].copy()

# Plot
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Value'})

ax.set_title('Performance Metrics Heatmap\n(Jan-Feb 2021)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Metric', fontsize=14)
ax.set_ylabel('Sensor', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_performance_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 04_performance_heatmap.png")

# ============================================================================
# GRAPH 5: Error Distribution Violin Plots
# ============================================================================
print("\n[7/10] Graph 5: Error Distribution Violin Plots...")

# Calculate errors
df_valid['pinn_error'] = (df_valid['pinn_pred'] - df_valid['ground_truth']).abs()
df_valid['nn2_error'] = (df_valid['final_pred'] - df_valid['ground_truth']).abs()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PINN errors
axes[0].violinplot([df_valid['pinn_error'].dropna()], positions=[0], 
                    showmeans=True, showmedians=True)
axes[0].set_ylabel('Absolute Error (ppb)', fontsize=14)
axes[0].set_title('PINN Error Distribution', fontsize=14, fontweight='bold')
axes[0].set_xticks([0])
axes[0].set_xticklabels(['PINN'])
axes[0].grid(True, alpha=0.3, axis='y')

# NN2 errors
axes[1].violinplot([df_valid['nn2_error'].dropna()], positions=[0], 
                    showmeans=True, showmedians=True)
axes[1].set_ylabel('Absolute Error (ppb)', fontsize=14)
axes[1].set_title('NN2 Error Distribution', fontsize=14, fontweight='bold')
axes[1].set_xticks([0])
axes[1].set_xticklabels(['NN2'])
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Error Distribution Comparison (Jan-Feb 2021)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_error_distribution_violins.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_error_distribution_violins.png")

# ============================================================================
# GRAPH 6: Correction Field Map ⭐⭐⭐ MOST BEAUTIFUL
# ============================================================================
print("\n[8/10] Graph 6: Correction Field Map...")

# Calculate average corrections per sensor
corrections_per_sensor = {}
for sensor_id in SENSORS.keys():
    sensor_data = df_valid[df_valid['sensor_id'] == sensor_id]
    avg_correction = (sensor_data['final_pred'] - sensor_data['pinn_pred']).mean()
    corrections_per_sensor[sensor_id] = avg_correction

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: PINN Predictions
for sensor_id, info in SENSORS.items():
    x, y = info['coords']
    avg_pinn = df_valid[df_valid['sensor_id'] == sensor_id]['pinn_pred'].mean()
    axes[0].scatter(x, y, s=300, c='red', edgecolors='black', linewidth=2, alpha=0.7)
    axes[0].text(x, y-1000, f"{avg_pinn:.1f}", ha='center', fontsize=9)

axes[0].set_title('PINN Predictions', fontsize=14, fontweight='bold')
axes[0].set_xlabel('X (meters)', fontsize=12)
axes[0].set_ylabel('Y (meters)', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Panel 2: NN2 Corrections
for sensor_id, info in SENSORS.items():
    x, y = info['coords']
    correction = corrections_per_sensor[sensor_id]
    color = 'blue' if correction > 0 else 'red'
    axes[1].scatter(x, y, s=abs(correction)*100, c=color, edgecolors='black', 
                    linewidth=2, alpha=0.7)
    axes[1].text(x, y-1000, f"{correction:+.1f}", ha='center', fontsize=9)

axes[1].set_title('NN2 Corrections', fontsize=14, fontweight='bold')
axes[1].set_xlabel('X (meters)', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

# Panel 3: Final Predictions
for sensor_id, info in SENSORS.items():
    x, y = info['coords']
    avg_final = df_valid[df_valid['sensor_id'] == sensor_id]['final_pred'].mean()
    axes[2].scatter(x, y, s=300, c='green', edgecolors='black', linewidth=2, alpha=0.7)
    axes[2].text(x, y-1000, f"{avg_final:.1f}", ha='center', fontsize=9)

axes[2].set_title('Final NN2 Predictions', fontsize=14, fontweight='bold')
axes[2].set_xlabel('X (meters)', fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].set_aspect('equal')

fig.suptitle('Correction Field Map (Jan-Feb 2021 Averages)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_correction_field_map.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 06_correction_field_map.png")

# ============================================================================
# GRAPH 7: Cross-Validation Summary (Placeholder - needs CV setup)
# ============================================================================
print("\n[9/10] Graph 7: Cross-Validation Summary...")
print("  ⚠️  Skipping - requires leave-one-out cross-validation")
print("  💡 Recommendation: Run separate CV script with held-out sensors")

# ============================================================================
# GRAPH 8: Feature Importance (Placeholder - needs SHAP or permutation)
# ============================================================================
print("\n[10/10] Graph 8: Feature Importance...")
print("  ⚠️  Skipping - requires feature importance analysis")
print("  💡 Recommendation: Use SHAP values or permutation importance")

# ============================================================================
# GRAPH 9: Residual Analysis
# ============================================================================
print("\n[11/10] Graph 9: Residual Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: PINN predictions vs actual
axes[0, 0].scatter(df_valid['ground_truth'], df_valid['pinn_pred'], 
                    alpha=0.3, s=10, c='red')
axes[0, 0].plot([0, df_valid['ground_truth'].max()], [0, df_valid['ground_truth'].max()], 
                 'k--', linewidth=2)
axes[0, 0].set_xlabel('Actual (ppb)', fontsize=12)
axes[0, 0].set_ylabel('PINN Predicted (ppb)', fontsize=12)
axes[0, 0].set_title('PINN: Predicted vs Actual', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Top-right: NN2 predictions vs actual
axes[0, 1].scatter(df_valid['ground_truth'], df_valid['final_pred'], 
                    alpha=0.3, s=10, c='blue')
axes[0, 1].plot([0, df_valid['ground_truth'].max()], [0, df_valid['ground_truth'].max()], 
                 'k--', linewidth=2)
axes[0, 1].set_xlabel('Actual (ppb)', fontsize=12)
axes[0, 1].set_ylabel('NN2 Predicted (ppb)', fontsize=12)
axes[0, 1].set_title('NN2: Predicted vs Actual', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Bottom-left: PINN residuals
pinn_residuals = df_valid['pinn_pred'] - df_valid['ground_truth']
axes[1, 0].scatter(df_valid['ground_truth'], pinn_residuals, 
                    alpha=0.3, s=10, c='red')
axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Actual (ppb)', fontsize=12)
axes[1, 0].set_ylabel('Residual (ppb)', fontsize=12)
axes[1, 0].set_title('PINN: Residuals', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Bottom-right: NN2 residuals
nn2_residuals = df_valid['final_pred'] - df_valid['ground_truth']
axes[1, 1].scatter(df_valid['ground_truth'], nn2_residuals, 
                    alpha=0.3, s=10, c='blue')
axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Actual (ppb)', fontsize=12)
axes[1, 1].set_ylabel('Residual (ppb)', fontsize=12)
axes[1, 1].set_title('NN2: Residuals', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Residual Analysis (Jan-Feb 2021)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_residual_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 09_residual_analysis.png")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION SUITE COMPLETE")
print("="*70)
print(f"Output directory: {OUTPUT_DIR}")
print("\nGenerated graphs:")
print("  ✓ Graph 2: Spatial Generalization Map")
print("  ✓ Graph 3: Before/After Time Series")
print("  ✓ Graph 4: Performance Heatmap")
print("  ✓ Graph 5: Error Distribution Violins")
print("  ✓ Graph 6: Correction Field Map")
print("  ✓ Graph 9: Residual Analysis")
print("\nSkipped (require additional data):")
print("  ⚠️  Graph 1: Training Dashboard (needs training logs)")
print("  ⚠️  Graph 7: Cross-Validation (needs CV setup)")
print("  ⚠️  Graph 8: Feature Importance (needs SHAP analysis)")
print("="*70)

# Save summary statistics
summary_stats = pd.DataFrame.from_dict(metrics_per_sensor, orient='index')
summary_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
print(f"✓ Saved summary statistics to: summary_statistics.csv")
