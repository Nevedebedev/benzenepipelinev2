
import matplotlib.pyplot as plt
import os

# Results
metrics = ['PINN Only', 'Hybrid (NN2 v2)']
mae_values = [55.67, 29.95]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, mae_values, color=['#e74c3c', '#2ecc71'])
plt.ylabel('MAE (ppb)', fontsize=12)
plt.title('Final Validation Performance (Jan - Mar 2019)', fontsize=14, pad=15)
plt.ylim(0, 70)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f} ppb', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path = "/Users/neevpratap/simpletesting/logs/mae_summary_v2.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
print(f"Saved plot to {save_path}")
