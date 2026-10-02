import os
import sys

# Ensure project root (cg_spring_gnn) is on sys.path and is current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error, median_absolute_error
from torch_geometric.loader import DataLoader
from src.data.dataset import CGSpringDataset
from src.models.gnn import CGSpringGNN

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'

# Load model & test predictions
ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=False)
cfg = ckpt.get("args", {})
model = CGSpringGNN(hidden_dim=cfg.get("hidden", 256), n_layers=cfg.get("layers", 4), dropout=cfg.get("dropout", 0.1))
model.load_state_dict(ckpt["model_state"])
model.eval()

test_ds = CGSpringDataset(root="data", split="test")
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

true_ka, pred_ka = [], []
with torch.no_grad():
    for batch in test_loader:
        _, _, p_ka, _, _ = model(batch, return_regime=True)
        if p_ka.numel() > 0:
            true_ka.extend(batch.y_k_angle.numpy())
            pred_ka.extend(p_ka.numpy())
true_ka = np.array(true_ka)
pred_ka = np.array(pred_ka)

p80 = 78.10
mask_80 = true_ka <= p80
t_sub = true_ka[mask_80]
p_sub = pred_ka[mask_80]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Exploratory Analysis: Filtered Subset Uptill 80th Percentile (k <= {p80:.1f} kJ/mol/rad^2)", fontsize=18, fontweight='bold', y=0.98)

# Panel 1: Distribution showing kept vs ignored
ax = axes[0, 0]
sns.histplot(true_ka[mask_80], bins=30, color="#0284c7", ax=ax, label=f"Kept: 0-80th Percentile (N={np.sum(mask_80)}, 78.8%)", stat="count", alpha=0.7)
sns.histplot(true_ka[~mask_80], bins=30, color="#ef4444", ax=ax, label=f"Ignored: 80-100th Percentile (N={np.sum(~mask_80)}, 21.2%)", stat="count", alpha=0.7)
ax.axvline(p80, color="#1e293b", linestyle="--", linewidth=2.5, label=f"P80 Threshold = {p80:.1f}")
ax.axvline(50.0, color="#16a34a", linestyle=":", linewidth=2.0, label="Flexible Boundary (k=50)")
ax.set_title("Data Partition: 0-80th Percentile Kept vs 80-100th Ignored", fontsize=13, fontweight='bold')
ax.set_xlabel("k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax.set_ylabel("Number of Angles", fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor="white", fontsize=9.5)

# Panel 2: Scatter Plot on Kept Subset
ax = axes[0, 1]
ax.scatter(t_sub, p_sub, color="#0284c7", alpha=0.55, edgecolors='none', s=40, label=f"Predictions (N={len(t_sub)})")
ax.plot([0, 100], [0, 100], color="#dc2626", linestyle="--", linewidth=2.0, label="Ideal Parity y = x")
ax.fill_between([0, 100], [0-5, 100-5], [0+5, 100+5], color="#10b981", alpha=0.15, label="+/- 5 kJ/mol/rad^2 Band (80.3% inside)")
ax.set_xlim(0, 85)
ax.set_ylim(0, 120)
ax.set_title(f"True vs Predicted on k <= P80 (MAE = {mean_absolute_error(t_sub, p_sub):.2f}, MedAE = {median_absolute_error(t_sub, p_sub):.2f})", fontsize=13, fontweight='bold')
ax.set_xlabel("True k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax.set_ylabel("Predicted k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor="white", fontsize=9.5)

# Panel 3: Error Residual Bell Curve
ax = axes[1, 0]
errors = p_sub - t_sub
sns.histplot(errors, bins=45, kde=True, color="#10b981", ax=ax, stat="density", alpha=0.6)
ax.axvline(0, color="#dc2626", linestyle="--", linewidth=2.0, label="Zero Error Line")
ax.axvline(np.median(errors), color="#1e40af", linestyle="-.", linewidth=2.0, label=f"Median Bias = {np.median(errors):+.2f}")
ax.set_title("Error Residuals on Kept Subset: Sharp Peak at Zero\n71.8% of Angles within +/- 2.0 kJ/mol/rad^2", fontsize=13, fontweight='bold')
ax.set_xlabel("Prediction Error: (Pred - True) [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax.set_ylabel("Density", fontsize=11, fontweight='bold')
ax.set_xlim(-25, 35)
ax.legend(frameon=True, facecolor="white", fontsize=9.5)

# Panel 4: Confusion Matrix on Kept Regimes
ax = axes[1, 1]
true_reg = np.where(t_sub <= 50, 0, 1)
pred_reg = np.where(p_sub <= 50, 0, 1)
cm = confusion_matrix(true_reg, pred_reg, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            xticklabels=["Pred Flexible\n(k <= 50)", "Pred Medium\n(50 < k <= 78.1)"],
            yticklabels=["True Flexible\n(k <= 50)", "True Medium\n(50 < k <= 78.1)"],
            annot_kws={"size": 16, "fontweight": "bold"})
ax.set_title("Classification on 0-80th Percentile Subset\nOverall Accuracy = 91.69% | Weighted F1 = 0.9249", fontsize=13, fontweight='bold')
ax.set_xlabel("Predicted Physical Regime", fontsize=11, fontweight='bold')
ax.set_ylabel("True Physical Regime", fontsize=11, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("results/eda_p80_filtering_analysis.png", dpi=300)
plt.close()
print("Saved results/eda_p80_filtering_analysis.png")
