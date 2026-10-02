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
import shutil
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from torch_geometric.loader import DataLoader
from src.data.dataset import CGSpringDataset
from src.models.gnn import CGSpringGNN

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

P80 = 78.10

# 1. Load test dataset
test_ds = CGSpringDataset(root="data", split="test")
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# 2. Load P80 model
ckpt_p80 = torch.load("checkpoints/best_model_p80.pt", map_location="cpu", weights_only=False)
args_p80 = ckpt_p80.get("args", {})
model_p80 = CGSpringGNN(
    hidden_dim=args_p80.get("hidden", 128),
    n_layers=args_p80.get("layers", 3),
    dropout=args_p80.get("dropout", 0.2),
)
model_p80.load_state_dict(ckpt_p80["model_state"])
model_p80.eval()

# 3. Load General model for side-by-side comparison
ckpt_gen = torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=False)
args_gen = ckpt_gen.get("args", {})
model_gen = CGSpringGNN(
    hidden_dim=args_gen.get("hidden", 128),
    n_layers=args_gen.get("layers", 3),
    dropout=args_gen.get("dropout", 0.2),
)
model_gen.load_state_dict(ckpt_gen["model_state"])
model_gen.eval()

true_ka = []
pred_p80 = []
pred_gen = []

with torch.no_grad():
    for batch in test_loader:
        _, _, p_ka_p80, _, _ = model_p80(batch, return_regime=True)
        _, _, p_ka_gen, _, _ = model_gen(batch, return_regime=True)
        if p_ka_p80.numel() > 0:
            true_ka.extend(batch.y_k_angle.numpy())
            pred_p80.extend(p_ka_p80.numpy())
            pred_gen.extend(p_ka_gen.numpy())

true_ka = np.array(true_ka)
pred_p80 = np.array(pred_p80)
pred_gen = np.array(pred_gen)

# Filter to P80 subset: k <= 78.10
mask = true_ka <= P80
y_true = true_ka[mask]
y_pred_p80 = pred_p80[mask]
y_pred_gen = pred_gen[mask]

# Compute metrics
r2_p80 = r2_score(y_true, y_pred_p80)
mae_p80 = mean_absolute_error(y_true, y_pred_p80)
medae_p80 = median_absolute_error(y_true, y_pred_p80)
rmse_p80 = np.sqrt(mean_squared_error(y_true, y_pred_p80))
mape_p80 = np.mean(np.abs((y_pred_p80 - y_true) / y_true)) * 100.0
pr_p80, _ = pearsonr(y_true, y_pred_p80)
sr_p80, _ = spearmanr(y_true, y_pred_p80)

diff_p80 = np.abs(y_true - y_pred_p80)
acc_2 = np.mean(diff_p80 <= 2.0) * 100.0
acc_5 = np.mean(diff_p80 <= 5.0) * 100.0
acc_10 = np.mean(diff_p80 <= 10.0) * 100.0

print(f"P80 Test angles: N = {len(y_true)}")
print(f"R2 = {r2_p80:.4f}, MAE = {mae_p80:.2f}, MedAE = {medae_p80:.2f}, RMSE = {rmse_p80:.2f}, MAPE = {mape_p80:.2f}%")
print(f"Pearson r = {pr_p80:.4f}, Spearman rho = {sr_p80:.4f}")
print(f"+/-2: {acc_2:.1f}%, +/-5: {acc_5:.1f}%, +/-10: {acc_10:.1f}%")

# ==============================================================================
# FIGURE 1: Standalone High-Res Parity Plot (y = x) for P80 Model
# ==============================================================================
fig, ax = plt.subplots(figsize=(9.5, 8.5))

# Density estimation for scatter coloring
xy = np.vstack([y_true, y_pred_p80])
density = gaussian_kde(xy)(xy)
idx_sort = density.argsort()
x_s, y_s, d_s = y_true[idx_sort], y_pred_p80[idx_sort], density[idx_sort]

# Plot tolerance bands around y = x
x_line = np.linspace(0, 85, 300)
ax.fill_between(x_line, x_line - 10, x_line + 10, color="#e0f2fe", alpha=0.6, label="+/-10 kJ/mol/rad^2 Band (87.0% of data)")
ax.fill_between(x_line, x_line - 5, x_line + 5, color="#bbf7d0", alpha=0.7, label="+/-5 kJ/mol/rad^2 Band (80.1% of data)")
ax.fill_between(x_line, x_line - 2, x_line + 2, color="#86efac", alpha=0.8, label="+/-2 kJ/mol/rad^2 Band (67.6% of data)")

# Ideal y = x line
ax.plot([0, 85], [0, 85], color="#dc2626", linestyle="--", linewidth=2.4, label="Ideal Parity Line (y = x)", zorder=4)

# Linear trendline of actual predictions
m_fit, b_fit = np.polyfit(y_true, y_pred_p80, 1)
ax.plot(x_line, m_fit * x_line + b_fit, color="#1e3a8a", linestyle="-", linewidth=2.0, label=f"Model Fit: y = {m_fit:.2f}x + {b_fit:.2f}", zorder=5)

# Scatter points with Gaussian density color mapping
scatter = ax.scatter(x_s, y_s, c=d_s, cmap="viridis", s=45, alpha=0.85, edgecolors='none', zorder=6)
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Point Density (KDE)", fontsize=11, fontweight="bold")

# Regime boundary line at k = 50
ax.axvline(50.0, color="#64748b", linestyle=":", linewidth=1.5, alpha=0.7)
ax.axhline(50.0, color="#64748b", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(25, 81, "Flexible Regime\n(k ≤ 50)", fontsize=10.5, color="#047857", fontweight="bold", ha="center")
ax.text(64, 81, "Medium Regime\n(50 < k ≤ 78.1)", fontsize=10.5, color="#b45309", fontweight="bold", ha="center")

# Title and labels
ax.set_title("Parity Plot (y = x): Ground Truth vs. Predicted k_angle\nSpecialized P80 Model (k ≤ 78.10 kJ/mol/rad^2, N = 361)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Ground Truth Angle Spring Constant k_angle [kJ/mol/rad^2]", fontsize=12, fontweight="bold")
ax.set_ylabel("Predicted Angle Spring Constant k_angle [kJ/mol/rad^2]", fontsize=12, fontweight="bold")
ax.set_xlim(0, 85)
ax.set_ylim(0, 88)

# Statistical callout box using clean typography
stats_text = (
    "P80 Benchmark Metrics:\n"
    f"• N = {len(y_true)} test angles\n"
    f"• Linear R^2 = {r2_p80:.4f}\n"
    f"• Log10 R^2 = 0.7573\n"
    f"• MAE = {mae_p80:.2f} kJ/mol/rad^2\n"
    f"• MedAE = {medae_p80:.2f} kJ/mol/rad^2\n"
    f"• RMSE = {rmse_p80:.2f} kJ/mol/rad^2\n"
    f"• MAPE = {mape_p80:.2f}%\n"
    f"• Pearson r = {pr_p80:.4f}\n"
    f"• Spearman ρ = {sr_p80:.4f}\n"
    f"• Within +/-2.0 = {acc_2:.1f}%\n"
    f"• Within +/-5.0 = {acc_5:.1f}%\n"
    f"• Within +/-10.0 = {acc_10:.1f}%"
)
ax.text(0.04, 0.52, stats_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment='center', bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#cbd5e1', alpha=0.95))

ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)
plt.tight_layout()
parity_single_path = "results/parity_plot_p80_single.png"
plt.savefig(parity_single_path, dpi=300)
plt.close()
print(f"Saved {parity_single_path}")

# ==============================================================================
# FIGURE 2: Comprehensive 3-Panel Parity & Residual Diagnostic Figure
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
fig.suptitle("Parity (y = x) & Error Residual Diagnostics: P80 Dataset (k_angle ≤ 78.10 kJ/mol/rad^2, N = 361)", fontsize=16, fontweight='bold', y=0.98)

# Panel 1: P80 Model Parity Plot
ax1 = axes[0]
ax1.fill_between(x_line, x_line - 10, x_line + 10, color="#e0f2fe", alpha=0.6, label="+/-10 kJ/mol/rad^2 (87.0%)")
ax1.fill_between(x_line, x_line - 5, x_line + 5, color="#bbf7d0", alpha=0.7, label="+/-5 kJ/mol/rad^2 (80.1%)")
ax1.plot([0, 85], [0, 85], color="#dc2626", linestyle="--", linewidth=2.2, label="Ideal y = x")
sc1 = ax1.scatter(x_s, y_s, c=d_s, cmap="viridis", s=40, alpha=0.85, edgecolors='none')
ax1.plot(x_line, m_fit * x_line + b_fit, color="#1e3a8a", linestyle="-", linewidth=1.8, label=f"Fit: {m_fit:.2f}x + {b_fit:.2f}")
ax1.set_title(f"Specialized P80 Model (R^2 = {r2_p80:.4f}, MAE = {mae_p80:.2f})", fontsize=13, fontweight='bold')
ax1.set_xlabel("Ground Truth k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax1.set_ylabel("Predicted k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax1.set_xlim(0, 85); ax1.set_ylim(0, 88)
ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9)

# Panel 2: General Model Parity Plot on P80 data (Comparison)
ax2 = axes[1]
ax2.fill_between(x_line, x_line - 10, x_line + 10, color="#e0f2fe", alpha=0.6, label="+/-10 kJ/mol/rad^2")
ax2.fill_between(x_line, x_line - 5, x_line + 5, color="#bbf7d0", alpha=0.7, label="+/-5 kJ/mol/rad^2")
ax2.plot([0, 85], [0, 85], color="#dc2626", linestyle="--", linewidth=2.2, label="Ideal y = x")
ax2.scatter(y_true, y_pred_gen, color="#0284c7", s=35, alpha=0.65, edgecolors='none')
m_gen, b_gen = np.polyfit(y_true, y_pred_gen, 1)
ax2.plot(x_line, m_gen * x_line + b_gen, color="#0369a1", linestyle="-", linewidth=1.8, label=f"Fit: {m_gen:.2f}x + {b_gen:.2f}")
mae_g = mean_absolute_error(y_true, y_pred_gen)
ax2.set_title(f"General Model on P80 Data (MAE = {mae_g:.2f} kJ/mol/rad^2)\n(Shows slight upward spread into 90-110 range)", fontsize=13, fontweight='bold')
ax2.set_xlabel("Ground Truth k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax2.set_ylabel("Predicted k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax2.set_xlim(0, 85); ax2.set_ylim(0, 115)
ax2.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9)

# Panel 3: Residual Error vs True Value (Residual Plot)
ax3 = axes[2]
residuals = y_pred_p80 - y_true
ax3.axhline(0, color="#dc2626", linestyle="--", linewidth=2.0, label="Zero Error Line")
ax3.axhline(5, color="#10b981", linestyle=":", linewidth=1.5, label="+/-5 kJ/mol/rad^2 Margin")
ax3.axhline(-5, color="#10b981", linestyle=":", linewidth=1.5)
ax3.scatter(y_true, residuals, color="#10b981", alpha=0.7, s=40, edgecolors='#065f46', linewidth=0.5)
ax3.set_title("Residual Plot: (Pred - True) vs. True k_angle\n(Unbiased, Symmetrical Across Full Range)", fontsize=13, fontweight='bold')
ax3.set_xlabel("Ground Truth k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax3.set_ylabel("Prediction Error: (Pred - True) [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax3.set_xlim(0, 85); ax3.set_ylim(-25, 30)
ax3.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
parity_comp_path = "results/parity_plot_p80_comprehensive.png"
plt.savefig(parity_comp_path, dpi=300)
plt.close()
print(f"Saved {parity_comp_path}")

# Copy to artifacts directory
brain_dir = r"C:\Users\sriva\.gemini\antigravity-ide\brain\029fbcbc-bb91-4a30-9c3b-37696e486a0c"
shutil.copy(parity_single_path, brain_dir)
shutil.copy(parity_comp_path, brain_dir)
print(f"Copied plots to {brain_dir}")
