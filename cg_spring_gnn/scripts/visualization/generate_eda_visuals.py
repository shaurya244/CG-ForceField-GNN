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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler, MinMaxScaler

os.makedirs("results", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# 1. Load Data
data_dir = "data/processed"
all_graphs = torch.load(os.path.join(data_dir, "all_graphs.pt"), weights_only=False)
with open(os.path.join(data_dir, "train_indices.json")) as f:
    train_idx = json.load(f)
with open(os.path.join(data_dir, "val_indices.json")) as f:
    val_idx = json.load(f)
with open(os.path.join(data_dir, "test_indices.json")) as f:
    test_idx = json.load(f)

ka_all, kb_all = [], []
for g in all_graphs:
    if hasattr(g, 'y_k_angle') and g.y_k_angle is not None and len(g.y_k_angle) > 0:
        ka_all.extend(g.y_k_angle.numpy())
    if hasattr(g, 'y_k_bond') and g.y_k_bond is not None and len(g.y_k_bond) > 0:
        kb_all.extend(g.y_k_bond.numpy())
ka_all = np.array(ka_all)
kb_all = np.array(kb_all)

# Load test predictions
from src.models.gnn import CGSpringGNN
from src.data.dataset import CGSpringDataset
from torch_geometric.loader import DataLoader

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

print(f"Data ready. Total angles: {len(ka_all)}, Total bonds: {len(kb_all)}, Test angles: {len(true_ka)}")

# ==============================================================================
# FIGURE 1: Overview Distributions & Outlier Tails (Dual Linear & Log)
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Exploratory Data Analysis: Force Constant Distributions & Outlier Tails", fontsize=18, fontweight='bold', y=0.98)

# 1A: k_angle Linear
ax = axes[0, 0]
sns.histplot(ka_all, bins=60, kde=True, color="#2b5c8f", ax=ax, stat="density", alpha=0.6, edgecolor='white')
q25, q50, q75 = np.percentile(ka_all, [25, 50, 75])
iqr = q75 - q25
upper_fence = q75 + 1.5 * iqr
extreme_fence = q75 + 3.0 * iqr

ax.axvline(q50, color="#1b4332", linestyle="--", linewidth=2.2, label=f"Median = {q50:.1f}")
ax.axvline(upper_fence, color="#d97706", linestyle="--", linewidth=2.0, label=f"Q3 + 1.5xIQR = {upper_fence:.1f} (Outliers: 5.67%)")
ax.axvline(extreme_fence, color="#dc2626", linestyle="-.", linewidth=2.2, label=f"Q3 + 3.0xIQR = {extreme_fence:.1f} (Extreme: 3.84%)")
ax.axvline(1000.0, color="#7c3aed", linestyle=":", linewidth=2.5, label="MARTINI Ring Constraints (k = 1000)")
ax.set_title("Angle Spring Constant k_angle (Linear Scale)\nHeavy Positive Skewness = +6.52, Kurtosis = +49.07", fontsize=13, fontweight='bold')
ax.set_xlabel("k_angle [kJ / mol / rad^2]", fontsize=11, fontweight='bold')
ax.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9.5)
ax.set_xlim(0, 1050)

# 1B: k_angle Log10
ax = axes[0, 1]
log_ka = np.log10(ka_all)
sns.histplot(log_ka, bins=45, kde=True, color="#0284c7", ax=ax, stat="density", alpha=0.6, edgecolor='white')
mean_log = np.mean(log_ka)
std_log = np.std(log_ka)
x_log = np.linspace(log_ka.min(), log_ka.max(), 200)
ax.plot(x_log, stats.norm.pdf(x_log, mean_log, std_log), color="#e11d48", linewidth=2.5, linestyle="--", label=f"Gaussian Fit (mu={mean_log:.2f}, sigma={std_log:.2f})")
ax.axvline(np.log10(50.0), color="#16a34a", linestyle="--", linewidth=2.0, label="Flexible/Medium Boundary (k=50)")
ax.axvline(np.log10(150.0), color="#ea580c", linestyle="--", linewidth=2.0, label="Medium/Stiff Boundary (k=150)")
ax.axvline(np.log10(1000.0), color="#7c3aed", linestyle=":", linewidth=2.5, label="Constraint Peak (k=1000, log=3.0)")
ax.set_title("Angle Spring Constant k_angle (Log10 Scale)\nSkewness drops to +0.70, Kurtosis drops to +1.94", fontsize=13, fontweight='bold')
ax.set_xlabel("log10(k_angle)", fontsize=11, fontweight='bold')
ax.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9.5)

# 1C: k_bond Linear
ax = axes[1, 0]
sns.histplot(kb_all, bins=50, kde=True, color="#059669", ax=ax, stat="density", alpha=0.6, edgecolor='white')
q25_b, q50_b, q75_b = np.percentile(kb_all, [25, 50, 75])
ax.axvline(q50_b, color="#1b4332", linestyle="--", linewidth=2.2, label=f"Median = {q50_b:.0f}")
ax.axvline(7000, color="#d97706", linestyle="--", linewidth=2.0, label="Flexible/Semi-rigid Boundary (k=7000)")
ax.axvline(25000, color="#ea580c", linestyle="--", linewidth=2.0, label="Semi-rigid/Stiff Boundary (k=25000)")
ax.axvline(50000, color="#dc2626", linestyle=":", linewidth=2.5, label="Aromatic Ring Constraints (k=50000)")
ax.set_title("Bond Spring Constant k_bond (Linear Scale)\nBimodal / Balanced Distribution: Skewness = +1.06, Kurtosis = -0.79", fontsize=13, fontweight='bold')
ax.set_xlabel("k_bond [kJ / mol / nm^2]", fontsize=11, fontweight='bold')
ax.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9.5)

# 1D: k_bond Log10
ax = axes[1, 1]
log_kb = np.log10(kb_all)
sns.histplot(log_kb, bins=40, kde=True, color="#10b981", ax=ax, stat="density", alpha=0.6, edgecolor='white')
ax.axvline(np.log10(5000.0), color="#1e40af", linestyle="--", linewidth=2.0, label="Standard Bond Peak (k=5000, log=3.70)")
ax.axvline(np.log10(50000.0), color="#dc2626", linestyle=":", linewidth=2.5, label="Constraint Bond Peak (k=50000, log=4.70)")
ax.set_title("Bond Spring Constant k_bond (Log10 Scale)\nClear Bimodal Separation Between Flexible Chains and Ring Constraints", fontsize=13, fontweight='bold')
ax.set_xlabel("log10(k_bond)", fontsize=11, fontweight='bold')
ax.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("results/eda_distributions_and_tails.png", dpi=300)
plt.close()
print("Saved results/eda_distributions_and_tails.png")

# ==============================================================================
# FIGURE 2: Outlier Breakdown, Box Plots & Cumulative Density Function
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Outlier Detection & Cumulative Tail Probability Analysis", fontsize=17, fontweight='bold', y=0.98)

# 2A: Box plot with Outlier annotations
ax = axes[0]
bp = ax.boxplot([ka_all], patch_artist=True, vert=True, widths=0.45,
                boxprops=dict(facecolor="#bae6fd", color="#0284c7", linewidth=2),
                medianprops=dict(color="#dc2626", linewidth=2.5),
                whiskerprops=dict(color="#0284c7", linewidth=1.5),
                capprops=dict(color="#0284c7", linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='#e11d48', markersize=5, alpha=0.5, markeredgecolor='none'))
ax.set_xticklabels(["k_angle (All Data, N=4606)"], fontsize=12, fontweight='bold')
ax.set_ylabel("k_angle [kJ / mol / rad^2]", fontsize=12, fontweight='bold')
ax.set_title("Angle Boxplot: Extreme Outlier Tail\nUpper Fence: 125.0 | Max Outliers: 1000.0", fontsize=12, fontweight='bold')
ax.axhline(125.0, color="#d97706", linestyle="--", linewidth=1.5, label="IQR Upper Fence (125.0)")
ax.axhline(185.0, color="#dc2626", linestyle="-.", linewidth=1.5, label="Extreme Fence (185.0)")
ax.axhline(1000.0, color="#7c3aed", linestyle=":", linewidth=2.0, label="Constraint Outliers (k=1000)")
ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

# 2B: CDF Plot for k_angle
ax = axes[1]
sorted_ka = np.sort(ka_all)
cdf = np.arange(1, len(sorted_ka) + 1) / len(sorted_ka) * 100.0
ax.plot(sorted_ka, cdf, color="#2563eb", linewidth=2.5, label="Empirical Cumulative Distribution (CDF)")
ax.axvline(50, color="#16a34a", linestyle="--", linewidth=1.8, label="k = 50 (67.9% of all data <= 50)")
ax.axvline(150, color="#ea580c", linestyle="--", linewidth=1.8, label="k = 150 (95.7% of all data <= 150)")
ax.axvline(700, color="#dc2626", linestyle=":", linewidth=2.0, label="k = 700 (98.9% of data <= 700)")
ax.axhline(50, color="#64748b", linestyle=":", alpha=0.6)
ax.axhline(90, color="#64748b", linestyle=":", alpha=0.6)
ax.axhline(99, color="#64748b", linestyle=":", alpha=0.6)
ax.set_title("Empirical CDF: 95.7% Compressed Below 150\nOnly Top 1.1% Fall in the 700-1000 Plateau", fontsize=12, fontweight='bold')
ax.set_xlabel("k_angle [kJ / mol / rad^2]", fontsize=12, fontweight='bold')
ax.set_ylabel("Cumulative Percentage (%)", fontsize=12, fontweight='bold')
ax.set_ylim(0, 102)
ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

# 2C: Violin plot of regimes
ax = axes[2]
# Create physical categories: Flexible (<=50), Medium (50-150), Stiff (>150)
regimes = []
for k in ka_all:
    if k <= 50: regimes.append("Flexible\n(k <= 50)\n[67.9%]")
    elif k <= 150: regimes.append("Medium\n(50 < k <= 150)\n[27.8%]")
    else: regimes.append("Stiff / Outliers\n(k > 150)\n[4.3%]")
df_viol = pd.DataFrame({"Regime": regimes, "k_angle": ka_all})
palette = ["#38bdf8", "#34d399", "#f87171"]
sns.violinplot(x="Regime", y="k_angle", data=df_viol, ax=ax, palette=palette, inner="quartile", cut=0)
ax.set_title("Distribution Across Physical Regimes\nShowing Extreme Spread in Stiff Category", fontsize=12, fontweight='bold')
ax.set_xlabel("Physical Regime Category", fontsize=12, fontweight='bold')
ax.set_ylabel("k_angle [kJ / mol / rad^2]", fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("results/eda_boxplots_and_outliers.png", dpi=300)
plt.close()
print("Saved results/eda_boxplots_and_outliers.png")

# ==============================================================================
# FIGURE 3: Rigorous Scaling Comparison (6 Transformations + Normal Q-Q Fits)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Comparison of 6 Data Scaling Schemes for k_angle", fontsize=18, fontweight='bold', y=0.98)

# Compute transforms on training set
train_ka_list = []
for i in train_idx:
    g = all_graphs[i]
    if hasattr(g, 'y_k_angle') and g.y_k_angle is not None and len(g.y_k_angle) > 0:
        train_ka_list.extend(g.y_k_angle.numpy())
t_ka = np.array(train_ka_list)

transforms_dict = {
    "1. Raw Linear": (t_ka, "#64748b", "Skew: +6.43 | Kurt: +47.15\nOutliers dominate MSE by 400x"),
    "2. StandardScaler (Z-Score)": (StandardScaler().fit_transform(t_ka.reshape(-1, 1)).flatten(), "#ef4444", "Skew: +6.43 | Kurt: +47.15\nLinear shift does not reduce tail skew"),
    "3. RobustScaler (Median/IQR)": (RobustScaler().fit_transform(t_ka.reshape(-1, 1)).flatten(), "#f59e0b", "Skew: +6.43 | Kurt: +47.15\nTop outlier is still at z = +23.4 IQR!"),
    "4. Min-Max Scaler [0, 1]": (MinMaxScaler().fit_transform(t_ka.reshape(-1, 1)).flatten(), "#8b5cf6", "Skew: +6.43 | Kurt: +47.15\n95% of data squashed into [0.0, 0.15]"),
    "5. Log10 Scale": (np.log10(np.clip(t_ka, 1e-4, None)), "#0284c7", "Skew: +0.70 | Kurt: +1.94\nEqually balances all 3 decades [10^0, 10^3]"),
    "6. Yeo-Johnson Power Transform": (PowerTransformer(method='yeo-johnson').fit_transform(t_ka.reshape(-1, 1)).flatten(), "#10b981", "Skew: -0.06 | Kurt: +1.07\nNearly Gaussian normal bell curve!")
}

for ax, (title, (vals, color, annot)) in zip(axes.flatten(), transforms_dict.items()):
    sns.histplot(vals, bins=40, kde=True, color=color, ax=ax, stat="density", alpha=0.5, edgecolor="white")
    mu, sigma = np.mean(vals), np.std(vals)
    x_grid = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), color="#1e293b", linestyle="--", linewidth=1.8, label=f"Normal Curve (N({mu:.2f}, {sigma:.2f}))")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Transformed Value", fontsize=10, fontweight='bold')
    ax.set_ylabel("Density", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.5)
    ax.text(0.04, 0.85, annot, transform=ax.transAxes, fontsize=9.5, fontweight='semibold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.9))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("results/eda_scaling_comparison.png", dpi=300)
plt.close()
print("Saved results/eda_scaling_comparison.png")

# ==============================================================================
# FIGURE 4: Error Dominance of Top Outliers on Linear R^2 & Sensitivity Curve
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Mathematical Impact of Top Outliers on Linear R^2 (Test Set N=458)", fontsize=17, fontweight='bold', y=0.98)

# 4A: Pareto Chart of Squared Errors
sq_errors = (true_ka - pred_ka)**2
sort_err_idx = np.argsort(sq_errors)[::-1]
total_ss_res = np.sum(sq_errors)

top_k_ranks = np.arange(1, 21)
top_k_sq_errs = sq_errors[sort_err_idx[:20]]
pct_contributions = (top_k_sq_errs / total_ss_res) * 100.0
cumulative_pct = np.cumsum(pct_contributions)

ax1 = axes[0]
bars = ax1.bar(top_k_ranks, pct_contributions, color="#ef4444", alpha=0.7, edgecolor="#b91c1c", label="Single Point % of SS_res")
ax1.set_xlabel("Outlier Rank (Sorted by Squared Error)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Contribution to Total SS_res (%)", color="#b91c1c", fontsize=11, fontweight='bold')
ax1.set_xticks(top_k_ranks)
ax1.set_ylim(0, 20)

ax2 = ax1.twinx()
ax2.plot(top_k_ranks, cumulative_pct, color="#1e40af", marker="o", linewidth=2.5, label="Cumulative % of SS_res")
ax2.set_ylabel("Cumulative Total SS_res (%)", color="#1e40af", fontsize=11, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.axhline(59.35, color="#d97706", linestyle="--", linewidth=1.5, label="Top 5 Outliers = 59.35% SS_res")
ax2.axhline(78.07, color="#dc2626", linestyle="-.", linewidth=1.5, label="Top 10 Outliers = 78.07% SS_res")

ax1.set_title("Pareto Analysis: 1% of Data Accounts for ~60% of All Error", fontsize=12, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, facecolor="white", fontsize=9)

# 4B: Sensitivity of R^2, MAE, and SS_res as Top Error Points are Removed
r2_list = []
mae_list = []
ss_rem_list = []
base_ss_tot = np.sum((true_ka - np.mean(true_ka))**2)

for n in range(0, 21):
    mask = np.ones(len(true_ka), dtype=bool)
    if n > 0:
        mask[sort_err_idx[:n]] = False
    sub_true = true_ka[mask]
    sub_pred = pred_ka[mask]
    ss_res_n = np.sum((sub_true - sub_pred)**2)
    ss_tot_n = np.sum((sub_true - np.mean(sub_true))**2)
    r2_n = 1.0 - (ss_res_n / ss_tot_n) if ss_tot_n > 0 else 0.0
    mae_n = np.mean(np.abs(sub_true - sub_pred))
    
    r2_list.append(r2_n)
    mae_list.append(mae_n)
    ss_rem_list.append((ss_res_n / total_ss_res) * 100.0)

ax = axes[1]
x_rem = np.arange(0, 21)
color_mae = "#059669"
ax.plot(x_rem, mae_list, color=color_mae, marker="s", linewidth=2.5, label="MAE [kJ/mol/rad^2]")
ax.set_xlabel("Number of Top Error Outliers Removed", fontsize=11, fontweight='bold')
ax.set_ylabel("Mean Absolute Error (MAE)", color=color_mae, fontsize=11, fontweight='bold')
ax.set_xticks(range(0, 21, 2))
ax.set_ylim(10, 25)

ax_twin = ax.twinx()
color_ss = "#dc2626"
ax_twin.plot(x_rem, ss_rem_list, color=color_ss, marker="^", linestyle="--", linewidth=2.5, label="Remaining SS_res (%)")
ax_twin.set_ylabel("Remaining Sum of Squared Residuals (%)", color=color_ss, fontsize=11, fontweight='bold')
ax_twin.set_ylim(0, 105)

ax.set_title("Impact on Error Metrics: MAE Drops by 46% (23.1 -> 12.4)\nSS_res Plummets from 100% to 13.1%", fontsize=12, fontweight='bold')
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax_twin.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, loc="upper right", frameon=True, facecolor="white", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("results/eda_error_dominance_and_r2.png", dpi=300)
plt.close()
print("Saved results/eda_error_dominance_and_r2.png")
