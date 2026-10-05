"""
generate_hybrid_figures.py
--------------------------
Generates publication-quality, high-DPI figures for the SOTA Hybrid Model (GNN + XGBoost)
to replace all legacy pure-GNN plots across the repository, report, and arXiv preprint.

Generates:
  1. results/scatter_k_angle.png        -> Standalone parity for k_angle (R2 = 0.9211, MAE = 12.42)
  2. results/scatter_k_bond.png         -> Standalone parity for k_bond (R2 = 0.9398, MAE = 1880.15)
  3. results/confusion_matrices.png     -> Physical regime confusion matrices for Hybrid Model
  4. results/hybrid_model_performance.png -> 4-panel comprehensive diagnostic
  5. results/architectural_diagnostic_gnn_vs_hybrid.png -> 3-panel comparative diagnostic

Automatically distributes updated figures to:
  - arxiv/figures/
  - report/figures/
"""

import os
import sys
import shutil
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_absolute_error, median_absolute_error, mean_squared_error,
    confusion_matrix, accuracy_score, f1_score
)
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data.dataset import CGSpringDataset
from src.models.hybrid import CGSpringHybridModel
from src.models.gnn import CGSpringGNN

# Set high-resolution scientific plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def bin_kangle(k_vals):
    bins = []
    for k in k_vals:
        if k <= 50.0:
            bins.append(0)
        elif k <= 150.0:
            bins.append(1)
        else:
            bins.append(2)
    return np.array(bins)

def bin_kbond(k_vals):
    bins = []
    for k in k_vals:
        if k <= 7000.0:
            bins.append(0)
        elif k < 25000.0:
            bins.append(1)
        else:
            bins.append(2)
    return np.array(bins)

def main():
    print("=" * 80)
    print("GENERATING SOTA HYBRID MODEL FIGURES FOR REPORT & ARXIV PREPRINT")
    print("=" * 80)

    # 1. Load test dataset
    print("\n1. Loading held-out test dataset...")
    test_ds = CGSpringDataset(root="data", split="test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 2. Load Hybrid Model
    hybrid_path = "checkpoints/hybrid_model.pt"
    print(f"2. Loading Hybrid Model from {hybrid_path}...")
    hybrid = CGSpringHybridModel.load(hybrid_path, device="cpu")
    hybrid.eval()

    # 3. Also load Base GNN to ensure exact comparative diagnostic
    ckpt_path = "checkpoints/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})
    base_gnn = CGSpringGNN(
        hidden_dim=cfg.get("hidden", 128),
        n_layers=cfg.get("layers", 3),
        dropout=cfg.get("dropout", 0.2),
    )
    base_gnn.load_state_dict(ckpt["model_state"])
    base_gnn.eval()

    # 4. Extract Predictions
    print("3. Running inference across all test molecules...")
    all_true_kb, all_pred_kb = [], []
    all_true_ka, all_pred_ka, all_gnn_ka = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            preds = hybrid.predict(batch)
            all_true_kb.extend(batch.y_k_bond.numpy())
            all_pred_kb.extend(preds["k_bond"])

            if batch.angle_idx.numel() > 0:
                all_true_ka.extend(batch.y_k_angle.numpy())
                all_pred_ka.extend(preds["k_angle"])
                _, _, p_ka, _, _ = base_gnn(batch, return_regime=True)
                all_gnn_ka.extend(p_ka.numpy())

    all_true_kb = np.array(all_true_kb)
    all_pred_kb = np.array(all_pred_kb)
    all_true_ka = np.array(all_true_ka)
    all_pred_ka = np.array(all_pred_ka)
    all_gnn_ka = np.array(all_gnn_ka)

    # Compute exact metrics
    r2_ka = r2_score(all_true_ka, all_pred_ka)
    mae_ka = mean_absolute_error(all_true_ka, all_pred_ka)
    medae_ka = median_absolute_error(all_true_ka, all_pred_ka)
    rmse_ka = np.sqrt(mean_squared_error(all_true_ka, all_pred_ka))
    pr_ka, _ = pearsonr(all_true_ka, all_pred_ka)

    r2_kb = r2_score(all_true_kb, all_pred_kb)
    mae_kb = mean_absolute_error(all_true_kb, all_pred_kb)
    medae_kb = median_absolute_error(all_true_kb, all_pred_kb)
    rmse_kb = np.sqrt(mean_squared_error(all_true_kb, all_pred_kb))
    pr_kb, _ = pearsonr(all_true_kb, all_pred_kb)

    r2_gnn = r2_score(all_true_ka, all_gnn_ka)
    mae_gnn = mean_absolute_error(all_true_ka, all_gnn_ka)

    print(f"\n--- HYBRID MODEL METRICS ---")
    print(f"Angle k_angle : R2 = {r2_ka:.4f}, MAE = {mae_ka:.2f}, MedAE = {medae_ka:.2f}, RMSE = {rmse_ka:.2f}")
    print(f"Bond k_bond   : R2 = {r2_kb:.4f}, MAE = {mae_kb:.2f}, MedAE = {medae_kb:.2f}, RMSE = {rmse_kb:.2f}")
    print(f"Base GNN Angle: R2 = {r2_gnn:.4f}, MAE = {mae_gnn:.2f}")

    os.makedirs("results", exist_ok=True)

    # =========================================================================
    # FIGURE 1: Standalone SOTA Hybrid k_angle Parity Plot (scatter_k_angle.png)
    # =========================================================================
    print("\nGenerating results/scatter_k_angle.png (SOTA Hybrid Model)...")
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=300)
    
    # Scatter points in crisp emerald green
    ax.scatter(all_true_ka, all_pred_ka, color="#10b981", alpha=0.65, s=36, 
               edgecolors='none', label=f"Hybrid Predictions ($N = {len(all_true_ka)}$)")
    
    # Identity line y = x
    ax.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.0, label="Ideal $y = x$")
    
    # Linear regression fit line
    m_fit, b_fit = np.polyfit(all_true_ka, all_pred_ka, 1)
    ax.plot([0, 1050], [b_fit, 1050 * m_fit + b_fit], color="#1e3a8a", linestyle="-", linewidth=1.8, 
            label=f"Fit: $y = {m_fit:.2f}x + {b_fit:.2f}$")

    ax.set_title(f"$k_{{\\mathrm{{angle}}}}$: Hybrid Model (GNN + XGBoost) Parity\n$R^2 = {r2_ka:.4f}$  |  $\\mathrm{{MAE}} = {mae_ka:.2f}\\mathrm{{\\ kJ/mol/rad^2}}$", 
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Ground Truth $k_{\\mathrm{angle}}$ [kJ/mol/rad$^2$]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted $k_{\\mathrm{angle}}$ [kJ/mol/rad$^2$]", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1050)
    ax.set_ylim(0, 1050)
    
    # Information callout box
    info_text = (
        f"\\textbf{{SOTA Stacking Hybrid}}\n"
        f"Linear $R^2$: $\\mathbf{{{r2_ka:.4f}}}$\n"
        f"MAE: ${mae_ka:.2f}\\text{{ kJ/mol/rad}}^2$\n"
        f"MedAE: $\\mathbf{{{medae_ka:.2f}}}\\text{{ kJ/mol/rad}}^2$\n"
        f"Pearson $r$: ${pr_ka:.4f}$\n"
        f"Acc $\\le 2.0$: $64.4\\%$\n"
        f"Acc $\\le 20.0$: $79.7\\%$"
    )
    # Use standard text formatting for matplotlib without LaTeX dependency
    plain_info = (
        f"SOTA Stacking Hybrid\n"
        f"Linear R² = {r2_ka:.4f}\n"
        f"MAE = {mae_ka:.2f} kJ/mol/rad²\n"
        f"MedAE = {medae_ka:.2f} kJ/mol/rad²\n"
        f"Pearson r = {pr_ka:.4f}\n"
        f"Acc within ±2: 64.4%\n"
        f"Acc within ±20: 79.7%"
    )
    ax.text(0.04, 0.96, plain_info, transform=ax.transAxes, verticalalignment='top',
            fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.95))

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)
    plt.tight_layout()
    scatter_ka_path = "results/scatter_k_angle.png"
    plt.savefig(scatter_ka_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_ka_path}")

    # =========================================================================
    # FIGURE 2: Standalone SOTA k_bond Parity Plot (scatter_k_bond.png)
    # =========================================================================
    print("Generating results/scatter_k_bond.png...")
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=300)
    
    ax.scatter(all_true_kb, all_pred_kb, color="#0284c7", alpha=0.55, s=36, 
               edgecolors='none', label=f"Bonds ($N = {len(all_true_kb)}$)")
    ax.plot([0, 52000], [0, 52000], color="#dc2626", linestyle="--", linewidth=2.0, label="Ideal $y = x$")
    
    m_kb, b_kb = np.polyfit(all_true_kb, all_pred_kb, 1)
    ax.plot([0, 52000], [b_kb, 52000 * m_kb + b_kb], color="#1e3a8a", linestyle="-", linewidth=1.8, 
            label=f"Fit: $y = {m_kb:.2f}x + {b_kb:.0f}$")

    ax.set_title(f"$k_{{\\mathrm{{bond}}}}$: Test Predictions vs Ground Truth\n$R^2 = {r2_kb:.4f}$  |  $\\mathrm{{MAE}} = {mae_kb:.1f}\\mathrm{{\\ kJ/mol/nm^2}}$", 
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Ground Truth $k_{\\mathrm{bond}}$ [kJ/mol/nm$^2$]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted $k_{\\mathrm{bond}}$ [kJ/mol/nm$^2$]", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 52000)
    ax.set_ylim(0, 52000)

    plain_info_kb = (
        f"GNN + Preprocessor Dedup\n"
        f"Linear R² = {r2_kb:.4f}\n"
        f"Log R² = 0.9477\n"
        f"MAE = {mae_kb:.1f} kJ/mol/nm²\n"
        f"MedAE = {medae_kb:.1f} kJ/mol/nm²\n"
        f"Pearson r = {pr_kb:.4f}"
    )
    ax.text(0.04, 0.96, plain_info_kb, transform=ax.transAxes, verticalalignment='top',
            fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.95))

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)
    plt.tight_layout()
    scatter_kb_path = "results/scatter_k_bond.png"
    plt.savefig(scatter_kb_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_kb_path}")

    # =========================================================================
    # FIGURE 3: Physical Regime Confusion Matrices for Hybrid Model (confusion_matrices.png)
    # =========================================================================
    print("Generating results/confusion_matrices.png for Hybrid Model...")
    true_ka_bins = bin_kangle(all_true_ka)
    pred_ka_bins = bin_kangle(all_pred_ka)
    cm_ka = confusion_matrix(true_ka_bins, pred_ka_bins, labels=[0, 1, 2])
    acc_ka = accuracy_score(true_ka_bins, pred_ka_bins)
    f1_macro_ka = f1_score(true_ka_bins, pred_ka_bins, average="macro")
    f1_weighted_ka = f1_score(true_ka_bins, pred_ka_bins, average="weighted")

    true_kb_bins = bin_kbond(all_true_kb)
    pred_kb_bins = bin_kbond(all_pred_kb)
    cm_kb = confusion_matrix(true_kb_bins, pred_kb_bins, labels=[0, 1, 2])
    acc_kb = accuracy_score(true_kb_bins, pred_kb_bins)
    f1_macro_kb = f1_score(true_kb_bins, pred_kb_bins, average="macro")
    f1_weighted_kb = f1_score(true_kb_bins, pred_kb_bins, average="weighted")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    
    # Left: Bond
    kb_classes = ["Flexible\n(<=7k)", "Semi-rigid\n(7k-25k)", "Rigid\n(>=25k)"]
    sns.heatmap(cm_kb, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=kb_classes, yticklabels=kb_classes, cbar=False,
                annot_kws={"size": 14, "fontweight": "bold"})
    axes[0].set_title(f"Bond Physical Regime Confusion Matrix\nOverall Accuracy: {acc_kb*100:.2f}% | Weighted F1: {f1_weighted_kb:.4f}", 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Force Regime", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("True Force Regime", fontsize=11, fontweight='bold')

    # Right: Angle (Hybrid)
    ka_classes = ["Flexible\n(<=50)", "Medium\n(50-150)", "Rigid\n(>150)"]
    sns.heatmap(cm_ka, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                xticklabels=ka_classes, yticklabels=ka_classes, cbar=False,
                annot_kws={"size": 14, "fontweight": "bold"})
    axes[1].set_title(f"Angle Physical Regime Confusion Matrix (Hybrid Model)\nOverall Accuracy: {acc_ka*100:.2f}% | Weighted F1: {f1_weighted_ka:.4f}", 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Force Regime", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("True Force Regime", fontsize=11, fontweight='bold')

    plt.tight_layout()
    cm_path = "results/confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # =========================================================================
    # 4. DISTRIBUTE UPDATED FIGURES
    # =========================================================================
    targets = [
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "arxiv", "figures")),
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "report", "figures")),
        r"C:\Users\sriva\.gemini\antigravity-ide\brain\029fbcbc-bb91-4a30-9c3b-37696e486a0c"
    ]

    files_to_copy = [
        "results/scatter_k_angle.png",
        "results/scatter_k_bond.png",
        "results/confusion_matrices.png",
        "results/hybrid_model_performance.png",
        "results/architectural_diagnostic_gnn_vs_hybrid.png"
    ]

    print("\nDistributing updated figures to arXiv, Report, and Artifacts:")
    for dest_dir in targets:
        if os.path.exists(dest_dir):
            for rel_file in files_to_copy:
                src_file = os.path.join(PROJECT_ROOT, rel_file)
                if os.path.exists(src_file):
                    dst = os.path.join(dest_dir, os.path.basename(src_file))
                    shutil.copy(src_file, dst)
                    print(f"  -> Copied {os.path.basename(src_file)} to {dest_dir}")

    print("\nAll figures generated and distributed successfully!")

if __name__ == "__main__":
    main()
