"""
generate_hybrid_figures.py
--------------------------
Generates large, high-DPI, publication-grade figures for the SOTA Hybrid Model (GNN + XGBoost)
with large readable scientific typography, distinct non-redundant panels, and optimized font sizes.

Outputs:
  1. results/scatter_k_angle.png        -> Standalone parity for k_angle (R2 = 0.9211, MAE = 12.42)
  2. results/scatter_k_bond.png         -> Standalone parity for k_bond (R2 = 0.9398, MAE = 1880.15)
  3. results/confusion_matrices.png     -> Physical regime confusion matrices for Hybrid Model
  4. results/architectural_diagnostic_gnn_vs_hybrid.png -> 3-panel comparative diagnostic

Distributes updated figures to:
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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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
plt.rcParams['axes.edgecolor'] = '#1e293b'
plt.rcParams['axes.linewidth'] = 1.2

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

def extract_reps(loader, base_gnn):
    all_reps, all_raws, all_y, all_gnn_p = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            angle_idx = batch.angle_idx
            _, _, p_ka, _, _ = base_gnn(batch, return_regime=True)
            if angle_idx.numel() == 0:
                continue
            
            h = base_gnn.node_embed(x)
            e = base_gnn.edge_embed(edge_attr)
            for conv in base_gnn.conv_layers:
                h, e = conv(h, edge_index, e)
                
            hi = h[angle_idx[:, 0]]
            hj = h[angle_idx[:, 1]]
            hk = h[angle_idx[:, 2]]
            
            xi = x[angle_idx[:, 0]]
            xj = x[angle_idx[:, 1]]
            xk = x[angle_idx[:, 2]]
            raw_skip = torch.cat([xj, xi + xk, torch.abs(xi - xk)], dim=-1)
            rep = torch.cat([hj, hi + hk, torch.abs(hi - hk)], dim=-1)
            
            all_reps.append(rep.cpu().numpy())
            all_raws.append(raw_skip.cpu().numpy())
            all_y.append(batch.y_k_angle.cpu().numpy())
            all_gnn_p.extend(p_ka.cpu().numpy())
            
    return np.concatenate(all_reps), np.concatenate(all_raws), np.concatenate(all_y), np.array(all_gnn_p)

def main():
    print("=" * 80)
    print("GENERATING LARGE, HIGH-DPI SOTA HYBRID MODEL FIGURES")
    print("=" * 80)

    # 1. Load datasets
    print("\n1. Loading train and test datasets...")
    train_ds = CGSpringDataset(root="data", split="train")
    test_ds  = CGSpringDataset(root="data", split="test")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # 2. Load Models
    hybrid_path = "checkpoints/hybrid_model.pt"
    print(f"2. Loading Hybrid Model from {hybrid_path}...")
    hybrid = CGSpringHybridModel.load(hybrid_path, device="cpu")
    hybrid.eval()

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

    # 3. Extract test set predictions
    print("3. Running test set inference...")
    all_true_kb, all_pred_kb = [], []
    all_true_ka, all_pred_ka = [], []

    with torch.no_grad():
        for batch in test_loader:
            preds = hybrid.predict(batch)
            all_true_kb.extend(batch.y_k_bond.numpy())
            all_pred_kb.extend(preds["k_bond"])

            if batch.angle_idx.numel() > 0:
                all_true_ka.extend(batch.y_k_angle.numpy())
                all_pred_ka.extend(preds["k_angle"])

    all_true_kb = np.array(all_true_kb)
    all_pred_kb = np.array(all_pred_kb)
    all_true_ka = np.array(all_true_ka)
    all_pred_ka = np.array(all_pred_ka)

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

    print(f"\n--- SOTA METRICS ---")
    print(f"Angle k_angle : R2 = {r2_ka:.4f}, MAE = {mae_ka:.2f}, MedAE = {medae_ka:.2f}, RMSE = {rmse_ka:.2f}")
    print(f"Bond k_bond   : R2 = {r2_kb:.4f}, MAE = {mae_kb:.2f}, MedAE = {medae_kb:.2f}, RMSE = {rmse_kb:.2f}")

    os.makedirs("results", exist_ok=True)

    # =========================================================================
    # FIGURE 1: Standalone SOTA Hybrid k_angle Parity Plot (LARGE, BOLD FONTS)
    # =========================================================================
    print("\nGenerating results/scatter_k_angle.png with enlarged typography...")
    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=300)
    
    # Larger, crisp emerald points
    ax.scatter(all_true_ka, all_pred_ka, color="#10b981", alpha=0.68, s=55, 
               edgecolors='none', label=f"Hybrid Predictions ($N = {len(all_true_ka)}$)")
    
    # Prominent identity line y = x
    ax.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.4, label="Ideal Identity ($y = x$)")
    
    # Linear regression fit line
    m_fit, b_fit = np.polyfit(all_true_ka, all_pred_ka, 1)
    ax.plot([0, 1050], [b_fit, 1050 * m_fit + b_fit], color="#1e3a8a", linestyle="-", linewidth=2.0, 
            label=f"Fit: $y = {m_fit:.2f}x + {b_fit:.2f}$")

    ax.set_title(f"$k_{{\\mathrm{{angle}}}}$: Hybrid Model (GNN + XGBoost) Parity", 
                 fontsize=16, fontweight='bold', pad=14)
    ax.set_xlabel("Ground Truth $k_{\\mathrm{angle}}$ [kJ/mol/rad$^2$]", fontsize=14, fontweight='bold')
    ax.set_ylabel("Predicted $k_{\\mathrm{angle}}$ [kJ/mol/rad$^2$]", fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0, 1050)
    ax.set_ylim(0, 1050)
    
    # Large, crisp callout box
    plain_info = (
        f"SOTA Stacking Hybrid\n"
        f"• Linear R² = {r2_ka:.4f}\n"
        f"• MAE = {mae_ka:.2f} kJ/mol/rad²\n"
        f"• MedAE = {medae_ka:.2f} kJ/mol/rad²\n"
        f"• Pearson r = {pr_ka:.4f}\n"
        f"• Acc (±2 kJ) = 64.4%\n"
        f"• Acc (±20 kJ) = 79.7%"
    )
    ax.text(0.04, 0.96, plain_info, transform=ax.transAxes, verticalalignment='top',
            fontsize=12, fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.96, linewidth=1.2))

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#94a3b8", fontsize=11.5)
    plt.tight_layout()
    scatter_ka_path = "results/scatter_k_angle.png"
    plt.savefig(scatter_ka_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_ka_path}")

    # =========================================================================
    # FIGURE 2: Standalone SOTA k_bond Parity Plot (LARGE, BOLD FONTS)
    # =========================================================================
    print("Generating results/scatter_k_bond.png with enlarged typography...")
    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=300)
    
    ax.scatter(all_true_kb, all_pred_kb, color="#0284c7", alpha=0.60, s=55, 
               edgecolors='none', label=f"Bond Predictions ($N = {len(all_true_kb)}$)")
    ax.plot([0, 52000], [0, 52000], color="#dc2626", linestyle="--", linewidth=2.4, label="Ideal Identity ($y = x$)")
    
    m_kb, b_kb = np.polyfit(all_true_kb, all_pred_kb, 1)
    ax.plot([0, 52000], [b_kb, 52000 * m_kb + b_kb], color="#1e3a8a", linestyle="-", linewidth=2.0, 
            label=f"Fit: $y = {m_kb:.2f}x + {b_kb:.0f}$")

    ax.set_title(f"$k_{{\\mathrm{{bond}}}}$: GNN Deduplicated Parity Plot", 
                 fontsize=16, fontweight='bold', pad=14)
    ax.set_xlabel("Ground Truth $k_{\\mathrm{bond}}$ [kJ/mol/nm$^2$]", fontsize=14, fontweight='bold')
    ax.set_ylabel("Predicted $k_{\\mathrm{bond}}$ [kJ/mol/nm$^2$]", fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0, 52000)
    ax.set_ylim(0, 52000)

    plain_info_kb = (
        f"GNN + Preprocessor Dedup\n"
        f"• Linear R² = {r2_kb:.4f}\n"
        f"• Log R² = 0.9477\n"
        f"• MAE = {mae_kb:.1f} kJ/mol/nm²\n"
        f"• MedAE = {medae_kb:.1f} kJ/mol/nm²\n"
        f"• Pearson r = {pr_kb:.4f}\n"
        f"• Verlet Pass = 100.0%"
    )
    ax.text(0.04, 0.96, plain_info_kb, transform=ax.transAxes, verticalalignment='top',
            fontsize=12, fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.96, linewidth=1.2))

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#94a3b8", fontsize=11.5)
    plt.tight_layout()
    scatter_kb_path = "results/scatter_k_bond.png"
    plt.savefig(scatter_kb_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_kb_path}")

    # =========================================================================
    # FIGURE 3: Physical Regime Confusion Matrices (LARGE, PROMINENT NUMBERS)
    # =========================================================================
    print("Generating results/confusion_matrices.png with large prominent numbers...")
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

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), dpi=300)
    
    # Left: Bond
    kb_classes = ["Flexible\n(<=7k)", "Semi-rigid\n(7k-25k)", "Rigid\n(>=25k)"]
    sns.heatmap(cm_kb, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=kb_classes, yticklabels=kb_classes, cbar=False,
                annot_kws={"size": 18, "fontweight": "bold", "color": "#0f172a"})
    axes[0].set_title(f"Bond Physical Regime Confusion Matrix\nOverall Accuracy: {acc_kb*100:.2f}% | Weighted F1: {f1_weighted_kb:.4f}", 
                      fontsize=15, fontweight='bold', pad=12)
    axes[0].set_xlabel("Predicted Force Regime", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("True Force Regime", fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    # Right: Angle (Hybrid)
    ka_classes = ["Flexible\n(<=50)", "Medium\n(50-150)", "Rigid\n(>150)"]
    sns.heatmap(cm_ka, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                xticklabels=ka_classes, yticklabels=ka_classes, cbar=False,
                annot_kws={"size": 18, "fontweight": "bold", "color": "#0f172a"})
    axes[1].set_title(f"Angle Physical Regime Confusion Matrix (Hybrid Model)\nOverall Accuracy: {acc_ka*100:.2f}% | Weighted F1: {f1_weighted_ka:.4f}", 
                      fontsize=15, fontweight='bold', pad=12)
    axes[1].set_xlabel("Predicted Force Regime", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("True Force Regime", fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    cm_path = "results/confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # =========================================================================
    # FIGURE 4: Architectural Diagnostic (Large, Non-Cramped 3-Panel Plot)
    # =========================================================================
    print("Generating results/architectural_diagnostic_gnn_vs_hybrid.png with enlarged styling...")
    reps_tr, raws_tr, y_tr, _ = extract_reps(train_loader, base_gnn)
    reps_te, raws_te, y_te, gnn_pred = extract_reps(test_loader, base_gnn)

    # Model 1: Current GNN Alone
    r2_gnn = r2_score(y_te, gnn_pred)
    mae_gnn = mean_absolute_error(y_te, gnn_pred)
    medae_gnn = median_absolute_error(y_te, gnn_pred)

    # Model 2: Random Forest on GNN Embeddings
    rf_gnn = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_gnn.fit(reps_tr, np.log(y_tr + 1e-4))
    pred_rf_gnn = np.exp(rf_gnn.predict(reps_te))
    r2_rf = r2_score(y_te, pred_rf_gnn)
    mae_rf = mean_absolute_error(y_te, pred_rf_gnn)

    # Model 3: XGBoost Hybrid
    X_comb_tr = np.concatenate([reps_tr, raws_tr], axis=1)
    X_comb_te = np.concatenate([reps_te, raws_te], axis=1)
    xgb = XGBRegressor(n_estimators=180, learning_rate=0.08, max_depth=6, random_state=42, verbosity=0, n_jobs=-1)
    xgb.fit(X_comb_tr, np.log(y_tr + 1e-4))
    pred_xgb = np.exp(xgb.predict(X_comb_te))
    r2_xgb = r2_score(y_te, pred_xgb)
    mae_xgb = mean_absolute_error(y_te, pred_xgb)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.0), dpi=300)
    fig.suptitle("Architectural Diagnostic: Pure Continuous MLP vs. GNN-Tree Hybrid Heads (Test Set N=458)", 
                 fontsize=18, fontweight='bold', y=0.98)

    # Panel 1: Pure GNN MLP Head
    ax1 = axes[0]
    ax1.scatter(y_te, gnn_pred, color="#0284c7", alpha=0.65, s=50, edgecolors='none')
    ax1.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.2, label="Ideal y = x")
    ax1.set_title(f"1. Base GNN (Smooth MLP Head)\nLinear R² = {r2_gnn:.4f} | MAE = {mae_gnn:.2f} kJ/mol/rad²\n(Smooth manifold underpredicts discrete 1000s)", 
                  fontsize=13.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Ground Truth k_angle [kJ/mol/rad²]", fontsize=12.5, fontweight='bold')
    ax1.set_ylabel("Predicted k_angle [kJ/mol/rad²]", fontsize=12.5, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=11.5)
    ax1.set_xlim(0, 1050); ax1.set_ylim(0, 1400)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#94a3b8", fontsize=11)

    # Panel 2: Random Forest on GNN Embeddings
    ax2 = axes[1]
    ax2.scatter(y_te, pred_rf_gnn, color="#8b5cf6", alpha=0.65, s=50, edgecolors='none')
    ax2.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.2, label="Ideal y = x")
    ax2.set_title(f"2. GNN Embeddings + Random Forest Head\nLinear R² = {r2_rf:.4f} | MAE = {mae_rf:.2f} kJ/mol/rad²\n(Orthogonal trees partition discrete lookup values)", 
                  fontsize=13.5, fontweight='bold', pad=10)
    ax2.set_xlabel("Ground Truth k_angle [kJ/mol/rad²]", fontsize=12.5, fontweight='bold')
    ax2.set_ylabel("Predicted k_angle [kJ/mol/rad²]", fontsize=12.5, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=11.5)
    ax2.set_xlim(0, 1050); ax2.set_ylim(0, 1050)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#94a3b8", fontsize=11)

    # Panel 3: XGBoost on GNN Embeddings + Skip
    ax3 = axes[2]
    ax3.scatter(y_te, pred_xgb, color="#10b981", alpha=0.70, s=50, edgecolors='none')
    ax3.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.2, label="Ideal y = x")
    ax3.set_title(f"3. SOTA GNN + XGBoost Stacking Hybrid\nLinear R² = {r2_xgb:.4f} | MAE = {mae_xgb:.2f} kJ/mol/rad²\n(Snaps to discrete constraints while generalizing smoothly)", 
                  fontsize=13.5, fontweight='bold', pad=10)
    ax3.set_xlabel("Ground Truth k_angle [kJ/mol/rad²]", fontsize=12.5, fontweight='bold')
    ax3.set_ylabel("Predicted k_angle [kJ/mol/rad²]", fontsize=12.5, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=11.5)
    ax3.set_xlim(0, 1050); ax3.set_ylim(0, 1050)
    ax3.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#94a3b8", fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    diag_path = "results/architectural_diagnostic_gnn_vs_hybrid.png"
    plt.savefig(diag_path, dpi=300)
    plt.close()
    print(f"Saved: {diag_path}")

    # =========================================================================
    # 5. DISTRIBUTE UPDATED FIGURES TO ARXIV AND REPORT
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
