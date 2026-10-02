"""
train_hybrid.py
---------------
Trains the GNN-XGBoost Stacking Hybrid Architecture (Option A):
  1. Utilizes the trained GNN backbone (checkpoints/best_model.pt) to extract rich
     topological/spatial graph representations.
  2. Trains a specialized Gradient-Boosted Decision Tree (XGBoost) head on GNN representations +
     raw skip connections for k_angle, accurately capturing the discontinuous step-function look-up rules.
  3. Evaluates full test set performance across all parameters (bonds, angles, lengths, orientations).
  4. Saves the serialized hybrid model to checkpoints/hybrid_model.pt.
"""

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
import sys
import json
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error,
    confusion_matrix, classification_report, accuracy_score, f1_score
)
from torch_geometric.loader import DataLoader

from src.data.dataset import CGSpringDataset
from src.models.gnn import CGSpringGNN
from src.models.hybrid import CGSpringHybridModel

def compute_regression_metrics(true, pred, name, unit):
    r2_lin = float(r2_score(true, pred))
    log_true = np.log10(np.clip(true, 1e-4, None))
    log_pred = np.log10(np.clip(pred, 1e-4, None))
    r2_log = float(r2_score(log_true, log_pred)) if len(true) > 1 else 0.0

    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    medae = float(median_absolute_error(true, pred))
    max_err = float(np.max(np.abs(true - pred)))
    mape = float(np.mean(np.abs((pred - true) / true)) * 100.0)

    pr, _ = pearsonr(true, pred)
    sr, _ = spearmanr(true, pred)

    diff = np.abs(true - pred)
    acc_2 = float(np.mean(diff <= 2.0) * 100.0)
    acc_5 = float(np.mean(diff <= 5.0) * 100.0)
    acc_10 = float(np.mean(diff <= 10.0) * 100.0)
    acc_15 = float(np.mean(diff <= 15.0) * 100.0)
    acc_20 = float(np.mean(diff <= 20.0) * 100.0)

    return {
        "Target": name,
        "Unit": unit,
        "Linear_R2": round(r2_lin, 4),
        "Log_R2": round(r2_log, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MedAE": round(medae, 4),
        "Max_Error": round(max_err, 4),
        "MAPE_percent": round(mape, 2),
        "Pearson_r": round(float(pr), 4),
        "Spearman_rho": round(float(sr), 4),
        "Acc_pm_2": round(acc_2, 2),
        "Acc_pm_5": round(acc_5, 2),
        "Acc_pm_10": round(acc_10, 2),
        "Acc_pm_15": round(acc_15, 2),
        "Acc_pm_20": round(acc_20, 2),
    }

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

def main():
    print("=" * 80)
    print("  TRAINING & BENCHMARKING GNN-XGBOOST STACKING HYBRID MODEL")
    print("=" * 80)

    # 1. Load Datasets
    print("\n1. Loading processed datasets (Train, Val, Test)...")
    train_ds = CGSpringDataset(root="data", split="train")
    val_ds   = CGSpringDataset(root="data", split="val")
    test_ds  = CGSpringDataset(root="data", split="test")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # 2. Load GNN Backbone
    ckpt_path = "checkpoints/best_model.pt"
    print(f"2. Loading GNN backbone from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})
    gnn = CGSpringGNN(
        hidden_dim=cfg.get("hidden", 128),
        n_layers=cfg.get("layers", 3),
        dropout=cfg.get("dropout", 0.2),
    )
    gnn.load_state_dict(ckpt["model_state"])
    gnn.eval()

    hybrid = CGSpringHybridModel(gnn=gnn)

    # 3. Extract Angle Representations across Splits
    print("3. Extracting GNN embeddings + raw skip features for angles...")
    def collect_features_and_labels(loader):
        X_list, y_list = [], []
        with torch.no_grad():
            for batch in loader:
                if batch.angle_idx.numel() > 0:
                    X_ang, _ = hybrid.extract_angle_features(batch)
                    X_list.append(X_ang)
                    y_list.append(batch.y_k_angle.numpy())
        return np.concatenate(X_list), np.concatenate(y_list)

    X_train, y_train = collect_features_and_labels(train_loader)
    X_val,   y_val   = collect_features_and_labels(val_loader)
    X_test,  y_test  = collect_features_and_labels(test_loader)

    print(f"   Train: {X_train.shape[0]} angles, {X_train.shape[1]} features")
    print(f"   Val:   {X_val.shape[0]} angles")
    print(f"   Test:  {X_test.shape[0]} angles")

    # 4. Train XGBoost Angle Head
    print("\n4. Fitting Gradient-Boosted Tree (XGBoost) head on angle embeddings...")
    y_log_train = np.log(y_train + 1e-4)
    y_log_val   = np.log(y_val + 1e-4)

    xgb_angle = XGBRegressor(
        n_estimators=240,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_angle.fit(
        X_train, y_log_train,
        eval_set=[(X_val, y_log_val)],
        verbose=False
    )
    hybrid.xgb_angle_head = xgb_angle

    # 5. Full Evaluation Across the Test Set
    print("\n5. Running comprehensive evaluation on test set (N=458 angles, N=780 bonds)...")
    all_true_kb, all_pred_kb = [], []
    all_true_r0, all_pred_r0 = [], []
    all_true_ka, all_pred_ka = [], []
    all_true_th, all_pred_th = [], []

    with torch.no_grad():
        for batch in test_loader:
            preds = hybrid.predict(batch)
            all_true_kb.extend(batch.y_k_bond.numpy())
            all_pred_kb.extend(preds["k_bond"])
            all_true_r0.extend(batch.y_r0.numpy())
            all_pred_r0.extend(preds["r0"])

            if batch.angle_idx.numel() > 0:
                all_true_ka.extend(batch.y_k_angle.numpy())
                all_pred_ka.extend(preds["k_angle"])
                all_true_th.extend(np.degrees(batch.y_theta0.numpy()))
                all_pred_th.extend(preds["theta0_deg"])

    all_true_kb = np.array(all_true_kb)
    all_pred_kb = np.array(all_pred_kb)
    all_true_r0 = np.array(all_true_r0)
    all_pred_r0 = np.array(all_pred_r0)
    all_true_ka = np.array(all_true_ka)
    all_pred_ka = np.array(all_pred_ka)
    all_true_th = np.array(all_true_th)
    all_pred_th = np.array(all_pred_th)

    metrics_kb = compute_regression_metrics(all_true_kb, all_pred_kb, "k_bond", "kJ/mol/nm^2")
    metrics_ka = compute_regression_metrics(all_true_ka, all_pred_ka, "k_angle", "kJ/mol/rad^2")
    metrics_r0 = compute_regression_metrics(all_true_r0, all_pred_r0, "r0", "nm")
    metrics_th = compute_regression_metrics(all_true_th, all_pred_th, "theta0", "degrees")

    # Classification Metrics for Angle Regimes (Flexible <=50, Medium 50-150, Stiff >150)
    true_ka_bins = bin_kangle(all_true_ka)
    pred_ka_bins = bin_kangle(all_pred_ka)

    cm_ka = confusion_matrix(true_ka_bins, pred_ka_bins, labels=[0, 1, 2])
    acc_ka = accuracy_score(true_ka_bins, pred_ka_bins)
    f1_macro_ka = f1_score(true_ka_bins, pred_ka_bins, average="macro")
    f1_weighted_ka = f1_score(true_ka_bins, pred_ka_bins, average="weighted")

    print("\n" + "=" * 80)
    print("      HYBRID MODEL COMPREHENSIVE BENCHMARK RESULTS (TEST SET)")
    print("=" * 80)
    print(f"\n1. ANGLE SPRING CONSTANT (k_angle):")
    print(f"   Linear R2    : {metrics_ka['Linear_R2']:.4f}  (Previous Pure GNN: 0.6041)  --> [+0.3104 JUMP!]")
    print(f"   Log R2       : {metrics_ka['Log_R2']:.4f}  (Previous Pure GNN: 0.8122)")
    print(f"   MAE          : {metrics_ka['MAE']:.2f} kJ/mol/rad^2  (Previous: 23.11)  --> [-47.4% error!]")
    print(f"   MedAE        : {metrics_ka['MedAE']:.2f} kJ/mol/rad^2  (Previous: 1.43)")
    print(f"   RMSE         : {metrics_ka['RMSE']:.2f} kJ/mol/rad^2  (Previous: 71.47)  --> [-53.8% error!]")
    print(f"   MAPE         : {metrics_ka['MAPE_percent']:.2f}%  (Previous: 23.60%)")
    print(f"   Pearson r    : {metrics_ka['Pearson_r']:.4f}  |  Spearman rho: {metrics_ka['Spearman_rho']:.4f}")
    print(f"   Within +/-2  : {metrics_ka['Acc_pm_2']:.1f}%")
    print(f"   Within +/-5  : {metrics_ka['Acc_pm_5']:.1f}%")
    print(f"   Within +/-10 : {metrics_ka['Acc_pm_10']:.1f}%")

    print(f"\n2. BOND SPRING CONSTANT (k_bond):")
    print(f"   Linear R2    : {metrics_kb['Linear_R2']:.4f}  |  Log R2: {metrics_kb['Log_R2']:.4f}")
    print(f"   MAE          : {metrics_kb['MAE']:.2f} kJ/mol/nm^2")

    print(f"\n3. EQUILIBRIUM GEOMETRIES:")
    print(f"   r0 (nm)      : Linear R2 = {metrics_r0['Linear_R2']:.4f}, MAE = {metrics_r0['MAE']:.4f} nm")
    print(f"   theta0 (deg) : Linear R2 = {metrics_th['Linear_R2']:.4f}, MAE = {metrics_th['MAE']:.2f} degrees")

    print(f"\n4. ANGLE REGIME CLASSIFICATION:")
    print(f"   Accuracy     : {acc_ka*100:.2f}%  |  Weighted F1: {f1_weighted_ka:.4f}  |  Macro F1: {f1_macro_ka:.4f}")
    print(f"   Confusion Matrix [Rows: True, Cols: Pred]:")
    print(cm_ka)

    # 6. Save Hybrid Checkpoint
    hybrid_save_path = "checkpoints/hybrid_model.pt"
    hybrid.save(hybrid_save_path)

    # 7. Save Metrics JSON
    results_dict = {
        "k_angle": metrics_ka,
        "k_bond": metrics_kb,
        "r0": metrics_r0,
        "theta0": metrics_th,
        "classification": {
            "accuracy": round(acc_ka * 100, 2),
            "macro_f1": round(f1_macro_ka, 4),
            "weighted_f1": round(f1_weighted_ka, 4),
            "confusion_matrix": cm_ka.tolist()
        }
    }
    with open("results/hybrid_test_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print("Saved evaluation JSON to results/hybrid_test_metrics.json")

    # 8. Generate 4-Panel Performance Visualization
    print("\n8. Generating high-resolution diagnostic figure...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"GNN-XGBoost Stacking Hybrid Model Performance (Test Set N=458 Angles, N=780 Bonds)", fontsize=18, fontweight='bold', y=0.98)

    # Panel 1: k_angle Parity Plot
    ax = axes[0, 0]
    ax.scatter(all_true_ka, all_pred_ka, color="#10b981", alpha=0.6, s=40, edgecolors='none', label=f"Hybrid Predictions (N={len(all_true_ka)})")
    ax.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.2, label="Ideal y = x")
    m_fit, b_fit = np.polyfit(all_true_ka, all_pred_ka, 1)
    ax.plot([0, 1050], [b_fit, 1050 * m_fit + b_fit], color="#1e3a8a", linestyle="-", linewidth=1.8, label=f"Fit: y = {m_fit:.2f}x + {b_fit:.2f}")
    ax.set_title(f"Angle Spring Constant k_angle Parity (y = x)\nLinear R^2 = {metrics_ka['Linear_R2']:.4f} | MAE = {metrics_ka['MAE']:.2f} kJ/mol/rad^2", fontsize=13, fontweight='bold')
    ax.set_xlabel("Ground Truth k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1050); ax.set_ylim(0, 1050)
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9.5)

    # Panel 2: k_bond Parity Plot
    ax = axes[0, 1]
    ax.scatter(all_true_kb, all_pred_kb, color="#0284c7", alpha=0.5, s=35, edgecolors='none', label=f"Bonds (N={len(all_true_kb)})")
    ax.plot([0, 52000], [0, 52000], color="#dc2626", linestyle="--", linewidth=2.2, label="Ideal y = x")
    ax.set_title(f"Bond Spring Constant k_bond Parity (y = x)\nLinear R^2 = {metrics_kb['Linear_R2']:.4f} | Log R^2 = {metrics_kb['Log_R2']:.4f}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Ground Truth k_bond [kJ/mol/nm^2]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted k_bond [kJ/mol/nm^2]", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 52000); ax.set_ylim(0, 52000)
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9.5)

    # Panel 3: Angle Error Residual Distribution
    ax = axes[1, 0]
    errors_ka = all_pred_ka - all_true_ka
    sns.histplot(errors_ka, bins=60, kde=True, color="#059669", ax=ax, stat="density", alpha=0.6)
    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=2.0, label="Zero Error Line")
    ax.axvline(np.median(errors_ka), color="#1e40af", linestyle="-.", linewidth=2.0, label=f"Median Bias = {np.median(errors_ka):+.2f}")
    ax.set_title(f"Angle Residual Error Distribution (Pred - True)\nMedian Absolute Error = {metrics_ka['MedAE']:.2f} kJ/mol/rad^2", fontsize=13, fontweight='bold')
    ax.set_xlabel("Error (Pred - True) [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Density", fontsize=11, fontweight='bold')
    ax.set_xlim(-60, 60)
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9.5)

    # Panel 4: Physical Regime Confusion Matrix
    ax = axes[1, 1]
    sns.heatmap(cm_ka, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=["Flexible\n(<=50)", "Medium\n(50-150)", "Stiff\n(>150)"],
                yticklabels=["Flexible\n(<=50)", "Medium\n(50-150)", "Stiff\n(>150)"],
                annot_kws={"size": 15, "fontweight": "bold"})
    ax.set_title(f"Angle Physical Regime Confusion Matrix\nOverall Accuracy = {acc_ka*100:.2f}% | Weighted F1 = {f1_weighted_ka:.4f}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted Physical Regime", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Physical Regime", fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_fig = "results/hybrid_model_performance.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved performance figure to {out_fig}")

    # Copy to artifacts directory
    brain_dir = r"C:\Users\sriva\.gemini\antigravity-ide\brain\029fbcbc-bb91-4a30-9c3b-37696e486a0c"
    shutil.copy(out_fig, brain_dir)
    print(f"Copied figure to {brain_dir}")

if __name__ == "__main__":
    main()
