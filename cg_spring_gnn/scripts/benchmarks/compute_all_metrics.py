"""
compute_all_metrics.py
----------------------
Computes comprehensive evaluation matrices and metrics for the CGSpringGNN model on the held-out test set:
1. Classification Matrices & Metrics:
   - F1-Score (Macro, Micro, Weighted, and Per-Class)
   - Precision & Recall (Macro, Micro, Weighted, and Per-Class)
   - Confusion Matrix (Actual vs. Predicted force regimes)
   - Classification Accuracy
2. Tolerance Band Accuracy Matrices:
   - Relative error tolerances (within 5%, 10%, 15%, 20%, 30%, 50%)
   - Absolute physical tolerances
3. Comprehensive Regression Metrics Matrix:
   - R^2 (Linear and Log-space)
   - RMSE, MAE, Median Absolute Error (MedAE)
   - MAPE (%)
   - Pearson Correlation (r) & Spearman Rank Correlation (rho)
   - Max Error
4. Confusion Matrix Visual Plot (results/confusion_matrices.png)
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
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, accuracy_score,
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
)
from scipy.stats import pearsonr, spearmanr

from src.data.dataset import CGSpringDataset
from src.models.gnn import CGSpringGNN

def bin_kbond(k_vals):
    """
    Bin k_bond into 3 physical force regimes:
    0: Flexible / aliphatic chain bonds (k <= 7000 kJ/mol/nm^2)
    1: Semi-rigid / carbohydrate / intermediate bonds (7000 < k < 25000 kJ/mol/nm^2)
    2: Ultra-stiff / aromatic / rigid ring constraints (k >= 25000 kJ/mol/nm^2)
    """
    bins = []
    for k in k_vals:
        if k <= 7000:
            bins.append(0)
        elif k < 25000:
            bins.append(1)
        else:
            bins.append(2)
    return np.array(bins)

def bin_kangle(k_vals):
    """
    Bin k_angle into 3 physical regimes:
    0: Flexible angles (k <= 50 kJ/mol/rad^2)
    1: Medium / standard angles (50 < k <= 150 kJ/mol/rad^2)
    2: Stiff / planar angles (k > 150 kJ/mol/rad^2)
    """
    bins = []
    for k in k_vals:
        if k <= 50:
            bins.append(0)
        elif k <= 150:
            bins.append(1)
        else:
            bins.append(2)
    return np.array(bins)

def compute_regression_metrics(true, pred, name, unit):
    r2_lin = float(r2_score(true, pred))
    log_true = np.log10(np.clip(true, 1e-6, None))
    log_pred = np.log10(np.clip(pred, 1e-6, None))
    r2_log = float(r2_score(log_true, log_pred)) if len(true) > 1 else 0.0
    
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    medae = float(median_absolute_error(true, pred))
    max_err = float(np.max(np.abs(true - pred)))
    
    # MAPE
    non_zero = true > 1e-4
    mape = float(np.mean(np.abs((pred[non_zero] - true[non_zero]) / true[non_zero])) * 100.0)
    
    pr, _ = pearsonr(true, pred)
    sr, _ = spearmanr(true, pred)
    
    return {
        "Target": name,
        "Unit": unit,
        "Linear_R2": round(r2_lin, 4),
        "Log_R2": round(r2_log, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Median_Absolute_Error": round(medae, 4),
        "Max_Error": round(max_err, 4),
        "MAPE_percent": round(mape, 2),
        "Pearson_r": round(float(pr), 4),
        "Spearman_rho": round(float(sr), 4)
    }

def compute_tolerance_accuracies(true, pred):
    rel_err = np.abs(pred - true) / np.clip(true, 1e-4, None)
    return {
        "within_5_percent": round(float(np.mean(rel_err <= 0.05) * 100.0), 2),
        "within_10_percent": round(float(np.mean(rel_err <= 0.10) * 100.0), 2),
        "within_15_percent": round(float(np.mean(rel_err <= 0.15) * 100.0), 2),
        "within_20_percent": round(float(np.mean(rel_err <= 0.20) * 100.0), 2),
        "within_25_percent": round(float(np.mean(rel_err <= 0.25) * 100.0), 2),
        "within_30_percent": round(float(np.mean(rel_err <= 0.30) * 100.0), 2),
        "within_50_percent": round(float(np.mean(rel_err <= 0.50) * 100.0), 2)
    }

def main():
    os.makedirs("results", exist_ok=True)
    
    ckpt_path = "checkpoints/best_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})
    model = CGSpringGNN(
        hidden_dim=cfg.get("hidden", 256),
        n_layers=cfg.get("layers", 4),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    test_ds = CGSpringDataset(root="data", split="test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    true_kb, pred_kb = [], []
    true_ka, pred_ka = [], []
    true_r0, pred_r0 = [], []
    true_th, pred_th = [], []
    
    pred_regime_logits_list = []
    
    with torch.no_grad():
        for batch in loader:
            p_kb, p_r0, p_ka, p_th, p_reg = model(batch, return_regime=True)
            true_kb.extend(batch.y_k_bond.numpy())
            pred_kb.extend(p_kb.numpy())
            true_r0.extend(batch.y_r0.numpy())
            pred_r0.extend(p_r0.numpy())
            if p_ka.numel() > 0:
                true_ka.extend(batch.y_k_angle.numpy())
                pred_ka.extend(p_ka.numpy())
                true_th.extend(batch.y_theta0.numpy())
                pred_th.extend(p_th.numpy())
                pred_regime_logits_list.extend(p_reg.numpy())
                
    true_kb, pred_kb = np.array(true_kb), np.array(pred_kb)
    true_ka, pred_ka = np.array(true_ka), np.array(pred_ka)
    true_r0, pred_r0 = np.array(true_r0), np.array(pred_r0)
    true_th, pred_th = np.array(true_th), np.array(pred_th)
    pred_head_cls = np.argmax(np.array(pred_regime_logits_list), axis=-1) if len(pred_regime_logits_list) > 0 else np.array([])
    
    print("=" * 70)
    print("      COMPREHENSIVE EVALUATION MATRICES & BENCHMARKS (TEST SET)")
    print("=" * 70)
    print(f"Evaluated on: {len(true_kb)} bonds, {len(true_ka)} angles across {len(test_ds)} molecules\n")
    
    # ── 1. REGRESSION MATRICES ──────────────────────────────────────────────
    reg_kb = compute_regression_metrics(true_kb, pred_kb, "k_bond", "kJ/mol/nm^2")
    reg_ka = compute_regression_metrics(true_ka, pred_ka, "k_angle", "kJ/mol/rad^2")
    reg_r0 = compute_regression_metrics(true_r0, pred_r0, "r0", "nm")
    reg_th = compute_regression_metrics(true_th, pred_th, "theta0", "degrees")
    
    regression_matrix = [reg_kb, reg_ka, reg_r0, reg_th]
    
    # ── 2. TOLERANCE BAND ACCURACY MATRICES ─────────────────────────────────
    tol_kb = compute_tolerance_accuracies(true_kb, pred_kb)
    tol_ka = compute_tolerance_accuracies(true_ka, pred_ka)
    tol_r0 = compute_tolerance_accuracies(true_r0, pred_r0)
    tol_th = compute_tolerance_accuracies(true_th, pred_th)
    
    abs_tol_kb = {
        "within_500_kJ_mol_nm2": round(float(np.mean(np.abs(pred_kb - true_kb) <= 500) * 100.0), 2),
        "within_1000_kJ_mol_nm2": round(float(np.mean(np.abs(pred_kb - true_kb) <= 1000) * 100.0), 2),
        "within_2500_kJ_mol_nm2": round(float(np.mean(np.abs(pred_kb - true_kb) <= 2500) * 100.0), 2),
        "within_5000_kJ_mol_nm2": round(float(np.mean(np.abs(pred_kb - true_kb) <= 5000) * 100.0), 2),
    }
    
    abs_tol_r0 = {
        "within_0.01_nm (0.1 A)": round(float(np.mean(np.abs(pred_r0 - true_r0) <= 0.01) * 100.0), 2),
        "within_0.02_nm (0.2 A)": round(float(np.mean(np.abs(pred_r0 - true_r0) <= 0.02) * 100.0), 2),
        "within_0.05_nm (0.5 A)": round(float(np.mean(np.abs(pred_r0 - true_r0) <= 0.05) * 100.0), 2),
    }
    
    abs_tol_th = {
        "within_5_deg": round(float(np.mean(np.abs(pred_th - true_th) <= 5.0) * 100.0), 2),
        "within_10_deg": round(float(np.mean(np.abs(pred_th - true_th) <= 10.0) * 100.0), 2),
        "within_15_deg": round(float(np.mean(np.abs(pred_th - true_th) <= 15.0) * 100.0), 2),
    }

    # ── 3. CLASSIFICATION MATRICES (F1, PRECISION, RECALL, CONFUSION MATRIX) ─
    # Bond force regimes
    y_true_kb_bins = bin_kbond(true_kb)
    y_pred_kb_bins = bin_kbond(pred_kb)
    kb_classes = ["Flexible (<=7k)", "Semi-Rigid (7k-25k)", "Rigid Ring (>=25k)"]
    
    cm_kb = confusion_matrix(y_true_kb_bins, y_pred_kb_bins, labels=[0, 1, 2])
    f1_macro_kb = float(f1_score(y_true_kb_bins, y_pred_kb_bins, average="macro"))
    f1_weighted_kb = float(f1_score(y_true_kb_bins, y_pred_kb_bins, average="weighted"))
    f1_per_class_kb = [round(float(x), 4) for x in f1_score(y_true_kb_bins, y_pred_kb_bins, average=None)]
    
    prec_macro_kb = float(precision_score(y_true_kb_bins, y_pred_kb_bins, average="macro"))
    prec_weighted_kb = float(precision_score(y_true_kb_bins, y_pred_kb_bins, average="weighted"))
    prec_per_class_kb = [round(float(x), 4) for x in precision_score(y_true_kb_bins, y_pred_kb_bins, average=None)]
    
    rec_macro_kb = float(recall_score(y_true_kb_bins, y_pred_kb_bins, average="macro"))
    rec_weighted_kb = float(recall_score(y_true_kb_bins, y_pred_kb_bins, average="weighted"))
    rec_per_class_kb = [round(float(x), 4) for x in recall_score(y_true_kb_bins, y_pred_kb_bins, average=None)]
    acc_kb = float(accuracy_score(y_true_kb_bins, y_pred_kb_bins))

    # Angle force regimes
    y_true_ka_bins = bin_kangle(true_ka)
    y_pred_ka_bins = bin_kangle(pred_ka)
    ka_classes = ["Flexible (<=50)", "Medium (50-150)", "Stiff (>150)"]
    
    cm_ka = confusion_matrix(y_true_ka_bins, y_pred_ka_bins, labels=[0, 1, 2])
    f1_macro_ka = float(f1_score(y_true_ka_bins, y_pred_ka_bins, average="macro"))
    f1_weighted_ka = float(f1_score(y_true_ka_bins, y_pred_ka_bins, average="weighted"))
    f1_per_class_ka = [round(float(x), 4) for x in f1_score(y_true_ka_bins, y_pred_ka_bins, average=None)]
    
    prec_macro_ka = float(precision_score(y_true_ka_bins, y_pred_ka_bins, average="macro"))
    prec_weighted_ka = float(precision_score(y_true_ka_bins, y_pred_ka_bins, average="weighted"))
    prec_per_class_ka = [round(float(x), 4) for x in precision_score(y_true_ka_bins, y_pred_ka_bins, average=None)]
    
    rec_macro_ka = float(recall_score(y_true_ka_bins, y_pred_ka_bins, average="macro"))
    rec_weighted_ka = float(recall_score(y_true_ka_bins, y_pred_ka_bins, average="weighted"))
    rec_per_class_ka = [round(float(x), 4) for x in recall_score(y_true_ka_bins, y_pred_ka_bins, average=None)]
    acc_ka = float(accuracy_score(y_true_ka_bins, y_pred_ka_bins))
    
    classification_results = {
        "k_bond_Classification": {
            "Classes": kb_classes,
            "Accuracy": round(acc_kb * 100.0, 2),
            "F1_Macro": round(f1_macro_kb, 4),
            "F1_Weighted": round(f1_weighted_kb, 4),
            "F1_Per_Class": dict(zip(kb_classes, f1_per_class_kb)),
            "Precision_Macro": round(prec_macro_kb, 4),
            "Precision_Weighted": round(prec_weighted_kb, 4),
            "Precision_Per_Class": dict(zip(kb_classes, prec_per_class_kb)),
            "Recall_Macro": round(rec_macro_kb, 4),
            "Recall_Weighted": round(rec_weighted_kb, 4),
            "Recall_Per_Class": dict(zip(kb_classes, rec_per_class_kb)),
            "Confusion_Matrix": cm_kb.tolist(),
        },
        "k_angle_Classification": {
            "Classes": ka_classes,
            "Accuracy": round(acc_ka * 100.0, 2),
            "F1_Macro": round(f1_macro_ka, 4),
            "F1_Weighted": round(f1_weighted_ka, 4),
            "F1_Per_Class": dict(zip(ka_classes, f1_per_class_ka)),
            "Precision_Macro": round(prec_macro_ka, 4),
            "Precision_Weighted": round(prec_weighted_ka, 4),
            "Precision_Per_Class": dict(zip(ka_classes, prec_per_class_ka)),
            "Recall_Macro": round(rec_macro_ka, 4),
            "Recall_Weighted": round(rec_weighted_ka, 4),
            "Recall_Per_Class": dict(zip(ka_classes, rec_per_class_ka)),
            "Confusion_Matrix": cm_ka.tolist(),
        }
    }

    # ── 4. VISUALIZE CONFUSION MATRICES ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.heatmap(cm_kb, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=kb_classes, yticklabels=kb_classes, cbar=False)
    axes[0].set_title(f"k_bond Physical Regime Confusion Matrix\nOverall Accuracy: {acc_kb*100:.1f}% | Macro F1: {f1_macro_kb:.3f}")
    axes[0].set_xlabel("Predicted Force Regime")
    axes[0].set_ylabel("True Force Regime")
    
    sns.heatmap(cm_ka, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                xticklabels=ka_classes, yticklabels=ka_classes, cbar=False)
    axes[1].set_title(f"k_angle Physical Regime Confusion Matrix\nOverall Accuracy: {acc_ka*100:.1f}% | Macro F1: {f1_macro_ka:.3f}")
    axes[1].set_xlabel("Predicted Force Regime")
    axes[1].set_ylabel("True Force Regime")
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] Saved: results/confusion_matrices.png")

    # ── 5. SAVE ALL METRICS TO JSON ──────────────────────────────────────────
    full_output = {
        "Regression_Metrics_Matrix": regression_matrix,
        "Relative_Tolerance_Accuracies": {
            "k_bond": tol_kb,
            "k_angle": tol_ka,
            "r0": tol_r0,
            "theta0": tol_th
        },
        "Absolute_Tolerance_Accuracies": {
            "k_bond": abs_tol_kb,
            "r0": abs_tol_r0,
            "theta0": abs_tol_th
        },
        "Classification_and_F1_Matrices": classification_results
    }
    
    with open("results/all_evaluation_matrices.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print("[OK] Saved: results/all_evaluation_matrices.json")
    
    # ── 6. PRINT FORMATTED SUMMARY TABLES ────────────────────────────────────
    print("\n--- 1. REGRESSION METRICS MATRIX ---")
    header = f"{'Target':<10} | {'Linear R2':<10} | {'Log R2':<8} | {'RMSE':<10} | {'MAE':<10} | {'MedAE':<8} | {'MAPE(%)':<8} | {'Pearson r':<9} | {'Spearman':<8}"
    print(header)
    print("-" * len(header))
    for m in regression_matrix:
        print(f"{m['Target']:<10} | {m['Linear_R2']:<10.4f} | {m['Log_R2']:<8.4f} | {m['RMSE']:<10.2f} | {m['MAE']:<10.2f} | {m['Median_Absolute_Error']:<8.2f} | {m['MAPE_percent']:<8.2f} | {m['Pearson_r']:<9.4f} | {m['Spearman_rho']:<8.4f}")
        
    print("\n--- 2. CLASSIFICATION & F1-SCORE METRICS (k_bond Regimes) ---")
    print(f"Overall Classification Accuracy: {acc_kb*100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro_kb:.4f}  |  Weighted F1-Score: {f1_weighted_kb:.4f}")
    print(f"Macro Precision:   {prec_macro_kb:.4f}  |  Macro Recall:       {rec_macro_kb:.4f}")
    print("\nPer-Class Breakdown:")
    for cls in kb_classes:
        print(f"  - {cls:<25}: F1 = {classification_results['k_bond_Classification']['F1_Per_Class'][cls]:.4f} | Precision = {classification_results['k_bond_Classification']['Precision_Per_Class'][cls]:.4f} | Recall = {classification_results['k_bond_Classification']['Recall_Per_Class'][cls]:.4f}")
        
    print("\nConfusion Matrix (k_bond) [Rows: True, Cols: Pred]:")
    print(f"{'':<25} | " + " | ".join([f"{c:<18}" for c in kb_classes]))
    for i, row in enumerate(cm_kb):
        print(f"{kb_classes[i]:<25} | " + " | ".join([f"{val:<18}" for val in row]))

    print("\n--- 3. CLASSIFICATION & F1-SCORE METRICS (k_angle Regimes) ---")
    print(f"Overall Classification Accuracy: {acc_ka*100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro_ka:.4f}  |  Weighted F1-Score: {f1_weighted_ka:.4f}")
    print(f"Macro Precision:   {prec_macro_ka:.4f}  |  Macro Recall:       {rec_macro_ka:.4f}")
    print("\nPer-Class Breakdown:")
    for cls in ka_classes:
        print(f"  - {cls:<25}: F1 = {classification_results['k_angle_Classification']['F1_Per_Class'][cls]:.4f} | Precision = {classification_results['k_angle_Classification']['Precision_Per_Class'][cls]:.4f} | Recall = {classification_results['k_angle_Classification']['Recall_Per_Class'][cls]:.4f}")

    print("\nConfusion Matrix (k_angle) [Rows: True, Cols: Pred]:")
    print(f"{'':<25} | " + " | ".join([f"{c:<18}" for c in ka_classes]))
    for i, row in enumerate(cm_ka):
        print(f"{ka_classes[i]:<25} | " + " | ".join([f"{val:<18}" for val in row]))

    print("\n--- 4. TOLERANCE BAND ACCURACY MATRIX (% of test predictions within tolerance) ---")
    print(f"{'Target':<10} | {'<= 5%':<8} | {'<= 10%':<8} | {'<= 15%':<8} | {'<= 20%':<8} | {'<= 25%':<8} | {'<= 30%':<8} | {'<= 50%':<8}")
    print("-" * 75)
    for name, tol in [("k_bond", tol_kb), ("k_angle", tol_ka), ("r0", tol_r0), ("theta0", tol_th)]:
        print(f"{name:<10} | {tol['within_5_percent']:<8.1f}% | {tol['within_10_percent']:<8.1f}% | {tol['within_15_percent']:<8.1f}% | {tol['within_20_percent']:<8.1f}% | {tol['within_25_percent']:<8.1f}% | {tol['within_30_percent']:<8.1f}% | {tol['within_50_percent']:<8.1f}%")

if __name__ == "__main__":
    main()
