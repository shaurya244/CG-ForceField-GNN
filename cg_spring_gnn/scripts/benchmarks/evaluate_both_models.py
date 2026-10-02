"""
evaluate_both_models.py
-----------------------
Evaluates and compares:
  1. General Model (checkpoints/best_model.pt) - trained on 0-100th percentile
  2. P80 Specialized Model (checkpoints/best_model_p80.pt) - trained on 0-80th percentile

Evaluates both models on:
  - Benchmark A: Full Test Set (N=458 angles)
  - Benchmark B: 0-80th Percentile Test Set (k <= 78.10 kJ/mol/rad^2, N=361 angles)

Generates:
  - results/model_comparison_p80.json
  - results/model_comparison_general_vs_p80.png
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
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error,
    confusion_matrix, classification_report, accuracy_score, f1_score
)
from torch_geometric.loader import DataLoader
from src.data.dataset import CGSpringDataset
from src.models.gnn import CGSpringGNN

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'

P80 = 78.10

def load_gnn(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})
    model = CGSpringGNN(
        hidden_dim=cfg.get("hidden", 256),
        n_layers=cfg.get("layers", 4),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def run_inference(model, loader):
    t_ka, p_ka, pred_logits = [], [], []
    t_kb, p_kb = [], []
    with torch.no_grad():
        for batch in loader:
            kb_p, r0_p, ka_p, th_p, reg_p = model(batch, return_regime=True)
            t_kb.extend(batch.y_k_bond.numpy())
            p_kb.extend(kb_p.numpy())
            if ka_p.numel() > 0:
                t_ka.extend(batch.y_k_angle.numpy())
                p_ka.extend(ka_p.numpy())
                pred_logits.extend(reg_p.numpy())
    return np.array(t_ka), np.array(p_ka), np.array(pred_logits), np.array(t_kb), np.array(p_kb)

def get_regression_metrics(true, pred):
    r2_lin = float(r2_score(true, pred))
    log_t = np.log10(np.clip(true, 1e-4, None))
    log_p = np.log10(np.clip(pred, 1e-4, None))
    r2_log = float(r2_score(log_t, log_p)) if len(true) > 1 else 0.0
    mae = float(mean_absolute_error(true, pred))
    medae = float(median_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
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
        "N": len(true),
        "Linear_R2": round(r2_lin, 4),
        "Log_R2": round(r2_log, 4),
        "MAE": round(mae, 2),
        "MedAE": round(medae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_percent": round(mape, 2),
        "Pearson_r": round(float(pr), 4),
        "Spearman_rho": round(float(sr), 4),
        "Accuracy_pm_2": round(acc_2, 2),
        "Accuracy_pm_5": round(acc_5, 2),
        "Accuracy_pm_10": round(acc_10, 2),
        "Accuracy_pm_15": round(acc_15, 2),
        "Accuracy_pm_20": round(acc_20, 2),
    }

def main():
    test_ds = CGSpringDataset(root="data", split="test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("Loading General Model...")
    gen_model = load_gnn("checkpoints/best_model.pt")
    t_ka_gen, p_ka_gen, logits_gen, _, _ = run_inference(gen_model, test_loader)

    print("Loading P80 Specialized Model...")
    p80_model = load_gnn("checkpoints/best_model_p80.pt")
    t_ka_p80, p_ka_p80, logits_p80, _, _ = run_inference(p80_model, test_loader)

    mask_p80 = t_ka_gen <= P80

    results = {
        "Benchmark_Full_Test_Set": {
            "General_Model": get_regression_metrics(t_ka_gen, p_ka_gen),
            "P80_Model": get_regression_metrics(t_ka_p80, p_ka_p80),
        },
        "Benchmark_P80_Subset_k_le_78": {
            "General_Model": get_regression_metrics(t_ka_gen[mask_p80], p_ka_gen[mask_p80]),
            "P80_Model": get_regression_metrics(t_ka_p80[mask_p80], p_ka_p80[mask_p80]),
        }
    }

    print("\n" + "=" * 80)
    print("      SIDE-BY-SIDE MODEL BENCHMARK: GENERAL MODEL vs P80 MODEL")
    print("=" * 80)
    print(f"\n1. ON P80 SUBSET (k <= {P80} kJ/mol/rad^2, N = {np.sum(mask_p80)}):")
    print(f"{'Metric':<25} | {'General Model':>15} | {'P80 Model':>15} | {'Delta':>15}")
    print("-" * 75)
    for m in ["MAE", "MedAE", "RMSE", "MAPE_percent", "Linear_R2", "Log_R2", "Pearson_r", "Spearman_rho", "Accuracy_pm_2", "Accuracy_pm_5", "Accuracy_pm_10"]:
        v_gen = results["Benchmark_P80_Subset_k_le_78"]["General_Model"][m]
        v_p80 = results["Benchmark_P80_Subset_k_le_78"]["P80_Model"][m]
        delta = v_p80 - v_gen
        sign = "+" if delta > 0 else ""
        print(f"{m:<25} | {v_gen:>15} | {v_p80:>15} | {sign}{delta:>14.2f}")

    print("\n" + "=" * 80)
    print(f"2. ON FULL TEST SET (0 to 100th Percentile, N = {len(t_ka_gen)}):")
    print(f"{'Metric':<25} | {'General Model':>15} | {'P80 Model':>15}")
    print("-" * 60)
    for m in ["MAE", "MedAE", "Linear_R2", "Log_R2", "MAPE_percent"]:
        v_gen = results["Benchmark_Full_Test_Set"]["General_Model"][m]
        v_p80 = results["Benchmark_Full_Test_Set"]["P80_Model"][m]
        print(f"{m:<25} | {v_gen:>15} | {v_p80:>15}")

    with open("results/model_comparison_p80.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved comparison to results/model_comparison_p80.json")

    # Generate 4-panel comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comparison: General Model vs Specialized P80 Model", fontsize=18, fontweight='bold', y=0.98)

    # Panel 1: Parity Plot on P80 Subset - General Model
    ax = axes[0, 0]
    sub_t = t_ka_gen[mask_p80]
    sub_p_gen = p_ka_gen[mask_p80]
    ax.scatter(sub_t, sub_p_gen, color="#0284c7", alpha=0.55, edgecolors='none', s=45)
    ax.plot([0, 85], [0, 85], color="#dc2626", linestyle="--", linewidth=2.0, label="Ideal y=x")
    ax.fill_between([0, 85], [0-5, 85-5], [0+5, 85+5], color="#10b981", alpha=0.15, label="+/- 5 Band")
    mae_g = results['Benchmark_P80_Subset_k_le_78']['General_Model']['MAE']
    med_g = results['Benchmark_P80_Subset_k_le_78']['General_Model']['MedAE']
    ax.set_title(f"General Model on k <= P80\nMAE = {mae_g:.2f}, MedAE = {med_g:.2f} kJ/mol/rad^2", fontsize=13, fontweight='bold')
    ax.set_xlabel("True k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted k_angle", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 85); ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", fontsize=9.5)

    # Panel 2: Parity Plot on P80 Subset - P80 Model
    ax = axes[0, 1]
    sub_p_p80 = p_ka_p80[mask_p80]
    ax.scatter(sub_t, sub_p_p80, color="#10b981", alpha=0.55, edgecolors='none', s=45)
    ax.plot([0, 85], [0, 85], color="#dc2626", linestyle="--", linewidth=2.0, label="Ideal y=x")
    ax.fill_between([0, 85], [0-5, 85-5], [0+5, 85+5], color="#10b981", alpha=0.15, label="+/- 5 Band")
    mae_p = results['Benchmark_P80_Subset_k_le_78']['P80_Model']['MAE']
    med_p = results['Benchmark_P80_Subset_k_le_78']['P80_Model']['MedAE']
    ax.set_title(f"Specialized P80 Model on k <= P80\nMAE = {mae_p:.2f}, MedAE = {med_p:.2f} kJ/mol/rad^2", fontsize=13, fontweight='bold')
    ax.set_xlabel("True k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted k_angle", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 85); ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", fontsize=9.5)

    # Panel 3: Error Distributions on P80 Subset
    ax = axes[1, 0]
    err_gen = sub_p_gen - sub_t
    err_p80 = sub_p_p80 - sub_t
    sns.kdeplot(err_gen, color="#0284c7", linewidth=2.5, ax=ax, label=f"General Model (Std = {np.std(err_gen):.2f})")
    sns.kdeplot(err_p80, color="#10b981", linewidth=2.5, ax=ax, label=f"P80 Model (Std = {np.std(err_p80):.2f})")
    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.8)
    ax.set_title("Error Residual Density on k <= P80 (Pred - True)\nSharper Zero-Centered Peak with Specialized Training", fontsize=13, fontweight='bold')
    ax.set_xlabel("Error (Pred - True) [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Density", fontsize=11, fontweight='bold')
    ax.set_xlim(-25, 30)
    ax.legend(frameon=True, facecolor="white", fontsize=9.5)

    # Panel 4: Tolerance Accuracy Comparison Bar Chart
    ax = axes[1, 1]
    tol_names = ["+/- 2.0", "+/- 5.0", "+/- 10.0", "+/- 15.0", "+/- 20.0"]
    accs_g = [results['Benchmark_P80_Subset_k_le_78']['General_Model'][f"Accuracy_pm_{x}"] for x in [2, 5, 10, 15, 20]]
    accs_p = [results['Benchmark_P80_Subset_k_le_78']['P80_Model'][f"Accuracy_pm_{x}"] for x in [2, 5, 10, 15, 20]]
    
    x = np.arange(len(tol_names))
    width = 0.35
    ax.bar(x - width/2, accs_g, width, label='General Model', color='#0284c7', alpha=0.8)
    ax.bar(x + width/2, accs_p, width, label='P80 Model', color='#10b981', alpha=0.8)
    ax.set_title("Tolerance Accuracy Comparison on k <= P80 (%)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tol_names, fontsize=10, fontweight='bold')
    ax.set_ylabel("Percentage within Tolerance (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    for i in range(len(x)):
        ax.text(x[i] - width/2, accs_g[i] + 1.2, f"{accs_g[i]:.1f}%", ha='center', fontsize=8.5, fontweight='bold')
        ax.text(x[i] + width/2, accs_p[i] + 1.2, f"{accs_p[i]:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color="#065f46")
    ax.legend(frameon=True, facecolor="white", fontsize=9.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/model_comparison_general_vs_p80.png", dpi=300)
    plt.close()
    print("Saved results/model_comparison_general_vs_p80.png")

if __name__ == "__main__":
    main()
