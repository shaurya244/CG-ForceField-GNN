"""
plot_results.py
---------------
Generate all result plots:
  1. Training loss curves           → results/loss_curves.png
  2. k_bond prediction scatter      → results/scatter_k_bond.png
  3. k_angle prediction scatter     → results/scatter_k_angle.png
  4. Spring constant distributions  → results/k_distribution.png
  5. Per-molecule error bar chart   → results/per_mol_error.png

Run AFTER training:
    python plot_results.py
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch_geometric.loader import DataLoader

# Plot style
mpl.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":    150,
})

COLORS = {"gnn": "#4C72B0", "rf": "#DD8452", "xgb": "#55A868", "true": "#888"}
os.makedirs("results", exist_ok=True)


# ─── 1. Training loss curves ──────────────────────────────────────────────────

def plot_loss_curves(log_path: str = "results/train_log.csv"):
    if not os.path.exists(log_path):
        print(f"[skip] {log_path} not found - run train.py first.")
        return

    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_total"], label="Train", color=COLORS["gnn"])
    axes[0].plot(df["epoch"], df["val_total"],   label="Val",   color=COLORS["rf"],
                 linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss"); axes[0].legend()

    axes[1].plot(df["epoch"], df["train_kb"],  label="Train k_bond",  color=COLORS["gnn"])
    axes[1].plot(df["epoch"], df["val_kb"],    label="Val k_bond",    color=COLORS["gnn"],
                 linestyle="--")
    axes[1].plot(df["epoch"], df["train_ka"],  label="Train k_angle", color=COLORS["rf"])
    axes[1].plot(df["epoch"], df["val_ka"],    label="Val k_angle",   color=COLORS["rf"],
                 linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Component Loss")
    axes[1].set_title("Component Losses"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/loss_curves.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/loss_curves.png")


# ─── 2. Scatter plots ─────────────────────────────────────────────────────────

def scatter_plot(pred: np.ndarray, true: np.ndarray,
                 label: str, unit: str, fname: str):
    """Predicted vs true scatter with identity line and R² annotation."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(true, pred, alpha=0.45, s=20, color=COLORS["gnn"],
               label=f"n = {len(true)}", edgecolors="none")

    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y=x")

    from sklearn.metrics import r2_score
    r2   = r2_score(true, pred)
    rmse = np.sqrt(np.mean((pred - true)**2))
    log_true = np.log10(np.clip(true, 1e-6, None))
    log_pred = np.log10(np.clip(pred, 1e-6, None))
    log_r2   = r2_score(log_true, log_pred) if len(true) > 1 else 0.0

    ax.text(0.05, 0.86, f"R^2 = {r2:.3f}\nlog-R^2 = {log_r2:.3f}\nRMSE = {rmse:.1f} {unit}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel(f"True {label} [{unit}]")
    ax.set_ylabel(f"Predicted {label} [{unit}]")
    ax.set_title(f"{label} Prediction")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ─── 3. k distribution ───────────────────────────────────────────────────────

def plot_k_distributions(train_ds, test_ds):
    k_bonds_tr, k_bonds_te = [], []
    k_angles_tr, k_angles_te = [], []

    for ds, kb_list, ka_list in [(train_ds, k_bonds_tr, k_angles_tr),
                                  (test_ds,  k_bonds_te, k_angles_te)]:
        for d in ds:
            kb_list.extend(d.y_k_bond.tolist())
            ka_list.extend(d.y_k_angle.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(k_bonds_tr, bins=50, alpha=0.6, color=COLORS["gnn"], label="Train")
    axes[0].hist(k_bonds_te, bins=50, alpha=0.6, color=COLORS["rf"],  label="Test")
    axes[0].set_xlabel("k_bond [kJ/mol/nm²]"); axes[0].set_ylabel("Count")
    axes[0].set_title("Bond Spring Constant Distribution"); axes[0].legend()

    if k_angles_tr:
        axes[1].hist(k_angles_tr, bins=50, alpha=0.6, color=COLORS["gnn"], label="Train")
        axes[1].hist(k_angles_te, bins=50, alpha=0.6, color=COLORS["rf"],  label="Test")
        axes[1].set_xlabel("k_angle [kJ/mol/rad²]"); axes[1].set_ylabel("Count")
        axes[1].set_title("Angle Spring Constant Distribution"); axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No angle data", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig("results/k_distribution.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/k_distribution.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from src.data.dataset import CGSpringDataset
    from src.models.gnn   import CGSpringGNN

    device   = torch.device("cpu")
    test_ds  = CGSpringDataset(root="data", split="test")
    train_ds = CGSpringDataset(root="data", split="train")
    loader   = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("\n-- Generating Result Plots --\n")

    # 1. Loss curves
    plot_loss_curves()

    # 2. Load model + collect predictions
    ckpt_path = "checkpoints/best_model.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        cfg  = ckpt.get("args", {})
        model = CGSpringGNN(
            hidden_dim=cfg.get("hidden", 256),
            n_layers  =cfg.get("layers", 4),
            dropout   =cfg.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        pred_kb, true_kb = [], []
        pred_ka, true_ka = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                p_kb, _, p_ka, _ = model(batch)
                pred_kb.extend(p_kb.numpy()); true_kb.extend(batch.y_k_bond.numpy())
                if p_ka.numel() > 0:
                    pred_ka.extend(p_ka.numpy()); true_ka.extend(batch.y_k_angle.numpy())

        scatter_plot(np.array(pred_kb), np.array(true_kb),
                     "k_bond", "kJ/mol/nm^2", "results/scatter_k_bond.png")
        if pred_ka:
            scatter_plot(np.array(pred_ka), np.array(true_ka),
                         "k_angle", "kJ/mol/rad^2", "results/scatter_k_angle.png")
    else:
        print(f"[skip] {ckpt_path} not found - run train.py first.")

    # 3. Distributions
    plot_k_distributions(train_ds, test_ds)

    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
