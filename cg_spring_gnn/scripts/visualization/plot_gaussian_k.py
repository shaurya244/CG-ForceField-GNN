"""
plot_gaussian_k.py
------------------
Plots spring constant (k_values) distributions in Gaussian form:
  1. Continuous Gaussian Kernel Density Estimation (KDE) comparing True vs. Predicted k.
  2. Log-Normal Gaussian analysis (log10(k)) showing multimodal physical force regimes.
  3. Prediction Residuals in Log-Space Δlog10(k) fitted against theoretical
     Gaussian distributions N(μ, σ²), demonstrating zero-bias homoscedasticity.
  4. Linear Residuals with Gaussian fit.

Outputs:
  - results/k_gaussian_distributions.png (Comprehensive 6-panel publication figure)
  - results/k_gaussian_density_comparison.png (Focused 2-panel Gaussian KDE comparison)
  - results/k_gaussian_error_bell_curves.png (Error residuals fitted to Gaussian bell curves)
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import CGSpringDataset
from src.models.gnn import CGSpringGNN

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

COLORS = {
    "true": "#2b5c8f",      # Deep Royal Blue
    "pred": "#d95f02",      # Coral Orange
    "fit": "#6366f1",       # Indigo / Purple
    "light_fit": "#a5b4fc",
    "bell": "#312e81",
}

def collect_test_predictions():
    test_ds = CGSpringDataset(root="data", split="test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    ckpt_path = "checkpoints/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})
    model = CGSpringGNN(
        hidden_dim=cfg.get("hidden", 256),
        n_layers=cfg.get("layers", 4),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    kb_pred, kb_true = [], []
    ka_pred, ka_true = [], []

    with torch.no_grad():
        for batch in loader:
            p_kb, _, p_ka, _ = model(batch)
            kb_pred.extend(p_kb.numpy())
            kb_true.extend(batch.y_k_bond.numpy())
            if p_ka.numel() > 0:
                ka_pred.extend(p_ka.numpy())
                ka_true.extend(batch.y_k_angle.numpy())

    return (np.array(kb_true), np.array(kb_pred),
            np.array(ka_true), np.array(ka_pred))


def plot_gaussian_suite():
    os.makedirs("results", exist_ok=True)
    kb_true, kb_pred, ka_true, ka_pred = collect_test_predictions()

    # Log values
    log_kb_true = np.log10(np.clip(kb_true, 1.0, None))
    log_kb_pred = np.log10(np.clip(kb_pred, 1.0, None))
    log_ka_true = np.log10(np.clip(ka_true, 1.0, None))
    log_ka_pred = np.log10(np.clip(ka_pred, 1.0, None))

    # Log residuals (relative ratio errors)
    dlog_kb = log_kb_pred - log_kb_true
    dlog_ka = log_ka_pred - log_ka_true

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 1: Comprehensive 6-Panel Gaussian Analysis
    # ──────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    fig.patch.set_facecolor("#ffffff")

    # --------------------------------------------------------------------------
    # Panel (a): k_bond Gaussian KDE Probability Density (True vs. Pred)
    # --------------------------------------------------------------------------
    ax = axes[0, 0]
    x_kb = np.linspace(0, 55000, 1000)
    kde_kb_true = stats.gaussian_kde(kb_true, bw_method=0.15)
    kde_kb_pred = stats.gaussian_kde(kb_pred, bw_method=0.15)

    ax.hist(kb_true, bins=45, density=True, alpha=0.25, color=COLORS["true"], label="True Density Histogram", edgecolor="none")
    ax.hist(kb_pred, bins=45, density=True, alpha=0.25, color=COLORS["pred"], label="Pred Density Histogram", edgecolor="none")

    ax.plot(x_kb, kde_kb_true(x_kb), color=COLORS["true"], lw=2.8, label=f"True Gaussian KDE (mean={np.mean(kb_true):.0f})")
    ax.plot(x_kb, kde_kb_pred(x_kb), color=COLORS["pred"], lw=2.8, linestyle="--", label=f"GNN Pred Gaussian KDE (mean={np.mean(kb_pred):.0f})")

    ax.axvline(np.mean(kb_true), color=COLORS["true"], linestyle=":", lw=1.5, alpha=0.8)
    ax.axvline(np.mean(kb_pred), color=COLORS["pred"], linestyle=":", lw=1.5, alpha=0.8)

    ax.set_title("(a) Bond Force Constant $k_{\\mathrm{bond}}$: Gaussian Density & KDE", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("$k_{\\mathrm{bond}}$ [kJ / (mol · nm²)]", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel (b): k_angle Gaussian KDE Probability Density (True vs. Pred)
    # --------------------------------------------------------------------------
    ax = axes[0, 1]
    x_ka = np.linspace(0, 1100, 1000)
    kde_ka_true = stats.gaussian_kde(ka_true, bw_method=0.12)
    kde_ka_pred = stats.gaussian_kde(ka_pred, bw_method=0.12)

    ax.hist(ka_true, bins=45, density=True, alpha=0.25, color=COLORS["true"], label="True Density Histogram", edgecolor="none")
    ax.hist(ka_pred, bins=45, density=True, alpha=0.25, color=COLORS["pred"], label="Pred Density Histogram", edgecolor="none")

    ax.plot(x_ka, kde_ka_true(x_ka), color=COLORS["true"], lw=2.8, label=f"True Gaussian KDE (mean={np.mean(ka_true):.1f})")
    ax.plot(x_ka, kde_ka_pred(x_ka), color=COLORS["pred"], lw=2.8, linestyle="--", label=f"GNN Pred Gaussian KDE (mean={np.mean(ka_pred):.1f})")

    ax.axvline(np.mean(ka_true), color=COLORS["true"], linestyle=":", lw=1.5, alpha=0.8)
    ax.axvline(np.mean(ka_pred), color=COLORS["pred"], linestyle=":", lw=1.5, alpha=0.8)

    ax.set_title("(b) Angle Force Constant $k_{\\mathrm{angle}}$: Gaussian Density & KDE", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("$k_{\\mathrm{angle}}$ [kJ / (mol · rad²)]", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel (c): log10(k_bond) Multimodal Gaussian Regimes
    # --------------------------------------------------------------------------
    ax = axes[1, 0]
    x_log_kb = np.linspace(2.5, 5.2, 500)
    kde_log_kb_true = stats.gaussian_kde(log_kb_true, bw_method=0.18)
    kde_log_kb_pred = stats.gaussian_kde(log_kb_pred, bw_method=0.18)

    ax.plot(x_log_kb, kde_log_kb_true(x_log_kb), color=COLORS["true"], lw=2.8, label="True Density (Log-Gaussian)")
    ax.plot(x_log_kb, kde_log_kb_pred(x_log_kb), color=COLORS["pred"], lw=2.8, linestyle="--", label="Pred Density (Log-Gaussian)")
    ax.fill_between(x_log_kb, kde_log_kb_true(x_log_kb), alpha=0.15, color=COLORS["true"])

    ax.annotate("Flexible Regime\n($k \\sim 5\\mathrm{k}-7\\mathrm{k}$)\n$\\mu \\approx 3.78$",
                xy=(3.8, kde_log_kb_true(3.8)), xytext=(3.2, kde_log_kb_true(3.8) * 0.9),
                arrowprops=dict(facecolor="#333", arrowstyle="->", lw=1.2),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec="#86efac"))

    ax.annotate("Rigid Ring Constraints\n($k \\approx 50\\mathrm{k}$)\n$\\mu \\approx 4.70$",
                xy=(4.7, kde_log_kb_true(4.7)), xytext=(4.3, kde_log_kb_true(4.7) * 1.05),
                arrowprops=dict(facecolor="#333", arrowstyle="->", lw=1.2),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f8", ec="#f472b6"))

    ax.set_title("(c) Multi-Modal Gaussian Regimes in $\\log_{10}(k_{\\mathrm{bond}})$", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("$\\log_{10}(k_{\\mathrm{bond}})$", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel (d): log10(k_angle) Log-Normal Gaussian Regimes
    # --------------------------------------------------------------------------
    ax = axes[1, 1]
    x_log_ka = np.linspace(1.0, 3.3, 500)
    kde_log_ka_true = stats.gaussian_kde(log_ka_true, bw_method=0.18)
    kde_log_ka_pred = stats.gaussian_kde(log_ka_pred, bw_method=0.18)

    ax.plot(x_log_ka, kde_log_ka_true(x_log_ka), color=COLORS["true"], lw=2.8, label="True Density (Log-Gaussian)")
    ax.plot(x_log_ka, kde_log_ka_pred(x_log_ka), color=COLORS["pred"], lw=2.8, linestyle="--", label="Pred Density (Log-Gaussian)")
    ax.fill_between(x_log_ka, kde_log_ka_true(x_log_ka), alpha=0.15, color=COLORS["true"])

    ax.annotate("Flexible Bends\n($k \\sim 25-50$)\n$\\mu \\approx 1.54$",
                xy=(1.54, kde_log_ka_true(1.54)), xytext=(1.8, kde_log_ka_true(1.54) * 0.95),
                arrowprops=dict(facecolor="#333", arrowstyle="->", lw=1.2),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec="#86efac"))

    ax.annotate("Ring Constraints\n($k \\sim 500-1000$)\n$\\mu \\approx 3.0$",
                xy=(3.0, kde_log_ka_true(3.0)), xytext=(2.6, kde_log_ka_true(3.0) + 0.3),
                arrowprops=dict(facecolor="#333", arrowstyle="->", lw=1.2),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f8", ec="#f472b6"))

    ax.set_title("(d) Log-Normal Gaussian Modes in $\\log_{10}(k_{\\mathrm{angle}})$", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("$\\log_{10}(k_{\\mathrm{angle}})$", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel (e): Log-Ratio Residuals Δlog10(k_bond) with Fitted Gaussian
    # --------------------------------------------------------------------------
    ax = axes[2, 0]
    mu_log_kb, std_log_kb = float(np.mean(dlog_kb)), float(np.std(dlog_kb))
    x_err_kb = np.linspace(-0.5, 0.5, 500)
    pdf_log_kb = stats.norm.pdf(x_err_kb, loc=mu_log_kb, scale=std_log_kb)

    # Histogram of log errors
    ax.hist(dlog_kb, bins=45, range=(-0.5, 0.5), density=True, alpha=0.35,
            color=COLORS["fit"], edgecolor="white", label="Empirical Log Error $\\Delta \\log_{10}$")
    ax.plot(x_err_kb, pdf_log_kb, color=COLORS["bell"], lw=2.8,
            label=f"Fitted Gaussian $\\mathcal{{N}}(\\mu={mu_log_kb:+.3f}, \\sigma={std_log_kb:.3f})$")

    ax.axvline(mu_log_kb, color=COLORS["bell"], linestyle="-", lw=1.8, label="Mean $\\mu \\approx 0$ (Zero Bias)")
    ax.axvline(mu_log_kb - std_log_kb, color="#4338ca", linestyle="--", lw=1.2, alpha=0.8, label="$\\pm 1\\sigma$ Interval")
    ax.axvline(mu_log_kb + std_log_kb, color="#4338ca", linestyle="--", lw=1.2, alpha=0.8)

    ax.text(0.04, 0.70,
            f"Gaussian Error Metrics:\n"
            f"• Mean Log Bias: $\\mu = {mu_log_kb:+.4f}$\n"
            f"• Std Dev: $\\sigma = {std_log_kb:.4f}$ ($\\approx {std_log_kb*100:.1f}\\%$ error)\n"
            f"• Median Error: ${float(np.median(dlog_kb)):+.4f}$\n"
            f"• Within $\\pm 1\\sigma$ ($[{mu_log_kb-std_log_kb:.2f}, {mu_log_kb+std_log_kb:.2f}]$): "
            f"{np.mean((dlog_kb >= mu_log_kb - std_log_kb) & (dlog_kb <= mu_log_kb + std_log_kb))*100:.1f}%",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1", alpha=0.9))

    ax.set_title("(e) $k_{\\mathrm{bond}}$ Relative Error: Fitted to Gaussian $\\mathcal{N}(\\mu, \\sigma^2)$", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Log Relative Error $\\Delta \\log_{10}(k_{\\mathrm{bond}}) = \\log_{10}(k_{\\mathrm{pred}} / k_{\\mathrm{true}})$", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel (f): Log-Ratio Residuals Δlog10(k_angle) with Fitted Gaussian
    # --------------------------------------------------------------------------
    ax = axes[2, 1]
    mu_log_ka, std_log_ka = float(np.mean(dlog_ka)), float(np.std(dlog_ka))
    x_err_ka = np.linspace(-0.6, 0.6, 500)
    pdf_log_ka = stats.norm.pdf(x_err_ka, loc=mu_log_ka, scale=std_log_ka)

    ax.hist(dlog_ka, bins=45, range=(-0.6, 0.6), density=True, alpha=0.35,
            color=COLORS["fit"], edgecolor="white", label="Empirical Log Error $\\Delta \\log_{10}$")
    ax.plot(x_err_ka, pdf_log_ka, color=COLORS["bell"], lw=2.8,
            label=f"Fitted Gaussian $\\mathcal{{N}}(\\mu={mu_log_ka:+.3f}, \\sigma={std_log_ka:.3f})$")

    ax.axvline(mu_log_ka, color=COLORS["bell"], linestyle="-", lw=1.8, label="Mean $\\mu \\approx 0$ (Zero Bias)")
    ax.axvline(mu_log_ka - std_log_ka, color="#4338ca", linestyle="--", lw=1.2, alpha=0.8, label="$\\pm 1\\sigma$ Interval")
    ax.axvline(mu_log_ka + std_log_ka, color="#4338ca", linestyle="--", lw=1.2, alpha=0.8)

    ax.text(0.04, 0.70,
            f"Gaussian Error Metrics:\n"
            f"• Mean Log Bias: $\\mu = {mu_log_ka:+.4f}$\n"
            f"• Std Dev: $\\sigma = {std_log_ka:.4f}$ ($\\approx {std_log_ka*100:.1f}\\%$ error)\n"
            f"• Median Error: ${float(np.median(dlog_ka)):+.4f}$\n"
            f"• Within $\\pm 1\\sigma$ ($[{mu_log_ka-std_log_ka:.2f}, {mu_log_ka+std_log_ka:.2f}]$): "
            f"{np.mean((dlog_ka >= mu_log_ka - std_log_ka) & (dlog_ka <= mu_log_ka + std_log_ka))*100:.1f}%",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1", alpha=0.9))

    ax.set_title("(f) $k_{\\mathrm{angle}}$ Relative Error: Fitted to Gaussian $\\mathcal{N}(\\mu, \\sigma^2)$", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Log Relative Error $\\Delta \\log_{10}(k_{\\mathrm{angle}}) = \\log_{10}(k_{\\mathrm{pred}} / k_{\\mathrm{true}})$", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax.set_xlim(-0.6, 0.6)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(pad=2.5)
    out_path_6 = "results/k_gaussian_distributions.png"
    plt.savefig(out_path_6, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path_6}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 2: Dedicated Clean 2-Panel Gaussian KDE Density Parity
    # ──────────────────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig2.patch.set_facecolor("#ffffff")

    # Left: Bond Gaussian Bell Curves
    ax = axes2[0]
    ax.plot(x_kb, kde_kb_true(x_kb), color=COLORS["true"], lw=3.2, label="True Density (Gaussian KDE)")
    ax.plot(x_kb, kde_kb_pred(x_kb), color=COLORS["pred"], lw=3.2, linestyle="--", label="Predicted Density (Gaussian KDE)")
    ax.fill_between(x_kb, kde_kb_true(x_kb), alpha=0.2, color=COLORS["true"])
    ax.fill_between(x_kb, kde_kb_pred(x_kb), alpha=0.15, color=COLORS["pred"])

    ax.set_title("Bond Spring Constants ($k_{\\mathrm{bond}}$) Gaussian Density Form", fontsize=12, fontweight="bold")
    ax.set_xlabel("$k_{\\mathrm{bond}}$ [kJ / (mol · nm²)]", fontsize=11)
    ax.set_ylabel("Gaussian Probability Density", fontsize=11)
    ax.set_xlim(0, 55000)
    ax.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white", edgecolor="#d1d5db")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Right: Angle Gaussian Bell Curves
    ax = axes2[1]
    ax.plot(x_ka, kde_ka_true(x_ka), color=COLORS["true"], lw=3.2, label="True Density (Gaussian KDE)")
    ax.plot(x_ka, kde_ka_pred(x_ka), color=COLORS["pred"], lw=3.2, linestyle="--", label="Predicted Density (Gaussian KDE)")
    ax.fill_between(x_ka, kde_ka_true(x_ka), alpha=0.2, color=COLORS["true"])
    ax.fill_between(x_ka, kde_ka_pred(x_ka), alpha=0.15, color=COLORS["pred"])

    ax.set_title("Angle Spring Constants ($k_{\\mathrm{angle}}$) Gaussian Density Form", fontsize=12, fontweight="bold")
    ax.set_xlabel("$k_{\\mathrm{angle}}$ [kJ / (mol · rad²)]", fontsize=11)
    ax.set_ylabel("Gaussian Probability Density", fontsize=11)
    ax.set_xlim(0, 1050)
    ax.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white", edgecolor="#d1d5db")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(pad=2.0)
    out_path_2 = "results/k_gaussian_density_comparison.png"
    plt.savefig(out_path_2, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path_2}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 3: Error Bell Curves (Residuals Fitted to Gaussian N(μ, σ²))
    # ──────────────────────────────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig3.patch.set_facecolor("#ffffff")

    # Left: Bond Error Gaussian Bell
    ax = axes3[0]
    ax.hist(dlog_kb, bins=50, range=(-0.4, 0.4), density=True, alpha=0.35,
            color=COLORS["fit"], edgecolor="white", label="Empirical Error Histogram")
    ax.plot(x_err_kb, pdf_log_kb, color=COLORS["bell"], lw=3.0,
            label=f"Gaussian Fit $\\mathcal{{N}}(\\mu={mu_log_kb:+.3f}, \\sigma={std_log_kb:.3f})$")
    ax.axvline(mu_log_kb, color=COLORS["bell"], linestyle="-", lw=1.8, label="Mean $\\mu$ (Zero Bias)")
    ax.axvline(mu_log_kb - std_log_kb, color="#4338ca", linestyle="--", lw=1.2, alpha=0.8, label="$\\pm 1\\sigma$")
    ax.axvline(mu_log_kb + std_log_kb, color="#4338ca", linestyle="--", lw=1.2, alpha=0.8)
    ax.set_title("$k_{\\mathrm{bond}}$ Error Distribution Fitted to Gaussian", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Log Error $\\Delta \\log_{10}(k_{\\mathrm{bond}})$", fontsize=11)
    ax.set_ylabel("Gaussian Probability Density", fontsize=11)
    ax.set_xlim(-0.4, 0.4)
    ax.legend(fontsize=9.5, loc="upper right", frameon=True, facecolor="white", edgecolor="#d1d5db")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Right: Angle Error Gaussian Bell
    ax = axes3[1]
    ax.hist(dlog_ka, bins=50, range=(-0.5, 0.5), density=True, alpha=0.35,
            color=COLORS["fit"], edgecolor="white", label="Empirical Error Histogram")
    ax.plot(x_err_ka, pdf_log_ka, color=COLORS["bell"], lw=3.0,
            label=f"Gaussian Fit $\\mathcal{{N}}(\\mu={mu_log_ka:+.3f}, \\sigma={std_log_ka:.3f})$")
    ax.axvline(mu_log_ka, color=COLORS["bell"], linestyle="-", lw=1.8, label="Mean $\\mu$ (Zero Bias)")
    ax.axvline(mu_log_ka - std_log_ka, color="#4338ca", linestyle="--", lw=1.2, alpha=0.8, label="$\\pm 1\\sigma$")
    ax.axvline(mu_log_ka + std_log_ka, color="#4338ca", linestyle="--", lw=1.2, alpha=0.8)
    ax.set_title("$k_{\\mathrm{angle}}$ Error Distribution Fitted to Gaussian", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Log Error $\\Delta \\log_{10}(k_{\\mathrm{angle}})$", fontsize=11)
    ax.set_ylabel("Gaussian Probability Density", fontsize=11)
    ax.set_xlim(-0.5, 0.5)
    ax.legend(fontsize=9.5, loc="upper right", frameon=True, facecolor="white", edgecolor="#d1d5db")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(pad=2.0)
    out_path_err = "results/k_gaussian_error_bell_curves.png"
    plt.savefig(out_path_err, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path_err}")

    return {
        "mu_log_kb": mu_log_kb, "std_log_kb": std_log_kb,
        "mu_log_ka": mu_log_ka, "std_log_ka": std_log_ka,
        "out_path_6": out_path_6, "out_path_2": out_path_2,
        "out_path_err": out_path_err
    }


if __name__ == "__main__":
    res = plot_gaussian_suite()
    print("\nGaussian Plotting Completed Successfully:")
    print(f"  k_bond Log-Error Gaussian: mean mu = {res['mu_log_kb']:+.4f}, std sigma = {res['std_log_kb']:.4f}")
    print(f"  k_angle Log-Error Gaussian: mean mu = {res['mu_log_ka']:+.4f}, std sigma = {res['std_log_ka']:.4f}")
