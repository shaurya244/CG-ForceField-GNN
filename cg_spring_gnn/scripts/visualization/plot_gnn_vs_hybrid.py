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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader
from src.data.dataset import CGSpringDataset
from src.models.gnn import CGSpringGNN

print("Loading datasets and model...")
train_ds = CGSpringDataset(root="data", split="train")
test_ds  = CGSpringDataset(root="data", split="test")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=False)
base_gnn = CGSpringGNN(hidden_dim=128, n_layers=3, dropout=0.2)
base_gnn.load_state_dict(ckpt["model_state"])
base_gnn.eval()

def extract_reps(loader):
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

reps_tr, raws_tr, y_tr, _ = extract_reps(train_loader)
reps_te, raws_te, y_te, gnn_pred = extract_reps(test_loader)

# Model 1: Current GNN Alone
r2_gnn = r2_score(y_te, gnn_pred)
mae_gnn = mean_absolute_error(y_te, gnn_pred)
medae_gnn = median_absolute_error(y_te, gnn_pred)

# Model 2: Random Forest on GNN Embeddings
print("Fitting Random Forest on GNN Embeddings...")
rf_gnn = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_gnn.fit(reps_tr, np.log(y_tr + 1e-4))
pred_rf_gnn = np.exp(rf_gnn.predict(reps_te))
r2_rf = r2_score(y_te, pred_rf_gnn)
mae_rf = mean_absolute_error(y_te, pred_rf_gnn)
medae_rf = median_absolute_error(y_te, pred_rf_gnn)

# Model 3: XGBoost on GNN Embeddings + Raw Skips
print("Fitting XGBoost on GNN Embeddings + Skip Features...")
X_comb_tr = np.concatenate([reps_tr, raws_tr], axis=1)
X_comb_te = np.concatenate([reps_te, raws_te], axis=1)
xgb = XGBRegressor(n_estimators=180, learning_rate=0.08, max_depth=6, random_state=42, verbosity=0, n_jobs=-1)
xgb.fit(X_comb_tr, np.log(y_tr + 1e-4))
pred_xgb = np.exp(xgb.predict(X_comb_te))
r2_xgb = r2_score(y_te, pred_xgb)
mae_xgb = mean_absolute_error(y_te, pred_xgb)
medae_xgb = median_absolute_error(y_te, pred_xgb)

print(f"\n--- DIAGNOSTIC RESULTS ---")
print(f"1. Current GNN: Linear R2 = {r2_gnn:.4f}, MAE = {mae_gnn:.2f}, MedAE = {medae_gnn:.2f}")
print(f"2. GNN Reps + RF Head: Linear R2 = {r2_rf:.4f}, MAE = {mae_rf:.2f}, MedAE = {medae_rf:.2f}")
print(f"3. GNN Reps + XGB Hybrid: Linear R2 = {r2_xgb:.4f}, MAE = {mae_xgb:.2f}, MedAE = {medae_xgb:.2f}")

# Plot side-by-side comparison
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
fig.suptitle("Architectural Diagnostic: Pure MLP Head vs. GNN-Tree Hybrid Heads (Test Set N=458)", fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Pure GNN MLP Head
ax1 = axes[0]
ax1.scatter(y_te, gnn_pred, color="#0284c7", alpha=0.6, s=35, edgecolors='none')
ax1.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.0, label="Ideal y = x")
ax1.set_title(f"1. Current GNN (Smooth MLP Head)\nLinear R2 = {r2_gnn:.4f} | MAE = {mae_gnn:.2f} kJ/mol/rad^2\n(Smooth interpolation underpredicts discrete 1000s)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Ground Truth k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax1.set_ylabel("Predicted k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax1.set_xlim(0, 1050); ax1.set_ylim(0, 1400)
ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9.5)

# Panel 2: Random Forest on GNN Embeddings
ax2 = axes[1]
ax2.scatter(y_te, pred_rf_gnn, color="#8b5cf6", alpha=0.6, s=35, edgecolors='none')
ax2.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.0, label="Ideal y = x")
ax2.set_title(f"2. GNN Reps + Random Forest Head\nLinear R2 = {r2_rf:.4f} | MAE = {mae_rf:.2f} kJ/mol/rad^2\n(Tree partitions capture discrete rules)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Ground Truth k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax2.set_ylabel("Predicted k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax2.set_xlim(0, 1050); ax2.set_ylim(0, 1050)
ax2.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9.5)

# Panel 3: XGBoost on GNN Embeddings + Skip
ax3 = axes[2]
ax3.scatter(y_te, pred_xgb, color="#10b981", alpha=0.65, s=35, edgecolors='none')
ax3.plot([0, 1050], [0, 1050], color="#dc2626", linestyle="--", linewidth=2.0, label="Ideal y = x")
ax3.set_title(f"3. GNN Reps + XGBoost Hybrid Head\nLinear R2 = {r2_xgb:.4f} | MAE = {mae_xgb:.2f} kJ/mol/rad^2\n(SOTA Hybrid: Captures constraints + smooth physics)", fontsize=12, fontweight='bold')
ax3.set_xlabel("Ground Truth k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax3.set_ylabel("Predicted k_angle [kJ/mol/rad^2]", fontsize=11, fontweight='bold')
ax3.set_xlim(0, 1050); ax3.set_ylim(0, 1050)
ax3.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out_plot = "results/architectural_diagnostic_gnn_vs_hybrid.png"
plt.savefig(out_plot, dpi=300)
plt.close()
print(f"Saved {out_plot}")

brain_dir = r"C:\Users\sriva\.gemini\antigravity-ide\brain\029fbcbc-bb91-4a30-9c3b-37696e486a0c"
shutil.copy(out_plot, brain_dir)
print(f"Copied to {brain_dir}")
