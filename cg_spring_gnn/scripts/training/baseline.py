"""
baseline.py
-----------
Trains three non-GNN baselines for comparison against the GNN:
  1. Mean predictor   (trivial baseline)
  2. Random Forest    (sklearn)
  3. XGBoost

Features for baselines: per-bond tabular features extracted from node/edge attributes.
Labels: k_bond (and k_angle separately).

Run:
    python baseline.py

Results are saved to results/baseline_metrics.json
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
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset  import CGSpringDataset
from src.utils.metrics import compute_metrics, print_metrics

from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from xgboost                  import XGBRegressor


# ─── Feature extraction from PyG graphs ──────────────────────────────────────

def extract_bond_features(dataset):
    """
    Extract per-bond tabular features from a PyG dataset.

    For each bond (i→j) we concatenate:
      [node_i features (NODE_DIM) | node_j features (NODE_DIM) | edge attr (EDGE_DIM)]

    Returns:
        X: np.ndarray [n_bonds, 2*NODE_DIM + EDGE_DIM]
        y_kb: np.ndarray [n_bonds]  bond spring constants
        y_r0: np.ndarray [n_bonds]  equilibrium bond lengths
    """
    X_list, y_kb_list, y_r0_list = [], [], []

    for data in dataset:
        x   = data.x.numpy()           # [N, NODE_DIM]
        ei  = data.edge_index.numpy()  # [2, 2B]
        ea  = data.edge_attr.numpy()   # [2B, EDGE_DIM]
        kb  = data.y_k_bond.numpy()    # [B]
        r0  = data.y_r0.numpy()        # [B]

        # Use forward edges only (even indices)
        n_bonds = len(kb)
        for idx in range(n_bonds):
            src = ei[0, 2*idx]
            dst = ei[1, 2*idx]
            feat = np.concatenate([x[src], x[dst], ea[2*idx]])
            X_list.append(feat)

        y_kb_list.extend(kb.tolist())
        y_r0_list.extend(r0.tolist())

    return (np.array(X_list, dtype=np.float32),
            np.array(y_kb_list, dtype=np.float32),
            np.array(y_r0_list, dtype=np.float32))


def extract_angle_features(dataset):
    """
    Extract per-angle tabular features.
    For each angle (i-j-k): [h_i | h_j | h_k | e_ji | e_jk]
    """
    X_list, y_ka_list, y_t0_list = [], [], []

    for data in dataset:
        x     = data.x.numpy()
        aidx  = data.angle_idx.numpy()  # [A, 3]
        ka    = data.y_k_angle.numpy()
        t0    = data.y_theta0.numpy()
        ea    = data.edge_attr.numpy() if hasattr(data, "edge_attr") else None
        ae    = data.angle_edge_idx.numpy() if hasattr(data, "angle_edge_idx") else None

        for a_i, (pi, pj, pk) in enumerate(aidx):
            parts = [x[pi], x[pj], x[pk]]
            if ae is not None and ea is not None and len(ae) > a_i:
                e1, e2 = ae[a_i]
                parts.extend([ea[e1], ea[e2]])
            feat = np.concatenate(parts)
            X_list.append(feat)
        y_ka_list.extend(ka.tolist())
        y_t0_list.extend(t0.tolist())

    if not X_list:
        return None, None, None
    return (np.array(X_list, dtype=np.float32),
            np.array(y_ka_list, dtype=np.float32),
            np.array(y_t0_list, dtype=np.float32))


# ─── Train + evaluate one model ───────────────────────────────────────────────

def train_and_eval(model_name, model, X_tr, y_tr, X_te, y_te,
                   label: str = "k_bond") -> dict:
    print(f"\n  [{model_name}] fitting on {len(X_tr)} samples ...", end=" ")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("done")
    metrics = compute_metrics(pred, y_te, name=f"{label}/{model_name}")
    print_metrics(metrics)
    return metrics


# --- Main ---------------------------------------------------------------------

def main():
    print("\n-- Baseline Models for CG Spring Constant Prediction --\n")

    train_ds = CGSpringDataset(root="data", split="train")
    val_ds   = CGSpringDataset(root="data", split="val")
    test_ds  = CGSpringDataset(root="data", split="test")

    # --- k_bond prediction -------------------------------------------------
    print("Extracting bond features ...")
    X_tr, y_kb_tr, _ = extract_bond_features(train_ds)
    X_te, y_kb_te, _ = extract_bond_features(test_ds)

    print(f"  Bond features: X_train={X_tr.shape}  X_test={X_te.shape}\n")
    print(f"  k_bond range: [{y_kb_tr.min():.1f}, {y_kb_tr.max():.1f}] kJ/mol/nm^2\n")

    all_metrics = {}

    # Work in log-space for k (matches loss function)
    y_log_tr = np.log(y_kb_tr + 1e-6)
    y_log_te = np.log(y_kb_te + 1e-6)

    models_kb = {
        "MeanPredictor": _MeanPredictor(),
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LinearRegression()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        ),
    }

    print("-- k_bond Prediction -----------------------------------")
    for name, model in models_kb.items():
        # Train in log-space, evaluate on original scale
        model.fit(X_tr, y_log_tr)
        log_pred = model.predict(X_te)
        pred     = np.exp(log_pred)
        m = compute_metrics(pred, y_kb_te, name=f"k_bond/{name}")
        print_metrics(m, header=name)
        all_metrics.update(m)

    # --- k_angle prediction ------------------------------------------------
    print("\nExtracting angle features ...")
    X_tr_a, y_ka_tr, _ = extract_angle_features(train_ds)
    X_te_a, y_ka_te, _ = extract_angle_features(test_ds)

    if X_tr_a is not None and len(X_tr_a) > 10:
        print(f"  Angle features: X_train={X_tr_a.shape}  X_test={X_te_a.shape}\n")
        y_log_tr_a = np.log(y_ka_tr + 1e-6)
        y_log_te_a = np.log(y_ka_te + 1e-6)

        print("-- k_angle Prediction ----------------------------------")
        for name, model in {
            "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0),
        }.items():
            model.fit(X_tr_a, y_log_tr_a)
            log_pred = model.predict(X_te_a)
            pred     = np.exp(log_pred)
            m = compute_metrics(pred, y_ka_te, name=f"k_angle/{name}")
            print_metrics(m, header=name)
            all_metrics.update(m)
    else:
        print("  Not enough angle data for baseline training (need >10 samples).")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_metrics.json", "w") as f:
        json.dump({k: round(float(v), 5) for k, v in all_metrics.items()}, f, indent=2)
    print("\nBaseline metrics saved to results/baseline_metrics.json")


class _MeanPredictor:
    """Trivial baseline: always predicts the training mean."""
    def fit(self, X, y):
        self.mean_ = np.mean(y)
    def predict(self, X):
        return np.full(len(X), self.mean_)


if __name__ == "__main__":
    main()
