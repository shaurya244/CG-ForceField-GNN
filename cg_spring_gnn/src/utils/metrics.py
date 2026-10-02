"""
metrics.py
----------
Evaluation metrics for spring constant prediction.

Functions:
  compute_metrics(pred, true)  →  dict with RMSE, MAE, R², MAPE
  print_metrics(metrics_dict)  →  formatted console output
"""

import numpy as np
from typing import Dict
from sklearn.metrics import r2_score


def compute_metrics(pred: np.ndarray, true: np.ndarray,
                    name: str = "k") -> Dict[str, float]:
    """
    Compute regression metrics between predicted and true spring constants.

    Args:
        pred: 1-D array of predicted values
        true: 1-D array of ground-truth values
        name: label prefix for the returned dict keys

    Returns dict with keys:
        {name}_RMSE, {name}_MAE, {name}_R2, {name}_MAPE
    """
    if len(pred) == 0:
        return {}

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae  = float(np.mean(np.abs(pred - true)))
    r2   = float(r2_score(true, pred))
    mape = float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-8))) * 100)

    # Log-space RMSE and Log-space R^2 (standard for wide-ranging chemical force constants)
    log_true = np.log10(np.clip(true, 1e-6, None))
    log_pred = np.log10(np.clip(pred, 1e-6, None))
    log_rmse = float(np.sqrt(np.mean((log_pred - log_true) ** 2)))
    log_r2   = float(r2_score(log_true, log_pred)) if len(true) > 1 else 0.0

    return {
        f"{name}_RMSE":     rmse,
        f"{name}_MAE":      mae,
        f"{name}_R2":       r2,
        f"{name}_logR2":    log_r2,
        f"{name}_MAPE(%)":  mape,
        f"{name}_logRMSE":  log_rmse,
    }


def print_metrics(metrics: Dict[str, float], header: str = ""):
    """Pretty-print a metrics dictionary."""
    if header:
        print(f"\n{'-'*50}")
        print(f"  {header}")
        print(f"{'-'*50}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")
