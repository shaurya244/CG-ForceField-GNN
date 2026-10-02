"""
loss.py
-------
Loss functions for CG spring constant prediction.

SpringConstantLoss:
  - Log-MSE for k_bond and k_angle   (handles 100x range in values)
  - Linear MSE for r0 and theta0     (narrow range, linear is fine)
  - Weighted sum with configurable λ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SpringConstantLoss(nn.Module):
    """
    Multi-task loss for joint prediction of:
      - k_bond   (bond spring constant)
      - r0       (equilibrium bond length)
      - k_angle  (angle spring constant)
      - theta0   (equilibrium angle)

    k values are trained in log-space because they span a wide range
    (100 – 10,000 kJ/mol/nm²), preventing large k values from dominating.
    """

    def __init__(self,
                 lambda_k_bond:  float = 1.0,
                 lambda_r0:      float = 0.5,
                 lambda_k_angle: float = 1.0,
                 lambda_theta0:  float = 0.5,
                 lambda_regime:  float = 0.1,
                 eps:            float = 1e-6):
        super().__init__()
        self.lam = {
            "k_bond":  lambda_k_bond,
            "r0":      lambda_r0,
            "k_angle": lambda_k_angle,
            "theta0":  lambda_theta0,
            "regime":  lambda_regime,
        }
        self.eps = eps

    def forward(self,
                pred:   Tuple[torch.Tensor, ...],
                target: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        """
        pred   = (pred_k_bond, pred_r0, pred_k_angle, pred_theta0, [optional: pred_regime_logits])
        target = (true_k_bond, true_r0, true_k_angle, true_theta0)

        Returns dict with keys: "total", "k_bond", "r0", "k_angle", "theta0", "regime"
        """
        if len(pred) == 5:
            pred_kb, pred_r0, pred_ka, pred_t0, pred_regime = pred
        else:
            pred_kb, pred_r0, pred_ka, pred_t0 = pred
            pred_regime = None

        true_kb, true_r0, true_ka, true_t0 = target

        # --- k_bond : Hybrid log-Huber + normalized relative linear penalty -
        if true_kb.numel() > 0:
            loss_kb_log = F.smooth_l1_loss(
                torch.log(pred_kb + self.eps),
                torch.log(true_kb + self.eps),
                beta=0.1
            )
            # Normalized relative error prevents large absolute errors on high-k bonds (e.g. 50,000)
            loss_kb_rel = torch.mean(torch.abs(pred_kb - true_kb) / (true_kb + 1000.0))
            loss_kb = loss_kb_log + 0.5 * loss_kb_rel
        else:
            loss_kb = pred_kb.sum() * 0.0

        # --- r0 : linear MSE ----------------------------------------------
        loss_r0 = F.mse_loss(pred_r0, true_r0) \
                  if true_r0.numel() > 0 else pred_r0.sum() * 0.0

        # --- k_angle : Unbiased log10-Huber loss + Boundary Margin Penalty ---
        if (true_ka.numel() > 0 and pred_ka.numel() > 0):
            log_pka = torch.log10(torch.clamp(pred_ka, min=1e-2))
            log_tka = torch.log10(torch.clamp(true_ka, min=1e-2))
            
            # 1. Base smooth L1 regression loss (unweighted to prevent artificial upward inflation)
            loss_ka_reg = F.smooth_l1_loss(log_pka, log_tka, beta=0.05)

            # 2. Boundary Margin Penalty: specifically penalizes predictions that cross the 50.0 boundary!
            # log10(50.0) ≈ 1.69897, log10(150.0) ≈ 2.17609
            log_50  = 1.69897
            log_150 = 2.17609

            bound_err = torch.zeros_like(true_ka)
            # True flexible (<= 50) predicting into medium (> 50)
            flex_mask = true_ka <= 50.0
            bound_err[flex_mask] = F.relu(log_pka[flex_mask] - log_50)
            
            # True medium (50-150) predicting into flexible (<= 50)
            med_mask = (true_ka > 50.0) & (true_ka <= 150.0)
            bound_err[med_mask] = F.relu(log_50 - log_pka[med_mask])
            
            # True stiff (> 150) predicting into medium/flexible (<= 150)
            stiff_mask = true_ka > 150.0
            bound_err[stiff_mask] = F.relu(log_150 - log_pka[stiff_mask])

            loss_boundary = torch.mean(bound_err)
            loss_ka = loss_ka_reg + 1.5 * loss_boundary
        else:
            loss_ka = torch.tensor(0.0, device=pred_kb.device)

        # ── theta0 : linear MSE ───────────────────────────────────────────
        loss_t0 = F.mse_loss(pred_t0, true_t0) \
                  if (true_t0.numel() > 0 and pred_t0.numel() > 0) else torch.tensor(0.0, device=pred_kb.device)

        # ── Auxiliary Angle Regime Classification Loss (Cross-Entropy) ─────
        if (pred_regime is not None and pred_regime.numel() > 0 and true_ka.numel() > 0):
            true_regime = torch.zeros(true_ka.shape[0], dtype=torch.long, device=true_ka.device)
            true_regime[(true_ka > 50.0) & (true_ka <= 150.0)] = 1
            true_regime[true_ka > 150.0] = 2

            # Balanced CrossEntropy weights (mild balance without prior shift)
            cls_weights = torch.tensor([1.0, 1.25, 1.5], device=true_ka.device)
            loss_regime = F.cross_entropy(pred_regime, true_regime, weight=cls_weights)
        else:
            loss_regime = torch.tensor(0.0, device=pred_kb.device)

        total = (self.lam["k_bond"]  * loss_kb +
                 self.lam["r0"]      * loss_r0  +
                 self.lam["k_angle"] * loss_ka  +
                 self.lam["theta0"]  * loss_t0  +
                 self.lam["regime"]  * loss_regime)

        return {
            "total":   total,
            "k_bond":  loss_kb.detach(),
            "r0":      loss_r0.detach(),
            "k_angle": loss_ka.detach(),
            "theta0":  loss_t0.detach(),
            "regime":  loss_regime.detach(),
        }
