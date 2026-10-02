"""
train_p80.py
------------
Trains a dedicated CGSpringGNN specialized on the 0-80th percentile angle spring constants (k_angle <= 78.10 kJ/mol/rad^2).
Initializes from checkpoints/best_model.pt and fine-tunes with an explicit P80 target filter and ceiling loss.

Saves:
    checkpoints/best_model_p80.pt
    results/train_log_p80.csv
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
import csv
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.data.dataset import CGSpringDataset
from src.models.gnn import CGSpringGNN

P80_THRESHOLD = 78.10

class SpringConstantLossP80(nn.Module):
    """Loss function that focuses k_angle training strictly on k <= P80 (78.10 kJ/mol/rad^2)."""
    def __init__(self,
                 lambda_k_bond=1.0,
                 lambda_r0=0.5,
                 lambda_k_angle=1.2,
                 lambda_theta0=0.5,
                 lambda_regime=0.15,
                 p80_thresh=P80_THRESHOLD,
                 eps=1e-6):
        super().__init__()
        self.lam = {
            "k_bond": lambda_k_bond,
            "r0": lambda_r0,
            "k_angle": lambda_k_angle,
            "theta0": lambda_theta0,
            "regime": lambda_regime,
        }
        self.p80 = p80_thresh
        self.eps = eps

    def forward(self, pred, target):
        if len(pred) == 5:
            pred_kb, pred_r0, pred_ka, pred_t0, pred_regime = pred
        else:
            pred_kb, pred_r0, pred_ka, pred_t0 = pred
            pred_regime = None

        true_kb, true_r0, true_ka, true_t0 = target

        # 1. k_bond loss (standard hybrid)
        if true_kb.numel() > 0:
            loss_kb_log = F.smooth_l1_loss(
                torch.log(pred_kb + self.eps),
                torch.log(true_kb + self.eps),
                beta=0.1
            )
            loss_kb_rel = torch.mean(torch.abs(pred_kb - true_kb) / (true_kb + 1000.0))
            loss_kb = loss_kb_log + 0.5 * loss_kb_rel
        else:
            loss_kb = pred_kb.sum() * 0.0

        # 2. r0 loss (linear MSE)
        loss_r0 = F.mse_loss(pred_r0, true_r0) if true_r0.numel() > 0 else pred_r0.sum() * 0.0

        # 3. k_angle loss: FILTERED TO k <= P80 (78.10)
        if true_ka.numel() > 0 and pred_ka.numel() > 0:
            valid_mask = true_ka <= self.p80
            if valid_mask.sum() > 0:
                t_ka_filt = true_ka[valid_mask]
                p_ka_filt = pred_ka[valid_mask]
                
                log_pka = torch.log10(torch.clamp(p_ka_filt, min=1e-2))
                log_tka = torch.log10(torch.clamp(t_ka_filt, min=1e-2))
                
                # Base smooth L1 regression loss on P80 angles
                loss_ka_reg = F.smooth_l1_loss(log_pka, log_tka, beta=0.05)
                
                # Boundary penalty at 50.0 (Flexible vs Medium)
                log_50 = 1.69897
                bound_err = torch.zeros_like(t_ka_filt)
                flex_mask = t_ka_filt <= 50.0
                bound_err[flex_mask] = F.relu(log_pka[flex_mask] - log_50)
                med_mask = (t_ka_filt > 50.0) & (t_ka_filt <= self.p80)
                bound_err[med_mask] = F.relu(log_50 - log_pka[med_mask])
                
                # Ceiling penalty: penalize predicting above P80!
                log_p80 = float(np.log10(self.p80))
                ceiling_err = F.relu(log_pka - log_p80)
                
                loss_ka = loss_ka_reg + 0.3 * torch.mean(bound_err) + 0.4 * torch.mean(ceiling_err)
            else:
                loss_ka = pred_ka.sum() * 0.0
        else:
            loss_ka = pred_ka.sum() * 0.0

        # 4. theta0 loss
        loss_t0 = F.mse_loss(pred_t0, true_t0) if true_t0.numel() > 0 else pred_t0.sum() * 0.0

        # 5. Regime classification loss (for valid angles)
        if pred_regime is not None and true_ka.numel() > 0:
            valid_mask = true_ka <= self.p80
            if valid_mask.sum() > 0:
                p_reg_filt = pred_regime[valid_mask]
                t_ka_filt = true_ka[valid_mask]
                
                # Regimes: 0 = Flexible (<=50), 1 = Medium (>50)
                reg_targets = torch.where(t_ka_filt <= 50.0,
                                          torch.zeros_like(t_ka_filt, dtype=torch.long),
                                          torch.ones_like(t_ka_filt, dtype=torch.long))
                cls_w = torch.tensor([1.0, 1.25, 1.0], device=true_ka.device)
                loss_reg = F.cross_entropy(p_reg_filt, reg_targets, weight=cls_w)
            else:
                loss_reg = pred_regime.sum() * 0.0
        else:
            loss_reg = torch.tensor(0.0, device=true_kb.device)

        total = (self.lam["k_bond"]  * loss_kb +
                 self.lam["r0"]      * loss_r0 +
                 self.lam["k_angle"] * loss_ka +
                 self.lam["theta0"]  * loss_t0 +
                 self.lam["regime"]  * loss_reg)

        return {
            "total": total,
            "k_bond": loss_kb,
            "r0": loss_r0,
            "k_angle": loss_ka,
            "theta0": loss_t0,
            "regime": loss_reg,
        }

def run_epoch(model, loader, loss_fn, optimizer, device, clip=5.0, is_train=True):
    model.train(is_train)
    totals = {"total": 0, "k_bond": 0, "r0": 0, "k_angle": 0, "theta0": 0, "regime": 0}
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            if is_train:
                optimizer.zero_grad()
            pred = model(batch, return_regime=True)
            targets = (batch.y_k_bond, batch.y_r0, batch.y_k_angle, batch.y_theta0)
            losses = loss_fn(pred, targets)
            if is_train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                optimizer.step()
            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
    base_args = base_ckpt.get("args", {})
    hidden = base_args.get("hidden", 128)
    layers = base_args.get("layers", 3)
    dropout = base_args.get("dropout", 0.2)

    print("=" * 70)
    print("  TRAINING SPECIALIZED 80th-PERCENTILE MODEL (k_angle <= 78.10 kJ/mol/rad^2)")
    print(f"  Initializing from checkpoints/best_model.pt (hidden={hidden}, layers={layers})")
    print(f"  Device: {device} | Epochs: {args.epochs} | LR: {args.lr}")
    print("=" * 70)

    train_ds = CGSpringDataset(root="data", split="train")
    val_ds   = CGSpringDataset(root="data", split="val")
    test_ds  = CGSpringDataset(root="data", split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CGSpringGNN(
        hidden_dim=hidden,
        n_layers=layers,
        dropout=dropout
    ).to(device)
    model.load_state_dict(base_ckpt["model_state"])

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn   = SpringConstantLossP80(p80_thresh=P80_THRESHOLD)

    ckpt_dir = "checkpoints"
    res_dir = "results"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    log_path = os.path.join(res_dir, "train_log_p80.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_total", "train_kb", "train_ka", "val_total", "val_kb", "val_ka", "lr", "time_s"])

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\nStarting {args.epochs} training epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_losses = run_epoch(model, train_loader, loss_fn, optimizer, device, is_train=True)
        vl_losses = run_epoch(model, val_loader, loss_fn, optimizer, device, is_train=False)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == 1 or vl_losses["total"] < best_val_loss:
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train: {tr_losses['total']:.4f} (ka={tr_losses['k_angle']:.3f}) | "
                  f"Val: {vl_losses['total']:.4f} (ka={vl_losses['k_angle']:.3f}) | LR: {lr_now:.1e} [{elapsed:.1f}s]")

        writer.writerow([epoch, tr_losses["total"], tr_losses["k_bond"], tr_losses["k_angle"],
                         vl_losses["total"], vl_losses["k_bond"], vl_losses["k_angle"], lr_now, round(elapsed, 2)])
        log_file.flush()

        if vl_losses["total"] < best_val_loss:
            best_val_loss = vl_losses["total"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "args": {**base_args, "p80_epochs": args.epochs, "p80_lr": args.lr},
                "p80_threshold": P80_THRESHOLD
            }, os.path.join(ckpt_dir, "best_model_p80.pt"))
            print(f"  --> [BEST P80 SAVED] epoch={epoch}, val_loss={best_val_loss:.4f}")

    log_file.close()
    print("\n" + "=" * 70)
    print(f"P80 Model Training Completed! Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Saved to: checkpoints/best_model_p80.pt")
    print("=" * 70)

if __name__ == "__main__":
    main()
