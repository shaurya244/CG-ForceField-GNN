"""
train.py
--------
Training script for CGSpringGNN.

Run:
    python train.py                          # uses default config
    python train.py --hidden 128 --layers 3  # custom hyperparams
    python train.py --help                   # show all options

Saves:
    checkpoints/best_model.pt   — best validation loss checkpoint
    results/train_log.csv       — per-epoch loss log
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
import numpy as np
from torch.optim          import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader   import DataLoader

# ── ensure project root is on PYTHONPATH ──────────────────────────────────────
from src.data.dataset  import CGSpringDataset, print_dataset_stats
from src.models.gnn    import CGSpringGNN
from src.models.loss   import SpringConstantLoss
from src.utils.metrics import compute_metrics, print_metrics


# ─── Argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Train GNN for CG spring constant prediction"
    )
    # Data
    p.add_argument("--data_root",  default="data",  help="Dataset root directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers",type=int, default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")

    # Model
    p.add_argument("--hidden",  type=int,   default=128, help="Hidden dimension")
    p.add_argument("--layers",  type=int,   default=3,   help="Message-passing layers")
    p.add_argument("--dropout", type=float, default=0.2)

    # Training
    p.add_argument("--epochs",  type=int,   default=200)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--wd",      type=float, default=1e-4, help="Weight decay")
    p.add_argument("--clip",    type=float, default=5.0,  help="Gradient clip norm")

    # Loss weights
    p.add_argument("--lam_kb", type=float, default=1.0, help="Weight for k_bond loss")
    p.add_argument("--lam_r0", type=float, default=0.5, help="Weight for r0 loss")
    p.add_argument("--lam_ka", type=float, default=1.0, help="Weight for k_angle loss")
    p.add_argument("--lam_t0", type=float, default=0.5, help="Weight for theta0 loss")

    # I/O
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--result_dir", default="results")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ─── Training helpers ─────────────────────────────────────────────────────────

def run_epoch(model, loader, loss_fn, optimizer, device,
              clip: float = 5.0, is_train: bool = True):
    """Run one epoch. Returns average total loss and component losses."""
    model.train(is_train)
    totals = {"total": 0, "k_bond": 0, "r0": 0, "k_angle": 0, "theta0": 0, "regime": 0}
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)

            if is_train:
                optimizer.zero_grad()

            pred    = model(batch, return_regime=True)
            targets = (batch.y_k_bond, batch.y_r0,
                       batch.y_k_angle, batch.y_theta0)
            losses  = loss_fn(pred, targets)

            if is_train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip
                )
                optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def evaluate(model, loader, device):
    """
    Collect all predictions and ground-truth values across the loader.
    Returns numpy arrays for computing metrics.
    """
    model.eval()
    all_pred_kb, all_true_kb = [], []
    all_pred_ka, all_true_ka = [], []
    all_pred_r0, all_true_r0 = [], []
    all_pred_t0, all_true_t0 = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_kb, pred_r0, pred_ka, pred_t0 = model(batch)

            all_pred_kb.extend(pred_kb.cpu().numpy())
            all_true_kb.extend(batch.y_k_bond.cpu().numpy())
            all_pred_r0.extend(pred_r0.cpu().numpy())
            all_true_r0.extend(batch.y_r0.cpu().numpy())

            if pred_ka.numel() > 0 and batch.y_k_angle.numel() > 0:
                all_pred_ka.extend(pred_ka.cpu().numpy())
                all_true_ka.extend(batch.y_k_angle.cpu().numpy())
                # Store in degrees for intuitive metrics
                all_pred_t0.extend(np.degrees(pred_t0.cpu().numpy()))
                all_true_t0.extend(np.degrees(batch.y_theta0.cpu().numpy()))

    return (np.array(all_pred_kb), np.array(all_true_kb),
            np.array(all_pred_ka), np.array(all_true_ka),
            np.array(all_pred_r0), np.array(all_true_r0),
            np.array(all_pred_t0), np.array(all_true_t0))


# ─── Main training loop ───────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CG Spring Constant GNN - Training")
    print(f"  Device : {device}")
    print(f"  Hidden : {args.hidden}   Layers : {args.layers}   Epochs : {args.epochs}")
    print(f"{'='*60}\n")

    # --- Datasets ---------------------------------------------------------
    print("Loading datasets ...")
    train_ds = CGSpringDataset(root=args.data_root, split="train")
    val_ds   = CGSpringDataset(root=args.data_root, split="val")
    test_ds  = CGSpringDataset(root=args.data_root, split="test")

    print(f"\nTrain  : {train_ds}")
    print_dataset_stats(train_ds)
    print(f"\nVal    : {val_ds}")
    print(f"Test   : {test_ds}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

    # ── Model ─────────────────────────────────────────────────────────────
    model = CGSpringGNN(
        hidden_dim = args.hidden,
        n_layers   = args.layers,
        dropout    = args.dropout,
    ).to(device)
    print(f"Model: {model}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn   = SpringConstantLoss(
        lambda_k_bond  = args.lam_kb,
        lambda_r0      = args.lam_r0,
        lambda_k_angle = args.lam_ka,
        lambda_theta0  = args.lam_t0,
    )

    # ── Logging ───────────────────────────────────────────────────────────
    os.makedirs(args.ckpt_dir,   exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "train_log.csv")
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch","train_total","train_kb","train_ka","train_r0",
                     "val_total","val_kb","val_ka","val_r0","lr","time_s"])

    best_val_loss = float("inf")
    best_epoch    = 0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_losses = run_epoch(model, train_loader, loss_fn,
                                  optimizer, device, is_train=True)
        val_losses   = run_epoch(model, val_loader,   loss_fn,
                                  optimizer, device, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        # Console log
        print(f"Epoch {epoch:3d}/{args.epochs}  |  "
              f"Train {train_losses['total']:.4f} "
              f"(kb={train_losses['k_bond']:.3f} ka={train_losses['k_angle']:.3f} reg={train_losses['regime']:.3f})  |  "
              f"Val {val_losses['total']:.4f} "
              f"(kb={val_losses['k_bond']:.3f} ka={val_losses['k_angle']:.3f} reg={val_losses['regime']:.3f})  |  "
              f"LR={lr_now:.1e}  [{elapsed:.1f}s]")

        # CSV log
        writer.writerow([
            epoch,
            train_losses["total"], train_losses["k_bond"],
            train_losses["k_angle"], train_losses["r0"],
            val_losses["total"],   val_losses["k_bond"],
            val_losses["k_angle"], val_losses["r0"],
            lr_now, round(elapsed, 2)
        ])
        log_file.flush()

        # Save best checkpoint
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_epoch    = epoch
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_loss":     best_val_loss,
                "args":         vars(args),
            }, os.path.join(args.ckpt_dir, "best_model.pt"))
            print(f"  [BEST] New best model saved  (val_loss={best_val_loss:.4f})")

    log_file.close()

    # --- Final test evaluation --------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training complete.  Best epoch: {best_epoch}  "
          f"Val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # Load best model for test evaluation
    ckpt = torch.load(os.path.join(args.ckpt_dir, "best_model.pt"),
                      weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    pred_kb, true_kb, pred_ka, true_ka, pred_r0, true_r0, pred_t0, true_t0 = \
        evaluate(model, test_loader, device)

    metrics_kb = compute_metrics(pred_kb, true_kb, name="k_bond")
    metrics_ka = compute_metrics(pred_ka, true_ka, name="k_angle")
    metrics_r0 = compute_metrics(pred_r0, true_r0, name="r0")
    metrics_t0 = compute_metrics(pred_t0, true_t0, name="theta0") if len(pred_t0) > 0 else {}

    print_metrics(metrics_kb, header="Test - k_bond")
    if len(pred_ka) > 0:
        print_metrics(metrics_ka, header="Test - k_angle")
    print_metrics(metrics_r0, header="Test - r0")
    if metrics_t0:
        print_metrics(metrics_t0, header="Test - theta0 (degrees)")

    # Save test metrics
    import json
    all_metrics = {**metrics_kb, **metrics_ka, **metrics_r0, **metrics_t0}
    with open(os.path.join(args.result_dir, "test_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nTest metrics saved to {args.result_dir}/test_metrics.json")


if __name__ == "__main__":
    args = get_args()
    main(args)
