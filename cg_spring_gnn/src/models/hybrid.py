"""
hybrid.py
---------
CGSpringHybridModel: Stacking Graph Neural Network (GNN) backbone with
Gradient Boosted Decision Trees (XGBoost) for coarse-grained force-field parameterization.

Key Architectural Principle:
  - Deep Graph MPNN learns contextualized geometric representations for nodes and incident edges.
  - Smooth MLP heads handle continuous equilibrium geometries (r0, theta0) and bond stiffness (k_bond).
  - Gradient-Boosted Trees (XGBoost) handle the discontinuous step-function look-up rules of MARTINI
    angle force constants (k_angle), bypassing the smooth-interpolation limitation of standard MLPs.
"""

import os
import io
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
from xgboost import XGBRegressor

from src.models.gnn import CGSpringGNN


class CGSpringHybridModel:
    """
    Hybrid GNN-GBDT model for MARTINI 3 CG parameter prediction.
    """

    def __init__(self,
                 gnn: CGSpringGNN,
                 xgb_angle_head: Optional[XGBRegressor] = None,
                 device: str = "cpu"):
        self.gnn = gnn.to(device)
        self.gnn.eval()
        self.xgb_angle_head = xgb_angle_head
        self.device = torch.device(device)

    def extract_angle_features(self, batch) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts GNN hidden states + raw node/edge skip connections for all angle triplets in batch.
        Returns:
            X_angles: [A, feature_dim] numpy array
            angle_idx: [A, 3] numpy array of bead indices
        """
        self.gnn.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            x          = batch.x
            edge_index = batch.edge_index
            edge_attr  = batch.edge_attr
            angle_idx  = batch.angle_idx

            if angle_idx.numel() == 0:
                return np.zeros((0, 1)), np.zeros((0, 3))

            # 1. Message passing representations
            h = self.gnn.node_embed(x)
            e = self.gnn.edge_embed(edge_attr)
            for conv in self.gnn.conv_layers:
                h, e = conv(h, edge_index, e)

            # Node triplets
            hi = h[angle_idx[:, 0]]
            hj = h[angle_idx[:, 1]]
            hk = h[angle_idx[:, 2]]

            h_sum  = hi + hk
            h_diff = torch.abs(hi - hk)

            # Incident edges
            if hasattr(batch, "angle_edge_idx") and batch.angle_edge_idx.numel() > 0:
                e_ji = e[batch.angle_edge_idx[:, 0]]
                e_jk = e[batch.angle_edge_idx[:, 1]]
                e_sum  = e_ji + e_jk
                e_diff = torch.abs(e_ji - e_jk)
            else:
                e_sum  = torch.zeros_like(hj)
                e_diff = torch.zeros_like(hj)

            # Raw skip features: un-smoothed one-hot bead identity and continuous properties
            xi = x[angle_idx[:, 0]]
            xj = x[angle_idx[:, 1]]
            xk = x[angle_idx[:, 2]]
            raw_skip = torch.cat([xj, xi + xk, torch.abs(xi - xk)], dim=-1)

            # Combined angle feature representation
            rep = torch.cat([hj, h_sum, h_diff, e_sum, e_diff, raw_skip], dim=-1)
            return rep.cpu().numpy(), angle_idx.cpu().numpy()

    def eval(self):
        """Sets the underlying GNN backbone to evaluation mode."""
        self.gnn.eval()
        return self

    def predict(self, batch) -> Dict[str, np.ndarray]:
        """
        End-to-end inference on a PyG batch or Data object.
        Returns dictionary of predicted parameters:
            - k_bond:   [B] kJ/mol/nm^2
            - r0:       [B] nm
            - k_angle:  [A] kJ/mol/rad^2
            - theta0:   [A] degrees
            - theta0_rad: [A] radians
        """
        self.gnn.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            p_kb, p_r0, gnn_ka, p_th0 = self.gnn(batch)

            out = {
                "k_bond": p_kb.cpu().numpy(),
                "r0": p_r0.cpu().numpy(),
                "theta0_rad": p_th0.cpu().numpy(),
                "theta0_deg": np.degrees(p_th0.cpu().numpy()),
            }

            # Angle stiffness prediction
            if batch.angle_idx.numel() > 0 and batch.angle_idx.shape[0] > 0:
                if self.xgb_angle_head is not None:
                    # Use SOTA hybrid tree head
                    X_ang, _ = self.extract_angle_features(batch)
                    log_pred = self.xgb_angle_head.predict(X_ang)
                    pred_ka = np.exp(log_pred)
                    # Bound to valid physical range [2.0, 1000.0]
                    pred_ka = np.clip(pred_ka, 2.0, 1000.0)
                else:
                    # Fallback to GNN MLP head
                    pred_ka = gnn_ka.cpu().numpy()
                out["k_angle"] = pred_ka
            else:
                out["k_angle"] = np.zeros(0, dtype=np.float32)

            return out

    def save(self, filepath: str):
        """
        Serializes the complete hybrid model (PyTorch GNN weights + XGBoost binary) into one file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        xgb_bytes = None
        if self.xgb_angle_head is not None:
            xgb_bytes = self.xgb_angle_head.get_booster().save_raw(raw_format="json")

        payload = {
            "gnn_state": self.gnn.state_dict(),
            "gnn_args": {
                "hidden_dim": self.gnn.hidden_dim,
                "n_layers": len(self.gnn.conv_layers),
            },
            "xgb_angle_head_json": xgb_bytes,
        }
        torch.save(payload, filepath)
        print(f"Hybrid model saved successfully to: {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "CGSpringHybridModel":
        """
        Loads the complete hybrid model from a saved checkpoint.
        """
        payload = torch.load(filepath, map_location=device, weights_only=False)
        gnn_args = payload.get("gnn_args", {})
        gnn = CGSpringGNN(
            hidden_dim=gnn_args.get("hidden_dim", 128),
            n_layers=gnn_args.get("n_layers", 3),
        )
        gnn.load_state_dict(payload["gnn_state"])

        xgb_head = None
        if payload.get("xgb_angle_head_json") is not None:
            xgb_head = XGBRegressor()
            xgb_head.load_model(bytearray(payload["xgb_angle_head_json"]))

        return cls(gnn=gnn, xgb_angle_head=xgb_head, device=device)
