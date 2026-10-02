"""
gnn.py
------
GNN model for predicting CG spring constants.

Architecture (SchNet-style message passing):
  Input:
    - node features x [N, NODE_DIM]
    - bond graph    edge_index [2, 2B]
    - edge features edge_attr  [2B, EDGE_DIM]
    - angle triplets angle_idx [A, 3]

  Processing:
    - Node + edge embedding layers
    - N_LAYERS of CGInteractionBlock (message passing)

  Output heads (trained simultaneously):
    - Bond head   → k_bond [B], r0 [B]       per-edge prediction
    - Angle head  → k_angle [A], theta0 [A]  per-triplet prediction

All spring constant outputs pass through softplus to guarantee > 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from src.data.featurize import NODE_DIM, EDGE_DIM


# ─── Utility: MLP block ───────────────────────────────────────────────────────

def mlp(in_dim: int, hidden_dim: int, out_dim: int,
        n_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    """Build a simple MLP with SiLU activations."""
    layers = []
    dims   = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


from typing import Tuple


# ─── Message Passing Block ────────────────────────────────────────────────────

class CGInteractionBlock(MessagePassing):
    """
    One round of message passing for CG spring constant prediction.

    Message     m_{i←j}  = MLP([h_i ‖ h_j ‖ e_{ij}])
    Node Update h_i'     = LayerNorm(h_i + MLP([h_i ‖ Σ_j m_{i←j}]))
    Edge Update e_{ij}'  = LayerNorm(e_{ij} + MLP([h_i' ‖ h_j' ‖ e_{ij}]))
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__(aggr="add")

        self.message_mlp = mlp(
            in_dim     = 3 * hidden_dim,   # h_i, h_j, e_ij (embedded)
            hidden_dim = hidden_dim,
            out_dim    = hidden_dim,
            n_layers   = 2,
            dropout    = dropout,
        )
        self.update_mlp = mlp(
            in_dim     = 2 * hidden_dim,   # h_i, agg
            hidden_dim = hidden_dim,
            out_dim    = hidden_dim,
            n_layers   = 2,
            dropout    = dropout,
        )
        self.edge_update_mlp = mlp(
            in_dim     = 3 * hidden_dim,   # h_i', h_j', e_ij
            hidden_dim = hidden_dim,
            out_dim    = hidden_dim,
            n_layers   = 2,
            dropout    = dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor,
                edge_index: torch.Tensor,
                e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h:           [N, hidden_dim]
        edge_index:  [2, 2B]
        e:           [2B, hidden_dim]   (embedded edge features)
        """
        agg   = self.propagate(edge_index, h=h, e=e)          # [N, hidden_dim]
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))  # [N, hidden_dim]
        h_out = self.norm(h + h_new)                          # residual + norm

        # Dynamic edge update: incorporate updated node features into edge states
        src, dst = edge_index[0], edge_index[1]
        e_in = torch.cat([h_out[src], h_out[dst], e], dim=-1) # [2B, 3*hidden]
        e_new = self.edge_update_mlp(e_in)
        e_out = self.edge_norm(e + e_new)

        return h_out, e_out

    def message(self, h_i: torch.Tensor,
                h_j: torch.Tensor,
                e:   torch.Tensor) -> torch.Tensor:
        return self.message_mlp(torch.cat([h_i, h_j, e], dim=-1))


# ─── Main GNN ─────────────────────────────────────────────────────────────────

class CGSpringGNN(nn.Module):
    """
    Graph Neural Network for predicting CG spring constants.

    Args:
        node_dim:   Input node feature dimension  (default: NODE_DIM from featurize.py)
        edge_dim:   Input edge feature dimension  (default: EDGE_DIM from featurize.py)
        hidden_dim: Hidden representation size    (default: 256)
        n_layers:   Number of message-passing rounds (default: 4)
        dropout:    Dropout probability            (default: 0.1)
    """

    def __init__(self,
                 node_dim:   int   = NODE_DIM,
                 edge_dim:   int   = EDGE_DIM,
                 hidden_dim: int   = 256,
                 n_layers:   int   = 4,
                 dropout:    float = 0.10):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ── Input embeddings ────────────────────────────────────────────────
        self.node_embed = mlp(node_dim, hidden_dim, hidden_dim, n_layers=2)
        self.edge_embed = mlp(edge_dim, hidden_dim, hidden_dim, n_layers=2)

        # ── Message-passing layers ──────────────────────────────────────────
        self.conv_layers = nn.ModuleList([
            CGInteractionBlock(hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # Degree / coordination / topology embedding for angle steric crowding & ring status (25 dims)
        self.deg_embed = nn.Sequential(
            nn.Linear(25, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

        # --- Prediction heads (distinct heads prevent gradient interference)
        self.k_bond_head         = mlp(3 * hidden_dim, hidden_dim, 1, n_layers=2, dropout=dropout)
        self.r0_head             = mlp(3 * hidden_dim, hidden_dim, 1, n_layers=2, dropout=dropout)
        self.k_angle_head        = mlp(5 * hidden_dim + 32, hidden_dim, 1, n_layers=2, dropout=dropout)
        self.theta0_head         = mlp(5 * hidden_dim + 32, hidden_dim, 1, n_layers=2, dropout=dropout)
        self.k_angle_regime_head = mlp(5 * hidden_dim + 32, hidden_dim, 3, n_layers=2, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --- Forward pass -----------------------------------------------------

    def forward(self, data, return_regime: bool = False):
        """
        data: PyG Data or Batch object with fields:
              x, edge_index, edge_attr, angle_idx, angle_edge_idx

        Returns:
            pred_k_bond:        [B]    bond spring constants      [kJ/mol/nm^2]
            pred_r0:            [B]    equilibrium bond lengths   [nm]
            pred_k_angle:       [A]    angle spring constants     [kJ/mol/rad^2]
            pred_theta0:        [A]    equilibrium angles         [radians]
            pred_regime_logits: [A, 3] regime logits (if return_regime=True)
        """
        x          = data.x            # [N, node_dim]
        edge_index = data.edge_index   # [2, 2B]
        edge_attr  = data.edge_attr    # [2B, edge_dim]
        angle_idx  = data.angle_idx    # [A, 3]

        # 1. Embed inputs
        h = self.node_embed(x)          # [N, hidden]
        e = self.edge_embed(edge_attr)  # [2B, hidden]

        # 2. Message passing
        for conv in self.conv_layers:
            h, e = conv(h, edge_index, e)

        # 3. Bond predictions - use only forward edges (even indices)
        fwd_mask = torch.arange(0, edge_index.shape[1], 2, device=x.device)
        src = edge_index[0, fwd_mask]    # [B]
        dst = edge_index[1, fwd_mask]    # [B]
        e_fwd = e[fwd_mask]              # [B, hidden]

        # Symmetrized representations: invariant to i <-> j exchange
        h_bond_sum  = h[src] + h[dst]
        h_bond_diff = torch.abs(h[src] - h[dst])
        bond_in     = torch.cat([h_bond_sum, h_bond_diff, e_fwd], dim=-1)  # [B, 3*hidden]

        # Log10-scale prediction for k_bond (starts at ~6300 kJ/mol/nm^2)
        log_kb      = self.k_bond_head(bond_in).squeeze(-1) + 3.8
        pred_k_bond = torch.pow(10.0, torch.clamp(log_kb, min=2.0, max=5.0))

        # Equilibrium bond length: centered around 0.35 nm, bounded [0.20, 0.50] nm
        pred_r0     = 0.35 + 0.15 * torch.tanh(self.r0_head(bond_in).squeeze(-1))

        # 4. Angle predictions (invariant to outer bead order: i-j-k == k-j-i)
        if angle_idx.numel() > 0 and angle_idx.shape[0] > 0:
            hi = h[angle_idx[:, 0]]                              # [A, hidden]
            hj = h[angle_idx[:, 1]]                              # [A, hidden] central bead
            hk = h[angle_idx[:, 2]]                              # [A, hidden]
            h_angle_outer_sum  = hi + hk
            h_angle_outer_diff = torch.abs(hi - hk)

            # Retrieve incident contextualized edge representations (j-i and j-k)
            if hasattr(data, "angle_edge_idx") and data.angle_edge_idx.numel() > 0:
                e_ji = e[data.angle_edge_idx[:, 0]]
                e_jk = e[data.angle_edge_idx[:, 1]]
                e_angle_sum  = e_ji + e_jk
                e_angle_diff = torch.abs(e_ji - e_jk)
            else:
                e_angle_sum  = torch.zeros_like(hj)
                e_angle_diff = torch.zeros_like(hj)

            # Degree / steric coordination features & physical properties
            deg = degree(edge_index[0], num_nodes=x.size(0), dtype=x.dtype).unsqueeze(-1) / 4.0
            deg_j = deg[angle_idx[:, 1]]
            deg_outer_sum = deg[angle_idx[:, 0]] + deg[angle_idx[:, 2]]
            deg_outer_diff = torch.abs(deg[angle_idx[:, 0]] - deg[angle_idx[:, 2]])

            # Bead mass (at index 71) and size_oh (at index -3:)
            mass = x[:, 71:72]
            size = x[:, -3:]

            mass_j = mass[angle_idx[:, 1]]
            size_j = size[angle_idx[:, 1]]
            mass_arms_sum = mass[angle_idx[:, 0]] + mass[angle_idx[:, 2]]
            mass_arms_diff = torch.abs(mass[angle_idx[:, 0]] - mass[angle_idx[:, 2]])
            size_arms_sum = size[angle_idx[:, 0]] + size[angle_idx[:, 2]]
            size_arms_diff = torch.abs(size[angle_idx[:, 0]] - size[angle_idx[:, 2]])

            # Ring cycle features from incident edges (-9:-4 in edge_attr: is_ring, is_c3, is_c4, is_c5, is_c6_plus)
            if hasattr(data, "angle_edge_idx") and data.angle_edge_idx.numel() > 0:
                e1 = data.angle_edge_idx[:, 0]
                e2 = data.angle_edge_idx[:, 1]
                ring1 = edge_attr[e1, -9:-4]
                ring2 = edge_attr[e2, -9:-4]
                ring_sum = ring1 + ring2
                ring_diff = torch.abs(ring1 - ring2)
            else:
                ring_sum = torch.zeros((angle_idx.size(0), 5), device=x.device)
                ring_diff = torch.zeros((angle_idx.size(0), 5), device=x.device)

            angle_topo = torch.cat([
                deg_j, deg_outer_sum, deg_outer_diff,   # 3
                mass_j, size_j,                         # 1 + 3 = 4
                mass_arms_sum, mass_arms_diff,          # 1 + 1 = 2
                size_arms_sum, size_arms_diff,          # 3 + 3 = 6
                ring_sum, ring_diff                     # 5 + 5 = 10
            ], dim=-1)                                  # Total: 25 dims

            deg_emb = self.deg_embed(angle_topo)        # [A, 32]

            angle_in = torch.cat([hj, h_angle_outer_sum, h_angle_outer_diff, e_angle_sum, e_angle_diff, deg_emb], dim=-1)  # [A, 5*hidden + 32]

            # Log10-scale prediction for k_angle (starts at ~40 kJ/mol/rad^2, max 1000 = 10^3.0)
            log_ka             = self.k_angle_head(angle_in).squeeze(-1) + 1.6
            pred_k_angle       = torch.pow(10.0, torch.clamp(log_ka, min=0.5, max=3.2))
            pred_theta0        = torch.sigmoid(self.theta0_head(angle_in).squeeze(-1)) * torch.pi
            pred_regime_logits = self.k_angle_regime_head(angle_in)  # [A, 3]
        else:
            pred_k_angle       = torch.zeros(0, device=x.device)
            pred_theta0        = torch.zeros(0, device=x.device)
            pred_regime_logits = torch.zeros((0, 3), device=x.device)

        if return_regime:
            return pred_k_bond, pred_r0, pred_k_angle, pred_theta0, pred_regime_logits
        return pred_k_bond, pred_r0, pred_k_angle, pred_theta0

    # ── Model info ────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"CGSpringGNN(\n"
                f"  hidden_dim={self.hidden_dim}, "
                f"  layers={len(self.conv_layers)}, "
                f"  params={self.count_parameters():,}\n)")


if __name__ == "__main__":
    # Sanity check with a random graph
    from torch_geometric.data import Data

    N, B, A = 6, 5, 4
    data = Data(
        x          = torch.randn(N, NODE_DIM),
        edge_index = torch.randint(0, N, (2, 2*B)),
        edge_attr  = torch.randn(2*B, EDGE_DIM),
        angle_idx  = torch.randint(0, N, (A, 3)),
        y_k_bond   = torch.rand(B) * 4000 + 1000,
        y_r0       = torch.rand(B) * 0.2 + 0.35,
        y_k_angle  = torch.rand(A) * 100,
        y_theta0   = torch.rand(A) * 2.0 + 0.5,
    )

    model = CGSpringGNN()
    print(model)

    kb, r0, ka, t0 = model(data)
    print(f"\nOutputs:")
    print(f"  k_bond  shape: {kb.shape},  values: {kb.detach()[:3]}")
    print(f"  r0      shape: {r0.shape},  values: {r0.detach()[:3]}")
    print(f"  k_angle shape: {ka.shape},  values: {ka.detach()[:3]}")
    print(f"  theta0  shape: {t0.shape},  values: {t0.detach()[:3]}")
