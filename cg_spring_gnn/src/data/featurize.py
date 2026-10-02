"""
featurize.py
------------
Converts a CGMolecule (from parse_itp) into a PyTorch Geometric Data object.

Node features (per CG bead):
  - One-hot bead type  (len=N_TYPES)
  - Normalized mass    (1)
  - Charge             (1)
  - Polarity category  (4, one-hot: apolar/intermediate/polar/charged)
  - Size category      (3, one-hot: tiny/small/regular)
  Total: N_TYPES + 9 = 36 features

Edge features (per CG bond, both directions):
  - Equilibrium distance r0  [nm]  (1)
  - One-hot bead type i            (N_TYPES)
  - One-hot bead type j            (N_TYPES)
  Total: 1 + 2*N_TYPES = 37 features

Labels:
  - y_k_bond   [E/2] bond spring constants  [kJ/mol/nm²]
  - y_r0       [E/2] equilibrium bond lengths [nm]
  - y_k_angle  [A]   angle spring constants  [kJ/mol/rad²]
  - y_theta0   [A]   equilibrium angles      [radians]
  - angle_idx  [A,3] triplet bead indices for angle prediction
"""

import math
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple
from src.data.parse_itp import CGMolecule, CGAtom


class CGData(Data):
    """PyG Data object that properly increments angle_idx across batches."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == "angle_idx":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)



# ─── Bead type vocabulary (MARTINI 3) ─────────────────────────────────────────

BEAD_TYPES = [
    # Regular beads (72 Da)
    "C1","C2","C3","C4","C5","C6",
    "N1","N2","N3","N4","N5","N6",
    "P1","P2","P3","P4","P5","P6",
    "Q1","Q2","Q3","Q4","Q5",
    "Qa","Qd","Qda",
    # Small beads (54 Da)
    "SC1","SC2","SC3","SC4","SC5","SC6",
    "SN1","SN2","SN3","SN4","SN5","SN6",
    "SP1","SP2","SP3","SP4","SP5","SP6",
    "SQ1","SQ2","SQ3","SQ4","SQ5",
    # Tiny beads (36 Da)
    "TC1","TC2","TC3","TC4","TC5",
    "TN1","TN2","TN3","TN4","TN5",
    "TP1","TP2","TP3","TP4","TP5",
    # Water / ions
    "W","WF","NA","CL","CA","MG",
    # Unknown / catch-all
    "UNK",
]
BEAD2IDX = {b: i for i, b in enumerate(BEAD_TYPES)}
N_TYPES   = len(BEAD_TYPES)

# Polarity: C/N-types → apolar/intermediate, P→polar, Q→charged
POLARITY = {
    "C": 0, "SC": 0, "TC": 0,
    "N": 1, "SN": 1, "TN": 1,
    "P": 2, "SP": 2, "TP": 2,
    "Q": 3, "SQ": 3, "W": 2, "WF": 2, "NA": 3, "CL": 3, "CA": 3, "MG": 3,
}

NODE_DIM = N_TYPES + 1 + 1 + 4 + 3  # = N_TYPES + 9
TOPO_DIM = 9
EDGE_DIM = 1 + 2 * N_TYPES + TOPO_DIM


# ─── Node featurizer ──────────────────────────────────────────────────────────

def bead_features(atom: CGAtom) -> torch.Tensor:
    """
    Build a feature vector for one CG bead.
    Returns tensor of shape [NODE_DIM].
    """
    # 1) One-hot bead type (robust to MARTINI 3 sublevels like C4a, SN2a)
    oh = torch.zeros(N_TYPES)
    btype_raw = atom.type.strip()
    btype_base = btype_raw.rstrip("abcdefghijklmnopqrstuvwxyz")
    idx = BEAD2IDX.get(btype_raw, BEAD2IDX.get(btype_base, BEAD2IDX["UNK"]))
    oh[idx] = 1.0

    # 2) Normalized mass  (regular=1.0, small≈0.75, tiny≈0.5)
    mass_norm = torch.tensor([atom.mass / 72.0])

    # 3) Charge (raw, typically ±1 or 0)
    charge = torch.tensor([atom.charge])

    # 4) Polarity (one-hot of 4 categories)
    pol_oh = torch.zeros(4)
    prefix = btype_base[:2] if len(btype_base) >= 2 and btype_base[:2] in POLARITY else btype_base[:1]
    pol_oh[POLARITY.get(prefix, 1)] = 1.0

    # 5) Size category (one-hot: [tiny, small, regular])
    size_oh = torch.zeros(3)
    if btype_base.startswith("T"):
        size_oh[0] = 1.0
    elif btype_base.startswith("S"):
        size_oh[1] = 1.0
    else:
        size_oh[2] = 1.0

    return torch.cat([oh, mass_norm, charge, pol_oh, size_oh])  # [NODE_DIM]


# --- Edge featurizer ----------------------------------------------------------

def bond_features(r0: float, type_i: str, type_j: str,
                  c_size: int = 0, deg_i: int = 2, deg_j: int = 2) -> torch.Tensor:
    """
    Build a feature vector for one CG bond (undirected, used for both directions).
    Includes:
      - Equilibrium distance r0 (1)
      - One-hot bead type i (N_TYPES)
      - One-hot bead type j (N_TYPES)
      - Graph topology features (TOPO_DIM = 9):
        * is_ring: 1 if in any cycle
        * is_c3: 1 if in 3-membered ring (e.g. rigid planar aromatic/sterol triangle)
        * is_c4: 1 if in 4-membered ring
        * is_c5: 1 if in 5-membered ring (e.g. sugar/furanose)
        * is_c6_plus: 1 if in 6+-membered ring (e.g. pyranose)
        * deg_min: min(deg_i, deg_j) / 6.0
        * deg_max: max(deg_i, deg_j) / 6.0
        * deg_sum: (deg_i + deg_j) / 12.0
        * deg_diff: abs(deg_i - deg_j) / 6.0
    Returns tensor of shape [EDGE_DIM].
    """
    r0_t = torch.tensor([r0])

    oh_i = torch.zeros(N_TYPES)
    ti_raw = type_i.strip()
    ti_base = ti_raw.rstrip("abcdefghijklmnopqrstuvwxyz")
    oh_i[BEAD2IDX.get(ti_raw, BEAD2IDX.get(ti_base, BEAD2IDX["UNK"]))] = 1.0

    oh_j = torch.zeros(N_TYPES)
    tj_raw = type_j.strip()
    tj_base = tj_raw.rstrip("abcdefghijklmnopqrstuvwxyz")
    oh_j[BEAD2IDX.get(tj_raw, BEAD2IDX.get(tj_base, BEAD2IDX["UNK"]))] = 1.0

    is_ring = 1.0 if c_size > 0 else 0.0
    is_c3 = 1.0 if c_size == 3 else 0.0
    is_c4 = 1.0 if c_size == 4 else 0.0
    is_c5 = 1.0 if c_size == 5 else 0.0
    is_c6_plus = 1.0 if c_size >= 6 else 0.0

    deg_min = float(min(deg_i, deg_j)) / 6.0
    deg_max = float(max(deg_i, deg_j)) / 6.0
    deg_sum = float(deg_i + deg_j) / 12.0
    deg_diff = float(abs(deg_i - deg_j)) / 6.0

    topo = torch.tensor([
        is_ring, is_c3, is_c4, is_c5, is_c6_plus,
        deg_min, deg_max, deg_sum, deg_diff
    ], dtype=torch.float)

    return torch.cat([r0_t, oh_i, oh_j, topo])  # [EDGE_DIM]


# ─── Molecule → PyG graph ─────────────────────────────────────────────────────

def molecule_to_graph(mol: CGMolecule) -> Data:
    """
    Convert a CGMolecule into a PyTorch Geometric Data object.

    Atoms are 1-indexed in ITP files; we convert to 0-indexed here.
    Edges are stored bidirectionally (i→j and j→i).
    Labels are only stored once per bond (forward direction).

    Returns:
        data.x           [N, NODE_DIM]  node features
        data.edge_index  [2, 2*B]       bidirectional bond graph
        data.edge_attr   [2*B, EDGE_DIM] edge features
        data.y_k_bond    [B]            bond spring constants
        data.y_r0        [B]            equilibrium bond lengths [nm]
        data.y_k_angle   [A]            angle spring constants
        data.y_theta0    [A]            equilibrium angles [radians]
        data.angle_idx   [A, 3]         bead index triplets for angles (0-indexed)
        data.mol_name    str
        data.n_atoms     int
    """
    n = mol.n_atoms
    if n == 0:
        return None

    # ── Node features ────────────────────────────────────────────────────
    node_feats = [bead_features(a) for a in mol.atoms]
    x = torch.stack(node_feats)  # [N, NODE_DIM]

    # ITP atom indices are 1-based; build mapping index→position
    idx_to_pos = {a.idx: pos for pos, a in enumerate(mol.atoms)}

    # ── Graph topology (cycles & degrees) via NetworkX ───────────────────
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for bond in mol.bonds:
        pi = idx_to_pos.get(bond.i)
        pj = idx_to_pos.get(bond.j)
        if pi is not None and pj is not None:
            G.add_edge(pi, pj)

    cycles = nx.cycle_basis(G)
    edge_min_cycle = {}
    for cycle in cycles:
        cl = len(cycle)
        for i in range(cl):
            ek = tuple(sorted((cycle[i], cycle[(i+1)%cl])))
            if ek not in edge_min_cycle or cl < edge_min_cycle[ek]:
                edge_min_cycle[ek] = cl

    # ── Edge index + edge features + bond labels ──────────────────────────
    edge_src, edge_dst = [], []
    edge_feats        = []
    k_bond_list       = []
    r0_list           = []

    for bond in mol.bonds:
        pi = idx_to_pos.get(bond.i)
        pj = idx_to_pos.get(bond.j)
        if pi is None or pj is None:
            continue   # skip bonds to atoms not in [ atoms ] section

        c_sz = edge_min_cycle.get(tuple(sorted((pi, pj))), 0)
        deg_i = G.degree[pi]
        deg_j = G.degree[pj]

        feat = bond_features(bond.r0, bond.atom_type_i, bond.atom_type_j,
                             c_size=c_sz, deg_i=deg_i, deg_j=deg_j)

        # Forward direction i→j (label stored here)
        edge_src.append(pi); edge_dst.append(pj)
        edge_feats.append(feat)
        k_bond_list.append(bond.k)
        r0_list.append(bond.r0)

        # Reverse direction j→i (same features, no label — handled in model)
        edge_src.append(pj); edge_dst.append(pi)
        edge_feats.append(feat)

    if not edge_src:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)  # [2, 2B]
    edge_attr  = torch.stack(edge_feats)                                 # [2B, EDGE_DIM]
    y_k_bond   = torch.tensor(k_bond_list, dtype=torch.float)            # [B]
    y_r0       = torch.tensor(r0_list,     dtype=torch.float)            # [B]

    # Map directed edge (u, v) -> index in edge_index
    edge_map = {(src, dst): idx for idx, (src, dst) in enumerate(zip(edge_src, edge_dst))}

    # ── Angle triplets + incident edges + labels ──────────────────────────
    angle_triplets      = []
    angle_edge_triplets = []
    k_angle_list        = []
    theta0_list         = []

    for angle in mol.angles:
        pi = idx_to_pos.get(angle.i)
        pj = idx_to_pos.get(angle.j)
        pk = idx_to_pos.get(angle.k)
        if pi is None or pj is None or pk is None:
            continue

        # Directed edge indices for incident bonds (pj -> pi) and (pj -> pk)
        e1 = edge_map.get((pj, pi), edge_map.get((pi, pj), 0))
        e2 = edge_map.get((pj, pk), edge_map.get((pk, pj), 0))

        angle_triplets.append([pi, pj, pk])
        angle_edge_triplets.append([e1, e2])
        k_angle_list.append(angle.k_ang)
        theta0_list.append(math.radians(angle.theta0))  # → radians

    if angle_triplets:
        angle_idx      = torch.tensor(angle_triplets, dtype=torch.long)       # [A, 3]
        angle_edge_idx = torch.tensor(angle_edge_triplets, dtype=torch.long)  # [A, 2]
        y_k_angle      = torch.tensor(k_angle_list,   dtype=torch.float)      # [A]
        y_theta0       = torch.tensor(theta0_list,     dtype=torch.float)      # [A]
    else:
        angle_idx      = torch.zeros((0, 3), dtype=torch.long)
        angle_edge_idx = torch.zeros((0, 2), dtype=torch.long)
        y_k_angle      = torch.zeros(0,       dtype=torch.float)
        y_theta0       = torch.zeros(0,       dtype=torch.float)

    data = CGData(
        x              = x,
        edge_index     = edge_index,
        edge_attr      = edge_attr,
        y_k_bond       = y_k_bond,
        y_r0           = y_r0,
        y_k_angle      = y_k_angle,
        y_theta0       = y_theta0,
        angle_idx      = angle_idx,
        angle_edge_idx = angle_edge_idx,
        mol_name       = mol.name,
        n_atoms        = n,
    )
    return data


# ─── Batch conversion ─────────────────────────────────────────────────────────

def molecules_to_graphs(molecules: List[CGMolecule],
                        verbose: bool = True) -> List[Data]:
    """Convert a list of CGMolecule objects to PyG Data graphs."""
    graphs, skipped = [], 0
    for mol in molecules:
        g = molecule_to_graph(mol)
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1
    if verbose:
        print(f"Converted {len(graphs)} molecules to graphs  "
              f"(skipped {skipped} empty/invalid)")
    return graphs


if __name__ == "__main__":
    from src.data.parse_itp import parse_all_martini
    mols   = parse_all_martini("data/raw/martini3", verbose=False)
    graphs = molecules_to_graphs(mols)
    if graphs:
        g = graphs[0]
        print(f"\nSample graph: {g.mol_name}")
        print(f"  Nodes:      {g.x.shape}")
        print(f"  Edges:      {g.edge_index.shape}")
        print(f"  Edge attr:  {g.edge_attr.shape}")
        print(f"  k_bond:     {g.y_k_bond[:3]}")
        print(f"  k_angle:    {g.y_k_angle[:3]}")
        print(f"  angle_idx:  {g.angle_idx[:3]}")
