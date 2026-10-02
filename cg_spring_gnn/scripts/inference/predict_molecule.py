"""
predict_molecule.py
-------------------
End-to-end inference pipeline for predicting MARTINI 3 coarse-grained
force-field parameters (k_bond, r0, k_angle, theta0) for any molecule
and generating a ready-to-run GROMACS .itp file.

Usage:
    # 1. Run demo on an unseen test molecule (e.g., Ibuprofen or Novel Lipid):
    python predict_molecule.py --demo

    # 2. Predict parameters for an existing unparameterized .itp:
    python predict_molecule.py --input path/to/molecule.itp --out predicted.itp

    # 3. Specify custom beads and bonds via CLI:
    python predict_molecule.py --name NOVEL_MOL --beads Q1 SC1 SC1 SP1 --bonds "1-2,2-3,3-4"
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
import argparse
import math
import torch
import numpy as np

# Ensure project root is on PYTHONPATH
from src.models.gnn import CGSpringGNN
from src.data.dataset import CGData
from src.data.featurize import bead_features, bond_features
from src.data.parse_itp import CGMolecule, CGAtom, CGBond, parse_itp


def build_graph_from_topology(mol_name: str, beads: list, bonds: list):
    """
    Constructs a PyG CGData object from a list of bead types and bond pairs.
    beads: list of bead type strings, e.g. ['Q1', 'SC1', 'SC1', 'SP1'] (1-indexed)
    bonds: list of (i, j) tuples (1-indexed)
    """
    n = len(beads)
    node_feats = []
    for idx, btype in enumerate(beads, start=1):
        # Default mass & charge estimation based on MARTINI bead
        mass = 36.0 if btype.startswith("T") else 54.0 if btype.startswith("S") else 72.0
        charge = -1.0 if "Qa" in btype else 1.0 if "Qd" in btype else -0.5 if "Q1" in btype else 0.5 if "Q2" in btype else 0.0
        atom = CGAtom(idx=idx, type=btype, resid=1, resname="MOL", name=f"B{idx}", charge=charge, mass=mass)
        feat = bead_features(atom)
        node_feats.append(feat)

    x = torch.stack(node_feats)

    # Build NetworkX graph for topology & cycle extraction
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j) in bonds:
        G.add_edge(i - 1, j - 1)

    cycles = nx.cycle_basis(G)
    edge_min_cycle = {}
    for cycle in cycles:
        cl = len(cycle)
        for idx in range(cl):
            ek = tuple(sorted((cycle[idx], cycle[(idx + 1) % cl])))
            if ek not in edge_min_cycle or cl < edge_min_cycle[ek]:
                edge_min_cycle[ek] = cl

    # Detect adjacency for bond features & angles
    adj = {i: [] for i in range(1, n + 1)}
    for (i, j) in bonds:
        adj[i].append(j)
        adj[j].append(i)

    edge_src, edge_dst, edge_feats = [], [], []
    for (i, j) in bonds:
        pi, pj = i - 1, j - 1
        btype_i = beads[pi]
        btype_j = beads[pj]
        c_sz = edge_min_cycle.get(tuple(sorted((pi, pj))), 0)
        deg_i = G.degree[pi]
        deg_j = G.degree[pj]
        feat = bond_features(0.35, btype_i, btype_j, c_size=c_sz, deg_i=deg_i, deg_j=deg_j)
        # Forward
        edge_src.append(pi); edge_dst.append(pj)
        edge_feats.append(feat)
        # Reverse
        edge_src.append(pj); edge_dst.append(pi)
        edge_feats.append(feat)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.stack(edge_feats)

    # Detect all angles (i - j - k) centered at j
    angle_triplets = []
    for j in range(1, n + 1):
        neighbors = sorted(adj[j])
        for idx1 in range(len(neighbors)):
            for idx2 in range(idx1 + 1, len(neighbors)):
                i = neighbors[idx1]
                k = neighbors[idx2]
                angle_triplets.append([i - 1, j - 1, k - 1])

    edge_map = {(src, dst): idx for idx, (src, dst) in enumerate(zip(edge_src, edge_dst))}
    angle_edge_triplets = []
    for (pi, pj, pk) in angle_triplets:
        e1 = edge_map.get((pj, pi), edge_map.get((pi, pj), 0))
        e2 = edge_map.get((pj, pk), edge_map.get((pk, pj), 0))
        angle_edge_triplets.append([e1, e2])

    if angle_triplets:
        angle_idx      = torch.tensor(angle_triplets, dtype=torch.long)
        angle_edge_idx = torch.tensor(angle_edge_triplets, dtype=torch.long)
    else:
        angle_idx      = torch.zeros((0, 3), dtype=torch.long)
        angle_edge_idx = torch.zeros((0, 2), dtype=torch.long)

    return CGData(
        x              = x,
        edge_index     = edge_index,
        edge_attr      = edge_attr,
        angle_idx      = angle_idx,
        angle_edge_idx = angle_edge_idx,
        mol_name       = mol_name,
        n_atoms        = n
    ), bonds, angle_triplets


MARTINI_KB_BINS = np.array([1250.0, 2500.0, 3000.0, 3800.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 50000.0])
MARTINI_KA_BINS = np.array([25.0, 35.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0])


def snap_to_martini(values: np.ndarray, bins: np.ndarray, rel_tol: float = 0.20) -> np.ndarray:
    """Snap continuous force constant predictions to canonical MARTINI bins if within tolerance."""
    if len(values) == 0:
        return values
    snapped = values.copy()
    for idx, v in enumerate(values):
        diffs = np.abs(bins - v) / (bins + 1e-6)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= rel_tol:
            snapped[idx] = bins[best_idx]
    return snapped


def predict_parameters(model, data, device, snap_ff: bool = False):
    """Run inference with the trained model (Hybrid or pure GNN), optionally snapping to MARTINI bins."""
    data = data.to(device)
    if hasattr(model, "xgb_angle_head"):
        # CGSpringHybridModel
        preds = model.predict(data)
        kb = preds["k_bond"]
        r0 = preds["r0"]
        ka = preds["k_angle"]
        t0_deg = preds["theta0_deg"]
    else:
        # Standard CGSpringGNN
        model.eval()
        with torch.no_grad():
            pred_kb, pred_r0, pred_ka, pred_t0 = model(data)
        kb = pred_kb.cpu().numpy()
        r0 = pred_r0.cpu().numpy()
        ka = pred_ka.cpu().numpy() if pred_ka.numel() > 0 else np.array([])
        t0_deg = np.degrees(pred_t0.cpu().numpy()) if pred_t0.numel() > 0 else np.array([])

    if snap_ff:
        kb = snap_to_martini(kb, MARTINI_KB_BINS, rel_tol=0.20)
        ka = snap_to_martini(ka, MARTINI_KA_BINS, rel_tol=0.20)

    return kb, r0, ka, t0_deg


def load_model_checkpoint(ckpt_path: str, device: str = "cpu"):
    """
    Intelligently loads either a CGSpringHybridModel or standard CGSpringGNN checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "xgb_angle_head_json" in ckpt:
        from src.models.hybrid import CGSpringHybridModel
        print(f"Loading SOTA CGSpringHybridModel (GNN + XGBoost) from {ckpt_path} ...")
        return CGSpringHybridModel.load(ckpt_path, device=device)
    else:
        print(f"Loading trained CGSpringGNN checkpoint from {ckpt_path} ...")
        model = CGSpringGNN(
            hidden_dim=ckpt.get("args", {}).get("hidden", 128),
            n_layers=ckpt.get("args", {}).get("layers", 3),
            dropout=ckpt.get("args", {}).get("dropout", 0.2)
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model


def write_gromacs_itp(output_path: str, mol_name: str, beads: list, bonds: list,
                       angle_triplets: list, kb: np.ndarray, r0: np.ndarray,
                       ka: np.ndarray, t0_deg: np.ndarray):
    """Generate a clean, standard GROMACS MARTINI .itp file."""
    with open(output_path, "w") as f:
        f.write(f"; GROMACS Topology for {mol_name}\n")
        f.write(f"; Predicted by CGSpringGNN (GNN Coarse-Grained Force Field Engine)\n\n")

        f.write("[ moleculetype ]\n")
        f.write("; molname      nrexcl\n")
        f.write(f"  {mol_name:<12s} 1\n\n")

        f.write("[ atoms ]\n")
        f.write("; nr   type   resnr  resname  atom   cgnr   charge   mass\n")
        for i, btype in enumerate(beads, start=1):
            mass = 36.0 if btype.startswith("T") else 54.0 if btype.startswith("S") else 72.0
            charge = -1.0 if "Qa" in btype else 1.0 if "Qd" in btype else -0.5 if "Q1" in btype else 0.5 if "Q2" in btype else 0.0
            f.write(f"  {i:<4d} {btype:<6s} 1      {mol_name[:4]:<6s} B{i:<4d} {i:<4d} {charge:>6.2f} {mass:>6.1f}\n")

        f.write("\n[ bonds ]\n")
        f.write("; i    j    funct   r0 [nm]   k [kJ/mol/nm^2]\n")
        for idx, (i, j) in enumerate(bonds):
            f.write(f"  {i:<4d} {j:<4d} 1       {r0[idx]:>7.4f}   {kb[idx]:>8.1f}\n")

        if len(angle_triplets) > 0:
            f.write("\n[ angles ]\n")
            f.write("; i    j    k    funct   theta0 [deg]   k [kJ/mol/rad^2]\n")
            for idx, (pi, pj, pk) in enumerate(angle_triplets):
                # +1 to restore 1-based indexing for GROMACS
                f.write(f"  {pi+1:<4d} {pj+1:<4d} {pk+1:<4d} 2       {t0_deg[idx]:>7.2f}        {ka[idx]:>7.2f}\n")

        f.write("\n")

    print(f"  [OK] Successfully written predicted GROMACS topology to: {output_path}")


def run_demo(model, device, snap_ff: bool = False):
    """Demonstrate inference on an unseen drug molecule: Coarse-Grained Ibuprofen (IBUP)."""
    print("\n" + "=" * 65)
    print("DEMO: INFERENCE ON NOVEL UNSEEN MOLECULE - COARSE-GRAINED IBUPROFEN")
    print("=" * 65)

    # MARTINI 3 representation of Ibuprofen:
    mol_name = "IBUP"
    beads = ["SC1", "TC5", "TC5", "Qa"]
    bonds = [(1, 2), (2, 3), (3, 4)]

    print(f"Molecule Name : {mol_name}")
    print(f"Beads ({len(beads)})    : {', '.join([f'B{i+1}:{b}' for i, b in enumerate(beads)])}")
    print(f"Bonds ({len(bonds)})    : {bonds}")

    data, bond_pairs, angle_triplets = build_graph_from_topology(mol_name, beads, bonds)
    print(f"Detected Angles ({len(angle_triplets)}) : {[[p[0]+1, p[1]+1, p[2]+1] for p in angle_triplets]}")

    print("\nRunning GNN forward pass ...")
    kb, r0, ka, t0_deg = predict_parameters(model, data, device, snap_ff=snap_ff)

    print("\nPredicted Bond Parameters:")
    for idx, (i, j) in enumerate(bonds):
        print(f"  Bond {i}-{j} ({beads[i-1]} - {beads[j-1]}):  r0 = {r0[idx]:.4f} nm ({r0[idx]*10:.2f} A)  |  k_bond = {kb[idx]:.1f} kJ/mol/nm^2")

    print("\nPredicted Angle Parameters:")
    for idx, (pi, pj, pk) in enumerate(angle_triplets):
        print(f"  Angle {pi+1}-{pj+1}-{pk+1} ({beads[pi]}-{beads[pj]}-{beads[pk]}):  theta0 = {t0_deg[idx]:.1f} deg  |  k_angle = {ka[idx]:.1f} kJ/mol/rad^2")

    os.makedirs("results", exist_ok=True)
    out_file = os.path.join("results", f"{mol_name}_predicted.itp")
    write_gromacs_itp(out_file, mol_name, beads, bonds, angle_triplets, kb, r0, ka, t0_deg)
    print("=" * 65 + "\n")


def main():
    p = argparse.ArgumentParser(description="Predict MARTINI 3 parameters using CGSpringGNN")
    p.add_argument("--demo", action="store_true", help="Run demonstration on novel test molecule")
    default_ckpt = "checkpoints/hybrid_model.pt" if os.path.exists("checkpoints/hybrid_model.pt") else "checkpoints/best_model.pt"
    p.add_argument("--snap_ff", action="store_true", help="Snap continuous predictions to canonical MARTINI FF bins")
    p.add_argument("--ckpt", default=default_ckpt, help=f"Model checkpoint path (default: {default_ckpt})")
    p.add_argument("--input", default=None, help="Input unparameterized .itp file")
    p.add_argument("--out", default="results/predicted.itp", help="Output .itp path")
    p.add_argument("--name", default="UNKN", help="Molecule name if using CLI beads/bonds")
    p.add_argument("--beads", nargs="+", help="Bead types (e.g. Q1 SC1 SC1 SP1)")
    p.add_argument("--bonds", default=None, help="Bond pairs as comma-separated pairs, e.g. '1-2,2-3,3-4'")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.ckpt):
        print(f"Error: checkpoint {args.ckpt} not found. Please run train.py or train_hybrid.py first.")
        sys.exit(1)

    model = load_model_checkpoint(args.ckpt, device=device)

    if args.demo or (args.input is None and args.beads is None):
        run_demo(model, device, snap_ff=args.snap_ff)
        return

    if args.beads and args.bonds:
        bond_pairs = [tuple(map(int, b.split("-"))) for b in args.bonds.split(",")]
        data, b_pairs, a_triplets = build_graph_from_topology(args.name, args.beads, bond_pairs)
        kb, r0, ka, t0_deg = predict_parameters(model, data, device, snap_ff=args.snap_ff)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        write_gromacs_itp(args.out, args.name, args.beads, b_pairs, a_triplets, kb, r0, ka, t0_deg)

    elif args.input:
        mols = parse_itp(args.input)
        if not mols:
            print(f"Error: no valid molecule found in {args.input}")
            sys.exit(1)
        mol = mols[0]
        beads = [a.type for a in mol.atoms]
        bonds = [(b.i, b.j) for b in mol.bonds]
        data, b_pairs, a_triplets = build_graph_from_topology(mol.name, beads, bonds)
        kb, r0, ka, t0_deg = predict_parameters(model, data, device, snap_ff=args.snap_ff)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        write_gromacs_itp(args.out, mol.name, beads, b_pairs, a_triplets, kb, r0, ka, t0_deg)


if __name__ == "__main__":
    main()
