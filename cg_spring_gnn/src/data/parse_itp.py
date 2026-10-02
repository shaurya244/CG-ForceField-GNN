"""
parse_itp.py
------------
Parses GROMACS/MARTINI .itp files to extract:
  - Molecule name & bead atoms (type, mass, charge)
  - Bond spring constants  : k_bond [kJ/mol/nm²], r0 [nm]
  - Angle spring constants : k_angle [kJ/mol/rad²], theta0 [deg]
  - Dihedral parameters    : k_dihedral, phi0, mult

Usage:
    from src.data.parse_itp import parse_itp, parse_all_martini
    mols = parse_all_martini("data/raw/martini3/")
"""

import re
import os
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class CGAtom:
    idx:    int
    type:   str    # MARTINI bead type, e.g. "C1", "P5", "Qd"
    resid:  int
    resname: str
    name:   str    # atom name within residue
    charge: float
    mass:   float


@dataclass
class CGBond:
    i:    int      # atom index (1-based as in ITP)
    j:    int
    func: int      # 1 = harmonic
    r0:   float    # equilibrium length [nm]
    k:    float    # force constant [kJ/mol/nm²]
    atom_type_i: str = ""
    atom_type_j: str = ""


@dataclass
class CGAngle:
    i:    int
    j:    int
    k:    int
    func: int      # 2 = cosine harmonic (MARTINI default); 1 = harmonic
    theta0: float  # equilibrium angle [degrees]
    k_ang:  float  # force constant [kJ/mol/rad²]
    atom_type_i: str = ""
    atom_type_j: str = ""
    atom_type_k: str = ""


@dataclass
class CGDihedral:
    i:    int
    j:    int
    k:    int
    l:    int
    func: int
    phi0: float    # equilibrium dihedral [degrees]
    k_dih: float   # force constant [kJ/mol]
    mult:  int = 1 # multiplicity


@dataclass
class CGMolecule:
    name:      str
    atoms:     List[CGAtom]    = field(default_factory=list)
    bonds:     List[CGBond]    = field(default_factory=list)
    angles:    List[CGAngle]   = field(default_factory=list)
    dihedrals: List[CGDihedral]= field(default_factory=list)
    source_file: str = ""

    @property
    def n_atoms(self): return len(self.atoms)
    @property
    def n_bonds(self): return len(self.bonds)
    @property
    def n_angles(self): return len(self.angles)


# Pre-defined named bond and angle types from MARTINI 3 parameter suites
NAMED_BONDS = {
    "b_C1_C1": (1, 0.47, 3800.0),
    "b_C1_C4": (1, 0.47, 3800.0),
    "b_C4_C1": (1, 0.47, 3800.0),
    "b_SN4a_C1": (1, 0.47, 5000.0),
    "b_C4_C4": (1, 0.47, 3800.0),
    "b_ET_C1_glyc": (1, 0.37, 5000.0),
    "b_SC1_C1": (1, 0.37, 2500.0),
}

NAMED_ANGLES = {
    "a_SN4a_C1_C1": (2, 180.0, 35.0),
    "a_SN4a_C4_C4": (2, 100.0, 10.0),
    "a_SN4a_C1_C4": (2, 180.0, 35.0),
    "a_C1_C4_C1": (2, 120.0, 35.0),
    "a_C1_C1_C4": (2, 180.0, 35.0),
    "a_C4_C1_C1": (2, 180.0, 35.0),
    "a_C1_C4_C4": (2, 100.0, 10.0),
    "a_C4_C4_C1": (2, 100.0, 10.0),
    "a_C4_C4_C4": (2, 100.0, 10.0),
    "a_C1_C1_C1": (2, 180.0, 35.0),
    "a_ET_C1_C4_glyc": (2, 150.0, 25.0),
    "a_ET_C1_C1_glyc": (2, 150.0, 25.0),
}


# --- Core Parser ---------------------------------------------------------------

def parse_itp(filepath: str) -> List[CGMolecule]:
    """
    Parse a GROMACS/MARTINI .itp or .ff file.

    Handles files with single OR MULTIPLE [ moleculetype ] definitions.
    Sections handled:
      [ moleculetype ], [ atoms ], [ bonds ], [ constraints ], [ angles ], [ dihedrals ]

    Returns a list of CGMolecule objects that have at least one bond.
    """
    molecules: List[CGMolecule] = []
    current_mol: Optional[CGMolecule] = None
    atom_map: Dict[int, CGAtom] = {}
    bond_pairs_seen: set = set()
    angle_triplets_seen: set = set()
    section = None

    def _flush_mol():
        nonlocal current_mol, atom_map, bond_pairs_seen, angle_triplets_seen
        if current_mol is not None and current_mol.n_bonds > 0:
            molecules.append(current_mol)
        current_mol = None
        atom_map = {}
        bond_pairs_seen = set()
        angle_triplets_seen = set()

    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                # Strip comments and whitespace
                line = raw_line.split(";")[0].strip()
                if not line:
                    continue

                # Parse #define if present
                if line.startswith("#define"):
                    tokens = line.split()
                    if len(tokens) >= 5 and tokens[1].startswith("b_"):
                        try:
                            NAMED_BONDS[tokens[1]] = (int(tokens[2]), float(tokens[3]), float(tokens[4]))
                        except ValueError:
                            pass
                    elif len(tokens) >= 5 and tokens[1].startswith("a_"):
                        try:
                            NAMED_ANGLES[tokens[1]] = (int(tokens[2]), float(tokens[3]), float(tokens[4]))
                        except ValueError:
                            pass
                    continue

                if line.startswith("#"):
                    continue

                # Detect section header
                m = re.match(r"^\[\s*(\w+)\s*\]", line)
                if m:
                    sec_name = m.group(1).lower()
                    if sec_name == "moleculetype":
                        _flush_mol()
                    section = sec_name
                    continue

                parts = line.split()

                # --- [ moleculetype ] -----------------------------------------
                if section == "moleculetype":
                    if current_mol is None:
                        current_mol = CGMolecule(name=parts[0], source_file=os.path.basename(filepath))
                    else:
                        current_mol.name = parts[0]

                # --- [ atoms ] ------------------------------------------------
                elif section == "atoms" and len(parts) >= 7:
                    if current_mol is None:
                        current_mol = CGMolecule(name="UNKNOWN", source_file=os.path.basename(filepath))
                    try:
                        idx     = int(parts[0])
                        btype   = parts[1]
                        resid   = int(parts[2])
                        resname = parts[3]
                        bname   = parts[4]
                        charge  = float(parts[6])
                        mass    = float(parts[7]) if len(parts) > 7 else _default_mass(btype)
                        atom    = CGAtom(idx, btype, resid, resname, bname, charge, mass)
                        current_mol.atoms.append(atom)
                        atom_map[idx] = atom
                    except (ValueError, IndexError):
                        pass

                # --- [ bonds ] or [ constraints ] -----------------------------
                elif section in ("bonds", "constraints"):
                    if current_mol is None:
                        continue
                    try:
                        if len(parts) >= 3 and parts[2] in NAMED_BONDS:
                            i, j = int(parts[0]), int(parts[1])
                            func, r0, k = NAMED_BONDS[parts[2]]
                        elif len(parts) >= 4:
                            i, j, func = int(parts[0]), int(parts[1]), int(parts[2])
                            r0 = float(parts[3])   # nm
                            if len(parts) >= 5:
                                k = float(parts[4])   # kJ/mol/nm^2
                            elif section == "constraints":
                                k = 50000.0          # Rigid constraint nominal spring constant
                            else:
                                continue
                        else:
                            continue

                        # Deduplicate: if this atom pair was already parsed (e.g. #ifdef FLEXIBLE vs [ constraints ]),
                        # retain the explicit [ bonds ] definition and skip the conflicting constraint.
                        pair = tuple(sorted((i, j)))
                        if pair in bond_pairs_seen:
                            continue
                        bond_pairs_seen.add(pair)

                        # Cap numerical pseudo-infinite constraints at standard MARTINI max stiff bond (50,000)
                        if k > 50000.0:
                            k = 50000.0

                        if k <= 0:
                            continue

                        bond = CGBond(
                            i=i, j=j, func=func, r0=r0, k=k,
                            atom_type_i=atom_map.get(i, CGAtom(0,"?",0,"?","?",0,72)).type,
                            atom_type_j=atom_map.get(j, CGAtom(0,"?",0,"?","?",0,72)).type,
                        )
                        current_mol.bonds.append(bond)
                    except (ValueError, IndexError):
                        pass

                # --- [ angles ] -----------------------------------------------
                elif section == "angles":
                    if current_mol is None:
                        continue
                    try:
                        if len(parts) >= 4 and parts[3] in NAMED_ANGLES:
                            i, j, k_ = int(parts[0]), int(parts[1]), int(parts[2])
                            func, theta0, k_ang = NAMED_ANGLES[parts[3]]
                        elif len(parts) >= 6:
                            i, j, k_, func = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                            theta0 = float(parts[4])  # degrees
                            k_ang  = float(parts[5])  # kJ/mol/rad^2
                        else:
                            continue

                        triplet = (j, min(i, k_), max(i, k_))
                        if triplet in angle_triplets_seen:
                            continue
                        angle_triplets_seen.add(triplet)

                        if k_ang <= 0:
                            continue
                        if k_ang > 1000.0:
                            k_ang = 1000.0
                        angle = CGAngle(
                            i=i, j=j, k=k_, func=func,
                            theta0=theta0, k_ang=k_ang,
                            atom_type_i=atom_map.get(i, CGAtom(0,"?",0,"?","?",0,72)).type,
                            atom_type_j=atom_map.get(j, CGAtom(0,"?",0,"?","?",0,72)).type,
                            atom_type_k=atom_map.get(k_, CGAtom(0,"?",0,"?","?",0,72)).type,
                        )
                        current_mol.angles.append(angle)
                    except (ValueError, IndexError):
                        pass

                # --- [ dihedrals ] --------------------------------------------
                elif section == "dihedrals" and len(parts) >= 7:
                    if current_mol is None:
                        continue
                    try:
                        i, j, k_, l = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        func  = int(parts[4])
                        phi0  = float(parts[5])
                        k_dih = float(parts[6])
                        mult  = int(parts[7]) if len(parts) > 7 else 1
                        current_mol.dihedrals.append(
                            CGDihedral(i, j, k_, l, func, phi0, k_dih, mult)
                        )
                    except (ValueError, IndexError):
                        pass

        _flush_mol()

    except (OSError, UnicodeDecodeError) as e:
        print(f"  [WARN] Could not read {filepath}: {e}")

    return molecules


def _default_mass(btype: str) -> float:
    """Return default mass for a MARTINI bead type."""
    if btype.startswith("T"):   return 36.0   # Tiny bead
    if btype.startswith("S"):   return 54.0   # Small bead
    return 72.0                                # Regular bead


# --- Batch Parser -------------------------------------------------------------

def parse_all_martini(data_dir: str, verbose: bool = True) -> List[CGMolecule]:
    """
    Recursively parse all .itp and .ff files under data_dir.

    Returns a list of clean CGMolecule objects that have at least one bond.
    """
    pattern_itp = glob.glob(os.path.join(data_dir, "**", "*.itp"), recursive=True)
    pattern_ff  = glob.glob(os.path.join(data_dir, "**", "*.ff"),  recursive=True)
    all_files   = sorted(set(pattern_itp + pattern_ff))

    if verbose:
        print(f"Found {len(all_files)} topology files in '{data_dir}'")

    molecules = []
    skipped_files = 0
    for fp in all_files:
        mols = parse_itp(fp)
        if mols:
            for mol in mols:
                molecules.append(mol)
                if verbose and len(molecules) <= 25:
                    print(f"  [OK] {mol.name:25s}  "
                          f"{mol.n_atoms:3d} beads  "
                          f"{mol.n_bonds:3d} bonds  "
                          f"{mol.n_angles:3d} angles  "
                          f"[{os.path.basename(fp)}]")
            if verbose and len(molecules) > 25 and len(mols) > 0:
                pass  # Avoid flooding console with hundreds of lines
        else:
            skipped_files += 1

    if verbose:
        print(f"\nParsed {len(molecules)} molecules  |  Skipped {skipped_files} non-molecule files")
    return molecules


# ─── Quick stats helper ────────────────────────────────────────────────────────

def dataset_stats(molecules: List[CGMolecule]) -> Dict:
    """Print summary statistics of the parsed dataset."""
    import numpy as np
    k_bonds  = [b.k       for m in molecules for b in m.bonds]
    r0_bonds = [b.r0      for m in molecules for b in m.bonds]
    k_angles = [a.k_ang   for m in molecules for a in m.angles]
    t0_angles= [a.theta0  for m in molecules for a in m.angles]

    stats = {
        "n_molecules":  len(molecules),
        "n_bonds":      len(k_bonds),
        "n_angles":     len(k_angles),
        "k_bond_mean":  float(np.mean(k_bonds))  if k_bonds  else 0,
        "k_bond_std":   float(np.std(k_bonds))   if k_bonds  else 0,
        "k_bond_min":   float(np.min(k_bonds))   if k_bonds  else 0,
        "k_bond_max":   float(np.max(k_bonds))   if k_bonds  else 0,
        "k_angle_mean": float(np.mean(k_angles)) if k_angles else 0,
        "k_angle_std":  float(np.std(k_angles))  if k_angles else 0,
    }
    print("\n-- Dataset Statistics ------------------------------")
    for k, v in stats.items():
        print(f"  {k:20s}: {v:.2f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    return stats


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/martini3"
    mols = parse_all_martini(data_dir)
    dataset_stats(mols)
