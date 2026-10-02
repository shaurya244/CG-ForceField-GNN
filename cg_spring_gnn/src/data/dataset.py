"""
dataset.py
----------
PyTorch Geometric Dataset class for CG spring constant prediction.

Workflow:
  1. Reads all .itp files from data/raw/martini3/
  2. Parses and featurizes each molecule (parse_itp + featurize)
  3. Splits into train / val / test  (80 / 10 / 10)
  4. Saves processed graphs to data/processed/  (avoids re-parsing)

Usage:
    from src.data.dataset import CGSpringDataset

    train = CGSpringDataset(root="data", split="train")
    val   = CGSpringDataset(root="data", split="val")
    test  = CGSpringDataset(root="data", split="test")

    print(train[0])  # Data(x=[N,36], edge_index=[2,2B], ...)
"""

import os
import json
import random
import torch
from torch_geometric.data import Dataset, Data
from typing import List, Optional

from src.data.parse_itp  import parse_all_martini
from src.data.featurize  import molecules_to_graphs


class CGData(Data):
    """PyG Data object that properly increments angle_idx and angle_edge_idx across batches."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == "angle_idx":
            return self.num_nodes
        if key == "angle_edge_idx":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


class CGSpringDataset(Dataset):
    """
    In-memory PyG Dataset for CG spring constants from MARTINI .itp files.
    Processed graphs are cached in {root}/processed/ for fast reload.
    """

    def __init__(self,
                 root:       str  = "data",
                 split:      str  = "train",   # "train" | "val" | "test"
                 raw_subdir: str  = "raw/martini3",
                 val_frac:   float = 0.10,
                 test_frac:  float = 0.10,
                 seed:       int   = 42,
                 transform         = None,
                 pre_transform     = None):

        self.split      = split
        self.raw_subdir = raw_subdir
        self.val_frac   = val_frac
        self.test_frac  = test_frac
        self.seed       = seed
        self._graphs: List[Data] = []

        # PyG Dataset.__init__ calls download() and process() if needed
        super().__init__(root, transform, pre_transform)

        # Load graphs for the requested split
        split_file = os.path.join(self.processed_dir, f"{split}_indices.json")
        all_graphs = torch.load(
            os.path.join(self.processed_dir, "all_graphs.pt"),
            weights_only=False
        )
        with open(split_file) as f:
            indices = json.load(f)
        self._graphs = [
            CGData(**all_graphs[i].to_dict()) if not isinstance(all_graphs[i], CGData) else all_graphs[i]
            for i in indices
        ]

    # ── PyG required properties ────────────────────────────────────────────

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_subdir)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        # PyG uses this to check if raw data exists; we just scan the folder
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["all_graphs.pt", "train_indices.json",
                "val_indices.json", "test_indices.json"]

    # ── Processing ────────────────────────────────────────────────────────

    def download(self):
        """Nothing to download automatically (user downloads MARTINI data)."""
        if not os.path.isdir(self.raw_dir) or not os.listdir(self.raw_dir):
            print(
                f"\n[!] No MARTINI data found in '{self.raw_dir}'.\n"
                f"    Please download MARTINI 3 .itp files:\n"
                f"    https://cgmartini.nl/images/martini3/martini_v3.0.0.tar.gz\n"
                f"    and extract them to:  {self.raw_dir}\n"
            )

    def process(self):
        """Parse .itp files -> featurize -> split -> save."""
        print(f"\n-- Processing MARTINI data from '{self.raw_dir}' --")

        # 1. Parse all ITP files
        molecules = parse_all_martini(self.raw_dir, verbose=True)
        if not molecules:
            raise RuntimeError(
                f"No molecules found in '{self.raw_dir}'. "
                f"Check that .itp files are present."
            )

        # 2. Featurize
        graphs = molecules_to_graphs(molecules, verbose=True)
        if not graphs:
            raise RuntimeError("Featurization produced no valid graphs.")

        # 3. Train / val / test split (stratified by molecule size is optional)
        random.seed(self.seed)
        indices = list(range(len(graphs)))
        random.shuffle(indices)

        n_test  = max(1, int(len(indices) * self.test_frac))
        n_val   = max(1, int(len(indices) * self.val_frac))
        n_train = len(indices) - n_val - n_test

        train_idx = indices[:n_train]
        val_idx   = indices[n_train : n_train + n_val]
        test_idx  = indices[n_train + n_val:]

        print(f"\nDataset split - train:{len(train_idx)}  "
              f"val:{len(val_idx)}  test:{len(test_idx)}")

        # 4. Save
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(graphs, os.path.join(self.processed_dir, "all_graphs.pt"))

        for name, idx_list in [("train", train_idx),
                                ("val",   val_idx),
                                ("test",  test_idx)]:
            with open(os.path.join(self.processed_dir, f"{name}_indices.json"), "w") as f:
                json.dump(idx_list, f)

        print("Saved processed data to", self.processed_dir)

    # ── Dataset interface ─────────────────────────────────────────────────

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        data = self._graphs[idx]
        if self.transform:
            data = self.transform(data)
        return data

    def __repr__(self) -> str:
        return (f"CGSpringDataset(split={self.split}, "
                f"n_molecules={len(self._graphs)})")


# ─── Dataset statistics helper ─────────────────────────────────────────────────

def print_dataset_stats(dataset: CGSpringDataset):
    """Print mean/std of spring constants across the dataset."""
    import numpy as np
    k_bonds, k_angles = [], []
    for data in dataset:
        if data.y_k_bond.numel() > 0:
            k_bonds.extend(data.y_k_bond.tolist())
        if data.y_k_angle.numel() > 0:
            k_angles.extend(data.y_k_angle.tolist())

    if k_bonds:
        print(f"  k_bond  — n={len(k_bonds):5d}  "
              f"mean={np.mean(k_bonds):.1f}  "
              f"std={np.std(k_bonds):.1f}  "
              f"min={np.min(k_bonds):.1f}  "
              f"max={np.max(k_bonds):.1f}  [kJ/mol/nm²]")
    if k_angles:
        print(f"  k_angle — n={len(k_angles):5d}  "
              f"mean={np.mean(k_angles):.1f}  "
              f"std={np.std(k_angles):.1f}  "
              f"min={np.min(k_angles):.1f}  "
              f"max={np.max(k_angles):.1f}  [kJ/mol/rad²]")


if __name__ == "__main__":
    ds = CGSpringDataset(root="data", split="train")
    print(ds)
    print_dataset_stats(ds)
    print("\nFirst graph:", ds[0])
