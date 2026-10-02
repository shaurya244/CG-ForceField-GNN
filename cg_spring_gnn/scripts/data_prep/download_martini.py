"""
download_martini.py
-------------------
Downloads MARTINI 3 .itp files from cgmartini.nl and organises them
into data/raw/martini3/.

What gets downloaded:
  1. martini_v3.0.0.itp         — core bead type parameters
  2. martini_v3.0_lipids.itp    — lipid molecule parameters
  3. martini_v3.0_proteins.itp  — protein residue parameters
  4. A collection of ~150 individual small-molecule/lipid .itp files
     hosted on the cgmartini.nl GitHub repository.

Run:
    python download_martini.py
    python download_martini.py --out data/raw/martini3
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
import urllib.request
import zipfile
import tarfile
import shutil

# Official MARTINI 3 Lipid parameter files from Martini-Force-Field-Initiative
M3_LIPIDS_BASE = (
    "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/"
    "M3-Lipid-Parameters/main/ITPs/"
)

M3_LIPID_ITPS = [
    "martini_v3.0.0_phospholipids_PC_v2.itp",
    "martini_v3.0.0_phospholipids_PE_v2.itp",
    "martini_v3.0.0_phospholipids_PG_v2.itp",
    "martini_v3.0.0_phospholipids_PI_v2.itp",
    "martini_v3.0.0_phospholipids_PS_v2.itp",
    "martini_v3.0.0_phospholipids_SM_v2.itp",
    "martini_v3.0.0_phospholipids_CL_v2.itp",
    "martini_v3.0.0_ceramides_v2.itp",
    "martini_v3.0.0_fattyacids_v2.itp",
    "martini_v3.0.0_sterols_v1.itp",
    "martini_v3.0.0_triglycerides_v2.itp",
    "martini_v3.0.0_diglycerides_v2.itp",
    "martini_v3.0.0_monoglycerides_v2.itp",
    "martini_v3.0.0_DOTAP_v2.itp",
    "martini_v3.0.0_hydrocarbons_v2.itp",
]


# --- Download helper ----------------------------------------------------------

def download_file(url: str, dest_path: str, desc: str = ""):
    """Download a single file with progress indication."""
    desc = desc or os.path.basename(dest_path)
    if os.path.exists(dest_path):
        print(f"  [skip] {desc} (already exists)")
        return True
    try:
        print(f"  [download] {desc} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest_path)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"done  ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"FAILED - {e}")
        return False


# --- Main downloader ----------------------------------------------------------

def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nDownloading MARTINI 3 real data suite -> {out_dir}\n")

    downloaded = 0
    failed     = []

    # 1. Download official MARTINI 3 lipids & biomolecules
    print(f"Downloading official MARTINI 3 lipid library ...")
    for itp_name in M3_LIPID_ITPS:
        url  = M3_LIPIDS_BASE + itp_name
        dest = os.path.join(out_dir, itp_name)
        ok   = download_file(url, dest, desc=itp_name)
        if ok:
            downloaded += 1
        else:
            failed.append(itp_name)

    # 2. Download small molecules if not present
    sm_file = "martini_v3.0.0_small_molecules_v1.itp"
    sm_dest = os.path.join(out_dir, sm_file)
    if not os.path.exists(sm_dest):
        sm_url = (
            "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/"
            "M3-Small-Molecules/main/models/" + sm_file
        )
        ok = download_file(sm_url, sm_dest, desc=sm_file)
        if ok:
            downloaded += 1
        else:
            failed.append(sm_file)

    # --- Summary ----------------------------------------------------------
    n_itp = len([f for f in os.listdir(out_dir) if f.endswith(".itp") or f.endswith(".ff")])
    print(f"\n{'-'*50}")
    print(f"  Downloaded : {downloaded} files")
    print(f"  Topologies : {n_itp} in '{out_dir}'")
    if failed:
        print(f"  Failed     : {len(failed)} files")
        for f in failed:
            print(f"    - {f}")
    print(f"\nNext step:")
    print(f"  python src/data/parse_itp.py {out_dir}")
    print(f"  python train.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/martini3",
                   help="Output directory for downloaded .itp files")
    args = p.parse_args()
    main(args.out)
