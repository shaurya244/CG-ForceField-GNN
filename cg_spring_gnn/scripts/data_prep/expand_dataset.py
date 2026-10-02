"""
expand_dataset.py
-----------------
Scales up the MARTINI 3 coarse-grained dataset by downloading:
1. Official M3 Sugars & Nucleobases & Small Molecules v2 from marrink-lab
2. Official M3 Optimized Small Molecules (mono- and poly-cyclic) from MFFI
3. Official M3 Ionizable Lipids library (230+ lipids) from MFFI
4. Expanded synthetic realistic library (lipids, branched polymers, small molecules)

Run:
    python expand_dataset.py
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
import io
import zipfile
import urllib.request
import shutil

RAW_DIR = os.path.join("data", "raw", "martini3")

def download_url(url: str, dest_path: str, desc: str = ""):
    desc = desc or os.path.basename(dest_path)
    if os.path.exists(dest_path):
        print(f"  [skip] {desc} (already exists)")
        return True
    try:
        print(f"  [download] {desc} ...", end=" ", flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"done ({size_kb:.1f} KB)", flush=True)
        return True
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        return False


def download_and_extract_zip(url: str, extract_dir: str, pattern: str, desc: str = ""):
    print(f"  [downloading archive] {desc} ...", flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        print(f"  [extracting {desc}] ({len(data)/1024/1024:.2f} MB) ...", flush=True)
        zf = zipfile.ZipFile(io.BytesIO(data))
        matched = 0
        os.makedirs(extract_dir, exist_ok=True)
        for member in zf.namelist():
            if pattern in member and member.endswith(".itp"):
                fname = os.path.basename(member)
                if not fname:
                    continue
                dest = os.path.join(extract_dir, fname)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                matched += 1
        print(f"  Extracted {matched} .itp files to {extract_dir}", flush=True)
        return matched
    except Exception as e:
        print(f"FAILED to download/extract {desc}: {e}", flush=True)
        return 0


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("=" * 60)
    print("EXPANDING MARTINI 3 DATASET")
    print("=" * 60)

    # 1. Sugars, Nucleobases, Small Molecules v2 from marrink-lab
    print("\n1. Downloading marrink-lab topologies (sugars, nucleobases)...")
    marrink_base = (
        "https://raw.githubusercontent.com/marrink-lab/martini-forcefields/main/"
        "martini_forcefields/regular/v3.0.0/gmx_files/"
    )
    for itp in [
        "martini_v3.0.0_sugars_v2.itp",
        "martini_v3.0.0_nucleobases_v1.itp",
        "martini_v3.0.0_small_molecules_v2.itp",
    ]:
        url = marrink_base + itp
        dest = os.path.join(RAW_DIR, itp)
        download_url(url, dest, desc=itp)

    # 2. M3-Small-Molecules (opt-mono and opt-poly)
    print("\n2. Downloading M3-Small-Molecules optimized library...")
    sm_zip = "https://github.com/Martini-Force-Field-Initiative/M3-Small-Molecules/archive/refs/heads/main.zip"
    sm_dir = os.path.join(RAW_DIR, "small_molecules_opt")
    download_and_extract_zip(sm_zip, sm_dir, pattern="models/itps/opt-", desc="M3-Small-Molecules (opt)")

    # 3. M3_Ionizable_Lipids (Single_itps)
    print("\n3. Downloading M3_Ionizable_Lipids library...")
    lip_zip = "https://github.com/Martini-Force-Field-Initiative/M3_Ionizable_Lipids/archive/refs/heads/main.zip"
    lip_dir = os.path.join(RAW_DIR, "ionizable_lipids")
    download_and_extract_zip(lip_zip, lip_dir, pattern="Single_itps", desc="M3_Ionizable_Lipids")

    # 4. Generate additional realistic synthetic molecules
    print("\n4. Generating expanded realistic synthetic molecules (lipids, polymers, small molecules)...")
    from scripts.data_prep import make_sample_data
    make_sample_data.main(n_per_class=100, out_dir=RAW_DIR)

    # Count total itp/ff files
    all_files = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".itp") or f.endswith(".ff"):
                all_files.append(os.path.join(root, f))

    print("\n" + "=" * 60)
    print(f"DATASET EXPANSION COMPLETE")
    print(f"Total raw topology files in {RAW_DIR}: {len(all_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
