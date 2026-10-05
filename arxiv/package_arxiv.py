#!/usr/bin/env python3
"""
Official arXiv Submission Packager
Complies with arXiv LaTeX submission guidelines:
- Bundles main.tex, main.bbl, references.bib, and all referenced figures.
- Excludes output PDFs, logs, intermediate files, and scripts.
- Produces both .tar.gz (arXiv preferred format) and .zip packages.
"""

import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path

def main():
    arxiv_dir = Path(__file__).resolve().parent
    repo_root = arxiv_dir.parent

    print("=" * 65)
    print(" arXiv Submission Bundle Generator")
    print("=" * 65)

    main_tex = arxiv_dir / "main.tex"
    main_bbl = arxiv_dir / "main.bbl"
    ref_bib = arxiv_dir / "references.bib"
    figures_dir = arxiv_dir / "figures"

    # Verification
    if not main_tex.exists():
        print(f"[ERROR] Missing {main_tex}")
        sys.exit(1)

    if not main_bbl.exists():
        print(f"[WARNING] main.bbl not found. Running compilation to generate it...")
        tectonic_exe = repo_root / "bin" / "tectonic.exe"
        if tectonic_exe.exists():
            os.system(f'"{tectonic_exe}" --keep-intermediates "{main_tex}"')
        else:
            print("[ERROR] Cannot find tectonic.exe to generate main.bbl")
            sys.exit(1)

    # Read main.tex and find all included figures
    with open(main_tex, "r", encoding="utf-8") as f:
        tex_content = f.read()

    # Check for absolute paths (excluding http/https URLs)
    clean_tex = re.sub(r'https?://\S+', '', tex_content)
    if re.search(r'\b[A-Za-z]:[\\/]|/(?:home|Users|tmp|var)/', clean_tex):
        print("[WARNING] Found potential absolute paths in main.tex! arXiv requires relative paths.")
    else:
        print("[*] Relative path check passed (no absolute paths found).")

    # Find referenced figures
    fig_matches = re.findall(r'\\includegraphics(?:\[.*?\])?\{(.*?)\}', tex_content)
    print(f"[*] Found {len(fig_matches)} figure inclusions in main.tex")

    missing_figs = []
    figures_to_include = set()
    for fig_path in fig_matches:
        # Normalize path
        norm_path = Path(fig_path)
        if not norm_path.suffix:
            # Check with .png, .pdf, .jpg
            for ext in [".png", ".pdf", ".jpg"]:
                cand = arxiv_dir / f"{fig_path}{ext}"
                if cand.exists():
                    norm_path = Path(f"{fig_path}{ext}")
                    break
        full_fig_path = arxiv_dir / norm_path
        if not full_fig_path.exists():
            missing_figs.append(fig_path)
        else:
            figures_to_include.add((norm_path.as_posix(), full_fig_path))

    if missing_figs:
        print(f"[ERROR] Missing referenced figures: {missing_figs}")
        sys.exit(1)

    # Items to package
    package_items = [
        ("main.tex", main_tex),
        ("main.bbl", main_bbl),
        ("references.bib", ref_bib),
    ]

    for rel_path, full_path in sorted(figures_to_include):
        package_items.append((rel_path, full_path))

    print(f"[*] Packaging {len(package_items)} items for arXiv:")
    total_bytes = 0
    for arcname, full_path in package_items:
        sz = full_path.stat().st_size
        total_bytes += sz
        print(f"  + {arcname:<35} ({sz / 1024:.1f} KB)")

    # 1. Create .tar.gz (arXiv standard)
    tar_path = arxiv_dir / "arxiv_submission.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for arcname, full_path in package_items:
            tar.add(full_path, arcname=arcname)
    print(f"\n[SUCCESS] Created {tar_path.name} ({tar_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Create .zip (alternative format)
    zip_path = arxiv_dir / "arxiv_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, full_path in package_items:
            zf.write(full_path, arcname=arcname)
    print(f"[SUCCESS] Created {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

    print("\n" + "=" * 65)
    print(" arXiv Upload Instructions")
    print("=" * 65)
    print("1. Go to https://arxiv.org/submit")
    print("2. Select Primary Subject: cs.LG (Machine Learning) or q-bio.BM (Biomolecules)")
    print("   Secondary Subjects: physics.chem-ph (Chemical Physics), cs.AI (Artificial Intelligence)")
    print("3. Upload 'arxiv_submission.tar.gz' (or 'arxiv_submission.zip')")
    print("4. arXiv's AutoTeX compiler will detect main.tex and use main.bbl automatically.")
    print("5. Preview the AutoTeX-generated PDF on arXiv before final sign-off.")
    print("=" * 65)

if __name__ == "__main__":
    main()
