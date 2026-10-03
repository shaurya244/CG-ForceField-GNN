#!/usr/bin/env python3
"""
Packaging script for EE798R Course Project Submission.
Generates the submission archive according to course guidelines:
Naming convention: <roll_number>_<name>.zip
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Package EE798R project submission zip")
    parser.add_argument("--roll", type=str, default="", help="Student Roll Number (e.g. 210999)")
    parser.add_argument("--name", type=str, default="Shaurya_Srivastava", help="Student Name (default: Shaurya_Srivastava)")
    args = parser.parse_args()

    if not args.roll:
        args.roll = input("Enter your Roll Number: ").strip()
        if not args.roll:
            args.roll = "ROLLNUMBER"

    report_dir = Path(__file__).resolve().parent
    repo_root = report_dir.parent
    zip_filename = f"{args.roll}_{args.name}.zip"
    zip_path = repo_root / zip_filename

    print(f"[*] Packaging submission to: {zip_path}")

    files_to_pack = [
        ("report/report_single_column.pdf", report_dir / "report_single_column.pdf"),
        ("report/report_single_column.tex", report_dir / "report_single_column.tex"),
        ("report/report.pdf", report_dir / "report.pdf"),
        ("report/report.tex", report_dir / "report.tex"),
        ("report/references.bib", report_dir / "references.bib"),
    ]

    # Include figures
    figures_dir = report_dir / "figures"
    for fig in figures_dir.glob("*.png"):
        files_to_pack.append((f"report/figures/{fig.name}", fig))

    # Include supplementary results
    results_dir = repo_root / "cg_spring_gnn" / "results"
    supplementary = [
        "hybrid_test_metrics.json",
        "benchmarking_matrices.json",
        "benchmarking_matrices.md",
        "all_evaluation_matrices.json",
        "model_comparison_p80.json",
        "boltzmann_overlap_results.json",
        "IBUP_predicted.itp",
    ]
    for sup in supplementary:
        sup_path = results_dir / sup
        if sup_path.exists():
            files_to_pack.append((f"supplementary/{sup}", sup_path))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, file_path in files_to_pack:
            if file_path.exists():
                print(f"  + Adding {arcname}")
                zf.write(file_path, arcname=arcname)
            else:
                print(f"  ! Missing {file_path}")

    print(f"[SUCCESS] Submission packaged: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
