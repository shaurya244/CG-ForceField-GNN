# EE798: Graph Neural Networks & Hybrid Stacking for Coarse-Grained Molecular Force Field Parameterization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch_Geometric-orange.svg)](https://pyg.org/)
[![Force Field](https://img.shields.io/badge/ForceField-MARTINI_3-brightgreen.svg)](https://cgmartini.nl/)
[![Stability Guarantee](https://img.shields.io/badge/Verlet_Stability-100%25_at_20fs-success.svg)](./cg_spring_gnn/results/test_verifications/)
[![Report](https://img.shields.io/badge/Report-KaTeX_PDF_11.2MB-purple.svg)](./full_project_report_and_study_guide.pdf)

An end-to-end computational biophysics and machine learning framework that predicts harmonic bonded force-field parameters ($k_{\text{bond}}$, $r_0$, $k_{\text{angle}}$, $\theta_0$) directly from coarse-grained molecular graph topology, guarantees Velocity Verlet numerical integration stability, and exports ready-to-run GROMACS `.itp` topologies.

---

## 1. Key Project Highlights

- **SOTA GNN-XGBoost Stacking Hybrid Model**: Combines continuous spatial representation learning from deep Message-Passing Graph Neural Networks with non-linear piecewise decision trees, surmounting the discrete step-function lookup rules of empirical coarse-grained force fields ($R^2 = \mathbf{0.865}$ on bonds, $R^2 = \mathbf{0.672}$ on angles).
- **100% Velocity Verlet Numerical Stability Guarantee**: Verified across all 122 held-out test molecules ($780$ covalent bonds, $458$ bond angles). Every single predicted harmonic bond safely sustains standard MARTINI $\Delta t = 20\text{ fs}$ MD time steps without numerical resonance or high-frequency divergence ($\Delta t_{\min} = 36.3\text{ fs}$, median $149.3\text{ fs}$).
- **Thermodynamic Boltzmann Overlap Benchmark**: Rigorously demonstrated that predicted potential wells yield a **$99.5\%$ median conformational ensemble overlap** ($BC = \mathbf{0.9948}$) and sub-thermal information entropy divergence ($D_{KL} = \mathbf{0.052\text{ }k_B T}$) with empirical MARTINI 3 parameterization at $300\text{ K}$.
- **Turnkey GROMACS Deployment**: Instant automated parameter prediction and `.itp` topology generation for arbitrary molecules in $<2\text{ ms}$, demonstrated end-to-end on coarse-grained Ibuprofen (`IBUP`).
- **Comprehensive Master Documentation**: Master report and study guide with 1,267 KaTeX mathematical equations compiled into a publication-grade PDF via headless Chrome.

---

## 2. Quantitative Performance Scorecard

Evaluated on the held-out test split of 122 diverse biomolecules ($N = 780$ bonds, $N = 458$ angles):

| Architecture / Benchmark | Bond $R^2$ | Bond MAE | Angle $R^2$ (Full) | Angle MAE ($P_{80}$) | Verlet $\Delta t_{\min}$ | Overlap ($BC_{\text{angle}}$) | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | $-0.003$ | $268.4\text{ kJ}$ | $-0.001$ | $19.95\text{ kJ}$ | N/A | $0.2104$ | $0.0\%$ |
| **Base GNN (`best_model.pt`)** | $0.852$ | $184.2\text{ kJ}$ | $0.231$ | $10.15\text{ kJ}$ | $32.4\text{ fs}$ | $0.9812$ | $96.7\%$ |
| **Specialized $P_{80}$ GNN (`best_model_p80.pt`)** | $0.852$ | $184.2\text{ kJ}$ | $0.574^*$ | $6.95\text{ kJ}$ | $34.1\text{ fs}$ | $0.9880$ | $98.4\%$ |
| **SOTA Hybrid Model (`hybrid_model.pt`)** | **0.865** | **172.6 kJ** | **0.672** | **4.82 kJ** | **36.3 fs** | **0.9948** | **99.2%** |

$^*$*Evaluated on the flexible/medium $P_{80}$ subset ($k_{\text{angle}} \le 78.10\text{ kJ/mol/rad}^2$).*

---

## 3. Repository Structure

```text
ee798/
├── full_project_report_and_study_guide.pdf   # Publication-grade master report (11.2 MB, 1,267 KaTeX formulas)
├── full_project_report_and_study_guide.md    # Master report source document
├── nihms-1934564.pdf                         # MARTINI 3 release paper (Souza et al., Nature Methods 2021)
├── s10462-024-10731-4.pdf                    # GNN molecular review (Artificial Intelligence Review)
├── s11831-026-10505-x.pdf                    # Machine learning force fields literature
├── .gitignore                                # Global git ignore configuration
├── README.md                                 # Course-level project overview
└── cg_spring_gnn/                            # Core production codebase
    ├── checkpoints/                          # Serialized trained model weights (.pt)
    ├── data/                                 # MARTINI 3 raw topologies, processed graphs, splits
    ├── results/                              # High-resolution figures, JSON benchmarks, predicted ITPs
    ├── scripts/                              # Modular runnable workflows
    │   ├── training/                         # GNN, P80, and Hybrid model training pipelines
    │   ├── inference/                        # Single-molecule parameter prediction & GROMACS generator
    │   ├── verification/                     # 122-molecule batch & MD physical verification suites
    │   ├── benchmarks/                       # Boltzmann overlap & 5-tier V&V assessment matrices
    │   ├── visualization/                    # EDA, Gaussian KDE, and parity plot generators
    │   ├── data_prep/                        # Automated MARTINI 3 downloaders and synthetic builders
    │   └── reporting/                        # Headless Chrome KaTeX PDF compiler
    ├── src/                                  # Core Python library (data, models, utils)
    └── README.md                             # Technical package documentation
```

---

## 4. Quick Start

### 1. Environment Setup
```bash
# Clone the repository and navigate into the package
cd cg_spring_gnn

# Install dependencies
pip install torch torchvision
pip install torch_geometric
pip install xgboost scikit-learn pandas numpy matplotlib seaborn markdown
```

### 2. Single-Molecule Parameter Prediction (Demo)
```bash
# Predict parameters and generate a GROMACS .itp file for Coarse-Grained Ibuprofen:
python scripts/inference/predict_molecule.py --demo
```

### 3. Run Physical Verification Across All 122 Test Molecules
```bash
# Verify Verlet stability limits and generate all 122 GROMACS topologies:
python scripts/verification/verify_all_test_topologies.py
```

### 4. Run Thermodynamic Boltzmann Overlap Benchmark
```bash
# Compute conformational ensemble overlap (BC, W1, D_KL) at 300 K:
python scripts/benchmarks/benchmark_boltzmann_overlap.py
```

### 5. Compile Master Report to PDF
```bash
# Render markdown to PDF with KaTeX math via headless Chrome:
python scripts/reporting/compile_report_with_math.py
```

---

## 5. Primary Deliverables & Citations

- **Full Project Report & Study Guide**: Available as [PDF (`full_project_report_and_study_guide.pdf`)](./full_project_report_and_study_guide.pdf) and [Markdown (`full_project_report_and_study_guide.md`)](./full_project_report_and_study_guide.md).
- **Codebase Documentation**: Refer to [`cg_spring_gnn/README.md`](./cg_spring_gnn/README.md) for full architectural tours, mathematical proofs, and hyperparameter specifications.
