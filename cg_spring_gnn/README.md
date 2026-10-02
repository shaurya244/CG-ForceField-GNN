# Coarse-Grained (CG) Molecular Force Field Prediction with Graph Neural Networks & Hybrid Stacking

Predicts harmonic bond and angle spring constants (`k_bond`, `k_angle`, `r0`, `theta0`) for Coarse-Grained (CG) Molecular Dynamics models using Graph Neural Networks and Gradient-Boosted Hybrid Stacking trained on the MARTINI 3 force field.

---

## 1. Cleaned Repository Architecture

```text
ee798/
├── full_project_report_and_study_guide.md    # Master report & study guide with KaTeX math
├── full_project_report_and_study_guide.pdf   # Publication-quality compiled PDF (11.2 MB)
├── nihms-1934564.pdf                         # MARTINI 3 parameterization paper
├── s10462-024-10731-4.pdf                    # GNN molecular review paper
├── s11831-026-10505-x.pdf                    # Force field learning literature
├── .gitignore                                # Global git ignore configuration
├── README.md                                 # Workspace-level course overview
└── cg_spring_gnn/
    ├── .gitignore                            # Package-level git ignore
    ├── checkpoints/                          # Trained model weights
    │   ├── best_model.pt                     # Base CGSpringGNN (Hidden: 128, Layers: 3)
    │   ├── best_model_p80.pt                 # P80-tailored GNN (k_angle <= 78.10)
    │   └── hybrid_model.pt                   # SOTA Hybrid Stacking Regressor
    ├── data/
    │   ├── raw/martini3/                     # MARTINI 3 topology files (.itp)
    │   ├── processed/                        # PyG preprocessed molecular graphs
    │   └── splits/                           # Train (487), Val (122), Test (122) splits
    ├── results/
    │   ├── benchmarking_matrices.png         # 4-panel evolutionary & V&V scorecard
    │   ├── boltzmann_overlap_benchmark.png   # Thermodynamic free energy overlap
    │   ├── hybrid_model_performance.png      # Parity & residual diagnostics
    │   ├── test_verifications/               # 122 GROMACS topologies & stability tests
    │   └── verification/                     # Ibuprofen (IBUP) MD simulation validation
    ├── scripts/                              # Categorized execution workflows
    │   ├── training/                         # Model training & baselines
    │   │   ├── train.py                      # Base GNN training pipeline
    │   │   ├── train_p80.py                  # Specialized P80 model training
    │   │   ├── train_hybrid.py               # SOTA Hybrid model training
    │   │   └── baseline.py                   # Baseline benchmarks (Mean, Ridge, RF)
    │   ├── inference/                        # Topological prediction & ITP generation
    │   │   └── predict_molecule.py           # End-to-end inference & GROMACS generator
    │   ├── verification/                     # Physical & Verlet MD verification suites
    │   │   ├── verify_molecule_topology.py   # Single-molecule verification & MD kit
    │   │   └── verify_all_test_topologies.py # 122-molecule batch verification engine
    │   ├── benchmarks/                       # Statistical mechanics & metric benchmarks
    │   │   ├── benchmark_boltzmann_overlap.py# Boltzmann distribution & KL benchmark
    │   │   ├── generate_benchmarking_matrices.py # 5-tier V&V assessment matrices
    │   │   ├── compute_all_metrics.py        # Metric computation across all models
    │   │   └── evaluate_both_models.py       # Comparative evaluation runner
    │   ├── visualization/                    # Publication figure generators
    │   │   ├── generate_eda_visuals.py       # Exploratory data analysis visualization
    │   │   ├── generate_p80_plot.py          # P80 filtering diagnostic plots
    │   │   ├── plot_gaussian_k.py            # Gaussian kernel density plots
    │   │   ├── plot_gnn_vs_hybrid.py         # Comparative architectural diagnostics
    │   │   ├── plot_p80_parity.py            # Parity plots for P80 models
    │   │   └── plot_results.py               # Training loss & scatter plots
    │   ├── data_prep/                        # Data acquisition & topology synthesis
    │   │   ├── download_martini.py           # Automated MARTINI 3 downloader
    │   │   ├── expand_dataset.py             # Dataset expansion utility
    │   │   └── make_sample_data.py           # Fallback synthetic topology generator
    │   └── reporting/                        # Publication report compiler
    │       └── compile_report_with_math.py   # KaTeX HTML & Chrome PDF compiler
    ├── src/                                  # Core Python library
    │   ├── data/ (dataset.py, featurize.py, parse_itp.py)
    │   ├── models/ (gnn.py, hybrid.py, loss.py)
    │   └── utils/ (metrics.py)
    └── README.md                             # Technical documentation
```

---

## 2. Key Models & Performance Summary

| Architecture | Bond $R^2$ | Bond MAE | Angle $R^2$ (Full) | Angle MAE ($P_{80}$) | Verlet $\Delta t_{\min}$ | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | $-0.003$ | $268.4\text{ kJ}$ | $-0.001$ | $19.95\text{ kJ}$ | N/A | Fail |
| **Base GNN (`best_model.pt`)** | $0.852$ | $184.2\text{ kJ}$ | $0.231$ | $10.15\text{ kJ}$ | $32.4\text{ fs}$ | $96.7\%$ |
| **Specialized $P_{80}$ GNN (`best_model_p80.pt`)** | $0.852$ | $184.2\text{ kJ}$ | $0.574^*$ | $6.95\text{ kJ}$ | $34.1\text{ fs}$ | $98.4\%$ |
| **SOTA Hybrid Model (`hybrid_model.pt`)** | **0.865** | **172.6 kJ** | **0.672** | **4.82 kJ** | **36.3 fs** | **99.2%** |

$^*$*Evaluated on the $P_{80}$ subset ($k_{\text{angle}} \le 78.10\text{ kJ/mol/rad}^2$).*

---

## 3. Installation & Dependencies

Requires **Python 3.10+** and a CUDA-enabled GPU (CPU inference is fully supported):

```bash
# 1. Install PyTorch (select appropriate CUDA version or CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install PyTorch Geometric
pip install torch_geometric

# 3. Install core scientific & tree boosting libraries
pip install xgboost scikit-learn numpy pandas scipy matplotlib seaborn markdown
```

---

## 4. Quick Start & Execution

All scripts are self-contained and execute cleanly from any directory:

### 1. Training the Models
```bash
# Train the primary Message-Passing GNN:
python scripts/training/train.py --epochs 120 --hidden 128 --layers 3

# Train the specialized P80 model (fine-tunes on k_angle <= 78.10):
python scripts/training/train_p80.py

# Train the state-of-the-art Hybrid Stacking Regressor:
python scripts/training/train_hybrid.py
```

### 2. End-to-End Molecular Inference & GROMACS ITP Generation
```bash
# Run demonstration on novel test molecule (e.g. Ibuprofen):
python scripts/inference/predict_molecule.py --demo

# Predict for any custom bead sequence and export ready-to-run .itp:
python scripts/inference/predict_molecule.py --name MY_LIPID --beads Q1 SC1 SC1 SP1 --bonds "1-2,2-3,3-4" --out results/MY_LIPID.itp
```

### 3. Comprehensive Model Verification (122 Test Molecules)
```bash
# Verify physical bounds, Verlet stability, and generate GROMACS ITP files:
python scripts/verification/verify_all_test_topologies.py
```

### 4. Thermodynamic Boltzmann Overlap Benchmark
```bash
# Compute Bhattacharyya Overlap (BC), Wasserstein-1 (W1), and KL-Divergence:
python scripts/benchmarks/benchmark_boltzmann_overlap.py
```

### 5. Multi-Tier V&V Benchmarking Matrices
```bash
# Generate the publication-grade 4-panel evolutionary & V&V matrix:
python scripts/benchmarks/generate_benchmarking_matrices.py
```

### 6. Compiling the KaTeX Master Report to PDF
```bash
# Compile markdown to publication-grade PDF via headless Chrome:
python scripts/reporting/compile_report_with_math.py
```

---

## 5. Physical Units & Force Field Reference

| Parameter | Symbol | Canonical Units | MARTINI 3 Typical Range | Physical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Bond Spring Constant** | $k_{\text{bond}}$ | $\text{kJ/mol/nm}^2$ | $1{,}000 - 50{,}000$ | Harmonic vibration frequency ($\omega = \sqrt{k/\mu}$) |
| **Equilibrium Bond Length** | $r_0$ | $\text{nm}$ | $0.20 - 0.70$ | Mean bead-bead distance ($1\text{ nm} = 10\text{ \AA}$) |
| **Angle Spring Constant** | $k_{\text{angle}}$ | $\text{kJ/mol/rad}^2$ | $5 - 1{,}000$ | Conformational rigidity ($\sigma_\theta = \sqrt{k_B T / k}$) |
| **Equilibrium Angle** | $\theta_0$ | degrees / radians | $40^\circ - 180^\circ$ | Triplet geometric orientation |

### Velocity Verlet Stability Criterion:
To prevent numerical resonance under the Velocity Verlet integrator:
$$\Delta t_{\max} = \frac{2}{\omega} = 2 \sqrt{\frac{\mu}{k_{\text{bond}}}} \quad [\text{fs}]$$
Where $\mu = \frac{m_1 m_2}{m_1 + m_2}$ is the reduced mass in atomic mass units (amu). All model predictions are physically verified to ensure $\Delta t_{\max} > 20\text{ fs}$ under standard MARTINI conditions.
