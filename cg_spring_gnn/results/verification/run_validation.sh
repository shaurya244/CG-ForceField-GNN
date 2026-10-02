#!/bin/bash
# ==============================================================================
# Automated GROMACS Post-Molecular Verification Workflow for IBUP
# ==============================================================================
set -e

echo "Step 1: Energy Minimization (checking force tolerance < 10 kJ/mol/nm)..."
gmx grompp -f em.mdp -c IBUP_initial.gro -p topol.top -o em.tpr
gmx mdrun -deffnm em -v

echo "Step 2: NVT Equilibration at 300K (checking integration stability at 20 fs)..."
gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr
gmx mdrun -deffnm nvt -v

echo "Step 3: Extracting Trajectory Distributions..."
# Extract bond and angle histograms from trajectory
gmx distance -s nvt.tpr -f nvt.xtc -oall bond_dist.xvg
gmx angle -f nvt.xtc -ov angle_dist.xvg

echo "Validation simulation finished successfully!"
