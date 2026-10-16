#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=rs_mbuild
#SBATCH --time=00:30:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/mech/mbuild_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/mech/mbuild_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/mech/src
make rs_mech 2>&1 | grep -v "^g++\|^/usr" | tail -30
ls -la rs_mech
