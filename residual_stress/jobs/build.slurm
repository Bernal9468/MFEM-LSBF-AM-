#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=rs_build
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/build_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/build_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/src
make -j4 fbke_enzyme 2>&1 | grep -i " error" | head
ls -la fbke_enzyme
