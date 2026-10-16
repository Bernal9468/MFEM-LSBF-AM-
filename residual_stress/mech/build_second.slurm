#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=rs_ksecond
#SBATCH --time=00:20:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/mech/ksecond_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/mech/ksecond_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/mech/src
make -f Makefile.test test_second 2>&1 | tail -5 && ./test_second 4000
