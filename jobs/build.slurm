#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fbe_build
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/build_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/build_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/src
make -j8 enzyme 2>&1 | grep -v "^g++\|^/usr" | tail -80
ls -la fbke_enzyme
