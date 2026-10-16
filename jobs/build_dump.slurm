#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fbe_blddump
#SBATCH --time=01:30:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/build_dump_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/build_dump_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/src_dump
make -j8 enzyme 2>&1 | grep -v "^g++\|^/usr\|^clang\|note:\|^ *\^\|OTI_CONSTEXPR" | tail -30
ls -la fbke_enzyme
