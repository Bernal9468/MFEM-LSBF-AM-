#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fbo_buildad
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/buildad_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/buildad_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/src
make -j2 fbko_ad1 fbko_addiag 2>&1 | grep -i " error" | head
ls -la fbko_ad1 fbko_addiag
