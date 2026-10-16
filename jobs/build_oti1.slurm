#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fbe_buildoti1
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/buildoti1_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/buildoti1_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/src
make -j8 fbko_opt_order1 2>&1 | grep -i " error" | head; cp fbko_opt_order1 fbko_blend_order1; ls -la fbko_blend_order1
