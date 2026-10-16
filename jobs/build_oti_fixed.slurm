#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fbe_buildoti
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/buildoti_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/buildoti_%j.err
# OTI order-2 binary from THIS source (no -DUSE_ENZYME_AD) with the powder blend added to the second-order blocks
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/src
make -j8 fbko_opt_order2 2>&1 | grep -i " error\|Error " | head; cp fbko_opt_order2 fbko_blend_order2; ls -la fbko_blend_order2
