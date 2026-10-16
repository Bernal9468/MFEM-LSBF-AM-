#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fbo_buildall
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/buildall_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/buildall_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/src
make -j4 fbke_enzyme fbko_opt_order1 fbko_diag_order2 fbko_hess_order2 2>&1 | grep -i " error" | head
ls -la fbke_enzyme fbko_opt_order1 fbko_diag_order2 fbko_hess_order2
