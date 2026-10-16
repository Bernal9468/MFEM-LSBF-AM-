#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=rs_mtest
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/mtest_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/mtest_%j.err
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
cd $R/inputs
$R/residual_stress/mech/src/rs_mech -m full_bridge.mesh -tie tie_weights_full.txt -snap $R/residual_stress/results/thermal_nominal/snap -o $R/residual_stress/results/mech_test -ninc 6
