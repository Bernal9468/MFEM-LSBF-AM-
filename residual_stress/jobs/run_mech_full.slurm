#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rs_mechfull
#SBATCH --time=1-12:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/mechfull_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/mechfull_%j.err
# Full thermo-mechanical history (all snapshots: heating window + 10 s cool-down): nominal + first-order + full second-order
# Enzyme sensitivities of the stress w.r.t. (E, nu, CTE, sigma_y, H). Probe element near x=2.5 mm, y=0.05, z=0.18.
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
cd $R/inputs
$R/residual_stress/mech/src/rs_mech -m full_bridge.mesh -tie tie_weights_full.txt -snap $R/residual_stress/results/thermal_nominal/snap \
   -o $R/residual_stress/results/mech_full -dorder 2 -px 2.5 -py 0.05 -pz 0.18 -ckpt-every 25
echo "exit=$?"
