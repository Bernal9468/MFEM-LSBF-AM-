#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rs_hscan
#SBATCH --time=02:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/hscan_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/hscan_%j.err
# finite-difference step scan for E (scale on Young's modulus): is the 1.5% gap at increment 5 a kink effect?
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
BIN=$R/residual_stress/mech/src/rs_mech; SNAP=$R/residual_stress/results/thermal_nominal/snap; V=$R/residual_stress/results/val_h; mkdir -p $V
cd $R/inputs
run() { name=$1; shift; mkdir -p $V/$name
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $BIN -m full_bridge.mesh -tie tie_weights_full.txt -snap $SNAP -o $V/$name -ninc 6 "$@" > $V/$name/run.log 2>&1; echo "$name done"; }
( run E_p1e-4 -dorder 0 -sE 1.0001; run E_m1e-4 -dorder 0 -sE 0.9999 ) &
( run E_p1e-5 -dorder 0 -sE 1.00001; run E_m1e-5 -dorder 0 -sE 0.99999 ) &
( run E_p1e-2 -dorder 0 -sE 1.01; run E_m1e-2 -dorder 0 -sE 0.99 ) &
wait; echo HSCAN_DONE
