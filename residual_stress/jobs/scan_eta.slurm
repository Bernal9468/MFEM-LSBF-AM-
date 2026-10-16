#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --job-name=rs_etascan
#SBATCH --time=03:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/etascan_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/etascan_%j.err
# Short-window (0.8 ms, laser passes the probe) scan of the absorptivity scale, NO powder: which effective power reproduces the Abaqus rise?
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; V=$R/residual_stress/results/eta_scan; mkdir -p $V
export OMP_PROC_BIND=false NEWTON_LS=auto
cd $IN
run() { eta=$1; d=$V/eta_$eta; mkdir -p $d
  env OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 $R/residual_stress/src/fbke_enzyme -m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 200 \
     -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -zshift 0.18 -tf 1.2530 -dorder 0 -etascale $eta -pcsv $d/probe.csv > $d/run.log 2>&1; echo "eta $eta done"; }
run 0.5 & run 0.6 & run 0.7 & wait; echo ETA_SCAN_DONE
