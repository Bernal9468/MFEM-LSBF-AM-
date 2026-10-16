#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbe_prof
#SBATCH --time=00:40:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/prof_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/prof_%j.err
# 12-step profile of the mixed (full Hessian) pass, 16 threads, exclusive node
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/results/profile; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 NEWTON_LS=auto
cp $R/src/fbke_enzyme $OUT/bin_$SLURM_JOB_ID; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.252212 -powderstart -zshift 0.18"
SENS_MIXED=1 taskset -c 0-15 $OUT/bin_$SLURM_JOB_ID $C -dorder 2 -pcsv $OUT/p.csv -scsv $OUT/s.csv > $OUT/mixed.log 2>&1
taskset -c 0-15 $OUT/bin_$SLURM_JOB_ID $C -dorder 2 -pcsv $OUT/p2.csv -scsv $OUT/s2.csv > $OUT/plain2.log 2>&1
grep "SENS2 timers\|^  Step" $OUT/mixed.log | tail -8; grep "SENS2 timers\|^  Step" $OUT/plain2.log | tail -4
