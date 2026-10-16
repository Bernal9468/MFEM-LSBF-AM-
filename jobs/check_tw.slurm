#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=fbe_tw
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/tw_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/tw_%j.err
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/results/check_tw; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 NEWTON_LS=auto
BIN=$OUT/bin_$SLURM_JOB_ID; cp $R/src/fbke_enzyme $BIN; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.05 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25226 -powderstart -zshift 0.18"
SENS_MIXED=1 $BIN $C -dorder 2 -pcsv $OUT/p.csv -scsv $OUT/s.csv > $OUT/mixed.log 2>&1; echo done
