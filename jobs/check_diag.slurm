#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbe_diag
#SBATCH --time=02:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/diag_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/diag_%j.err
# diagonal second order, 3 variants side by side (8 cores each): blend-consistent/direct, blend-consistent/GMRES, reference blocks/GMRES
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/results/check_diag; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NEWTON_LS=auto
BIN=$OUT/bin_$SLURM_JOB_ID; cp $R/src/fbke_enzyme $BIN; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.05 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25234 -powderstart -zshift 0.18"
run() { name=$1; cpus=$2; shift 2; env "$@" taskset -c $cpus $BIN $C -dorder 2 -pcsv $OUT/$name.probe.csv -scsv $OUT/$name.sens.csv > $OUT/$name.log 2>&1; echo "$name exit=$?"; }
run hess_direct 0-7  X=1 &
run hess_gmres  8-15 SENS_DIRECT=0 &
run ref_gmres   16-23 SENS_DIRECT=0 FB_REFERENCE2=1 &
run hess_direct_fdsrc 24-31 SRC_FD_LEGACY=1 &
wait; echo DIAG_DONE
