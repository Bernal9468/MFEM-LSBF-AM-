#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbe_d2
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/d2_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/d2_%j.err
# Full Hessian (-dorder 2, SENS_MIXED=1, 20 pure + 190 cross terms), all 40 cores, restartable (checkpoint now includes ever_melted + last_dir).
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; D=$R/results/deriv_second_order; mkdir -p $D
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 NEWTON_LS=auto
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25660 -powderstart -zshift 0.18"
RESTART=""; [ -f $D/d2.ckpt ] && RESTART="-restart $D/d2.ckpt"
cd $IN
SENS_MIXED=1 timeout 170000 $R/results/fbke_enzyme_frozen2 $C -dorder 2 $RESTART -ckpt $D/d2.ckpt -ckpt-every 100 -pcsv $D/d2_probe.csv -scsv $D/d2_sens.csv >> $D/d2.log 2>&1
if tail -n 12 $D/d2.log | grep -q "Wall time"; then touch $D/DONE; echo DONE; else sbatch $R/jobs/run_d2.slurm; fi
