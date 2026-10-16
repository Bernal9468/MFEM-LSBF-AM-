#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=oti_fixed_d2
#SBATCH --time=3-00:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/otifull_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/otifull_%j.err
# OTI second order (diagonal Hessian, 20 params) on the full window with the powder blend fixed in (a2)(b2)(c2);
# 40 threads, checkpointed, resubmits itself if it hits the time limit.
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/results/oti_blend_fixed; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 NEWTON_LS=auto
cp -n $R/src/fbko_blend_order2 $OUT/bin_frozen; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25660 -powderstart -zshift 0.18"
RESTART=""; [ -f $OUT/d2.ckpt ] && RESTART="-restart $OUT/d2.ckpt"
timeout 255000 $OUT/bin_frozen $C -dorder 2 $RESTART -ckpt $OUT/d2.ckpt -ckpt-every 100 -pcsv $OUT/d2_probe.csv -scsv $OUT/d2_sens.csv >> $OUT/d2.log 2>&1
if tail -n 12 $OUT/d2.log | grep -q "Wall time"; then touch $OUT/DONE; echo DONE; else sbatch $R/jobs/run_oti_fixed.slurm; fi
