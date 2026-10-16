#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_diag
#SBATCH --time=02:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/diag_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/diag_%j.err
# OPT-5: diagonal-only OTI type vs the full-multivariate OTI type (n2 from compare_oti), 40-step window, 16 threads
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/opt/results/diag; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
cp $R/opt/src/fbko_diag_order2 $OUT/dg; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25224 -powderstart -zshift 0.18"
NT=16
/usr/bin/time -o $OUT/diag_d2.time -f "WALL_S %e" taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT $OUT/dg $C -dorder 2 -pcsv $OUT/diag_d2.p.csv -scsv $OUT/diag_d2.s.csv > $OUT/diag_d2.log 2>&1
echo "diag_d2 $(cat $OUT/diag_d2.time)"
echo DONE
