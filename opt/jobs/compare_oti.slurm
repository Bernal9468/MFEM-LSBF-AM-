#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_cmpoti
#SBATCH --time=04:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/cmpoti_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/cmpoti_%j.err
# OTI driver: original (GMRES, no FTZ, no chunking) vs optimised, same 40-step window, 16 threads, exclusive node
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/opt/results/compare_oti; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
cp $R/src/fbko_blend_order1 $OUT/o1; cp $R/src/fbko_blend_order2 $OUT/o2; cp $R/opt/src/fbko_opt_order1 $OUT/n1; cp $R/opt/src/fbko_opt_order2 $OUT/n2; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25224 -powderstart -zshift 0.18"
NT=16
t() { name=$1; bin=$2; ord=$3; shift 3
  /usr/bin/time -o $OUT/$name.time -f "WALL_S %e" taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT "$@" $OUT/$bin $C -dorder $ord -pcsv $OUT/$name.p.csv -scsv $OUT/$name.s.csv > $OUT/$name.log 2>&1
  echo "$name $(cat $OUT/$name.time)"; }
t old_d1     o1 1 X=1
t new_d1     n1 1 X=1
t new_d1_ftzoff n1 1 FTZ_OFF=1
t old_d2     o2 2 X=1
t new_d2     n2 2 X=1
t new_d2_noDirect n2 2 SENS_DIRECT=0
echo DONE
