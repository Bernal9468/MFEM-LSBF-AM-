#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_cmp
#SBATCH --time=03:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/cmpall_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/cmpall_%j.err
# original vs optimised Enzyme driver, same 40-step window, 16 threads, exclusive node: nominal / d1 / d2 diag / d2 full Hessian
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/opt/results/compare; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
cp $R/src/fbke_enzyme $OUT/orig; cp $R/opt/src/fbke_enzyme $OUT/opt; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25224 -powderstart -zshift 0.18"
NT=16
t() { name=$1; bin=$2; ord=$3; shift 3
  /usr/bin/time -o $OUT/$name.time -f "WALL_S %e" taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT "$@" $OUT/$bin $C -dorder $ord -pcsv $OUT/$name.p.csv -scsv $OUT/$name.s.csv > $OUT/$name.log 2>&1
  echo "$name $(cat $OUT/$name.time)"; }
for B in orig opt; do
  t ${B}_nominal $B 0 X=1
  t ${B}_d1      $B 1 X=1
  t ${B}_d2diag  $B 2 X=1
  t ${B}_d2full  $B 2 SENS_MIXED=1
done
echo DONE
