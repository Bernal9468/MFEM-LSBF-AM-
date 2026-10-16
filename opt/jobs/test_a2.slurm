#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_a2
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/a2_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/a2_%j.err
# OPT-2 test: coloured (atomic-free) a2 scatter, full Hessian, 12-step window, 16 threads; compare with scan32 (chunk32, atomics)
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/opt/results/a2; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
cp $R/opt/src/fbke_enzyme $OUT/enz; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.252212 -powderstart -zshift 0.18 -dorder 2"
NT=16
for V in color atomic; do
  name=$V; X=""; [ $V = atomic ] && X="A2_ATOMIC=1"
  /usr/bin/time -o $OUT/$name.time -f "WALL_S %e" timeout 1500 taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT SENS_MIXED=1 $X \
     $OUT/enz $C -pcsv $OUT/$name.p.csv -scsv $OUT/$name.s.csv > $OUT/$name.log 2>&1
  echo "$name $(cat $OUT/$name.time)"
done
echo DONE
