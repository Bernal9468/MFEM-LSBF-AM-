#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_solvepar
#SBATCH --time=02:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/solvechunk_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/solvechunk_%j.err
# OPT-1 test: concurrent single-thread PARDISO solves (AO_SOLVE_PAR=T) vs baseline loop, full Hessian, 40-step window, 16 threads
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/opt/results/solvepar; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
cp $R/opt/src/fbke_enzyme $OUT/enz2; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25224 -powderstart -zshift 0.18 -dorder 2"
NT=16
for P in 4 16; do
  name=chunk$P
  /usr/bin/time -o $OUT/$name.time -f "WALL_S %e" timeout 1500 taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT SENS_MIXED=1 AO_CHUNK=$P \
     $OUT/enz2 $C -pcsv $OUT/$name.p.csv -scsv $OUT/$name.s.csv > $OUT/$name.log 2>&1
  echo "$name $(cat $OUT/$name.time)"
done
echo DONE
