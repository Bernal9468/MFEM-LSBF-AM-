#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_otihess3
#SBATCH --time=03:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/otihess3_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/otihess3_%j.err
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/opt/results/otihess3; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
cp $R/opt/src/fbko_hess_order2 $OUT/oh; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -powderstart -zshift 0.18 -dorder 2"
NT=16
for TF in 1.252212 1.25224; do
  name=oh_$TF
  /usr/bin/time -o $OUT/$name.time -f "WALL_S %e" taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT SENS_MIXED=1 \
     $OUT/oh $C -tf $TF -pcsv $OUT/$name.p.csv -scsv $OUT/$name.s.csv > $OUT/$name.log 2>&1
  echo "$name $(cat $OUT/$name.time)"
done
cp $R/opt/src/fbko_opt_order1 $OUT/o1
C1="${C% -dorder 2}"
/usr/bin/time -o $OUT/d1.time -f "WALL_S %e" taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT $OUT/o1 $C1 -dorder 1 -tf 1.25224 -pcsv $OUT/d1.p.csv -scsv $OUT/d1.s.csv > $OUT/d1.log 2>&1
echo "d1 $(cat $OUT/d1.time)"
echo DONE
