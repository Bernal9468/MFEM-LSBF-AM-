#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_smokead
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/smokead_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/smokead_%j.err
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/opt/results/smokead; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
cp $R/opt/src/fbko_ad1 $OUT/a1; cp $R/opt/src/fbko_addiag $OUT/ad; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.252212 -powderstart -zshift 0.18"
r() { n=$1; b=$2; o=$3; shift 3; taskset -c 0-15 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 "$@" $OUT/$b $C -dorder $o -pcsv $OUT/$n.p.csv -scsv $OUT/$n.s.csv > $OUT/$n.log 2>&1; echo "$n done"; }
r a1_d1 a1 1 X=1
r ad_d2 ad 2 X=1
echo DONE
