#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=rs_testcool
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/testcool_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/testcool_%j.err
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/residual_stress/results/testcool; rm -rf $OUT; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cp $R/residual_stress/src/fbke_enzyme $OUT/enz; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -powderstart -zshift 0.18"
$OUT/enz $C -tf 21.25224 -laser-off 1.25224 -cool-grow 1.3 -cool-dtmax 0.5 -snapdir $OUT/snap -snap-every 10 -dorder 0 -pcsv $OUT/probe.csv > $OUT/run.log 2>&1
echo "exit=$?"
grep -E "^\s+Step" $OUT/run.log | tail -30 | cut -c1-140
ls $OUT/snap | wc -l
