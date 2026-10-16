#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=rs_thermal
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/thermal_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/thermal_%j.err
# Nominal thermal history for the residual-stress work: the validated single-layer window (t=1.2522..1.2566, 4400 steps of 1 us),
# laser off at 1.2566, then a 10 s cool-down with growing dt (x1.3, cap 0.5 s). Temperature snapshots every 20 steps while heating,
# every step while cooling. Checkpointed; resubmits itself if unfinished.
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/residual_stress/results/thermal_nominal; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cp -n $R/residual_stress/src/fbke_enzyme $OUT/enz; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 100 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -powderstart -zshift 0.18"
RESTART=""; [ -f $OUT/th.ckpt ] && RESTART="-restart $OUT/th.ckpt"
$OUT/enz $C -tf 11.2566 -laser-off 1.2566 -cool-grow 1.3 -cool-dtmax 0.5 -snapdir $OUT/snap -snap-every 20 -dorder 0 $RESTART \
   -ckpt $OUT/th.ckpt -ckpt-every 200 -pcsv $OUT/probe.csv >> $OUT/run.log 2>&1
echo "exit=$?" >> $OUT/run.log
if tail -n 12 $OUT/run.log | grep -q "Wall time"; then touch $OUT/DONE; else sbatch $R/residual_stress/jobs/run_thermal_nominal.slurm; fi
