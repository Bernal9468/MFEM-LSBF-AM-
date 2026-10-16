#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=fbe_viz
#SBATCH --time=14:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/viz_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/viz_%j.err
# First-order (-dorder 1, 20 wired params) run of the single-layer full bridge, window t=1.2522..1.2566 (4400 steps), same flags as run_full.slurm's first-order stage, with a field dump (T + dT/dp_i)
# every 20 steps for the sensitivity-colour videos.  Restartable: re-submit to resume from the checkpoint (written every 100 steps).
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme; IN=$R/inputs; O=$R/results/viz_first_order; mkdir -p $O/dump; cd $IN; export OMP_STACKSIZE=512M
CORES=$(/work/gyj329/MFEM/implementations/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh | tr ' ' ','); cp -n $R/src_dump/fbke_enzyme $O/fbke_enzyme_frozen
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25660 -powderstart -zshift 0.18"
restart=""; [ -f $O/d1.ckpt ] && restart="-restart $O/d1.ckpt"
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 NEWTON_LS=auto FIELD_DUMP_DIR=$O/dump FIELD_DUMP_EVERY=20 taskset -c $CORES $O/fbke_enzyme_frozen $C -dorder 1 $restart \
  -ckpt $O/d1.ckpt -ckpt-every 100 -pcsv $O/d1_probe.csv -scsv $O/d1_sens.csv >> $O/d1.log 2>&1
tail -n 12 $O/d1.log | grep -q "Wall time" && touch $O/DONE && echo DONE; ls $O/dump | wc -l
