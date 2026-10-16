#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_full_enzyme
#SBATCH --time=2-18:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/full_enzyme_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/full_enzyme_%j.err
# FULL window t=1.2522..1.2566 (4400 steps), OPTIMISED enzyme drivers, 16 threads on an exclusive node, stages run one after another.
# Every stage checkpoints and restarts (-restart); the job resubmits itself if a stage is unfinished. Timing = sum of per-step walls in the log.
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; RES=$R/opt/results/full_enzyme; mkdir -p $RES
cp -n $R/opt/src/fbke_enzyme $RES/enz
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25660 -powderstart -zshift 0.18"
cd $IN
NT=16
stage() { local name=$1 bin=$2 dord=$3 envs=$4
  [ -f $RES/$name.DONE ] && return
  local restart=""; [ -f $RES/$name.ckpt ] && restart="-restart $RES/$name.ckpt"
  timeout 230000 taskset -c 0-$((NT-1)) env $envs OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT $RES/$bin $C -dorder $dord $restart \
      -ckpt $RES/$name.ckpt -ckpt-every 100 -pcsv $RES/${name}_probe.csv -scsv $RES/${name}_sens.csv >> $RES/$name.log 2>&1
  echo "exit=$?" >> $RES/$name.log
  tail -n 12 $RES/$name.log | grep -q "Wall time" && touch $RES/$name.DONE
}
stage nominal enz 0 "X=1"
stage d1 enz 1 "X=1"
stage d2diag enz 2 "X=1"
stage d2full enz 2 "SENS_MIXED=1"
ALL=1; for n in nominal d1 d2diag d2full; do [ -f $RES/$n.DONE ] || ALL=0; done
if [ $ALL -eq 1 ]; then echo ALL_DONE > $RES/ALL_DONE; else sbatch $R/opt/jobs/full_enzyme.slurm; fi
