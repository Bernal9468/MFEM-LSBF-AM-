#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbe_full
#SBATCH --time=2-18:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/full_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/full_%j.err
# Full window t=1.2522..1.2566 (4400 steps), Enzyme forward mode, 20 wired parameters:
#   nominal (-dorder 0) | first order (-dorder 1) | full Hessian incl. cross terms (-dorder 2, SENS_MIXED=1)
# ONE job, one 40-core node.  Phase 1: the three stages run side by side (4 / 12 / 24 cores).  The Hessian stage is the long pole:
# as soon as nominal and first order are DONE it is stopped at its last checkpoint (written every 50 steps) and restarted on all 40
# cores.  Every stage restarts from its checkpoint (-restart), so the job can also simply be resubmitted if it is ever killed.
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; RES=$R/results
mkdir -p $RES/nominal $RES/deriv_first_order $RES/deriv_second_order
cp -n $R/src/fbke_enzyme $RES/fbke_enzyme_frozen     # one frozen binary for the whole chain
BIN=$RES/fbke_enzyme_frozen
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25660 -powderstart -zshift 0.18"
cd $IN
# stage name cpus nthreads dorder "ENV=.." resultdir ckpt_every
stage() {
  local name=$1 cpus=$2 nthr=$3 dord=$4 envs=$5 dir=$RES/$6 ck=$7
  local restart=""; [ -f $dir/$name.ckpt ] && restart="-restart $dir/$name.ckpt"
  env $envs OMP_NUM_THREADS=$nthr MKL_NUM_THREADS=$nthr taskset -c $cpus $BIN $C -dorder $dord $restart \
      -ckpt $dir/$name.ckpt -ckpt-every $ck -pcsv $dir/${name}_probe.csv -scsv $dir/${name}_sens.csv >> $dir/$name.log 2>&1
  echo "exit=$? (cpus $cpus)" >> $dir/$name.log
  if tail -n 12 $dir/$name.log | grep -q "Wall time"; then touch $dir/DONE; fi
}
[ -f $RES/nominal/DONE ]          || stage nominal 0-3   4  0 "X=1" nominal 200 &
[ -f $RES/deriv_first_order/DONE ] || stage d1     4-15  12 1 "X=1" deriv_first_order 100 &
# Hessian stage, with re-launch on all 40 cores once the others are finished
(
  while [ ! -f $RES/deriv_second_order/DONE ]; do
    if [ -f $RES/nominal/DONE ] && [ -f $RES/deriv_first_order/DONE ]; then CP=0-39; NT=40; else CP=16-39; NT=24; fi
    stage d2 $CP $NT 2 "SENS_MIXED=1" deriv_second_order 50 &
    spid=$!
    while kill -0 $spid 2>/dev/null; do
      sleep 60
      if [ $NT -eq 24 ] && [ -f $RES/nominal/DONE ] && [ -f $RES/deriv_first_order/DONE ]; then
        pkill -TERM -f "fbke_enzyme_frozen.*d2.ckpt"; wait $spid 2>/dev/null; echo "[orchestrator] relaunching d2 on 40 cores" >> $RES/deriv_second_order/d2.log; break
      fi
    done
    wait $spid 2>/dev/null
    [ -f $RES/deriv_second_order/DONE ] && break
    if tail -n 3 $RES/deriv_second_order/d2.log | grep -q "relaunching"; then continue; fi
    sleep 5; break   # crashed for another reason: do not loop forever
  done
) &
wait
if [ -f $RES/nominal/DONE ] && [ -f $RES/deriv_first_order/DONE ] && [ -f $RES/deriv_second_order/DONE ]; then touch $RES/ALL_DONE; echo ALL_DONE
else echo "NOT ALL DONE - resubmit jobs/run_full.slurm"; fi
