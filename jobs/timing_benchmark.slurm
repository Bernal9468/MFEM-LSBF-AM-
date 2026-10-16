#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbe_timing
#SBATCH --time=16:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/timing_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/timing_%j.err
# CONTROLLED CPU-overhead / speed-up benchmark, OTI vs Enzyme: same source, same node type, EXCLUSIVE node, one run at a time,
# SAME core count for every run (16, then 40), same 250-step window (t=1.2522..1.25245, includes the laser approach and first melting),
# same probe, same source code for OTI and Enzyme, no other job on the node.
# runs: nominal | first order (20 params) | second order diagonal | second order full Hessian (Enzyme only)
# overhead = wall(derivative run) / wall(nominal run, same implementation).   speed-up = wall(OTI) / wall(Enzyme).
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/results/timing_benchmark; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
cp $R/src/fbke_enzyme $OUT/enz; cp $R/src/fbko_blend_order2 $OUT/oti2; cp $R/src/fbko_blend_order1 $OUT/oti1; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25245 -powderstart -zshift 0.18"
t() { name=$1; bin=$2; ord=$3; shift 3
  /usr/bin/time -o $OUT/$name.time -f "WALL_S %e" taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT "$@" $OUT/$bin $C -dorder $ord -pcsv $OUT/$name.p.csv -scsv $OUT/$name.s.csv > $OUT/$name.log 2>&1
  echo "$name threads=$NT $(cat $OUT/$name.time) driver: $(grep 'Wall time' $OUT/$name.log | tail -1)"; }
for NT in 16 40; do
  t enz_nominal_t$NT   enz  0 X=1
  t oti_nominal_t$NT   oti1 0 X=1
  t enz_d1_t$NT        enz  1 X=1
  t oti_d1_t$NT        oti1 1 X=1
  t enz_d2diag_t$NT    enz  2 X=1
  t oti_d2diag_t$NT    oti2 2 X=1
  t enz_d2full_t$NT    enz  2 SENS_MIXED=1
done
echo TIMING_DONE
