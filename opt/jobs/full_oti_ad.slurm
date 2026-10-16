#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbo_otiad
#SBATCH --time=1-12:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/otiad_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/opt/jobs/otiad_%j.err
# OTI first-order and second-order diagonal with the AD laser source (OTI_NODAL + USE_ENZYME_AD).
# Step 0: 12-step smoke check against the validated Enzyme results; abort if any column differs by more than 1e-6 (relative to column max).
# Then the FULL window (4400 steps), 16 threads, exclusive node, checkpointed; resubmits itself if a stage is unfinished.
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; RES=$R/opt/results/full_oti_ad; mkdir -p $RES
cp -n $R/opt/src/fbko_ad1 $RES/a1; cp -n $R/opt/src/fbko_addiag $RES/ad
export OMP_PROC_BIND=close OMP_PLACES=cores NEWTON_LS=auto
C0="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -powderstart -zshift 0.18"
cd $IN; NT=16
if [ ! -f $RES/SMOKE_OK ]; then
  for v in "a1 1" "ad 2"; do set -- $v
    taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT $RES/$1 $C0 -tf 1.252212 -dorder $2 -pcsv $RES/smoke_$1.p.csv -scsv $RES/smoke_$1.s.csv > $RES/smoke_$1.log 2>&1
  done
  python3 - <<PY
import numpy as np, sys
def L(f): return np.loadtxt(f,delimiter=',',skiprows=1)
ref=L("$R/opt/results/a2v2/color.s.csv")      # validated Enzyme optimised, same 12-step window, AD for every parameter
bad=0
for n,cols in (("a1",30),("ad",60)):
    b=L("$RES/smoke_%s.s.csv"%n); a=ref[:,:b.shape[1]]
    c=np.abs(a-b).max(0)/np.maximum(np.abs(a).max(0),1e-300); print(n,"worst col rel err %.2e"%c.max(), "n cols >1e-6:",(c>1e-6).sum()); bad+=(c>1e-6).sum()
sys.exit(1 if bad else 0)
PY
  if [ $? -eq 0 ]; then touch $RES/SMOKE_OK; else echo "SMOKE CHECK FAILED" > $RES/SMOKE_FAILED; exit 1; fi
fi
C="$C0 -tf 1.25660"
stage() { local name=$1 bin=$2 dord=$3
  [ -f $RES/$name.DONE ] && return
  local restart=""; [ -f $RES/$name.ckpt ] && restart="-restart $RES/$name.ckpt"
  timeout 120000 taskset -c 0-$((NT-1)) env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT $RES/$bin $C -dorder $dord $restart \
      -ckpt $RES/$name.ckpt -ckpt-every 100 -pcsv $RES/${name}_probe.csv -scsv $RES/${name}_sens.csv >> $RES/$name.log 2>&1
  echo "exit=$?" >> $RES/$name.log
  tail -n 12 $RES/$name.log | grep -q "Wall time" && touch $RES/$name.DONE
}
stage d1 a1 1
stage d2diag ad 2
if [ -f $RES/d1.DONE ] && [ -f $RES/d2diag.DONE ]; then echo ALL_DONE > $RES/ALL_DONE; else sbatch $R/opt/jobs/full_oti_ad.slurm; fi
