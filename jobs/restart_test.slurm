#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --job-name=fbe_rst
#SBATCH --time=01:30:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/rst_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/rst_%j.err
# restart == continuous? first-order run to step 700 (laser has passed the probe, powder->melt->cool), (A) continuous, (B) killed after ckpt at 300 and resumed
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/results/restart_test; rm -rf $OUT; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 NEWTON_LS=auto
BIN=$OUT/bin; cp $R/src/fbke_enzyme $BIN; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.05 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25270 -powderstart -zshift 0.18"
# A: continuous
taskset -c 0-19 $BIN $C -dorder 1 -pcsv $OUT/A_p.csv -scsv $OUT/A_s.csv > $OUT/A.log 2>&1 &
# B: stop at step ~310 (ckpt every 100 -> resume from 300), then resume to the end
( timeout -s TERM 25 true; taskset -c 20-39 $BIN $C -dorder 1 -ckpt $OUT/B.ckpt -ckpt-every 100 -pcsv $OUT/B_p.csv -scsv $OUT/B_s.csv > $OUT/B1.log 2>&1 & p=$!
  while ! grep -q "^  Step   310" $OUT/B1.log 2>/dev/null; do sleep 2; done; kill -TERM $p; wait $p 2>/dev/null
  taskset -c 20-39 $BIN $C -dorder 1 -restart $OUT/B.ckpt -ckpt $OUT/B.ckpt -ckpt-every 100 -pcsv $OUT/B_p.csv -scsv $OUT/B_s.csv > $OUT/B2.log 2>&1 ) &
wait
python3 - <<'PY'
import csv, numpy as np
OUT='/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/results/restart_test'
def load(f):
    r=list(csv.reader(open(f))); h=r[0]; rows={}
    for row in r[1:]:
        if len(row)==len(h): rows[round(float(row[0]),9)]=[float(x) for x in row]
    return h,np.array([rows[k] for k in sorted(rows)])
for f in ('p','s'):
    a=load(f'{OUT}/A_{f}.csv'); b=load(f'{OUT}/B_{f}.csv'); n=min(len(a[1]),len(b[1]))
    d=np.max(np.abs(a[1][:n]-b[1][:n])/(np.abs(a[1][:n])+1e-30)); print(f, 'rows',len(a[1]),len(b[1]),'max rel diff', d)
PY
