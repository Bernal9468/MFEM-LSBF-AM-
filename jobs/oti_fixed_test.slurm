#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=oti_fixtest
#SBATCH --time=03:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/otitest_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/otitest_%j.err
# 70 steps (probe x=2.05): OTI order 2 with the powder blend added to (a2)(b2)(c2), vs the Enzyme blend-consistent diagonal
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; OUT=$R/results/oti_fixed_test; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 NEWTON_LS=auto
BIN=$OUT/bin_$SLURM_JOB_ID; cp $R/src/fbko_blend_order2 $BIN; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.05 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25227 -powderstart -zshift 0.18"
$BIN $C -dorder 2 -pcsv $OUT/p.csv -scsv $OUT/s.csv > $OUT/run.log 2>&1
python3 - <<'PY'
import csv, numpy as np
def load(f):
    r=list(csv.reader(open(f))); h=r[0]; return h,np.array([[float(x) for x in row] for row in r[1:] if len(row)==len(h)])
o=load('/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/results/oti_fixed_test/s.csv')
e=load('/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/results/check_diag/hess_direct_fdsrc.sens.csv')
n=min(len(o[1]),len(e[1])); res=[]
for j,c in enumerate(o[0]):
    if c=='time_native' or c not in e[0]: continue
    x=o[1][:n,j]; y=e[1][:n,e[0].index(c)]; d=np.sqrt(np.mean(x**2))
    if d>1e-30: res.append((np.sqrt(np.mean((x-y)**2))/d*100,c))
res.sort(reverse=True); print(n,'rows; OTI(blend fixed) vs Enzyme:'); [print(f'  {c:20s} {v:.2e} %') for v,c in res]
PY
