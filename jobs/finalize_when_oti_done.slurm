#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=fbe_final
#SBATCH --time=2-12:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/final_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/final_%j.err
# waits for the corrected OTI second-order run (results/oti_blend_fixed/DONE), then regenerates report + figures and writes the
# second-order NRMSE table (Enzyme vs OTI, 20 pure terms)
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
until [ -f $R/results/oti_blend_fixed/DONE ] && [ -f $R/results/oti_first_order/DONE ] && [ -f $R/results/deriv_second_order/DONE ]; do sleep 300; done
cd $R
python3 jobs/make_report.py > /dev/null
python3 jobs/compare_table.py > $R/COMPARISON_TABLE.txt
python3 jobs/make_fig_vs_x.py
python3 - > $R/SECOND_ORDER_COMPARISON.txt <<'PY'
import csv, numpy as np
R='/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/results'
def load(f):
    r=list(csv.reader(open(f))); h=r[0]; rows={}
    for row in r[1:]:
        if len(row)==len(h): rows[round(float(row[0]),9)]=[float(x) for x in row]
    return h,np.array([rows[k] for k in sorted(rows)])
e=load(f'{R}/deriv_second_order/d2_sens.csv'); o=load(f'{R}/oti_blend_fixed/d2_sens.csv'); n=min(len(e[1]),len(o[1]))
print(f'second order, Enzyme (full Hessian run) vs OTI with blend fixed, {n} steps, NRMSE per column')
res=[]
for j,c in enumerate(o[0]):
    if c.startswith('d2T_') and c in e[0]:
        x=o[1][:n,j]; y=e[1][:n,e[0].index(c)]; d=np.sqrt(np.mean(x**2))
        if d>0: res.append((np.sqrt(np.mean((x-y)**2))/d*100,c))
for v,c in sorted(res,reverse=True): print(f'  {c:22s} {v:.3e} %')
PY
