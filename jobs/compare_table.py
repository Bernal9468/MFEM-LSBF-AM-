"""PASS/FAIL table (NRMSE <= 1 %) Enzyme vs OTI for all 20 first-order and 20 pure second-order columns.
NRMSE = sqrt(mean((ref-enz)^2)) / sqrt(mean(ref^2)) * 100 over the common steps."""
import csv, sys, os, numpy as np
B = '/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/results'
REF = '/work/gyj329/MFEM/implementations/full_bridge_20_param_single_layer/results'
def load(f):
    r = list(csv.reader(open(f))); h = r[0]; rows = {}
    for row in r[1:]:
        if len(row) == len(h):
            try: rows[round(float(row[0]), 9)] = [float(x) for x in row]
            except ValueError: pass
    return h, np.array([rows[k] for k in sorted(rows)])
names = ['k','cp','rho','Lfus','Ts','Tliq','hc','T0','P','Eta','Rx','Ry','Rz','X0','Vx','TL','Tw','hp','kp','Rhop']
tag = {'Eta': 'Eta', 'Rhop': 'Rhop'}
notes = {'Tw': 'OTI/reference misses the substrate IC seed (Enzyme copy fixed it)', 'Ry': '|value| ~1e-10, below solver noise',
         'X0': 'OTI 2nd order is exactly 0 (not implemented)', 'Vx': 'OTI 2nd order is exactly 0 (not implemented)', 'TL': 'OTI 2nd order is exactly 0 (not implemented)'}
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
def table(title, enz, ref, pre):
    n = min(len(enz[1]), len(ref[1]), NMAX); print(f'\n{title}  ({n} common steps)')
    print(f'{"param":6s} {"NRMSE [%]":>12s}  {"verdict":8s} note')
    npass = 0
    for p in names:
        c = f'{pre}{p}2_row1' if pre.startswith('d2') else f'{pre}{p}_row1'
        if c not in ref[0] or c not in enz[0]: print(f'{p:6s} {"n/a":>12s}'); continue
        x = ref[1][:n, ref[0].index(c)]; y = enz[1][:n, enz[0].index(c)]; d = np.sqrt(np.mean(x**2))
        e = np.sqrt(np.mean((x - y)**2)) / d * 100 if d > 0 else (0.0 if not np.any(y) else float('inf'))
        ok = e <= 1.0; npass += ok
        print(f'{p:6s} {e:12.3e}  {"PASS" if ok else "FAIL":8s} {"" if ok else notes.get(p, "")}')
    print(f'-> {npass}/{len(names)} within 1 %')
e1 = load(f'{B}/deriv_first_order/d1_sens.csv'); r1 = load(f'{B}/oti_first_order/d1_sens.csv') if os.path.exists(f'{B}/oti_first_order/DONE') else load(f'{REF}/deriv_first_order/d1_sens.csv')
table('FIRST ORDER  Enzyme vs OTI reference', e1, r1, 'dT_d')
import os
f2 = f'{B}/oti_blend_fixed/d2_sens.csv'
if os.path.exists(f2):
    e2 = load(f'{B}/deriv_second_order/d2_sens.csv'); r2 = load(f2)
    table('SECOND ORDER (pure terms)  Enzyme vs OTI with powder blend fixed', e2, r2, 'd2T_d')
