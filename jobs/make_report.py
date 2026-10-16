"""Post-process the full-window Enzyme runs: de-duplicate restarted CSVs, compare with the OTI reference, write figures.
usage: python3 make_report.py [results_dir]
"""
import csv, sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = sys.argv[1] if len(sys.argv) > 1 else '/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/results'
REF = '/work/gyj329/MFEM/implementations/full_bridge_20_param_single_layer/results'
FIG = os.path.join(os.path.dirname(R.rstrip('/')), 'figures'); os.makedirs(FIG, exist_ok=True)

def load(f):
    """CSV -> (header, array), last occurrence of every time wins (a restart re-appends the rows after its checkpoint)."""
    r = list(csv.reader(open(f))); h = r[0]
    rows = {}
    for row in r[1:]:
        if len(row) == len(h):
            try: rows[round(float(row[0]), 9)] = [float(x) for x in row]
            except ValueError: pass
    return h, np.array([rows[k] for k in sorted(rows)])

def nrmse(x, y):
    n = min(len(x), len(y)); x, y = x[:n], y[:n]; d = np.sqrt(np.mean(x**2))
    return np.sqrt(np.mean((x - y)**2)) / d * 100 if d > 0 else (0.0 if not np.any(y) else np.inf)

out = []
def say(s=''): print(s); out.append(s)

# ---- nominal ----
nom = load(f'{R}/nominal/nominal_probe.csv')
say(f'nominal: {len(nom[1])} steps, T_max = {nom[1][:, 2].max():.1f} K')
fig_data = {'t': nom[1][:, 0], 'T': nom[1][:, 1], 'Tmax': nom[1][:, 2]}
rp = f'{REF}/nominal_single_layer/probe.csv'
if os.path.exists(rp):
    rh, ra = load(rp); n = min(len(ra), len(nom[1]))
    say(f'  vs OTI reference nominal: probe NRMSE {nrmse(ra[:n,1], nom[1][:n,1]):.3e} %, T_max NRMSE {nrmse(ra[:n,2], nom[1][:n,2]):.3e} %, '
        f'peak {ra[:n,2].max():.2f} K (ref) vs {nom[1][:n,2].max():.2f} K')
# ---- first order ----
d1 = load(f'{R}/deriv_first_order/d1_sens.csv'); say(f'first order: {len(d1[1])} steps')
r1 = f'{REF}/deriv_first_order/d1_sens.csv'
if os.path.exists(r1):
    rh, ra = load(r1); n = min(len(ra), len(d1[1])); say(f'  vs OTI reference first order ({n} common steps):')
    res = sorted(((nrmse(ra[:n, j], d1[1][:n, d1[0].index(c)]), c) for j, c in enumerate(rh) if c != 'time_native' and c in d1[0] and np.any(ra[:n, j])), reverse=True)
    for e, c in res: say(f'    {c:22s} NRMSE {e:10.3e} %')
# ---- second order ----
hf = f'{R}/deriv_second_order/d2_sens.csv.hess.csv'
if os.path.exists(hf):
    hh = load(hf); say(f'second order (full Hessian): {len(hh[1])} steps, {len(hh[0]) - 1} entries (20 pure + 190 mixed)')
    d2 = load(f'{R}/deriv_second_order/d2_sens.csv')
    # pure terms of the Hessian file == d2 columns of the sens file
    for nm, cc in (('k', 'dk2'), ('cp', 'dcp2'), ('P', 'dP2')):
        a = hh[1][:, hh[0].index(f'd2T_d{nm}_d{nm}')]; b = d2[1][:len(a), d2[0].index(f'd2T_{cc}_row1')]
        say(f'  pure d2T/d{nm}2 in hess.csv vs sens.csv: NRMSE {nrmse(a, b):.2e} %')

# ---- figures (white background, no grid, no title, thick lines, one row per figure) ----
plt.rcParams.update({'font.size': 14, 'axes.linewidth': 2, 'lines.linewidth': 3, 'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
t = (fig_data['t'] - fig_data['t'][0]) * 1e3
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(t, fig_data['T'], color='#1f77b4'); ax[0].set_xlabel('time [native x 1e-3]'); ax[0].set_ylabel('probe T [K]')
ax[1].plot(t, fig_data['Tmax'], color='#d62728'); ax[1].set_xlabel('time [native x 1e-3]'); ax[1].set_ylabel('max T [K]')
fig.tight_layout(); fig.savefig(f'{FIG}/01_nominal.png', dpi=200); plt.close(fig)

names = ['k', 'cp', 'rho', 'P', 'Rx', 'Rz', 'X0', 'kp', 'hp']
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
tt = (d1[1][:, 0] - d1[1][0, 0]) * 1e3
cols = {'cp': 'dT_dcp_row1', 'P': 'dT_dP_row1', 'X0': 'dT_dX0_row1', 'Rz': 'dT_dRz_row1', 'kp': 'dT_dkp_row1', 'Rx': 'dT_dRx_row1', 'k': 'dT_dk_row1', 'rho': 'dT_drho_row1'}
for lab, c in cols.items():
    v = d1[1][:, d1[0].index(c)]
    ax[0].plot(tt, v, label=lab)
ax[0].set_yscale('symlog', linthresh=1e-6); ax[0].legend(frameon=False, ncol=2); ax[0].set_xlabel('time [native x 1e-3]'); ax[0].set_ylabel('dT/dp (row-1 probe)')
if os.path.exists(hf):
    th = (hh[1][:, 0] - hh[1][0, 0]) * 1e3
    for a_, b_ in (('cp', 'cp'), ('P', 'P'), ('X0', 'X0'), ('Rz', 'Rz'), ('kp', 'kp')):
        ax[1].plot(th, hh[1][:, hh[0].index(f'd2T_d{a_}_d{b_}')], label=f'{a_},{b_}')
    ax[1].set_yscale('symlog', linthresh=1e-3); ax[1].legend(frameon=False, ncol=2); ax[1].set_xlabel('time [native x 1e-3]'); ax[1].set_ylabel('pure d2T/dp2')
    for a_, b_ in (('cp', 'P'), ('cp', 'X0'), ('P', 'X0'), ('cp', 'kp'), ('P', 'kp'), ('Rz', 'X0')):
        c = f'd2T_d{a_}_d{b_}' if f'd2T_d{a_}_d{b_}' in hh[0] else f'd2T_d{b_}_d{a_}'
        ax[2].plot(th, hh[1][:, hh[0].index(c)], label=f'{a_},{b_}')
    ax[2].set_yscale('symlog', linthresh=1e-3); ax[2].legend(frameon=False, ncol=2); ax[2].set_xlabel('time [native x 1e-3]'); ax[2].set_ylabel('mixed d2T/dp_i dp_j')
fig.tight_layout(); fig.savefig(f'{FIG}/02_first_and_second_order.png', dpi=200); plt.close(fig)
open(f'{os.path.dirname(R.rstrip("/"))}/REPORT.txt', 'w').write('\n'.join(out) + '\n')
