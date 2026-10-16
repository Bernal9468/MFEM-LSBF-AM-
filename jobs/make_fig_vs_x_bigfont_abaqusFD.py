"""First/second-order derivatives of the row-1 probe temperature versus the laser x position (2x2: k, cp, rho, Lfus)."""
import csv, sys, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = '/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme'
REF = '/work/gyj329/MFEM/implementations/full_bridge_20_param_single_layer/results'
def load(f):
    r = list(csv.reader(open(f))); h = r[0]; rows = {}
    for row in r[1:]:
        if len(row) == len(h):
            try: rows[round(float(row[0]), 9)] = [float(x) for x in row]
            except ValueError: pass
    return h, np.array([rows[k] for k in sorted(rows)])
L = np.array([[float(v) for v in l.replace(' ', '').split(',')[:5]] for l in open(f'{B}/inputs/es_bridge_laser.inp') if l.strip()])
xlas = lambda t: np.interp(t, L[:, 0], L[:, 1])
plt.rcParams.update({'font.size': 22, 'axes.linewidth': 2, 'lines.linewidth': 5.5, 'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
def fig(order):
    if order == 1:
        h, a = load(f'{B}/results/deriv_first_order/d1_sens.csv'); rf = f'{B}/opt/results/full_oti_ad/d1_sens.csv'   # single-pass OTI run (the old single_layer reference resumed from a checkpoint at step 3400 and diverged after it)
        cols = [('dT_dk_row1', r'$\partial T/\partial k$'), ('dT_dcp_row1', r'$\partial T/\partial c_p$'), ('dT_drho_row1', r'$\partial T/\partial \rho$'), ('dT_dLfus_row1', r'$\partial T/\partial L$')]
    else:
        h, a = load(f'{B}/results/deriv_second_order/d2_sens.csv'); rf = f'{B}/results/oti_blend_fixed/d2_sens.csv'
        cols = [('d2T_dk2_row1', r'$\partial^2 T/\partial k^2$'), ('d2T_dcp2_row1', r'$\partial^2 T/\partial c_p^2$'), ('d2T_drho2_row1', r'$\partial^2 T/\partial \rho^2$'), ('d2T_dLfus2_row1', r'$\partial^2 T/\partial L^2$')]
    ref = load(rf) if rf and os.path.exists(rf) else None
    f, ax = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True)
    for k, (c, lab) in enumerate(cols):
        A = ax[k // 2, k % 2]
        A.plot(xlas(a[:, 0]), a[:, h.index(c)], 'r-', label='Enzyme/RESIDUAL')
        if ref is not None and c in ref[0]:
            A.plot(xlas(ref[1][:, 0]), ref[1][:, ref[0].index(c)], 'k--', lw=4.5, label='Abaqus FD')
            n = min(len(ref[1]), len(a)); x_, y_ = ref[1][:n, ref[0].index(c)], a[:n, h.index(c)]
            A.plot([], [], ' ', label=f'NRMSE = {np.sqrt(np.mean((x_ - y_)**2)) / np.sqrt(np.mean(x_**2)) * 100:.2e} %')
        A.set_ylabel(lab)
        lo_, hi_ = A.get_ylim(); A.set_ylim(lo_, hi_ + 0.62 * (hi_ - lo_))   # headroom so the legend never sits on a curve
        A.legend(frameon=False, loc='upper left', fontsize=19)
        if k >= 2: A.set_xlabel('x [mm]')
    f.tight_layout(); out = f'{B}/figures/{"first" if order == 1 else "second"}_order_vs_x_2x2_residual_bigfont_abaqusFD.png'; f.savefig(out, dpi=170); plt.close(f); print(out)
fig(1)
if 'both' in sys.argv: fig(2)
