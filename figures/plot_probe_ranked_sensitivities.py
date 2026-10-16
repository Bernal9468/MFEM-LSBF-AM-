"""Ranked sensitivities at the probe top (single-layer full bridge, 20-param Enzyme first-order run), paper Fig. 6(b) style: stacked cumulative ranked sensitivities (left, %),
nominal temperature response as a black line (right, K), legend below.  Sensitivities = relative-parameter dT/dp at the probe from results/deriv_first_order/d1_sens.csv (every step),
nominal T from results/nominal/nominal_probe.csv.  All 30 parameters of the problem (paper Table 1 set) are in the legend; 10 geometry ones are structurally zero.  Each |dT/dp| is multiplied by a perturbation (DELTA) so all are comparable in K.
Same colours/notation as the tiny-patch figures (paper Fig. 6)."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
R = "/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme"; T0W = 1.2522
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 22, 'axes.linewidth': 2, 'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
# all 30 parameters of the problem (paper Table 1 set), in the order of the d1_sens.csv columns; the 10 geometry ones (y0, Vy, lL, T0p, Bp, Lp, Wp, B, L, W) are structurally zero in this model but stay in the legend
NAMES = ["k", "cp", "rho", "Lfus", "Ts", "Tliq", "hc", "T0", "P", "Eta", "Rx", "Ry", "Rz", "X0", "Vx", "TL", "Y0", "Vy", "lL", "T0p", "Tw", "hp", "kp", "Rhop", "Bp", "Lp", "Wp", "B", "L", "W"]
PAPER = {"k": "#2f6e9f", "cp": "#2e8b2e", "rho": "#c8202a", "Lfus": "#00d400", "Ts": "#ff3b3b", "Tliq": "#e69f00", "hc": "#00e5ee", "T0": "#ff00ff", "P": "#a65628", "Eta": "#f781bf",
         "Rx": "#e41a1c", "Ry": "#3a7bb5", "Rz": "#4daf4a", "X0": "#9c3fb5", "Vx": "#ffff33", "TL": "#999999", "Y0": "#ff7f00", "Vy": "#b5c900", "lL": "#00a087", "T0p": "#fdd9c4",
         "Tw": "#19a8e0", "hp": "#eeeeee", "kp": "#e377c2", "Rhop": "#1f6fe0", "Bp": "#7ee081", "Lp": "#0000cc", "Wp": "#ff8fa3", "B": "#ffa500", "L": "#1e90ff", "W": "#4b3621"}
DISPN = {"k": "$k_s$", "cp": "$c_{p_s}$", "rho": "$\\rho_s$", "Lfus": "$L$", "Ts": "$T_s$", "Tliq": "$T_l$", "hc": "$h_c$", "T0": "$T_0$", "P": "$P$", "Eta": "$\\eta$", "Rx": "$r_x$", "Ry": "$r_y$", "Rz": "$r_z$",
         "X0": "$x_0$", "Vx": "$V_x$", "TL": "$t_L$", "Y0": "$y_0$", "Vy": "$V_y$", "lL": "$l_L$", "T0p": "$T_{0_p}$", "Tw": "$T_w$", "hp": "$h_p$", "kp": "$k_p$", "Rhop": "$\\rho_p$",
         "Bp": "$b_p$", "Lp": "$l_p$", "Wp": "$w_p$", "B": "$b$", "L": "$l$", "W": "$w$"}
# perturbation each |dT/dp| is multiplied by, so that all 30 are compared in the same unit (K caused by a plausible change of that parameter).  Defaults, edit here:
#   relative-parameter derivatives (dT/dln p, K per 100%): 5 %;  Vx: 5 % (its derivative is also per unit relative speed);  X0: 10 um = 0.01 mm (derivative per mm);
#   TL: a 10 um shift of the path = 1.1e-5 s at 0.9 m/s (derivative per second of laser delay).
DELTA = {n: 0.05 for n in NAMES}; DELTA["X0"] = 0.01; DELTA["TL"] = 1.1e-5
COL = [mcolors.to_rgba(PAPER[n]) for n in NAMES]; DISP = [DISPN[n] for n in NAMES]
h = open(f"{R}/results/deriv_first_order/d1_sens.csv").readline().strip().split(","); d = np.genfromtxt(f"{R}/results/deriv_first_order/d1_sens.csv", delimiter=",", skip_header=1)
nom = np.genfromtxt(f"{R}/results/nominal/nominal_probe.csv", delimiter=",", skip_header=1)
t = (d[:, 0] - T0W) * 1e3; tn = (nom[:, 0] - T0W) * 1e3; Tp = nom[:, 1]; TEND = t.max()
A = np.abs(np.stack([d[:, h.index(f"dT_d{n}_row1")] * DELTA[n] for n in NAMES], 1)); share = A / np.maximum(A.sum(1, keepdims=True), 1e-300)
order = np.argsort(-share.mean(0)); TF = np.linspace(t[0], t[-1], 1600)
SH = np.clip(PchipInterpolator(t, share, axis=0)(TF), 0, None); SH = gaussian_filter1d(SH, sigma=5, axis=0, mode="nearest"); SH /= np.maximum(SH.sum(1, keepdims=True), 1e-300)
print("30 parameters;", int((A.max(0) == 0).sum()), "structurally zero:", ", ".join(DISP[k] for k in range(len(NAMES)) if A[:, k].max() == 0))
print("probe RMS ranking (K per perturbation):", ", ".join(f"{DISP[k]}={np.sqrt((A[:, k] ** 2).mean()):.0f}" for k in np.argsort(-(A ** 2).mean(0))[:8]))
f, ax = plt.subplots(figsize=(10.5, 8.8))
ax.stackplot(TF, SH[:, order].T, colors=[COL[i] for i in order], linewidth=0, edgecolor="none")
ax.set_xlim(0, TEND); ax.set_ylim(0, 1); ax.set_yticks([0, 1]); ax.set_yticklabels(["0%", "100%"]); ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xticklabels(["0", "1", "2", "3", "4"])
ax.set_ylabel("Cumulative\nRanked\nSensitivities", fontstyle="italic"); ax.yaxis.set_label_coords(-0.045, 0.5); ax.set_xlabel("$t$ [ms]", fontstyle="italic")
ax2 = ax.twinx(); ax2.plot(tn, Tp, color="black", lw=4.5, solid_capstyle="round"); ax2.set_xlim(0, TEND)
Tlo = 300.0; Thi = float(np.ceil((Tp.max() + 100) / 100) * 100); ax2.set_ylim(Tlo, Thi); ax2.set_yticks([Tlo, 1000, 2000, Thi]) if Thi > 2150 else ax2.set_yticks([Tlo, 1000, Thi])
ax2.set_ylabel("$\\boldsymbol{T}$ [K]", fontweight="bold", fontstyle="italic", rotation=90, va="bottom", labelpad=22)
for a in (ax, ax2):
    a.tick_params(width=2, length=6)
    for sp in a.spines.values(): sp.set_linestyle((0, (3.5, 2.5))); sp.set_color("0.45"); sp.set_linewidth(3.0)
f.legend(handles=[Patch(facecolor=COL[i], edgecolor="black", linewidth=1.2, label=DISP[i]) for i in order], loc="lower center", ncol=8, frameon=False, fontsize=20, handlelength=1.2, handleheight=1.0, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.6)
f.tight_layout(rect=(0.0, 0.20, 1, 1)); f.savefig(f"{R}/figures/probe_ranked_sensitivities.png", dpi=200); f.savefig(f"{R}/figures/probe_ranked_sensitivities.pdf"); plt.close(f); print("written", f"{R}/figures/probe_ranked_sensitivities.png", "T axis", Tlo, Thi)
