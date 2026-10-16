import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
B = "/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme"; R = B + "/residual_stress"
L = open(B + "/inputs/full_bridge.mesh").read().split("\n")
i = L.index("elements"); ne = int(L[i + 1]); E = np.array([l.split() for l in L[i + 2:i + 2 + ne]], dtype=np.int64)
v = L.index("vertices"); nv = int(L[v + 1]); V = np.array([l.split() for l in L[v + 3:v + 3 + nv]], dtype=float)
attr, conn = E[:, 0], E[:, 2:]; lo = np.array([V[c].min(0) for c in conn]); hi = np.array([V[c].max(0) for c in conn])
S = np.fromfile(R + "/results/mech_full/final_sigma_elem.bin").reshape(ne, 6)
p = np.loadtxt(R + "/results/mech_full/probe_hist.csv", delimiter=",")
names = ["E", "ν", "CTE", "σ_y", "H"]
plt.rcParams.update({"font.size": 16, "axes.linewidth": 1.5})
# 1. residual sigma_xx on the top layer of the first patch
top = np.where((attr == 2) & (hi[:, 2] > 0.19) & (lo[:, 0] < 8.5) & (hi[:, 0] < 8.5))[0]
quads = [[(lo[e, 0], lo[e, 1]), (hi[e, 0], lo[e, 1]), (hi[e, 0], hi[e, 1]), (lo[e, 0], hi[e, 1])] for e in top]
fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="white")
pc = PolyCollection(quads, array=S[top, 0], cmap="coolwarm", edgecolors="none"); vm = np.abs(S[top, 0]).max(); pc.set_clim(-vm, vm); ax.add_collection(pc)
ax.set_xlim(-0.2, 8.4); ax.set_ylim(-0.2, 5.2); ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
cb = fig.colorbar(pc, ax=ax, shrink=0.85); cb.set_label("Residual σ_xx [MPa]")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(R + "/figures/rs1_residual_sigma_xx_map.png", dpi=170); plt.close(fig)
# 2. probe stress history (time since the start of the window)
t = p[:, 2] - 1.2522
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
for k, (lab, c) in enumerate((("σ_xx", "#c0392b"), ("σ_yy", "#2a6fc9"), ("σ_zz", "#1e8449"))): ax.plot(t, p[:, 3 + k], color=c, lw=3.5, label=lab)
ax.set_xscale("log"); ax.set_xlabel("Time since laser start [s]"); ax.set_ylabel("Stress at the probe element [MPa]"); ax.legend(frameon=False)
ax.axvline(0.0044, color="k", lw=1.5, ls="--")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(R + "/figures/rs2_probe_stress_history.png", dpi=170); plt.close(fig)
# 3. first-order sensitivities of the final probe stress
d1 = np.array([[p[-1, 9 + 6 * j + k] for j in range(5)] for k in range(3)])
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white"); x = np.arange(5); bw = 0.26
for k, (lab, c) in enumerate((("σ_xx", "#c0392b"), ("σ_yy", "#2a6fc9"), ("σ_zz", "#1e8449"))):
    ax.bar(x + (k - 1) * bw, d1[k], bw, color=c, edgecolor="black", lw=1.5, label=lab)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("dσ / d(scale) [MPa]"); ax.legend(frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(R + "/figures/rs3_first_order_sensitivity.png", dpi=170); plt.close(fig)
# 4. second-order Hessian of sigma_xx at the probe (pairs ordered (0,0),(0,1),...,(4,4))
pairs = [(i, j) for i in range(5) for j in range(i, 5)]; Hm = np.zeros((5, 5))
for q, (i, j) in enumerate(pairs): Hm[i, j] = Hm[j, i] = p[-1, 39 + 6 * q]
fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor="white"); vmax = np.abs(Hm).max()
im = ax.imshow(Hm, cmap="coolwarm", vmin=-vmax, vmax=vmax)
for a in range(5):
    for b in range(5): ax.text(b, a, "%.0f" % Hm[a, b], ha="center", va="center", fontsize=14)
ax.set_xticks(range(5)); ax.set_xticklabels(names); ax.set_yticks(range(5)); ax.set_yticklabels(names)
cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("d²σ_xx / d(scale)² [MPa]")
fig.tight_layout(); fig.savefig(R + "/figures/rs4_second_order_hessian_sigma_xx.png", dpi=170); plt.close(fig)
print("probe final sigma:", np.round(p[-1, 3:9], 1)); print("first-order dsigma_xx:", dict(zip(names, np.round(d1[0], 1)))); print("Hessian sigma_xx:\n", np.round(Hm, 1))
