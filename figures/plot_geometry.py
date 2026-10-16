import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection, LineCollection
B = "/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme"
L = open(f"{B}/inputs/full_bridge.mesh").read().split("\n")
i = L.index("elements"); ne = int(L[i + 1]); E = [l.split() for l in L[i + 2:i + 2 + ne]]
v = L.index("vertices"); nv = int(L[v + 1]); V = np.array([[float(t) for t in l.split()] for l in L[v + 3:v + 3 + nv]])
bb = L.index("boundary"); nb = int(L[bb + 1]); BF = np.array([[int(t) for t in l.split()[2:6]] for l in L[bb + 2:bb + 2 + nb]])
attr = np.array([int(e[0]) for e in E]); C = np.array([[int(t) for t in e[2:]] for e in E])
lo = np.array([V[c].min(0) for c in C]); hi = np.array([V[c].max(0) for c in C])
SUB, BR, PATH, PROBE = "#9aa5b1", "#d98c1f", "#c0392b", "#1f4e9c"
plt.rcParams.update({"font.size": 14, "font.weight": "bold", "axes.labelweight": "bold"})

# ---- figure 1: 3D geometry built from the real boundary faces of the bridge elements (z of the bridge drawn x6 so it is visible; ticks are true mm) ----
ZE = 6.0
# exterior faces of the bridge elements = faces owned by exactly one bridge element (the free-surface list in the mesh file omits the top faces)
HF = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
bel = np.where(attr == 2)[0]
fk = {}
for e in bel:
    for f in HF:
        key = tuple(sorted(C[e, list(f)])); fk[key] = fk.get(key, 0) + 1
quads = []; cols = []
for e in bel:
    for f in HF:
        ids = C[e, list(f)]
        if fk[tuple(sorted(ids))] != 1: continue
        P = V[ids].copy()
        if P[:, 2].max() < 1e-9: continue            # interface with the substrate
        flat_top = (P[:, 2].max() - P[:, 2].min() < 1e-9)
        P[:, 2] *= ZE; quads.append(P); cols.append("#e9a13b" if flat_top else "#a8680f")
def box(x0, x1, y0, y1, z0, z1, col):
    P = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    F = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    return Poly3DCollection([P[f] for f in F], facecolor=col, edgecolor="k", linewidth=1.2)
fig = plt.figure(figsize=(12, 6.5), facecolor="white"); ax = fig.add_subplot(111, projection="3d"); ax.computed_zorder = False
sub = box(-7.5, 82.5, -5, 10, -12.7, 0, "#c9d1d9"); sub.set_zorder(1); ax.add_collection3d(sub)
br = Poly3DCollection(quads, facecolor=cols, edgecolor="none"); br.set_zorder(2); ax.add_collection3d(br)
zt = 0.2 * ZE + 0.05
ax.set_xlim(-7.5, 82.5); ax.set_ylim(-5, 10); ax.set_zlim(-12.7, 6)
ax.set_box_aspect((90, 15 * 2.4, 12.7 * 2.2))
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.tick_params(axis="x", pad=2); ax.tick_params(axis="y", pad=2); pass
ax.set_axis_off()
ax.view_init(elev=24, azim=-38)
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis): a_.pane.set_facecolor("white"); a_.pane.set_edgecolor("white")
ax.grid(False)
import matplotlib.patches as mp
H = [mp.Patch(facecolor="#c9d1d9", edgecolor="k", label="Substrate  90 x 15 x 12.7 mm"), mp.Patch(facecolor="#e9a13b", edgecolor="k", label="Bridge layer  75 x 5 x 0.2 mm")]
fig.legend(handles=H, loc="upper right", frameon=False, fontsize=14, bbox_to_anchor=(0.98, 0.98))
# coordinate triad with its origin in the upper-left corner (directions taken from the actual 3D projection)
from mpl_toolkits.mplot3d import proj3d
fig.subplots_adjust(left=0, right=1, bottom=0, top=1); fig.canvas.draw()
M = ax.get_proj(); lim = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
def scr(pt):
    x2, y2, _ = proj3d.proj_transform(pt[0], pt[1], pt[2], M)
    return np.array(fig.transFigure.inverted().transform(ax.transData.transform((x2, y2))))
c0 = np.array([np.mean(l) for l in lim]); p0 = scr(c0); dirs = []
for k in range(3):
    e = c0.copy(); e[k] += 0.3 * (lim[k][1] - lim[k][0]); d = scr(e) - p0
    dirs.append(d / np.hypot(*d))
Lt = 0.085; O = np.array([0.075, 0.74])
tips = [O + Lt * d for d in dirs]
allx = [t[0] for t in tips] + [O[0]]; ally = [t[1] for t in tips] + [O[1]]
O = O + np.array([max(0.0, 0.045 - min(allx)), min(0.0, 0.95 - max(ally))])
for d, lab in zip(dirs, ("x [mm]", "y [mm]", "z [mm]")):
    tip = O + Lt * d
    fig.add_artist(matplotlib.patches.FancyArrowPatch(O, tip, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, lw=3, color="k"))
    fig.text(tip[0] + 0.012 * np.sign(d[0] if abs(d[0]) > 0.2 else 1), tip[1] + 0.012 * np.sign(d[1] if abs(d[1]) > 0.2 else 1), lab, fontsize=15, ha="left" if d[0] >= 0 else "right", va="bottom" if d[1] >= 0 else "top")
fig.savefig(f"{B}/figures/09_geometry_3d.png", dpi=170); plt.close(fig)

# ---- figure 1b: top view of the whole bridge layer ----
bot = np.where((attr == 2) & (lo[:, 2] < 0.05))[0]
q = [[(lo[e, 0], lo[e, 1]), (hi[e, 0], lo[e, 1]), (hi[e, 0], hi[e, 1]), (lo[e, 0], hi[e, 1])] for e in bot]
fig, ax = plt.subplots(figsize=(16, 3.6), facecolor="white")
ax.add_collection(PolyCollection(q, facecolor=BR, edgecolor="none"))
ax.set_xlim(-1, 76); ax.set_ylim(-1.0, 5.5); ax.set_aspect("equal"); ax.set_xlabel("x, mm"); ax.set_ylabel("y, mm")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{B}/figures/10_bridge_topview.png", dpi=170); plt.close(fig)

# ---- figure 2: mesh close-ups (top view of the bridge start, side section through y = 0) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5), facecolor="white", gridspec_kw={"width_ratios": [1.15, 1]})
m = (attr == 2) & (np.abs(hi[:, 2] - 0.2) < 1e-5) & (lo[:, 0] < 8.5)
quads = [[(lo[e, 0], lo[e, 1]), (hi[e, 0], lo[e, 1]), (hi[e, 0], hi[e, 1]), (lo[e, 0], hi[e, 1])] for e in np.where(m)[0]]
a1.add_collection(PolyCollection(quads, facecolor="#f4d9a6", edgecolor="k", linewidth=0.4))
a1.set_xlim(-0.2, 8.4); a1.set_ylim(-0.2, 5.2); a1.set_aspect("equal"); a1.set_xlabel("x, mm"); a1.set_ylabel("y, mm")
s = (lo[:, 1] <= 1e-9) & (hi[:, 1] > 1e-9) & (hi[:, 0] > -8) & (lo[:, 0] < 9) & (hi[:, 2] > -4)
quads = [[(lo[e, 0], lo[e, 2]), (hi[e, 0], lo[e, 2]), (hi[e, 0], hi[e, 2]), (lo[e, 0], hi[e, 2])] for e in np.where(s)[0]]
cols = [BR if attr[e] == 2 else "#c9d1d9" for e in np.where(s)[0]]
a2.add_collection(PolyCollection(quads, facecolor=cols, edgecolor="k", linewidth=0.4))
a2.set_xlim(-8, 9); a2.set_ylim(-4, 0.6); a2.set_xlabel("x, mm"); a2.set_ylabel("z, mm")
for a in (a1, a2):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{B}/figures/11_mesh_closeup.png", dpi=170); plt.close(fig)
med = lambda a, k: np.median((hi[attr == a] - lo[attr == a])[:, k])
print("elements", ne, "nodes", nv, "bridge", (attr == 2).sum(), "substrate", (attr == 1).sum())
print("median element size bridge", [round(med(2, k), 4) for k in range(3)], "substrate", [round(med(1, k), 4) for k in range(3)])
