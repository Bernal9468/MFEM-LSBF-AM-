#!/usr/bin/env python3
"""Sensitivity-colour / temperature videos for the single-layer full bridge (cropped around the first block, where the laser runs in the 4.4 ms window).
usage: make_bridge_video.py <deriv|temp> <dump dir> <out.mp4> [fps=20] [nproc=8] [stride=1] [tmax_ms=1e9]
 deriv: 3D + top view coloured by the highest-ranked parameter (|dT/dp|, 16 physical params; X0,Vx,TL excluded; P and Eta merged) at each location, probe-history stack (panel a) with the nominal
        temperature as a black line and a moving time cursor, colour legend below.  Same notation/colours as the tiny-patch videos (paper Fig. 6).
 temp : 3D + top view coloured by temperature (Abaqus-style 12 bands, 300-2400 K), laser marker, time and T max.
 dumps: f_<step>.bin from src_dump/fbke_enzyme (FIELD_DUMP_DIR): T[ndof], dT/dp_i [20][ndof] (relative-parameter derivatives, LI_ order k,cp,rho,Lfus,Ts,Tliq,hc,T0,P,Eta,Rx,Ry,Rz,X0,Vx,TL,Tw,hp,kp,Rhop).
 crop box and probe node are set below (CROP, PROBE)."""
import sys, os, glob, subprocess, shutil, numpy as np
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
mode, dumpdir, out = sys.argv[1:4]; FPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20; NP = int(sys.argv[5]) if len(sys.argv) > 5 else 8
STRIDE = int(sys.argv[6]) if len(sys.argv) > 6 else 1; TMAXMS = float(sys.argv[7]) if len(sys.argv) > 7 else 1e9
R = "/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme"; MESH = f"{R}/inputs/full_bridge.mesh"; LASER = f"{R}/inputs/es_bridge_laser.inp"
T0W = 1.2522; CROP = (-1.5, 8.5, -2.0, 7.0, -1.5); PROBE = 193323; ZTOP = 0.2; SMOOTH_PX = 5.0; GNX = 760
# all 30 parameters of the problem (paper Table 1 set); the dump holds the 20 wired ones (LI_ order k..TL, Tw, hp, kp, Rhop); the 10 geometry ones (Y0, Vy, lL, T0p, Bp, Lp, Wp, B, L, W) are structurally zero
NAMES = ["k", "cp", "rho", "Lfus", "Ts", "Tliq", "hc", "T0", "P", "Eta", "Rx", "Ry", "Rz", "X0", "Vx", "TL", "Y0", "Vy", "lL", "T0p", "Tw", "hp", "kp", "Rhop", "Bp", "Lp", "Wp", "B", "L", "W"]
SRC30 = list(range(16)) + [20, 21, 22, 23]                    # dump column i (0..19) -> position in NAMES
PAPER = {"k": "#2f6e9f", "cp": "#2e8b2e", "rho": "#c8202a", "Lfus": "#00d400", "Ts": "#ff3b3b", "Tliq": "#e69f00", "hc": "#00e5ee", "T0": "#ff00ff", "P": "#a65628", "Eta": "#f781bf",
         "Rx": "#e41a1c", "Ry": "#3a7bb5", "Rz": "#4daf4a", "X0": "#9c3fb5", "Vx": "#ffff33", "TL": "#999999", "Y0": "#ff7f00", "Vy": "#b5c900", "lL": "#00a087", "T0p": "#fdd9c4",
         "Tw": "#19a8e0", "hp": "#eeeeee", "kp": "#e377c2", "Rhop": "#1f6fe0", "Bp": "#7ee081", "Lp": "#0000cc", "Wp": "#ff8fa3", "B": "#ffa500", "L": "#1e90ff", "W": "#4b3621"}
DISPN = {"k": "$k_s$", "cp": "$c_{p_s}$", "rho": "$\\rho_s$", "Lfus": "$L$", "Ts": "$T_s$", "Tliq": "$T_l$", "hc": "$h_c$", "T0": "$T_0$", "P": "$P$", "Eta": "$\\eta$", "Rx": "$r_x$", "Ry": "$r_y$", "Rz": "$r_z$",
         "X0": "$x_0$", "Vx": "$V_x$", "TL": "$t_L$", "Y0": "$y_0$", "Vy": "$V_y$", "lL": "$l_L$", "T0p": "$T_{0_p}$", "Tw": "$T_w$", "hp": "$h_p$", "kp": "$k_p$", "Rhop": "$\\rho_p$",
         "Bp": "$b_p$", "Lp": "$l_p$", "Wp": "$w_p$", "B": "$b$", "L": "$l$", "W": "$w$"}
# perturbation each |dT/dp| is multiplied by so that all 30 are comparable in K (defaults; same table as figures/plot_probe_ranked_sensitivities.py):
# relative-parameter derivatives 5 %, Vx 5 %, X0 10 um (= 0.01 mm), TL a 10 um path shift (1.1e-5 s at 0.9 m/s).
DELTA = np.array([{"X0": 0.01, "TL": 1.1e-5}.get(n, 0.05) for n in NAMES]); NP_ = len(NAMES)
PCOL = [mcolors.to_rgba(PAPER[n]) for n in NAMES]; DISP = [DISPN[n] for n in NAMES]; cmap_cat = mcolors.ListedColormap(PCOL)
def a30(D, idx=None):
    """|dT/dp| * DELTA for all 30 parameters (rows), at all nodes or at the node list idx; zero rows for the structurally-zero geometry parameters; P wins exact P/Eta ties."""
    cols = slice(None) if idx is None else idx
    ncol = D.shape[1] if idx is None else (1 if np.isscalar(idx) else len(idx))
    out = np.zeros((NP_, ncol)); out[SRC30] = np.abs(D[:, cols]).reshape(len(SRC30), ncol) * DELTA[SRC30][:, None]; out[NAMES.index("Eta")] *= (1 - 1e-9)
    return out
# ---------------------------------------------------------------- mesh, crop, boundary of the cropped solid
L_ = open(MESH).read().split("\n"); iv = L_.index("vertices"); nv = int(L_[iv + 1]); ie = L_.index("elements"); ne = int(L_[ie + 1])
V = np.array([l.split() for l in L_[iv + 3: iv + 3 + nv]], float); EL = np.array([l.split() for l in L_[ie + 2: ie + 2 + ne]], int); ATT = EL[:, 0]; E = EL[:, 2:10]; del L_
cen = V[E].mean(1); X0, X1, Y0, Y1, Z0 = CROP
sub = np.where((cen[:, 0] >= X0) & (cen[:, 0] <= X1) & (cen[:, 1] >= Y0) & (cen[:, 1] <= Y1) & (cen[:, 2] >= Z0))[0]
HEXF = np.array([(3, 2, 1, 0), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)])
FN = E[sub][:, HEXF].reshape(-1, 4); FE = np.repeat(sub, 6); FI = np.tile(np.arange(6), len(sub)); key = np.sort(FN, 1)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); inv = inv.ravel(); bnd = cnt[inv] == 1
B = FN[bnd]; BF = FI[bnd]; BE = FE[bnd]                                                       # exterior faces of the cropped solid
_zf = V[B][:, :, 2]; _top = (np.ptp(_zf, axis=1) < 1e-9) & (_zf[:, 0] > cen[BE][:, 2])                       # top (+z) faces, found geometrically (local face numbering of this mesh is not MFEM's)
UP = B[_top]; UPE = BE[_top]
LAYER_TOP = UPE[:, None] * 0 + (ATT[UPE] == 2)[:, None]
print(f"crop: {len(sub)} elements, {len(B)} exterior faces, {len(UP)} top faces", flush=True)
# ---------------------------------------------------------------- top-view pixel grid (axis-aligned quads, bilinear weights)
XLO, XHI, YLO, YHI = X0, X1, Y0, Y1; GNY = int(round(GNX * (YHI - YLO) / (XHI - XLO)))
gx = XLO + (np.arange(GNX) + 0.5) * (XHI - XLO) / GNX; gy = YLO + (np.arange(GNY) + 0.5) * (YHI - YLO) / GNY
PIXIN = np.zeros(GNX * GNY, bool); PIXN = np.zeros((GNX * GNY, 4), int); PIXW = np.zeros((GNX * GNY, 4)); dxp = (XHI - XLO) / GNX; dyp = (YHI - YLO) / GNY
for q in UP:
    p = V[q]; x0, x1, y0, y1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
    c = lambda xx, yy: q[int(np.argmin((p[:, 0] - xx) ** 2 + (p[:, 1] - yy) ** 2))]
    n00, n10, n11, n01 = c(x0, y0), c(x1, y0), c(x1, y1), c(x0, y1)
    i0 = int(np.ceil((x0 - XLO) / dxp - 0.5)); i1 = int(np.floor((x1 - XLO) / dxp - 0.5)); j0 = int(np.ceil((y0 - YLO) / dyp - 0.5)); j1 = int(np.floor((y1 - YLO) / dyp - 0.5))
    for j in range(max(j0, 0), min(j1, GNY - 1) + 1):
        ii = np.arange(max(i0, 0), min(i1, GNX - 1) + 1)
        if len(ii) == 0: continue
        k_ = j * GNX + ii; u = (gx[ii] - x0) / (x1 - x0); v = (gy[j] - y0) / (y1 - y0)
        PIXIN[k_] = True; PIXN[k_] = (n00, n10, n11, n01); PIXW[k_] = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], 1)
# ---------------------------------------------------------------- 3D faces: NSUBF x NSUBF sub-faces with bilinear weights on the parent face nodes
NSUBF = 2 if len(B) < 12000 else 1
corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
SK = []; SAB = []
for k in range(len(B)):
    for i in range(NSUBF):
        for j in range(NSUBF):
            h_ = 1.0 / NSUBF; SK.append(k); SAB.append([(i * h_, j * h_), ((i + 1) * h_, j * h_), ((i + 1) * h_, (j + 1) * h_), (i * h_, (j + 1) * h_)])
SK = np.array(SK); SAB = np.array(SAB)
bil = lambda a, b, v: (1 - a) * (1 - b) * v[0] + a * (1 - b) * v[1] + a * b * v[2] + (1 - a) * b * v[3]
P3 = np.array([[bil(a, b, V[B[k]]) for (a, b) in SAB[s]] for s, k in enumerate(SK)])
Wsf = np.zeros((len(SK), 4))
for s in range(len(SK)):
    for (a, b) in SAB[s]: Wsf[s] += np.array([(1 - a) * (1 - b), a * (1 - b), a * b, (1 - a) * b]) / 4.0
FACES_NODES = B[SK]
# ---------------------------------------------------------------- outlines: crop box (visible edges) + the layer (attr 2) top outline
def box_edges(x0, x1, y0, y1, z0, z1):
    c = lambda x, y, z: (x, y, z); E_ = []
    for (a_, b_) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))): E_.append((c(*a_, z1), c(*b_, z1)))
    for (x_, y_) in ((x1, y0), (x1, y1), (x0, y0)): E_.append((c(x_, y_, z0), c(x_, y_, z1)))
    E_ += [(c(x1, y0, z0), c(x1, y1, z0)), (c(x1, y0, z0), c(x0, y0, z0))]
    return E_
zlo = V[B][:, :, 2].min(); ztop = V[B][:, :, 2].max()
BOX_E = box_edges(X0, X1, Y0, Y1, zlo, 0.0)
lt = UP[ATT[UPE] == 2]                                                                   # layer top faces: boundary edges of that set = outline of the layer
ed = np.concatenate([lt[:, [0, 1]], lt[:, [1, 2]], lt[:, [2, 3]], lt[:, [3, 0]]]); ek = np.sort(ed, 1); _, ei, ec = np.unique(ek, axis=0, return_inverse=True, return_counts=True); ei = ei.ravel()
LAY_SEG = ed[ec[ei] == 1]                                                                # (nseg, 2) node ids
LAY_E3 = [(tuple(V[a]), tuple(V[b])) for a, b in LAY_SEG]; LAY_E2 = [(V[a][:2], V[b][:2]) for a, b in LAY_SEG]
# ---------------------------------------------------------------- toolpath
LR = np.array([[float(v) for v in l.split(",")[:5]] for l in open(LASER).read().split("\n")[:12] if l.strip()])
def laser_xy(t):
    k = int(np.clip(np.searchsorted(LR[:, 0], t, side="right") - 1, 0, len(LR) - 2)); a = min(max((t - LR[k, 0]) / max(LR[k + 1, 0] - LR[k, 0], 1e-300), 0.0), 1.0)
    return LR[k, 1] + a * (LR[k + 1, 1] - LR[k, 1]), LR[k, 2] + a * (LR[k + 1, 2] - LR[k, 2]), LR[k, 4] > 0
# ---------------------------------------------------------------- dumps
def dump_files(d): return sorted(glob.glob(f"{d}/f_*.bin"), key=lambda f: int(os.path.basename(f)[2:-4]))
def read_dump(fn, der=True):
    with open(fn, "rb") as f:
        n, st, n1, n2, nm = np.fromfile(f, dtype=np.int32, count=5); t = np.fromfile(f, dtype=np.float64, count=1)[0]; T = np.fromfile(f, dtype=np.float64, count=n)
        D = np.fromfile(f, dtype=np.float64, count=n1 * n).reshape(n1, n) if (der and n1) else None
    return t, T, D
files = dump_files(dumpdir)
files = [f for f in files if (read_dump(f, False)[0] - T0W) * 1e3 <= TMAXMS + 1e-9][::STRIDE]
if mode == "deriv":
    nom = np.genfromtxt(f"{R}/results/nominal/nominal_probe.csv", delimiter=",", skip_header=1); tn = (nom[:, 0] - T0W) * 1e3; Tp = nom[:, 1]
    pt = []; pD = []
    for fn in files:
        t, T, D = read_dump(fn); pt.append((t - T0W) * 1e3); pD.append(a30(D, PROBE)[:, 0])
    pt = np.array(pt); pD = np.array(pD); share = pD / np.maximum(pD.sum(1, keepdims=True), 1e-300); order = np.argsort(-share.mean(0)); TF = np.linspace(pt[0], pt[-1], 1600)
    SH = np.clip(PchipInterpolator(pt, share, axis=0)(TF), 0, None); SH = gaussian_filter1d(SH, sigma=5, axis=0, mode="nearest"); SH /= np.maximum(SH.sum(1, keepdims=True), 1e-300)
    STK = SH[:, order].T; STK_COL = [PCOL[k] for k in order]; TEND = pt[-1]; LEG_ORDER = list(order)
ABQ = [(0, 16, 253), (0, 92, 252), (0, 169, 251), (0, 215, 244), (0, 209, 138), (0, 186, 71), (106, 207, 0), (200, 222, 0), (255, 208, 0), (255, 159, 1), (255, 94, 9), (255, 29, 3)]
cmap_T = mcolors.ListedColormap([tuple(c / 255.0 for c in r) for r in ABQ]); TMIN_, TMAX_ = 300.0, 2400.0; normT = mcolors.Normalize(TMIN_, TMAX_)
tmp = os.path.splitext(out)[0] + "_frames"; os.makedirs(tmp, exist_ok=True)
def draw(i):
    t, T, D = read_dump(files[i], mode == "deriv"); tms = (t - T0W) * 1e3; lx, ly, on = laser_xy(t)
    fig = plt.figure(figsize=(20, 9.5) if mode == "deriv" else (16, 7.2), dpi=80, facecolor="white")
    ax = fig.add_axes([0.005, 0.29, 0.50, 0.70] if mode == "deriv" else [0.10, 0.05, 0.50, 0.9], projection="3d")
    if mode == "deriv":
        amag = a30(D); cf = np.argmax(np.einsum("pfc,fc->pf", amag[:, FACES_NODES], Wsf), axis=0); fc = [PCOL[c] for c in cf]
    else:
        Tq = T[FACES_NODES]; fc = cmap_T(normT(np.einsum("fc,fc->f", Tq, Wsf)))
    ax.add_collection3d(Poly3DCollection(P3, facecolors=fc, edgecolors="none", linewidths=0, zorder=1))
    ax.add_collection3d(Line3DCollection(BOX_E + LAY_E3, colors="black", linewidths=1.6, zorder=5))
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_zlim(zlo, ztop + 1.0)
    ax.set_box_aspect((X1 - X0, Y1 - Y0, max((ztop - zlo + 1.0) * 1.2, 2.0)), zoom=0.9); ax.view_init(elev=32, azim=-58); ax.set_axis_off(); ax.computed_zorder = False
    ax.scatter([lx], [ly], [ztop + 0.45], marker="v", s=300, c="black" if on else "white", edgecolors="black", linewidths=2.2, depthshade=False, zorder=50)
    if mode == "deriv": ax.text2D(0.02, 0.96, "highest-ranked parameter at each location", transform=ax.transAxes, fontsize=14, fontweight="bold")
    a2 = fig.add_axes([0.52, 0.32, 0.46, 0.58] if mode == "deriv" else [0.64, 0.12, 0.34, 0.62])
    if mode == "deriv":
        best = np.full((GNY, GNX), -1.0); bidx = np.zeros((GNY, GNX), int)                                     # streaming argmax over the parameters (keeps memory small)
        for p in range(NP_):
            if not amag[p].any(): continue
            v = gaussian_filter((amag[p][PIXN] * PIXW).sum(1).reshape(GNY, GNX), sigma=SMOOTH_PX, mode="nearest"); m_ = v > best; best[m_] = v[m_]; bidx[m_] = p
        rgb = np.array(PCOL)[:, :3][bidx]
    else:
        tv = (T[PIXN] * PIXW).sum(1).reshape(GNY, GNX); rgb = cmap_T(normT(tv))[:, :, :3]
    rgb[~PIXIN.reshape(GNY, GNX)] = 1.0
    a2.imshow(rgb, extent=(XLO, XHI, YLO, YHI), origin="lower", interpolation="bilinear", zorder=1); a2.set_xlim(XLO, XHI); a2.set_ylim(YLO, YHI); a2.set_aspect("equal"); a2.set_axis_off()
    a2.add_patch(plt.Rectangle((XLO, YLO), XHI - XLO, YHI - YLO, fill=False, ec="black", lw=2.2, zorder=20))
    for (p_, q_) in LAY_E2: a2.plot([p_[0], q_[0]], [p_[1], q_[1]], color="black", lw=1.8, zorder=21, solid_capstyle="butt")
    a2.plot([lx], [ly], marker="v", ms=15, mfc="black" if on else "white", mec="black", mew=2.2, zorder=30); a2.set_title("top view", fontsize=14, loc="left")
    if mode == "deriv":
        fig.text(0.52, 0.945, f"t = {tms:6.3f} ms", fontsize=22, fontweight="bold"); fig.text(0.72, 0.947, "Laser ON" if on else "Laser OFF", fontsize=16, fontweight="bold", color="red" if on else "0.35")
        ap = fig.add_axes([0.055, 0.07, 0.52, 0.20]); ap.stackplot(TF, STK, colors=STK_COL, linewidth=0.0, edgecolor="none")
        ap.set_ylim(0, 1); ap.set_xlim(TF[0], TF[-1]); ap.set_yticks([0, 0.5, 1]); ap.set_yticklabels(["0%", "50%", "100%"], fontsize=12); ap.set_xlabel("t [ms]", fontsize=13); ap.set_ylabel("Cumulative ranked\nsensitivities", fontsize=12)
        at = ap.twinx(); at.plot(tn, Tp, color="black", lw=2.5); at.set_ylim(300, 2400); at.set_ylabel("T [K]", fontsize=13); at.tick_params(labelsize=11); ap.tick_params(labelsize=11)
        ap.axvline(tms, color="black", lw=2.2, ls="--"); ap.set_title("(a) probe: top midpoint of the first block", fontsize=13, loc="left")
        lg = fig.add_axes([0.645, 0.01, 0.35, 0.28]); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1); lg.text(0.0, 0.97, "Parameters (all 30; the geometry ones, y0 to w, are zero in this model)", fontsize=11, fontweight="bold", va="center")
        for rk, k in enumerate(LEG_ORDER):
            c_, r_ = divmod(rk, 6); x0 = c_ * 0.2; y0 = 0.84 - r_ * 0.15
            lg.add_patch(plt.Rectangle((x0, y0 - 0.045), 0.05, 0.10, facecolor=PCOL[k], edgecolor="black", lw=0.5)); lg.text(x0 + 0.065, y0 + 0.005, DISP[k], fontsize=13, va="center")
    else:
        lgx = fig.add_axes([0.0, 0.0, 0.12, 1.0]); lgx.set_xlim(0, 1); lgx.set_ylim(0, 1); lgx.axis("off"); y0, y1 = 0.20, 0.90; hb = (y1 - y0) / 12; LEV = np.linspace(TMIN_, TMAX_, 13)
        lgx.text(0.04, 0.95, "NT11", fontsize=18, fontweight="bold", family="monospace", va="center")
        for k in range(12): lgx.add_patch(plt.Rectangle((0.05, y0 + k * hb), 0.22, hb, facecolor=cmap_T(k), edgecolor="black", linewidth=0.6))
        for k in range(13): lgx.text(0.30, y0 + k * hb, f"{LEV[k]:+.3e}", fontsize=11.5, va="center", ha="left")
        fig.text(0.64, 0.92, f"t = {tms:6.3f} ms", fontsize=22, fontweight="bold"); fig.text(0.64, 0.875, f"T max (crop) = {T[np.unique(B)].max():6.0f} K", fontsize=16)
        fig.text(0.64, 0.835, "Laser ON" if on else "Laser OFF", fontsize=16, fontweight="bold", color="red" if on else "0.35")
    fig.savefig(f"{tmp}/fr_{i:05d}.png", dpi=80, facecolor="white"); plt.close(fig); return i
if __name__ == "__main__":
    print(len(files), "frames", mode, flush=True)
    with Pool(NP) as pl: pl.map(draw, range(len(files)), chunksize=2)
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/fr_%05d.png", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "20", out])
    shutil.rmtree(tmp); print("wrote", out)
