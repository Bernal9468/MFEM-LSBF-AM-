import re, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "results/timing_benchmark/"
def w(n): return float(re.search(r"WALL_S\s+([\d.]+)", open(D + n + ".time").read()).group(1))
cats = ["First-order", "Second-order diagonal"]
keys = ["d1", "d2diag"]
OTI, ENZ, BLUE, RED = "#c0392b", "#2a6fc9", None, None
plt.rcParams.update({"font.size": 14, "font.weight": "bold", "axes.labelweight": "bold"})
for thr in (16, 40):
    on, en = w(f"oti_nominal_t{thr}"), w(f"enz_nominal_t{thr}")
    oti = [w(f"oti_{k}_t{thr}") / on for k in keys]
    enz = [w(f"enz_{k}_t{thr}") / en for k in keys]
    full = w(f"enz_d2full_t{thr}") / en
    # overhead
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white"); x = np.arange(3); bw = 0.36
    ov_o = oti + [np.nan]; ov_e = enz + [full]
    for xs, v, c, l in ((x - bw/2, ov_o, OTI, "OTI/HYPAD"), (x + bw/2, ov_e, ENZ, "Enzyme/HYPAD")):
        ax.bar(xs, v, bw, color=c, label=l)
        for xi, vi in zip(xs, v):
            if not np.isnan(vi): ax.text(xi, vi, f"{vi:.1f}x", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(cats + ["Full Hessian"])
    ax.set_ylabel("CPU overhead vs nominal, x"); ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"figures/03_cpu_overhead_t{thr}.png", dpi=170); plt.close(fig)
    # speed-up
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white"); x = np.arange(2)
    sp = [o / e for o, e in zip(oti, enz)]
    ax.bar(x, sp, 0.5, color=ENZ)
    for xi, v in zip(x, sp): ax.text(xi, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=13)
    ax.axhline(1, color="k", lw=3.5)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("Speed-up of Enzyme/HYPAD over OTI/HYPAD, x")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"figures/04_speedup_t{thr}.png", dpi=170); plt.close(fig)
    print(thr, "overhead OTI", oti, "Enz", enz, "full", full, "speedup", sp)
