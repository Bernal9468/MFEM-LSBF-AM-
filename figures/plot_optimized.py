import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# 40-step window, 16 threads, exclusive node (opt/results/compare, compare_oti, diag). Nominal: Enzyme driver (OTI nominal measured equal to Enzyme within 1% in results/timing_benchmark).
nom_o, nom_n = 57.45, 54.06
W = {  # wall s: (original, optimised)
 "OTI/HYPAD":    {"d1": (274.84, 113.88), "d2": (1494.66, 154.07), "full": (None, 489.23)},
 "Enzyme/HYPAD": {"d1": (126.46, 107.14), "d2": (198.78, 140.34), "full": (448.15, 218.93)},
}
cats = ["First-order", "Second-order diagonal", "Full Hessian"]; keys = ["d1", "d2", "full"]
RED, LRED, BLUE, LBLUE = "#c0392b", "#e6a39b", "#2a6fc9", "#a7c3ec"
plt.rcParams.update({"font.size": 14, "font.weight": "bold", "axes.labelweight": "bold"})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
# overhead
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white"); x = np.arange(3); bw = 0.2
series = [("OTI/HYPAD", 0, LRED, "OTI/HYPAD original"), ("OTI/HYPAD", 1, RED, "OTI/HYPAD optimized"),
          ("Enzyme/HYPAD", 0, LBLUE, "Enzyme/HYPAD original"), ("Enzyme/HYPAD", 1, BLUE, "Enzyme/HYPAD optimized")]
for si, (m, v, c, lab) in enumerate(series):
    vals = [W[m][k][v] / (nom_o if v == 0 else nom_n) if (k in W[m] and W[m][k][v] is not None) else np.nan for k in keys]
    xs = x + (si - 1.5) * bw
    ax.bar(xs, vals, bw, color=c, label=lab)
    for xi, vi in zip(xs, vals):
        if not np.isnan(vi): ax.text(xi, vi, f"{vi:.1f}x", ha="center", va="bottom", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("CPU overhead vs nominal, x"); ax.legend(frameon=False, loc="upper left"); clean(ax)
fig.tight_layout(); fig.savefig("figures/05_overhead_original_vs_optimized.png", dpi=170); plt.close(fig)
# speed-up of the optimisation
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white"); bw = 0.36
for si, (m, c) in enumerate((("OTI/HYPAD", RED), ("Enzyme/HYPAD", BLUE))):
    vals = [W[m][k][0] / W[m][k][1] if (k in W[m] and W[m][k][0] is not None) else np.nan for k in keys]
    xs = x + (si - 0.5) * bw
    ax.bar(xs, vals, bw, color=c, label=m)
    for xi, vi in zip(xs, vals):
        if not np.isnan(vi): ax.text(xi, vi, f"{vi:.1f}x", ha="center", va="bottom", fontsize=12)
ax.axhline(1, color="k", lw=3.5); ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("Speed-up optimized over original, x"); ax.legend(frameon=False, loc="upper left"); clean(ax)
fig.tight_layout(); fig.savefig("figures/06_speedup_optimized_over_original.png", dpi=170); plt.close(fig)
for m in W:
    for k in W[m]: print(m, k, W[m][k], "opt overhead %.1fx" % (W[m][k][1]/nom_n))
