import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Full window (4400 steps, t=1.2522..1.2566), optimised drivers, 16 threads, exclusive node. Hours = sum of per-step walls.
H = {"OTI/HYPAD":    {"nom": 1.3572, "d1": 3.0840, "d2": 3.8916, "full": 15.3169},
     "Enzyme/HYPAD": {"nom": 1.3347, "d1": 2.9231, "d2": 3.9298, "full": 7.2030}}
cats = ["First-order", "Full Hessian"]; keys = ["d1", "full"]
RED, BLUE, GREEN = "#F28E2B", "#287FB8", "#1e8449"
plt.rcParams.update({"font.size": 16, "font.weight": "normal", "axes.labelweight": "normal"})
x = np.arange(2); bw = 0.3
fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
for si, (m, c) in enumerate((("OTI/HYPAD", RED), ("Enzyme/HYPAD", BLUE))):
    v = [H[m][k] / H[m]["nom"] for k in keys]; xs = x + (si - 0.5) * bw
    ax.bar(xs, v, bw, color=c, edgecolor="black", lw=2, label=f"{m.replace('HYPAD','RESIDUAL') if m.startswith('Enzyme') else m} overhead")
    for xi, vi in zip(xs, v): ax.text(xi, vi, f"{vi:.1f}×", ha="center", va="bottom", fontsize=13)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("CPU overhead"); ax.set_ylim(0, 13.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
# right axis: speed-up of Enzyme/HYPAD over OTI/HYPAD, in the same figure
ax2 = ax.twinx()
sp = [H["OTI/HYPAD"][k] / H["Enzyme/HYPAD"][k] for k in keys]
ax2.plot(x, sp, color=GREEN, lw=3.5, marker="o", ms=11, label="Speed-up Enzyme/HYPAD over OTI/HYPAD", zorder=5)
for xi, si_ in zip(x, sp):
    ax2.text(xi + 0.09, si_ + 0.07, f"{si_:.2f}×", ha="left", va="bottom", fontsize=13, color=GREEN,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=6)
ax2.set_ylim(0, 13.5 / 5.0); ax2.set_ylabel("Speed-up", color=GREEN)
ax2.tick_params(axis="y", colors=GREEN); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_color(GREEN)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1, l1, frameon=False, loc="upper left", fontsize=16)
fig.tight_layout(); fig.savefig("figures/07_fullwindow_overhead_speedup_residual_style.png", dpi=170)
for m in H: print(m, {k: "%.2fx" % (H[m][k] / H[m]["nom"]) for k in keys})
print("speed-up", ["%.2fx" % s for s in sp])
