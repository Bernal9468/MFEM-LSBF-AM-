import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Full window (4400 steps, t=1.2522..1.2566), optimised drivers, 16 threads, exclusive node. Hours = sum of per-step walls in the logs.
# Style copied from 07_fullwindow_overhead_speedup_residual_style.png. Series names as in that reference figure.
NAME_OTI, NAME_ENZ = "OTI/HYPAD", "Enzyme/RESIDUAL"
H = {NAME_OTI: {"nom": 1.3572, "d1": 3.0840, "full": 15.3169}, NAME_ENZ: {"nom": 1.3347, "d1": 2.9231, "full": 7.2030}}
# extra-only wall time: first order = d1 - nominal ; second order = full - d1
ext = lambda m: [H[m]["d1"] - H[m]["nom"], H[m]["full"] - H[m]["d1"]]
cats = ["First-order (extra)", "Second-order (extra)"]; keys = ["d1", "full"]
ORANGE, BLUE, GREEN = "#F28E2B", "#287FB8", "#1e8449"
plt.rcParams.update({"font.size": 16, "font.weight": "normal", "axes.labelweight": "normal"})
x = np.arange(2); bw = 0.3; YMAX = 14.0; RATIO = 6.5
fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
for si, (m, c) in enumerate(((NAME_OTI, ORANGE), (NAME_ENZ, BLUE))):
    v = ext(m); xs = x + (si - 0.5) * bw
    ax.bar(xs, v, bw, color=c, edgecolor="black", lw=2, label=f"{m} wall time")
    for xi, vi in zip(xs, v): ax.text(xi, vi, f"{vi:.1f} h", ha="center", va="bottom", fontsize=13)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("Wall time [h]"); ax.set_ylim(0, YMAX)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
sp = [a / b for a, b in zip(ext(NAME_OTI), ext(NAME_ENZ))]
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, frameon=False, loc="upper left", fontsize=16)
fig.tight_layout(); fig.savefig("figures/12_fullwindow_walltime_extra_only.png", dpi=170)
for m in H: print(m, ["%.2f h" % v for v in ext(m)])
print("speed-up", ["%.2fx" % s for s in sp])
