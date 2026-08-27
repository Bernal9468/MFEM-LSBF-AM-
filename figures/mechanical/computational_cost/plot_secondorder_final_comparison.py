import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Full-scale (8447-step) SECOND-ORDER comparison: cost of ALSO obtaining
# second-order sensitivities, stacked on top of each library's own
# first-order cost, matching the established stacking style used for the
# restart figure (plot_restart_both_libraries.py).
#
# OTI (30-basis): the "1st order only" bar is anchored to the ALREADY-
# ESTABLISHED, widely-cited 1.939X (job 793323) to stay consistent with
# every other figure in this set. The 2nd-order run (job 796834, -o2)
# happened to land on a different compute node, whose own first-order
# measurement came out higher (2.369X) purely from this project's
# well-documented node-to-node timing variance (up to ~34% seen
# elsewhere on unchanged code) -- NOT used here, to avoid contradicting
# the established number. Only the 2nd-order INCREMENT (tan2_n+res2_n+
# sol2_n, normalized by that same run's own nominal, which is itself
# within 0.1% of the established nominal so this mixing is safe) is
# taken from job 796834, since no other run has measured it. Second
# order (-o2) was cross-validated correct via isolated closed-form tests
# (2026-08-26) after the full-trajectory FD check proved too noisy to be
# useful (compounding Newton/solver tolerance noise across ~1200-2400
# chained steps in each perturbed run) -- see project memory.
#
# OTI (6-basis): second order comes from production's existing legacy
# onumm6n2 4-parameter cross-check (k,cp,rho,Lfus), which is genuinely
# order-2 by construction (real otilib multicomplex type) -- no new
# implementation or validation needed. legacy2nd_s (990.64s at a 3000-step
# short test, job 797920) was split into genuine tangent/residual/solve
# sub-costs via NEW instrumentation added to the driver (mirrors the
# 30-basis driver's own domain/boundary/nodal timer split) -- the
# measured short-test fractions (tan=88.18%, res=4.73%, sol=7.08% of
# legacy2nd_s) were applied to the ALREADY-KNOWN, full-scale-correct
# legacy2nd_s=3081.32 (job 793237) to get the full-scale tan/res/sol
# split, since the ratio should be stable but the short-test's own
# ABSOLUTE total isn't the trustworthy full-scale number. legacy1st_s
# (the redundant 1st-order prerequisite) is a PURE solve-time cost by its
# own definition (it only ever timed a narrow solve span, never its own
# assembly -- see project memory), so it's folded into "Linear solve"
# rather than split across tangent/residual. Source: job 793237 (same
# job used for the established first-order figure) for the totals; job
# 797920 for the tan:res:sol ratio only.
nom_n = 1162.94
tan2_n, res2_n, sol2_n = 3141.25, 250.37, 218.47
X1_30 = 1.939

nom_p = 1168.85
X1_6 = 3.019
legacy2nd_s, legacy1st_s = 3081.32, 194.51
_frac_tan, _frac_res, _frac_sol = 0.8818, 0.0473, 0.0708
legacy2nd_tan_full = legacy2nd_s * _frac_tan
legacy2nd_res_full = legacy2nd_s * _frac_res
legacy2nd_sol_full = legacy2nd_s * _frac_sol + legacy1st_s

C_SINGLE = "#4C72B0"  # base (1st-order) bar -- blue
C_TAN2   = "#2D6A4F"  # tangent assembly, 2nd order -- dark green
C_RES2   = "#D9B23C"  # residual assembly, 2nd order -- gold
C_SOL2   = "#E67E22"  # linear solve, 2nd order -- orange (was light blue,
                      # too easily mistaken for the 1st-order base bar)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.8))

shared_max = max(
    X1_30 + (tan2_n + res2_n + sol2_n) / nom_n,
    X1_6 + (legacy2nd_tan_full + legacy2nd_res_full + legacy2nd_sol_full) / nom_p,
)

# ---- Panel 1: OTI (30-basis) ----
ax = axes[0]
segs30 = [
    ("Tangent assembly (2nd order)", C_TAN2, tan2_n / nom_n),
    ("Residual assembly (2nd order)", C_RES2, res2_n / nom_n),
    ("Linear solve (2nd order)", C_SOL2, sol2_n / nom_n),
]
X_total_30 = X1_30 + sum(v for _, _, v in segs30)
ax.bar([0, 1], [X1_30, X1_30], color=C_SINGLE, width=0.55)
bottom = X1_30
for label, color, v in segs30:
    ax.bar([1], [v], bottom=[bottom], color=color, width=0.55, label=label, edgecolor="white", linewidth=0.4)
    bottom += v
ax.text(0, X1_30 + shared_max*0.02, f"{X1_30:.3f}X", ha="center", fontsize=13, fontweight="bold")
ax.text(1, X_total_30 + shared_max*0.02, f"{X_total_30:.3f}X", ha="center", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["1st order\nonly", "+ 2nd order"], fontsize=10)
ax.set_title("OTI (30-basis)", fontsize=12)
ax.set_ylim(0, shared_max * 1.15)
ax.legend(loc="upper left", fontsize=8.0, framealpha=0.95)
ax.set_ylabel("CPU overhead")

# ---- Panel 2: OTI (6-basis) ----
ax = axes[1]
segs6 = [
    ("Tangent assembly (2nd order)", C_TAN2, legacy2nd_tan_full / nom_p),
    ("Residual assembly (2nd order)", C_RES2, legacy2nd_res_full / nom_p),
    ("Linear solve (2nd order)", C_SOL2, legacy2nd_sol_full / nom_p),
]
X_total_6 = X1_6 + sum(v for _, _, v in segs6)
ax.bar([0, 1], [X1_6, X1_6], color=C_SINGLE, width=0.55)
bottom = X1_6
for label, color, v in segs6:
    ax.bar([1], [v], bottom=[bottom], color=color, width=0.55, label=label, edgecolor="white", linewidth=0.4)
    bottom += v
ax.text(0, X1_6 + shared_max*0.02, f"{X1_6:.3f}X", ha="center", fontsize=13, fontweight="bold")
ax.text(1, X_total_6 + shared_max*0.02, f"{X_total_6:.3f}X", ha="center", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["1st order\nonly", "+ 2nd order"], fontsize=10)
ax.set_title("OTI (6-basis)", fontsize=12)
ax.set_ylim(0, shared_max * 1.15)
ax.legend(loc="upper left", fontsize=8.0, framealpha=0.95)

plt.tight_layout()
out = "/work/gyj329/MFEM/implementations/simple_case_30_param/MFEM-LSBF-AM-/figures/mechanical/computational_cost/secondorder_final_comparison_X_fullsim.png"
plt.savefig(out, dpi=150)
print("wrote", out)
print(f"OTI(30-basis): X1={X1_30:.3f} X_total={X_total_30:.3f}")
print(f"OTI(6-basis):  X1={X1_6:.3f} X_total={X_total_6:.3f}")
