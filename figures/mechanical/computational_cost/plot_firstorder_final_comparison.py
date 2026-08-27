import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Full-scale (8447-step) first-order sensitivity cost comparison, both
# libraries on MKL PARDISO, both genuinely first-order-only. Bars show
# EXTRA cost only (nominal solve subtracted) -- X = extra / nominal,
# matching this project's own established convention and the reference
# paper's own (Rincon-Tabares et al.) reporting.
#
# Both runs confirmed bit-identical trajectories (8447 accepted / 16
# rejected steps, 355 reassemblies) and exactly ONE PARDISO factorize_s
# debug line each (symbolic-analysis reuse held for the whole run).
#
# *** IMPORTANT CAVEAT (2026-08-24) ***
# Every full-scale run this session landed on a DIFFERENT compute node on
# this shared, non-exclusive cluster. A same-driver, same-code check
# (production with vs without the jac30_s timer -- a PURELY ADDITIVE
# instrumentation change that cannot alter any other timed region) showed
# its OWN tangent-assembly category swing by ~34% between two nodes with
# no code difference at all. So the specific X values below carry real,
# currently-unquantified node-to-node uncertainty on top of the measured
# numbers -- they are the best available full-scale numbers, using each
# driver's most complete/correct instrumentation, but should NOT be read
# as more precise than that until a matched-node (or multi-repeat) rerun
# is done.

# ---- Production (6-basis label), job 793237, N=19 wired directions.
# NOW INCLUDES jac30_s (PARDISO factorize), previously silently missing
# from this driver's own reported total -- see project memory "FINAL fair
# full-simulation comparison" and the jac30_s fairness-fix section. ----
nom_p_parts = dict(reassembly=130.33, residual=219.97, jacsetup=1.90, cgsolve=815.18, datasave=1.47)
nom_p = sum(nom_p_parts.values())
tan_p, res_p, jac_p, solve_p = 2003.23, 88.73, 511.14, 925.29   # [FIRSTORDER30 BREAKDOWN], N=19
extra_p = tan_p + res_p + jac_p + solve_p
N_p = 19

# ---- New/compact (30-basis label), job 793323, N=16 wired directions,
# ALL 7 fixes (compact-basis + batched multi-RHS PARDISO + boundary
# geom_cache + hrad_q skip + domain loop-interchange + gdot-hoisting +
# epp-fill-gating). Fixes 6+7 alone gave -27.9% on tan_dom_s at full scale
# (LARGER than their own same-node-verified short-test result of -21.3%,
# consistent with the predicted mechanism: gdot-hoisting's benefit grows
# as more directions survive the early-exit later in the simulation). ----
nom_n_parts = dict(reassembly=129.51, residual=218.54, jacsetup=2.04, cgsolve=810.55, datasave=0.93)
nom_n = sum(nom_n_parts.values())
tan_n, res_n, jac_n, solve_n = 1351.98, 175.53, 509.27, 215.91   # [FIRSTORDER30 BREAKDOWN], N=16
extra_n = tan_n + res_n + jac_n + solve_n
N_n = 16

def X(x, nom): return x / nom

# Same category = same color across both bars -- both drivers now have
# PARDISO factorize broken out separately (production's jac30_s fix put it
# on equal footing with the new library's own jacsetup2_s).
C_TAN  = "#2D6A4F"  # tangent assembly (domain+boundary OTI derivative RHS)
C_RES  = "#D9B23C"  # residual / nodal capacitive+latent assembly
C_JAC  = "#7A5DC7"  # PARDISO factorize (Refactorize)
C_SOL  = "#3E8E7E"  # linear solve (PARDISO triangular solve)

fig, ax = plt.subplots(figsize=(8.6, 7.2))
bar_x = [0, 1]
bar_labels = ["OTI (6 basis)", "OTI (30 basis)"]

bottoms = [0.0, 0.0]
segs = [
    ("Tangent assembly",           C_TAN, X(tan_p, nom_p),   X(tan_n, nom_n)),
    ("Residual / nodal assembly",  C_RES, X(res_p, nom_p),   X(res_n, nom_n)),
    ("PARDISO factorize",          C_JAC, X(jac_p, nom_p),   X(jac_n, nom_n)),
    ("Linear solve (PARDISO)",     C_SOL, X(solve_p, nom_p), X(solve_n, nom_n)),
]
for label, color, vP, vN in segs:
    vals = [vP, vN]
    ax.bar(bar_x, vals, bottom=bottoms, color=color, width=0.55,
           label=label, edgecolor="white", linewidth=0.4)
    bottoms = [b + v for b, v in zip(bottoms, vals)]

for i, tot in enumerate(bottoms):
    ax.text(bar_x[i], tot + 0.04, f"{tot:.3f}X", ha="center", fontsize=13, fontweight="bold")

ax.set_xticks(bar_x); ax.set_xticklabels(bar_labels, fontsize=10.5)
ax.set_ylabel("CPU overhead")
ax.legend(loc="upper right", fontsize=8.6, framealpha=0.95)
ax.set_ylim(0, max(bottoms) * 1.30)
ax.set_xlim(-0.5, 1.5)

speedup = (extra_p/nom_p) / (extra_n/nom_n)

plt.tight_layout()
out = "/work/gyj329/MFEM/implementations/simple_case_30_param/MFEM-LSBF-AM-/figures/mechanical/computational_cost/firstorder_final_comparison_X_fullsim.png"
plt.savefig(out, dpi=150)
print("wrote", out)
print(f"Production (6-basis, N={N_p}): nom={nom_p:.2f}s extra={extra_p:.2f}s X={X(extra_p,nom_p):.3f}")
print(f"New (30-basis, N={N_n}):        nom={nom_n:.2f}s extra={extra_n:.2f}s X={X(extra_n,nom_n):.3f}")
print(f"Speedup (new vs production, extra-cost basis): {speedup:.3f}x")
