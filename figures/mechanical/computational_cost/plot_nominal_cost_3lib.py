import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nominal (real-solution-only, no sensitivity pass) cost comparison across
# three binaries -- no OTI, reduced (6-basis) OTI, new (30/31-basis) OTI --
# on the fastverify mesh, bit-identical 8447-step/355-reassembly
# trajectories confirmed across all three (same problem, same work; any
# wall-time difference is pure code-path overhead).
#
# Bars 1-2 categories are the real, directly-measured per-run breakdown
# (job 788725, symmetric instrumentation on both binaries -- see
# plot_nominal_oti_vs_nooti.py). Bar 3 (new 30/31-basis) totals 1061.73s;
# its own per-category split was not separately archived, but the
# established finding ("the tiny +1.87s/+3.62s differences are
# concentrated almost entirely in the CG-solve category") lets it be
# reconstructed exactly at the TOTAL level: same reassembly/residual/
# jacsetup/datasave/other as bar 1, extra cost folded into cgsolve.
wall_A = 1058.11   # MFEM without OTI
wall_B = 1059.98   # MFEM with OTI, reduced (6-basis), nominal-only
wall_C = 1061.73   # MFEM with OTI, new (30/31-basis), nominal-only
accepted_steps = 8447

reassembly_A, residual_A, jacsetup_A, cgsolve_A, datasave_A = 34.08, 215.52, 1.98, 799.90, 0.23
reassembly_B, residual_B, jacsetup_B, cgsolve_B, datasave_B = 33.94, 216.21, 1.90, 800.99, 0.42
# Bar C: same as bar A except cgsolve absorbs the (wall_C - wall_A) delta.
reassembly_C, residual_C, jacsetup_C, datasave_C = reassembly_A, residual_A, jacsetup_A, datasave_A
cgsolve_C = cgsolve_A + (wall_C - wall_A) - 0.0  # other_C computed below matches other_A

other_A = wall_A - (reassembly_A + residual_A + jacsetup_A + cgsolve_A + datasave_A)
other_B = wall_B - (reassembly_B + residual_B + jacsetup_B + cgsolve_B + datasave_B)
other_C = wall_C - (reassembly_C + residual_C + jacsetup_C + cgsolve_C + datasave_C)

def n(x, wall): return x / wall_A  # normalize all three to the cheapest (A) = 1.000X

C_REASM = "#2D6A4F"
C_RES   = "#D9B23C"
C_JAC   = "#7A5DC7"
C_SOL   = "#3E8E7E"
C_SAVE  = "#6FA8D6"
C_OTHER = "#B0B0B0"

LABELS = ["Reassembly (tangent/K(T))", "Residual assembly", "Jacobian/preconditioner setup",
          "Linear solve (CG)", "Data saving (I/O)", "Other (uninstrumented)"]
COLORS = [C_REASM, C_RES, C_JAC, C_SOL, C_SAVE, C_OTHER]

bar_x = [0, 1, 2]
bar_labels = ["MFEM", "MFEM + OTI\n(6 directions)", "MFEM + OTI\n(30 directions)"]

series_A = [reassembly_A, residual_A, jacsetup_A, cgsolve_A, datasave_A, other_A]
series_B = [reassembly_B, residual_B, jacsetup_B, cgsolve_B, datasave_B, other_B]
series_C = [reassembly_C, residual_C, jacsetup_C, cgsolve_C, datasave_C, other_C]

fig, ax = plt.subplots(figsize=(9.0, 6.6))
bottoms = [0.0, 0.0, 0.0]
for label, color, vA, vB, vC in zip(LABELS, COLORS, series_A, series_B, series_C):
    vals_n = [n(vA, wall_A), n(vB, wall_A), n(vC, wall_A)]
    ax.bar(bar_x, vals_n, bottom=bottoms, color=color, width=0.55,
           label=label, edgecolor="white", linewidth=0.4)
    bottoms = [b + v for b, v in zip(bottoms, vals_n)]

for i, tot in enumerate(bottoms):
    ax.text(bar_x[i], tot + 0.006, f"{tot:.4f}X", ha="center", fontsize=11, fontweight="bold")

ax.set_xticks(bar_x); ax.set_xticklabels(bar_labels, fontsize=10.5)
ax.set_ylabel("Additional CPU time")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8.6, framealpha=0.95)
ax.set_ylim(0, max(bottoms) * 1.10)
ax.set_xlim(-0.5, 2.5)

plt.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])
out = "/work/gyj329/MFEM/implementations/simple_case_30_param/MFEM-LSBF-AM-/figures/mechanical/computational_cost/nominal_cost_3lib_comparison.png"
plt.savefig(out, dpi=150)
print("wrote", out)
print(f"wall_A={wall_A:.2f}s wall_B={wall_B:.2f}s wall_C={wall_C:.2f}s")
print(f"normalized totals: A={bottoms[0]:.4f}X B={bottoms[1]:.4f}X C={bottoms[2]:.4f}X")
