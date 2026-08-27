import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Checkpoint/restart cost comparison, decomposed by category (same style/
# legend as firstorder_final_comparison_X_fullsim.png). 500-step short
# test, all 7 fixes (same driver build as job 793323's full-scale 1.939X
# result -- see plot_firstorder_final_comparison.py).
#
# Straight-through (job 793250): X = extra/nominal from ONE run, same 4
# categories as the first-derivative figure.
# Restart (job 793327, Phases A+B): X = Phase_B_wall / Phase_A_wall.
# Phase B's own wall-clock carries a 5th category with no straight-through
# counterpart -- rebuilding the full-sparse PARDISO operator representation
# from scratch, since Phase A (nominal-only) never needed it. In the
# straight-through run this same rebuild happens ONCE, alongside the
# partial-assembly build Newton needs, inside its own reassembly_s (which
# counts as NOMINAL there, hence excluded from ITS bar). Shown here as its
# own segment so the reason restart's X is higher is visible, not hidden.
#
# Restart mechanism verified correct: bit-identical sensitivity CSVs
# (mod noise-floor) vs the reference, 118 reassemblies reproduced exactly.

phaseA_wall = 114.48
# Phase B categories:
reasm_B, tan_B, res_B, jac_B, sol_B = 43.32, 62.73, 8.97, 30.97, 12.97

# Straight-through categories (job 793250):
nom_st = 43.06 + 20.26 + 0.20 + 82.22 + 0.05
tan_st, res_st, jac_st, sol_st = 64.15, 8.77, 31.01, 12.87

def X(x, denom): return x / denom

C_TAN   = "#2D6A4F"  # tangent assembly
C_RES   = "#D9B23C"  # residual / nodal assembly
C_JAC   = "#7A5DC7"  # PARDISO factorize
C_SOL   = "#3E8E7E"  # linear solve (PARDISO)
C_REASM = "#C96A3B"  # reassembly (full-sparse rebuild) -- restart-only segment

fig, ax = plt.subplots(figsize=(7.6, 7.0))
bar_x = [0, 1]
bar_labels = ["Straight-through\n(single run)", "Restart\n(Phase B / Phase A)"]

bottoms = [0.0, 0.0]
segs = [
    ("Tangent assembly",                    C_TAN,   X(tan_st, nom_st),  X(tan_B, phaseA_wall)),
    ("Residual / nodal assembly",           C_RES,   X(res_st, nom_st),  X(res_B, phaseA_wall)),
    ("PARDISO factorize",                   C_JAC,   X(jac_st, nom_st),  X(jac_B, phaseA_wall)),
    ("Linear solve (PARDISO)",              C_SOL,   X(sol_st, nom_st),  X(sol_B, phaseA_wall)),
    ("Reassembly (full-sparse rebuild)\n-- restart-only, see caption", C_REASM, 0.0, X(reasm_B, phaseA_wall)),
]
for label, color, vST, vR in segs:
    vals = [vST, vR]
    ax.bar(bar_x, vals, bottom=bottoms, color=color, width=0.55,
           label=label, edgecolor="white", linewidth=0.4)
    bottoms = [b + v for b, v in zip(bottoms, vals)]

for i, tot in enumerate(bottoms):
    ax.text(bar_x[i], tot + 0.03, f"{tot:.3f}X", ha="center", fontsize=13, fontweight="bold")

ax.set_xticks(bar_x); ax.set_xticklabels(bar_labels, fontsize=10.5)
ax.set_ylabel("X (extra cost as a multiple of the nominal cost)")
ax.set_title("Extra-cost X, by category: straight-through vs restart")
ax.legend(loc="upper left", fontsize=8.2, framealpha=0.95)
ax.set_ylim(0, max(bottoms) * 1.30)
ax.set_xlim(-0.5, 1.5)

fig.text(0.5, 0.005,
        "500-step test, all 7 fixes (same build as the full-scale 1.939X result). Restart mechanism verified "
        "correct: bit-identical sensitivity CSVs,\n"
        "118 reassemblies reproduced exactly. Restart's extra orange segment is the real cost of building the "
        "PARDISO operator representation\n"
        "SEPARATELY in Phase B, since Phase A (nominal-only) never needed it -- in one combined run this build "
        "happens once (~43-44s total,\n"
        "counted as NOMINAL there). Total end-to-end cost: straight-through 262.6s vs restart(A+B) 274.0s "
        "(+4.3%) -- restart's real value is\n"
        "amortizing Phase A's cost across MULTIPLE Phase-B re-runs, not reducing cost for one one-shot "
        "calculation like this comparison.",
        ha="center", va="bottom", fontsize=7.1, color="#444444")

plt.tight_layout(rect=[0.0, 0.14, 1.0, 1])
out = "/work/gyj329/MFEM/implementations/simple_case_30_param/MFEM-LSBF-AM-/figures/mechanical/computational_cost/restart_vs_straightthrough.png"
plt.savefig(out, dpi=150)
print("wrote", out)
print(f"Straight-through: X={bottoms[0] if False else X(tan_st+res_st+jac_st+sol_st, nom_st):.3f}")
print(f"Restart:          X={X(tan_B+res_B+jac_B+sol_B+reasm_B, phaseA_wall):.3f}")
