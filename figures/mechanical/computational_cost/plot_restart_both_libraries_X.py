import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same full-scale (8447-step) single-run-vs-restart X comparison, now with
# each bar decomposed by category using the SAME legend/colors as
# firstorder_final_comparison_X_fullsim.png (Tangent assembly / Residual-
# nodal / PARDISO factorize / Linear solve), plus whatever extra categories
# are needed so each bar's stack honestly sums to its real measured total
# (never leaving an unexplained gap, matching this project's convention).

phaseA_30, phaseB_30 = 1086.16, 2425.94
nom_single_30 = 1161.57
tan_s30, res_s30, jac_s30, sol_s30 = 1351.98, 175.53, 509.27, 215.91          # single run, job 793323
reasm_B30, tan_B30, res_B30, jac_B30, sol_B30 = 129.88, 1390.79, 176.64, 506.86, 215.77  # Phase B, job 793353

phaseA_6, phaseB_6 = 1082.49, 7904.42
nom_single_6 = 1168.85
tan_s6, res_s6, jac_s6, sol_s6 = 2003.23, 88.73, 511.14, 925.29               # single run, job 793237
reasm_B6, tan_B6, res_B6, jac_B6, sol_B6 = 131.38, 2134.67, 91.03, 511.80, 929.05  # Phase B, job 793355
legacy_B6 = 195.26 + 2910.89
other_B6 = phaseB_6 - (reasm_B6+tan_B6+res_B6+jac_B6+sol_B6+legacy_B6)   # uninstrumented remainder, ~999s -- see caption
other_B30 = phaseB_30 - (reasm_B30+tan_B30+res_B30+jac_B30+sol_B30)      # ~5s, negligible but included for honesty

C_TAN   = "#2D6A4F"
C_RES   = "#D9B23C"
C_JAC   = "#7A5DC7"
C_SOL   = "#3E8E7E"
C_REASM = "#C96A3B"
C_LEGACY= "#8B4A9C"
C_OTHER = "#B0B0B0"

def X(x, denom): return x / denom

fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.2))

# ---- Panel 1: OTI (30-basis) ----
ax = axes[0]
bottoms = [0.0, 0.0]
segs30 = [
    ("Tangent assembly",          C_TAN,   X(tan_s30, nom_single_30), X(tan_B30, phaseA_30)),
    ("Residual / nodal assembly", C_RES,   X(res_s30, nom_single_30), X(res_B30, phaseA_30)),
    ("PARDISO factorize",         C_JAC,   X(jac_s30, nom_single_30), X(jac_B30, phaseA_30)),
    ("Linear solve (PARDISO)",    C_SOL,   X(sol_s30, nom_single_30), X(sol_B30, phaseA_30)),
    ("Reassembly (restart only)", C_REASM, 0.0,                       X(reasm_B30, phaseA_30)),
    ("Other (uninstrumented)",    C_OTHER, 0.0,                       X(other_B30, phaseA_30)),
]
for label, color, vS, vR in segs30:
    vals = [vS, vR]
    ax.bar([0, 1], vals, bottom=bottoms, color=color, width=0.55, label=label, edgecolor="white", linewidth=0.4)
    bottoms = [b + v for b, v in zip(bottoms, vals)]
for i, tot in enumerate(bottoms):
    ax.text(i, tot + 0.03, f"{tot:.3f}X", ha="center", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Single run", "Restart\n(Phase B / Phase A)"], fontsize=10)
ax.set_title("OTI (30-basis)", fontsize=12)
ax.set_ylabel("X (extra cost as a multiple of the nominal cost)")
ax.legend(loc="upper left", fontsize=7.6, framealpha=0.95)
ax.set_ylim(0, max(bottoms) * 1.22)
ax.annotate("", xy=(0.80, 1.06), xytext=(0.20, 1.06), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.2))
ax.text(0.5, 1.085, "independent runs, NOT additive", ha="center", fontsize=7.8,
        color="#666666", style="italic", transform=ax.transAxes)

# ---- Panel 2: OTI (6-basis) ----
ax = axes[1]
bottoms = [0.0, 0.0]
segs6 = [
    ("Tangent assembly",          C_TAN,    X(tan_s6, nom_single_6), X(tan_B6, phaseA_6)),
    ("Residual / nodal assembly", C_RES,    X(res_s6, nom_single_6), X(res_B6, phaseA_6)),
    ("PARDISO factorize",         C_JAC,    X(jac_s6, nom_single_6), X(jac_B6, phaseA_6)),
    ("Linear solve (PARDISO)",    C_SOL,    X(sol_s6, nom_single_6), X(sol_B6, phaseA_6)),
    ("Reassembly (restart only)", C_REASM,  0.0,                     X(reasm_B6, phaseA_6)),
    ("Legacy 4-param + 2nd order (restart only)", C_LEGACY, 0.0,     X(legacy_B6, phaseA_6)),
    ("Other (uninstrumented)",    C_OTHER,  0.0,                     X(other_B6, phaseA_6)),
]
for label, color, vS, vR in segs6:
    vals = [vS, vR]
    ax.bar([0, 1], vals, bottom=bottoms, color=color, width=0.55, label=label, edgecolor="white", linewidth=0.4)
    bottoms = [b + v for b, v in zip(bottoms, vals)]
for i, tot in enumerate(bottoms):
    ax.text(i, tot + 0.08, f"{tot:.3f}X", ha="center", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Single run", "Restart\n(Phase B / Phase A)"], fontsize=10)
ax.set_title("OTI (6-basis)", fontsize=12)
ax.legend(loc="upper left", fontsize=7.6, framealpha=0.95)
ax.set_ylim(0, max(bottoms) * 1.22)
ax.annotate("", xy=(0.80, 1.06), xytext=(0.20, 1.06), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.2))
ax.text(0.5, 1.085, "independent runs, NOT additive", ha="center", fontsize=7.8,
        color="#666666", style="italic", transform=ax.transAxes)

fig.suptitle("Extra-cost X, by category: single run vs. checkpoint/restart, full 8447-step simulation", fontsize=12.5)
fig.text(0.5, 0.005,
        "The two bars in each panel are INDEPENDENT measurements from separate runs, not a base-plus-increment "
        "relationship -- the restart bar's own\n"
        "tangent/residual/factorize/solve values come from Phase B's own run, not copied from the single-run bar; "
        "they simply measure nearly the same\n"
        "real computation twice, via two different workflows.\n"
        "X_restart = Phase B / Phase A. Both mechanisms verified correct at full scale (bit-identical sensitivity "
        "CSVs, exact reassembly-count match).\n"
        "Single-run bars use this project's standard convention (legacy/2nd-order excluded); restart bars show "
        "the SAME 4 categories plus what Phase B\n"
        "additionally carries: the reassembly it must redo separately (both libraries), and for OTI(6-basis) "
        "also its always-on legacy cross-check\n"
        "(2.87X of Phase A alone) -- explaining most of that panel's larger gap. \"Other\" is wall-clock minus "
        "all named categories: negligible for\n"
        "OTI(30-basis) (~0.2%) but a real, not-yet-investigated ~12.6% for OTI(6-basis)'s restart Phase B, "
        "flagged honestly rather than hidden.",
        ha="center", va="bottom", fontsize=7.0, color="#444444")

plt.tight_layout(rect=[0.0, 0.20, 1.0, 0.93])
out = "/work/gyj329/MFEM/implementations/simple_case_30_param/MFEM-LSBF-AM-/figures/mechanical/computational_cost/restart_both_libraries_X.png"
plt.savefig(out, dpi=150)
print("wrote", out)
