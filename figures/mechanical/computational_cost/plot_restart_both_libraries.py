import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Full-scale (8447-step) single-run-vs-restart comparison, in X terms,
# matching this project's ESTABLISHED convention: X = extra / nominal,
# a single bar height per side -- same meaning as
# firstorder_final_comparison_X_fullsim.png.
#
# Single run: X_single = extra_single / nominal_single (own reference).
# Restart: X_restart = Phase B / Phase A, the same structure applied to
# restart's own pair of independent runs.
#
# OTI (6-basis) CORRECTION: Phase B's raw total bundles production's
# always-on legacy 4-parameter cross-check (legacy1st_s+legacy2nd_s,
# counted) PLUS a further ~999s of legacy's own domain/boundary assembly
# that is never wired into any printed total (found by direct code
# inspection; confirmed present at nearly the same magnitude in the
# single run too, ~1178s, so it is NOT restart-specific). Legacy-related
# cost this way accounts for over half (51.9%) of Phase B's raw total.
# Excluding it from Phase B (matching the single-run bar's own
# legacy-excluded convention) gives the fair, apples-to-apples number:
# X_restart = 3.510X, much closer to single-run's 3.019X and in the same
# relative range as OTI(30-basis)'s own 1.939X -> 2.234X gap.

nom_single_30, extra_single_30 = 1161.57, 2252.69
phaseA_30, phaseB_30 = 1086.16, 2425.94
tan_s30, res_s30, jac_s30, sol_s30 = 1351.98, 175.53, 509.27, 215.91
reasm_B30, tan_B30, res_B30, jac_B30, sol_B30 = 129.88, 1390.79, 176.64, 506.86, 215.77

nom_single_6, extra_single_6 = 1168.85, 3528.39
phaseA_6 = 1082.49
tan_s6, res_s6, jac_s6, sol_s6 = 2003.23, 88.73, 511.14, 925.29
reasm_B6, tan_B6, res_B6, jac_B6, sol_B6 = 131.38, 2134.67, 91.03, 511.80, 929.05
datasave_B6 = 1.43
firstorder_only_6 = tan_B6 + res_B6 + jac_B6 + sol_B6
phaseB_6_legacy_excluded = firstorder_only_6 + reasm_B6 + datasave_B6

C_SINGLE = "#4C72B0"
C_REASM  = "#C96A3B"
C_TAN    = "#2D6A4F"
C_RES    = "#D9B23C"
C_JAC    = "#7A5DC7"
C_SOL    = "#6FA8D6"
C_SAVE   = "#B0B0B0"
C_OTHER  = "#5A5A5A"

fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.8))

shared_max = max(phaseB_30 / phaseA_30, phaseB_6_legacy_excluded / phaseA_6)

for ax, name, nom_single, extra_single, phaseA, tan_s, res_s, jac_s, sol_s, reasm_B, tan_B, res_B, jac_B, sol_B, save_B in [
    (axes[0], "OTI (30-basis)", nom_single_30, extra_single_30, phaseA_30,
     tan_s30, res_s30, jac_s30, sol_s30, reasm_B30, tan_B30, res_B30, jac_B30, sol_B30, 0.0),
    (axes[1], "OTI (6-basis)",  nom_single_6,  extra_single_6,  phaseA_6,
     tan_s6, res_s6, jac_s6, sol_s6, reasm_B6, tan_B6, res_B6, jac_B6, sol_B6, datasave_B6),
]:
    X_single = extra_single / nom_single
    phaseB_total = phaseB_30 if phaseA == phaseA_30 else phaseB_6_legacy_excluded
    X_restart = phaseB_total / phaseA
    other_B = phaseB_total - (reasm_B + tan_B + res_B + jac_B + sol_B + save_B)

    # Restart bar: bottom segment = single-run total (same height/color as
    # the single-run bar), then the difference stacked as: (1) Reassembly --
    # the one piece with NO counterpart in the single run's own accounting,
    # since a combined run builds the operator once and never has to redo it
    # -- and (2) per-category deltas (Phase B's own category cost, minus the
    # single run's own category cost, each normalized by its own run's
    # nominal reference). Those deltas are read against TWO independently-
    # timed runs, so part of their size reflects ordinary node-to-node
    # timing variance (documented elsewhere in this project at up to ~34%),
    # not a guaranteed mechanistic restart cost -- only Reassembly is.
    segs = [
        ("Reassembly",         C_REASM, reasm_B / phaseA),
        ("Tangent assembly",   C_TAN,   tan_B / phaseA - tan_s / nom_single),
        ("Residual assembly",  C_RES,   res_B / phaseA - res_s / nom_single),
        ("PARDISO factorize",  C_JAC,   jac_B / phaseA - jac_s / nom_single),
        ("Linear solve",       C_SOL,   sol_B / phaseA - sol_s / nom_single),
    ]
    if save_B > 0:
        segs.append(("Data saving", C_SAVE, save_B / phaseA))
    if other_B > 0.001 * phaseA:
        segs.append(("Other", C_OTHER, other_B / phaseA))

    ax.bar([0, 1], [X_single, X_single], color=C_SINGLE, width=0.55)
    bottom = X_single
    for label, color, v in segs:
        ax.bar([1], [v], bottom=[bottom], color=color, width=0.55, label=label, edgecolor="white", linewidth=0.4)
        bottom += v

    ax.text(0, X_single + shared_max*0.02, f"{X_single:.3f}X", ha="center", fontsize=13, fontweight="bold")
    ax.text(1, X_restart + shared_max*0.02, f"{X_restart:.3f}X", ha="center", fontsize=13, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Single run", "Restart\n(Real + derivatives)"], fontsize=10)
    ax.set_title(name, fontsize=12)
    ax.set_ylim(0, shared_max * 1.22)
    ax.legend(loc="upper left", fontsize=7.3, framealpha=0.95)

axes[0].set_ylabel("CPU overhead")

plt.tight_layout()
out = "/work/gyj329/MFEM/implementations/simple_case_30_param/MFEM-LSBF-AM-/figures/mechanical/computational_cost/restart_both_libraries_fullscale.png"
plt.savefig(out, dpi=150)
print("wrote", out)
print(f"OTI(30-basis): X_single={extra_single_30/nom_single_30:.3f} X_restart={phaseB_30/phaseA_30:.3f}")
print(f"OTI(6-basis):  X_single={extra_single_6/nom_single_6:.3f} X_restart={phaseB_6_legacy_excluded/phaseA_6:.3f}")
