import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# AD (this project's OTI libraries) speedup over central finite differences,
# for the N=30-parameter target -- same methodology as the reference slide
# (Morales-Bernal D.F., AMMS Lab/UT San Antonio): "Speed up first-order
# derivatives", Speedup(N) = FD_cost(N) / AD_total(N).
#
# FD_cost = 2*N_target = 2*30 = 60 nominal-solve-equivalents, FIXED for BOTH
# libraries regardless of how many directions each currently has wired --
# central differences for the full 30-parameter target always needs 60 full
# nonlinear solves (2 per parameter: forward + backward perturbation), and
# that's the real work being compared against, not a partial subset.
#
# AD_total(library) = 1 + X_single(library), using this project's own
# established, full-scale, single-run extra/nominal X values (matching
# firstorder_final_comparison_X_fullsim.png / restart_both_libraries_fullscale.png):
#   OTI (6-basis):  X_single = 3.019  ->  AD_total = 4.019
#   OTI (30-basis): X_single = 1.939  ->  AD_total = 2.939

nom_single_30, extra_single_30 = 1161.57, 2252.69
nom_single_6,  extra_single_6  = 1168.85, 3528.39

FD_COST = 2 * 30  # fixed N=30 target, both libraries

X_single_30 = extra_single_30 / nom_single_30
X_single_6  = extra_single_6  / nom_single_6

AD_total_30 = 1.0 + X_single_30
AD_total_6  = 1.0 + X_single_6

speedup_30 = FD_COST / AD_total_30
speedup_6  = FD_COST / AD_total_6

C_6  = "#E07B39"
C_30 = "#5B7DB1"

fig, ax = plt.subplots(figsize=(6.4, 6.0))

labels = ["OTI (6-basis)", "OTI (30-basis)"]
values = [speedup_6, speedup_30]
colors = [C_6, C_30]

bars = ax.bar([0, 1], values, color=colors, width=0.5, edgecolor="white", linewidth=0.6)
for i, v in enumerate(values):
    ax.text(i, v + max(values)*0.02, f"{v:.2f}x", ha="center", fontsize=13, fontweight="bold")

ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Speedup")
ax.set_ylim(0, max(values) * 1.18)

plt.tight_layout()
out = "/work/gyj329/MFEM/implementations/simple_case_30_param/MFEM-LSBF-AM-/figures/mechanical/computational_cost/speedup_ad_vs_fd_N30.png"
plt.savefig(out, dpi=150)
print("wrote", out)
print(f"OTI(6-basis):  AD_total={AD_total_6:.3f} speedup={speedup_6:.3f}x")
print(f"OTI(30-basis): AD_total={AD_total_30:.3f} speedup={speedup_30:.3f}x")
