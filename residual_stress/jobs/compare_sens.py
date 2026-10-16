import numpy as np, os, sys
V = "/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/results/val"
H = 1e-3
names = ["E", "nu", "CTE", "sigma_y", "H"]
def last(run):
    f = f"{V}/{run}/probe_hist.csv"
    if not os.path.exists(f): return None
    a = np.loadtxt(f, delimiter=",", ndmin=2); return a[-1]
ad = last("ad_d2"); nom = last("nominal")
sig0 = ad[3:9]
print("nominal probe stress (AD run vs nominal run), max|diff| = %.2e MPa" % np.abs(sig0 - nom[3:9]).max())
print("probe stress components [MPa]:", np.round(sig0, 3))
print("\nFIRST ORDER  d sigma / d scale   (AD vs central FD, h=%g)" % H)
for i in range(5):
    p, m = last(f"p{i}_plus"), last(f"p{i}_minus")
    if p is None or m is None: print(" ", names[i], "FD runs missing"); continue
    fd = (p[3:9] - m[3:9]) / (2 * H); a = ad[9 + 6 * i: 15 + 6 * i]
    print("  %-8s AD %s\n  %-8s FD %s   max|AD-FD|=%.2e  rel=%.2e" % (names[i], np.array2string(a, precision=4, suppress_small=True), "", np.array2string(fd, precision=4, suppress_small=True), np.abs(a - fd).max(), np.abs(a - fd).max() / max(np.abs(fd).max(), 1e-30)))
print("\nSECOND ORDER  d2 sigma / d scale_i d scale_j")
pairs = [(i, j) for i in range(5) for j in range(i, 5)]
for pidx, (i, j) in enumerate(pairs):
    a = ad[39 + 6 * pidx: 45 + 6 * pidx]
    if i == j:
        p, m = last(f"p{i}_plus"), last(f"p{i}_minus")
        if p is None or m is None: continue
        fd = (p[3:9] - 2 * sig0 + m[3:9]) / H ** 2          # uses AD-run nominal stress (same as nominal run to round-off)
    else:
        runs = [last(f"m{i}{j}_{s}") for s in ("pp", "pm", "mp", "mm")]
        if any(r is None for r in runs): continue
        fd = (runs[0][3:9] - runs[1][3:9] - runs[2][3:9] + runs[3][3:9]) / (4 * H ** 2)
    print("  d2/(%s,%s): max|AD-FD|=%.2e  scale %.2e  rel=%.2e" % (names[i], names[j], np.abs(a - fd).max(), np.abs(fd).max(), np.abs(a - fd).max() / max(np.abs(fd).max(), 1e-30)))
