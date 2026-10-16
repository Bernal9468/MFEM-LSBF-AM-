import numpy as np, os
R = "/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/results/"
H = 1e-4; ne = 188648; names = ["E", "nu", "CTE", "sigma_y", "H"]
ld = lambda f: np.fromfile(f).reshape(ne, 6)
S0 = ld(R + "mech_full/final_sigma_elem.bin")
done = lambda r: os.path.exists(R + f"fd_full/{r}/DONE")
def rel(a, b):
    return np.sqrt(((a - b) ** 2).sum()) / max(np.sqrt((b ** 2).sum()), 1e-300), np.abs(a - b).max(), np.abs(b).max()
print("FIRST ORDER (whole final stress field, 188,648 elements x 6 components): AD vs central FD (h=%g)" % H)
for i in range(5):
    if not (done(f"p{i}_plus") and done(f"p{i}_minus")): print("  ", names[i], "FD runs not finished"); continue
    fd = (ld(R + f"fd_full/p{i}_plus/final_sigma_elem.bin") - ld(R + f"fd_full/p{i}_minus/final_sigma_elem.bin")) / (2 * H)
    ad = ld(R + f"mech_full/final_dsigma_elem_{i}.bin"); e = rel(ad, fd)
    print("  d/d%-8s relative L2 error %.2e | max|AD-FD| %.3e (max|d sigma| %.3e MPa)" % (names[i], *e))
print("\nSECOND ORDER, pure terms: AD vs (s+ - 2 s0 + s-)/h^2")
for i in range(5):
    if not (done(f"p{i}_plus") and done(f"p{i}_minus")): continue
    fd = (ld(R + f"fd_full/p{i}_plus/final_sigma_elem.bin") - 2 * S0 + ld(R + f"fd_full/p{i}_minus/final_sigma_elem.bin")) / H ** 2
    ad = ld(R + f"mech_full/final_d2sigma_elem_{i}{i}.bin"); e = rel(ad, fd)
    print("  d2/d%s2: relative L2 error %.2e | max|AD-FD| %.3e (max|d2 sigma| %.3e MPa)" % (names[i], *e))
print("\nSECOND ORDER, mixed terms: AD vs (s++ - s+- - s-+ + s--)/(4 h^2)")
for (i, j) in ((0, 3), (1, 2), (3, 4), (0, 4)):
    rs = [f"m{i}{j}_{s}" for s in ("pp", "pm", "mp", "mm")]
    if not all(done(r) for r in rs): print("  (%s,%s): FD runs not finished" % (names[i], names[j])); continue
    pp, pm, mp, mm = [ld(R + f"fd_full/{r}/final_sigma_elem.bin") for r in rs]
    fd = (pp - pm - mp + mm) / (4 * H ** 2); ad = ld(R + f"mech_full/final_d2sigma_elem_{i}{j}.bin"); e = rel(ad, fd)
    print("  d2/d%s d%s: relative L2 error %.2e | max|AD-FD| %.3e (max|d2 sigma| %.3e MPa)" % (names[i], names[j], *e))
