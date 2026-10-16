#!/usr/bin/env python3
"""First-order AD (ad_d1, -dorder 1) vs central finite differences of the nominal probe temperature (layer1_top).

usage: compare_fd.py --dir OUT [--tol 1e-2] [--summary FILE]

For each parameter and each step h in fd_meta.json: FD_h = (T(+h) - T(-h)) / (2 unit(h)); with two or more steps the
two finest are Richardson-extrapolated (central FD error is O(h^2)), FD_R. Errors are NRMSE over the window:
rms(AD - FD) / rms(FD). Verdicts:
  PASS          NRMSE(AD, FD_R) <= tol
  ZERO          AD and FD both below the FD noise floor (parameter not exercised in this window)
  INCONCLUSIVE  AD disagrees but the FD itself is not converged in h (|FD_h1 - FD_h2| > tol/2): change h / window
  FAIL          AD disagrees with a converged FD
  MISSING       a run did not finish (see its .log / .status)
Writes fd_report.txt and fd_vs_ad.csv (time, AD and FD_R per parameter) into --dir. Exit 1 on FAIL / MISSING.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_params as F  # noqa: E402
import tcsv  # noqa: E402

PROBE = "layer1_top"


def ok_run(d, name):
    st = os.path.join(d, name + ".status")
    return os.path.exists(st) and open(st).read().startswith("exit=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    D = a.dir
    meta = json.load(open(os.path.join(D, "fd_meta.json")))
    hs, names = meta["hs"], meta["params"]
    P = F.params(meta["scan_speed"])
    out = []
    say = lambda s="": (print(s), out.append(s))

    if not (ok_run(D, "ad_d1") and ok_run(D, "fd_base")):
        say("FAIL  ad_d1 or fd_base did not finish -- nothing to compare")
        open(os.path.join(D, "fd_report.txt"), "w").write("\n".join(out) + "\n")
        sys.exit(1)
    AD = tcsv.load(os.path.join(D, "ad_d1.sens.csv"))
    ADp = tcsv.load(os.path.join(D, "ad_d1.probe.csv"))
    B = tcsv.load(os.path.join(D, "fd_base.probe.csv"))
    t, x, y = tcsv.common(ADp, B)
    Tmax = float(np.max(tcsv.col(B, PROBE)))
    base_dev = float(np.max(np.abs(x[:, ADp[0].index(PROBE)] - y[:, B[0].index(PROBE)]))) / Tmax
    say(f"first-order AD vs central FD, probe {PROBE}, {len(t)} steps, t = {meta['ts']} .. {meta['tf']}, h = {hs}")
    say(f"  probe T: max {Tmax:.1f} K (solidus {F.T_SOLIDUS:.0f} K, liquidus {F.T_LIQUIDUS:.0f} K) -> "
        f"{'melt band reached' if Tmax > F.T_SOLIDUS else 'NO MELTING: Lfus/Ts/Tliq/rho are not exercised'}")
    say(f"  nominal probe of the AD run vs fd_base: max rel diff {base_dev:.2e} (same discrete state expected)")
    # FD noise floor: probe written with 12 significant digits
    dT = 5e-13 * Tmax
    say("")
    say(f"{'param':6s} {'verdict':12s} {'NRMSE(AD,FD_R)':>14s} " + " ".join(f"{'h=' + format(h, 'g'):>10s}" for h in hs)
        + f" {'FD conv':>9s} {'rms AD':>10s}  note")
    nfail = 0
    series = {"time_native": None}
    for p in names:
        col, argf, unit, note = P[p]
        runs = [(f"fd_{p}_{i}_p", f"fd_{p}_{i}_m") for i in range(len(hs))]
        if not all(ok_run(D, r) for pair in runs for r in pair):
            say(f"{p:6s} {'MISSING':12s} (a perturbed run failed; see fd_{p}_*.log)")
            nfail += 1
            continue
        tabs = [(tcsv.load(os.path.join(D, rp + ".probe.csv")), tcsv.load(os.path.join(D, rm + ".probe.csv"))) for rp, rm in runs]
        # align AD and all FD runs on common times
        tt = np.round(AD[1][:, 0], 9)
        for tp, tm in tabs:
            tt = np.intersect1d(tt, np.round(tp[1][:, 0], 9))
            tt = np.intersect1d(tt, np.round(tm[1][:, 0], 9))
        pick = lambda tab, c: tab[1][np.searchsorted(np.round(tab[1][:, 0], 9), tt), tab[0].index(c)]
        ad = pick(AD, col)
        fds = [(pick(tp, PROBE) - pick(tm, PROBE)) / (2.0 * unit(h)) for (tp, tm), h in zip(tabs, hs)]
        if len(hs) >= 2:
            r = (hs[-2] / hs[-1]) ** 2
            fdr = fds[-1] + (fds[-1] - fds[-2]) / (r - 1.0)
        else:
            fdr = fds[-1]
        floor = 10.0 * dT / unit(min(hs))
        nr = lambda u, v: tcsv.rms(u - v) / max(tcsv.rms(v), floor)
        e_r = nr(ad, fdr)
        e_h = [nr(ad, fd) for fd in fds]
        conv = nr(fds[-2], fds[-1]) if len(fds) >= 2 else float("nan")
        if tcsv.rms(ad) <= floor and tcsv.rms(fdr) <= floor:
            verdict = "ZERO"
        elif e_r <= a.tol:
            verdict = "PASS"
        elif len(fds) >= 2 and conv > 0.5 * a.tol:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FAIL"
            nfail += 1
        say(f"{p:6s} {verdict:12s} {e_r:14.3e} " + " ".join(f"{e:10.3e}" for e in e_h)
            + f" {conv:9.2e} {tcsv.rms(ad):10.3e}  {note}")
        if series["time_native"] is None or len(series["time_native"]) == len(tt):
            series["time_native"] = tt
            series[f"AD_{p}"] = ad
            series[f"FD_{p}"] = fdr
    say("")
    say("NRMSE = rms(AD - FD) / rms(FD) over the window; 'FD conv' = NRMSE between the two finest FD steps "
        "(the FD truncation error, ~ (h1/h2)^2 times the AD-FD_R error when the AD is right).")
    ok = nfail == 0
    line = f"{'PASS' if ok else 'FAIL'}  first-order AD vs FD: {len(names)} parameters, {nfail} failing/missing, tol {a.tol:g} (see fd_report.txt)"
    say(line)
    open(os.path.join(D, "fd_report.txt"), "w").write("\n".join(out) + "\n")
    if series["time_native"] is not None:
        keys = list(series)
        np.savetxt(os.path.join(D, "fd_vs_ad.csv"), np.column_stack([series[k] for k in keys]), delimiter=",",
                   header=",".join(keys), comments="", fmt="%.12e")
    if a.summary:
        with open(a.summary, "a") as fh:
            fh.write(line + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
