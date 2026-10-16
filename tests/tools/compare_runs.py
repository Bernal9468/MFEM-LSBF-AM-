#!/usr/bin/env python3
"""Compare every common data column of two driver CSVs (probe or sensitivity files) and print PASS/FAIL.

usage: compare_runs.py --test A.csv --ref B.csv --label NAME [--tol 1e-10] [--metric maxrel|nrmse]
                       [--nrows N] [--cols REGEX] [--zero 1e-12] [--summary FILE]

A column passes when its error (default: max |A-B| / max |B|) is <= tol. Columns that are identically ~0 in both
files (|value| <= --zero everywhere, e.g. the structurally zero y0/Vy/... slots) are reported as ZERO and pass.
Appends one line per comparison to --summary and exits 1 if any column fails or the files share no rows.
"""
import argparse
import re
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tcsv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--metric", choices=("maxrel", "nrmse"), default="maxrel")
    ap.add_argument("--nrows", type=int, default=None, help="compare only the first N common time rows")
    ap.add_argument("--cols", default=None, help="regex: compare only matching columns")
    ap.add_argument("--zero", type=float, default=1e-12)
    ap.add_argument("--floor", type=float, default=0.0,
                    help="absolute floor of the error normalisation: err = |A-B| / max(|B|, floor); lets columns "
                         "that are tiny compared to the others (e.g. dT/dRy ~ 1e-10) pass on an absolute basis")
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()

    def done(ok, msg):
        line = f"{'PASS' if ok else 'FAIL'}  {a.label}: {msg}"
        print(line)
        if a.summary:
            with open(a.summary, "a") as f:
                f.write(line + "\n")
        sys.exit(0 if ok else 1)

    try:
        A = tcsv.load(a.test)
        B = tcsv.load(a.ref)
    except (OSError, ValueError) as e:
        done(False, f"cannot load: {e}")
    t, da, db = tcsv.common(A, B, a.nrows)
    if len(t) == 0:
        done(False, "no common time rows")
    names = [c for c in A[0][1:] if c in B[0]]
    if a.cols:
        names = [c for c in names if re.search(a.cols, c)]
    if not names:
        done(False, "no common columns")
    worst, worst_col, nfail, nzero = 0.0, "", 0, 0
    print(f"{a.label}: {len(t)} common rows, {len(names)} columns, metric {a.metric}, tol {a.tol:g}")
    for c in names:
        x = da[:, A[0].index(c)]
        y = db[:, B[0].index(c)]
        if np.max(np.abs(x)) <= a.zero and np.max(np.abs(y)) <= a.zero:
            nzero += 1
            print(f"  {c:24s} ZERO")
            continue
        m = tcsv.metrics(x, y, a.floor)
        e = m[a.metric]
        bad = not (e <= a.tol)
        nfail += bad
        if not (e <= worst):
            worst, worst_col = e, c
        print(f"  {c:24s} {a.metric} {e:.3e}  {'FAIL' if bad else 'ok'}")
    done(nfail == 0, f"{len(names) - nzero} columns ({nzero} zero), {len(t)} rows, worst {a.metric} "
                     f"{worst:.2e} ({worst_col}), tol {a.tol:g}, {nfail} failing")


if __name__ == "__main__":
    main()
