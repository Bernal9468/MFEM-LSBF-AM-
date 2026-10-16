"""Shared helpers for the HPC test comparisons: load the driver's probe / sensitivity CSVs and compare columns."""
import csv
import numpy as np


def load(path):
    """Return (header, rows) with rows sorted by time and de-duplicated (a resumed run re-writes rows after its
    checkpoint; the last occurrence wins). Rows with a wrong field count (a run killed mid-write) are skipped."""
    with open(path) as f:
        r = list(csv.reader(f))
    if not r:
        raise ValueError(f"{path}: empty file")
    h = [c.strip() for c in r[0]]
    rows = {}
    for row in r[1:]:
        if len(row) != len(h):
            continue
        try:
            vals = [float(x) for x in row]
        except ValueError:
            continue
        rows[round(vals[0], 9)] = vals
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return h, np.array([rows[k] for k in sorted(rows)])


def common(a, b, nrows=None):
    """Align two loaded tables on their common time values (optionally only the first nrows of them)."""
    ta = np.round(a[1][:, 0], 9)
    tb = np.round(b[1][:, 0], 9)
    t = np.intersect1d(ta, tb)
    if nrows:
        t = t[:nrows]
    ia = np.searchsorted(ta, t)
    ib = np.searchsorted(tb, t)
    return t, a[1][ia], b[1][ib]


def col(table, name):
    h, d = table
    if name not in h:
        raise KeyError(f"column {name} not found")
    return d[:, h.index(name)]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def metrics(test, ref, floor=0.0):
    """Relative errors of a test series against a reference series (normalisation never below `floor`)."""
    d = test - ref
    sref = max(rms(ref), floor, 1e-300)
    mref = max(float(np.max(np.abs(ref))) if len(ref) else 0.0, floor, 1e-300)
    return {
        "nrmse": rms(d) / sref,                                   # rms error / rms reference
        "maxrel": float(np.max(np.abs(d))) / mref if len(d) else 0.0,   # max error / max |reference|
        "rms_ref": rms(ref),
        "rms_test": rms(test),
    }
