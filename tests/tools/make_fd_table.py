#!/usr/bin/env python3
"""Write the job table (name|env|args) for the first-order finite-difference check and its metadata (fd_meta.json).

usage: make_fd_table.py --out DIR --laser es_bridge_laser.inp --ts T0 --tf T1 [--hs 5e-3,1e-3] [--params k,cp,...]

Runs: fd_base (nominal), fd_<param>_<i>_p / _m (central pairs, step index i into --hs), ad_d1 (-dorder 1).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_params as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--laser", required=True)
    ap.add_argument("--ts", type=float, required=True)
    ap.add_argument("--tf", type=float, required=True)
    ap.add_argument("--hs", default="5e-3,1e-3")
    ap.add_argument("--params", default="all")
    a = ap.parse_args()
    hs = [float(x) for x in a.hs.split(",") if x]
    v = F.scan_speed(a.laser, a.ts, a.tf)
    P = F.params(v)
    names = list(P) if a.params == "all" else [p for p in a.params.split(",") if p]
    unknown = [p for p in names if p not in P]
    if unknown:
        sys.exit(f"unknown parameters: {unknown}; known: {list(P)}")
    os.makedirs(a.out, exist_ok=True)
    lines = ["ad_d1||-dorder 1", "fd_base||-dorder 0"]   # the long AD run first, so it overlaps the FD runs
    for p in names:
        argf = P[p][1]
        for i, h in enumerate(hs):
            lines.append(f"fd_{p}_{i}_p||-dorder 0 {argf(+h)}")
            lines.append(f"fd_{p}_{i}_m||-dorder 0 {argf(-h)}")
    with open(os.path.join(a.out, "fd_jobs.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "fd_meta.json"), "w") as fh:
        json.dump({"hs": hs, "params": names, "scan_speed": v, "ts": a.ts, "tf": a.tf}, fh, indent=1)
    print(f"{len(lines)} runs ({len(names)} parameters x {len(hs)} steps x 2 + base + AD); scan speed {v:.6g} mm/time unit")


if __name__ == "__main__":
    main()
