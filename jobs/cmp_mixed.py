"""Validate the Enzyme full Hessian (rows of *.hess.csv) against 4-point central finite differences of the nominal probe."""
import csv, sys, itertools, glob, numpy as np
D = sys.argv[1]; H = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
HN = {'x0': 0.002}                      # absolute x0 step [mm]; everything else is a relative step H
hstep = lambda n: HN.get(n, H)
hess_file = sys.argv[3] if len(sys.argv) > 3 else f'{D}/enz_d2_mixed.sens.csv.hess.csv'
NFD = int(sys.argv[4]) if len(sys.argv) > 4 else 140
def probe(name):
    r = list(csv.reader(open(f'{D}/fd_{name}.probe.csv')))
    a = np.array([[float(x) for x in row[:2]] for row in r[1:] if len(row) >= 2])
    if len(a) < NFD: raise FileNotFoundError(name)
    return a
rows = list(csv.reader(open(hess_file))); hdr = rows[0]
A = np.array([[float(x) for x in r] for r in rows[1:] if len(r) == len(hdr)])
colmap = {'k':'k','cp':'cp','rho':'rho','P':'P','rx':'Rx','rz':'Rz','Ts':'Ts','x0':'X0','kp':'kp','rhop':'Rhop','hp':'hp','T0':'T0','Tw':'Tw','hc':'hc'}
def col(a, b):
    for x, y in ((a, b), (b, a)):
        n = f'd2T_d{colmap[x]}_d{colmap[y]}'
        if n in hdr: return hdr.index(n)
    raise KeyError((a, b))
def tag(sc): return '_'.join(f'{n}{"p" if v > 0 else "m"}' for n, v in sc) or 'base'
names = list(colmap)
base = probe('base')
n = min(len(base), len(A)); ts = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else [int(f * n) - 1 for f in (0.25, 0.5, 0.75, 1.0)]
print(f'FD h={H} relative; Hessian rows={len(A)}, common={n}; time rows compared: {ts}')
out = []
def fd_pair(a, b):
    if a == b:
        p, m = probe(tag([(a, 1)])), probe(tag([(a, -1)]))
        return (p[:, 1] - 2 * base[:, 1] + m[:, 1]) / hstep(a)**2
    key = lambda sa, sb: tag(sorted([(a, sa), (b, sb)]))
    try:
        pp, pm, mp, mm = (probe(key(sa, sb)) for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
    except FileNotFoundError:
        return None
    return (pp[:, 1] - pm[:, 1] - mp[:, 1] + mm[:, 1]) / (4 * hstep(a) * hstep(b))
for a, b in itertools.combinations_with_replacement(names, 2):
    try:
        v = fd_pair(a, b)
    except FileNotFoundError:
        v = None
    if v is None: continue
    try: c = col(a, b)
    except KeyError: continue
    for t in ts:
        e, f = A[t, c], v[t]
        out.append((a, b, t, e, f))
print(f'{"pair":10s} {"row":>4s} {"Enzyme":>14s} {"FD":>14s} {"rel.err":>10s}')
for a, b, t, e, f in out:
    sc = max(abs(f), 1e-12)
    print(f'{a+","+b:10s} {t:4d} {e:14.6e} {f:14.6e} {abs(e-f)/sc:10.2e}')
