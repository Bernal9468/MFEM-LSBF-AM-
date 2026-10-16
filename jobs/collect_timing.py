"""Wall-time table from the production logs (sum of the per-step 'wall=' values and the driver's final 'Wall time')."""
import re, os, sys
B = '/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/results'
REF = '/work/gyj329/MFEM/implementations/full_bridge_20_param_single_layer/results'
runs = [  # label, log, threads, concurrent load note
 ('Enzyme nominal',            f'{B}/nominal/nominal.log',                    4,  '3 stages share a 40-core node (4/12/24 cores)'),
 ('Enzyme first order',        f'{B}/deriv_first_order/d1.log',               12, '3 stages share a 40-core node'),
 ('Enzyme full Hessian',       f'{B}/deriv_second_order/d2.log',              40, 'node shared with nothing else after the relaunch; 40 threads'),
 ('OTI nominal (reference)',   f'{REF}/nominal_single_layer/nominal.log',     16, 'exclusive node'),
 ('OTI first order (1 pass)',  f'{B}/oti_first_order/d1.log',                 40, 'exclusive node, 40 threads'),
 ('OTI 2nd order blend-fixed', f'{B}/oti_blend_fixed/d2.log',                 40, 'exclusive node, 40 threads'),
]
print(f'{"run":28s} {"steps":>6s} {"sum step wall [s]":>18s} {"s/step":>8s} {"driver Wall time [s]":>21s} {"threads":>8s}  load')
rows = []
for lab, log, thr, note in runs:
    if not os.path.exists(log): continue
    steps = 0; tot = 0.0; wt = None
    seen = {}
    for l in open(log, errors='ignore'):
        m = re.match(r'\s+Step\s+(\d+).*wall=(\d+)ms', l)
        if m: seen[int(m.group(1))] = int(m.group(2)) / 1000.0     # last occurrence per step (restart-safe)
        m = re.search(r'Wall time\s*:\s*([\d.]+) s', l)
        if m: wt = float(m.group(1))
    steps = len(seen); tot = sum(seen.values())
    print(f'{lab:28s} {steps:6d} {tot:18.1f} {tot/max(steps,1):8.2f} {wt if wt else float("nan"):21.1f} {thr:8d}  {note}')
