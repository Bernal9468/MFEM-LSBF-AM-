import re, glob, os
D='/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/results/timing_benchmark'
def wall(n):
    f=f'{D}/{n}.log'
    if not os.path.exists(f): return None
    m=[float(x) for x in re.findall(r'Wall time\s*:\s*([\d.]+) s', open(f,errors='ignore').read())]
    return m[-1] if m else None
print('threads  stage            OTI [s]  Enzyme [s]  speed-up(OTI/Enz)  OTI overhead  Enzyme overhead')
for t in (16,40):
    en,on=wall(f'enz_nominal_t{t}'),wall(f'oti_nominal_t{t}')
    for st,lab in (('nominal','nominal'),('d1','first order'),('d2diag','2nd order diag')):
        e,o=wall(f'enz_{st}_t{t}'),wall(f'oti_{st}_t{t}')
        if e and o: print(f'{t:7d}  {lab:15s} {o:8.1f} {e:10.1f} {o/e:14.2f}x {o/on:14.2f} {e/en:14.2f}')
    ef=wall(f'enz_d2full_t{t}')
    if ef and en: print(f'{t:7d}  Enzyme full Hessian (210 fields): {ef:.1f} s, overhead {ef/en:.2f}x')
