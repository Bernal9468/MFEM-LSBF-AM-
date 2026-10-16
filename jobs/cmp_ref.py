import csv, sys, numpy as np
def load(f):
    r=list(csv.reader(open(f))); h=r[0]; a=np.array([[float(x) for x in row] for row in r[1:] if len(row)==len(h)])
    return h,a
ref1=load('/work/gyj329/MFEM/implementations/full_bridge_20_param_single_layer/results/deriv_bench/d1_sens.csv')
ref2=load('/work/gyj329/MFEM/implementations/full_bridge_20_param_single_layer/results/deriv_bench/d2_sens.csv')
def cmp(name, ref, ours, label):
    hr,ar=ref; ho,ao=ours
    n=min(len(ar),len(ao))
    print(f'== {label}: {name}  ({n} common rows)')
    bad=[]
    for j,c in enumerate(hr):
        if c=='time_native' or c not in ho: continue
        x=ar[:n,j]; y=ao[:n,ho.index(c)]
        den=np.sqrt(np.mean(x**2))
        if den==0 and np.all(y==0): continue
        nr=np.sqrt(np.mean((x-y)**2))/den*100 if den>0 else float('inf')
        bad.append((nr,c))
    for nr,c in sorted(bad,reverse=True): print(f'   {c:22s} NRMSE {nr:10.3e} %')
d=sys.argv[1]
for nm in sys.argv[2:]:
    o=load(f'{d}/{nm}.sens.csv')
    cmp(nm,ref1,o,'first-order vs OTI bench')
    cmp(nm,ref2,o,'second-order vs OTI bench')
