import csv, sys, numpy as np
D=sys.argv[1]; enz=sys.argv[2] if len(sys.argv)>2 else 'enz_d2_mixed'
def load(f):
    r=list(csv.reader(open(f))); h=r[0]; a=np.array([[float(x) for x in row] for row in r[1:] if len(row)==len(h)]); return h,a
oh,oa=load(f'{D}/oti_cross.sens.csv'); eh,ea=load(f'{D}/{enz}.sens.csv.hess.csv')
n=min(len(oa),len(ea)); print('rows',n)
res=[]
for j,c in enumerate(oh):
    if c=='time_native': continue
    name=c.replace('_row1','')
    if name not in eh: print('missing',name); continue
    x=oa[:n,j]; y=ea[:n,eh.index(name)]
    den=np.sqrt(np.mean(x**2)); 
    res.append((np.sqrt(np.mean((x-y)**2))/den*100 if den>0 else (0.0 if not np.any(y) else np.inf), name, den))
res.sort(reverse=True)
for e,nm,d in res: print(f'{nm:22s} rms(ref)={d:10.3e}  NRMSE={e:10.3e} %')
