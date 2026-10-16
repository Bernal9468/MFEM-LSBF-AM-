#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=fbe_val
#SBATCH --time=04:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/val_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/val_%j.err
# Validation on the 200-step pre-melt window of the OTI reference benchmark (same flags as
# full_bridge_20_param_single_layer/jobs/deriv_bench.slurm).  All runs packed into this ONE node job, 8 cores each.
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
REF=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_layer
IN=$R/inputs; OUT=$R/results/validate_short; mkdir -p $OUT
export OMP_PROC_BIND=close OMP_PLACES=cores MKL_NUM_THREADS=8 NEWTON_LS=auto
BIN=$OUT/bin_enz_$SLURM_JOB_ID; cp $R/src/fbke_enzyme $BIN
cp $REF/src/fbko_opt_order2 $OUT/bin_oti2_$SLURM_JOB_ID
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25240 -powderstart -zshift 0.18"
cd $IN
# job table: name|env|extra args
cat > $OUT/jobs.txt <<EOF
enz_d2_plain||-dorder 2
enz_d2_fdsrc|SRC_FD_LEGACY=1|-dorder 2
enz_d2_mixed_noblend|SENS_MIXED=1 FB_NOBLEND2=1|-dorder 2
enz_d2_mixed|SENS_MIXED=1|-dorder 2
EOF
# FD nominal runs: central 4-point stencils for mixed pairs among (k,cp,P,rx,rho), h=0.02 relative (x0: 0.005 mm abs)
python3 - >> $OUT/jobs.txt <<'PY'
import itertools
h=0.02
def args(sc):  # sc: dict name->relative offset
    a=[]
    for n,v in sc.items():
        f=1.0+v
        if n=='k': a+=['-kscale',str(f)]
        elif n=='cp': a+=['-cpscale',str(f)]
        elif n=='rho': a+=['-rhoscale',str(f)]
        elif n=='P': a+=['-pscale',str(f)]
        elif n=='rx': a+=['-gaf',str(0.1*f),'-gar',str(0.1*f)]
        elif n=='ry': a+=['-gb',str(0.1*f)]
        elif n=='rz': a+=['-gc',str(0.1*f)]
    return ' '.join(a)
names=['k','cp','rho','P','rx','rz']
runs={}
def add(sc):
    key=tuple(sorted((n,round(v,6)) for n,v in sc.items() if v!=0.0))
    runs.setdefault(key,sc)
add({})
for n in names:
    add({n:h}); add({n:-h})
for a,b in itertools.combinations(names,2):
    for sa in (h,-h):
        for sb in (h,-h): add({a:sa,b:sb})
for key,sc in runs.items():
    nm='fd_'+('base' if not key else '_'.join(f'{n}{"p" if v>0 else "m"}' for n,v in key))
    print(f'{nm}||{"-dorder 0 " + args({k:v for k,v in sc.items()})}')
PY
cat > $OUT/runone.sh <<'EOS'
#!/bin/bash
# usage: runone.sh slot "name|env|args"
slot=$1; IFS='|' read -r name envs args <<< "$2"
lo=$((slot*8)); hi=$((lo+7))
cd $IN
env OMP_NUM_THREADS=8 $envs taskset -c $lo-$hi /usr/bin/time -o $OUT/$name.time -f "WALL_S %e" \
  $BIN $C $args -pcsv $OUT/$name.probe.csv -scsv $OUT/$name.sens.csv > $OUT/$name.log 2>&1
echo "$name exit=$?"
EOS
chmod +x $OUT/runone.sh; export IN OUT BIN C
NSLOT=$(( $(nproc) / 8 )); echo "slots=$NSLOT jobs=$(wc -l < $OUT/jobs.txt)"
# simple slot scheduler
i=0; declare -A pid
while IFS= read -r line; do
  while true; do
    for s in $(seq 0 $((NSLOT-1))); do
      if [ -z "${pid[$s]:-}" ] || ! kill -0 ${pid[$s]} 2>/dev/null; then
        $OUT/runone.sh $s "$line" & pid[$s]=$!; break 2
      fi
    done
    sleep 5
  done
done < $OUT/jobs.txt
wait
echo VALIDATE_SHORT_DONE
