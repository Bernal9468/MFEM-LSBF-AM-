#!/bin/bash
#SBATCH --job-name=t04_restart
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=tests/logs/%x_%j.out
# 04 -- checkpoint/restart reproduces a continuous run (the 2026-10-09 restart bug: ever_melted and the laser's last
# direction were not checkpointed). First-order run of NSTEPS_RST steps through the melt at probe x = 2.05:
#   A   continuous
#   B1  stops at step NSPLIT_RST, writing a checkpoint there (-tf at the split, -ckpt-every NSPLIT_RST)
#   B2  resumes from that checkpoint to the end (appends to B's CSVs)
# Deterministic split (no kill timing). Same thread count for A and B so only the restart differs.
set -uo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/tests/config.sh"
O=$OUT_ROOT/04_restart; rm -rf "$O"; mkdir -p "$O"
[ -x "$ENZ_OPT" ] || { summary "FAIL  restart: missing $ENZ_OPT (run 00_build)"; exit 1; }
export NEWTON_LS=pardiso
NCPU=${SLURM_CPUS_ON_NODE:-$(nproc)}; H=$(( NCPU / 2 ))
run() {   # run <cpus> <log> <args...>
  local cpus=$1 log=$2; shift 2
  ( cd "$IN_DIR" && env OMP_NUM_THREADS=$H MKL_NUM_THREADS=$H taskset -c "$cpus" "$ENZ_OPT" $COMMON $PROBE_FD -dorder 1 "$@" ) > "$log" 2>&1
}
TF=$(tf_after "$NSTEPS_RST"); TSPLIT=$(tf_after "$NSPLIT_RST")
run "0-$((H-1))" "$O/A.log" -tf "$TF" -pcsv "$O/A.probe.csv" -scsv "$O/A.sens.csv" & PA=$!
run "$H-$((2*H-1))" "$O/B1.log" -tf "$TSPLIT" -ckpt "$O/B.ckpt" -ckpt-every "$NSPLIT_RST" -pcsv "$O/B.probe.csv" -scsv "$O/B.sens.csv"; R1=$?
if [ $R1 -ne 0 ] || [ ! -s "$O/B.ckpt" ]; then
  wait $PA; summary "FAIL  restart: first half did not finish or wrote no checkpoint (exit $R1, see $O/B1.log)"; exit 1
fi
run "$H-$((2*H-1))" "$O/B2.log" -tf "$TF" -restart "$O/B.ckpt" -pcsv "$O/B.probe.csv" -scsv "$O/B.sens.csv"; R2=$?
wait $PA; RA=$?
if [ $RA -ne 0 ] || [ $R2 -ne 0 ]; then summary "FAIL  restart: run failed (A exit $RA, B2 exit $R2; see $O/*.log)"; exit 1; fi
grep -q "RESUMED from" "$O/B2.log" || { summary "FAIL  restart: B2.log has no 'RESUMED from' line -- -restart was not honoured"; exit 1; }

C="python3 $REPO/tests/tools/compare_runs.py --summary $SUMMARY --metric maxrel"
rc=0
$C --label "restart: probe temperatures, resumed == continuous ($NSTEPS_RST steps, split $NSPLIT_RST)" --test "$O/B.probe.csv" --ref "$O/A.probe.csv" --tol 1e-10 || rc=1
$C --label "restart: first-order sensitivities, resumed == continuous" --test "$O/B.sens.csv" --ref "$O/A.sens.csv" --floor 1e-3 --tol 1e-8 || rc=1
# both files must cover the whole window (a resume that silently restarted from t0 would still "match" its head)
nA=$(( $(wc -l < "$O/A.probe.csv") - 1 )); nB=$(python3 -c "import sys; sys.path.insert(0,'$REPO/tests/tools'); import tcsv; print(len(tcsv.load('$O/B.probe.csv')[1]))")
if [ "$nA" -eq "$nB" ] && [ "$nA" -ge "$NSTEPS_RST" ]; then summary "PASS  restart: resumed run covers all $nB steps"
else summary "FAIL  restart: row count continuous $nA vs resumed $nB (expected $NSTEPS_RST)"; rc=1; fi
exit $rc
