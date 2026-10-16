#!/bin/bash
#SBATCH --job-name=t02_fd_first
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --output=tests/logs/%x_%j.out
# 02 -- first-order sensitivities of ALL 20 wired parameters vs central finite differences of the nominal run.
# Window NSTEPS_FD steps from t = 1.2522 with probe x = 2.05 (the laser passes it, ~step 78), two relative steps
# (FD_HS, Richardson-extrapolated), tight Newton tolerance (FD_NEWTON) and the direct solver for every run, so AD and
# FD differentiate the same discrete solution. ~80 nominal runs packed SLOT_CORES cores each on one node; resubmit
# the job to finish an interrupted table (finished runs are skipped). Report: tests/out/02_fd_first_order/fd_report.txt
set -uo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/tests/config.sh"
O=$OUT_ROOT/02_fd_first_order; mkdir -p "$O"
[ -x "$ENZ_OPT" ] || { summary "FAIL  fd: missing $ENZ_OPT (run 00_build)"; exit 1; }
TF=$(tf_after "$NSTEPS_FD")
export NEWTON_LS=pardiso SENS_DIRECT=1 IN_DIR
python3 "$REPO/tests/tools/make_fd_table.py" --out "$O" --laser "$IN_DIR/es_bridge_laser.inp" --ts "$TS" --tf "$TF" \
  --hs "$FD_HS" --params "$FD_PARAMS" || { summary "FAIL  fd: could not build the job table"; exit 1; }

# preflight: can Newton reach the tight FD tolerance? (10 steps; a failure here would otherwise cost ~80 runs)
if [ ! -f "$O/PREFLIGHT_OK" ]; then
  ( cd "$IN_DIR" && OMP_NUM_THREADS=$SLOT_CORES MKL_NUM_THREADS=$SLOT_CORES "$ENZ_OPT" $COMMON $PROBE_FD $FD_NEWTON \
      -tf "$(tf_after 10)" -dorder 0 -pcsv "$O/preflight.probe.csv" ) > "$O/preflight.log" 2>&1
  if [ $? -ne 0 ]; then
    summary "FAIL  fd: preflight with FD_NEWTON='$FD_NEWTON' failed ($(grep -m1 ERROR "$O/preflight.log")); loosen FD_NEWTON and resubmit"
    exit 1
  fi
  touch "$O/PREFLIGHT_OK"
fi

bash "$REPO/tests/tools/pack.sh" "$O/fd_jobs.txt" "$O" "$ENZ_OPT" "$COMMON $PROBE_FD -tf $TF $FD_NEWTON" "$SLOT_CORES"
python3 "$REPO/tests/tools/compare_fd.py" --dir "$O" --tol "$FD_TOL" --summary "$SUMMARY"
