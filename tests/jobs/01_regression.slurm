#!/bin/bash
#SBATCH --job-name=t01_regression
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=tests/logs/%x_%j.out
# 01 -- regression of the nominal temperature and first-order sensitivities against the stored full-window runs.
# Production flags (opt/jobs/full_enzyme.slurm), probe x = 2.5, first NSTEPS_REG steps:
#   opt nominal / d1      vs opt/results/full_enzyme/{nominal_probe,d1_sens}.csv   (same driver, stored run)
#   opt d1 probe          vs opt nominal probe          (the sensitivity pass must not change the nominal solve)
#   residual_stress n/d1  vs opt n/d1                   (rs driver = opt + flag-guarded cool-down: identical by default)
#   opt d1                vs results/deriv_first_order  (older src/ driver lineage, looser tolerance)
# Bitwise identity is not expected across thread counts (OpenMP atomics in block (a), threaded PARDISO), hence
# tight-but-nonzero tolerances.
set -uo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/tests/config.sh"
O=$OUT_ROOT/01_regression; mkdir -p "$O"
for b in "$ENZ_OPT" "$ENZ_RS"; do [ -x "$b" ] || { summary "FAIL  regression: missing $b (run 00_build)"; exit 1; }; done
TF=$(tf_after "$NSTEPS_REG")
export NEWTON_LS=auto IN_DIR
cat > "$O/jobs.txt" <<EOF
opt_d1||-dorder 1 $PROBE_REF -tf $TF
rs_d1||-dorder 1 $PROBE_REF -tf $TF|$ENZ_RS
opt_nominal||-dorder 0 $PROBE_REF -tf $TF
rs_nominal||-dorder 0 $PROBE_REF -tf $TF|$ENZ_RS
EOF
bash "$REPO/tests/tools/pack.sh" "$O/jobs.txt" "$O" "$ENZ_OPT" "$COMMON" "${SLOT_CORES_REG:-10}"

C="python3 $REPO/tests/tools/compare_runs.py --summary $SUMMARY"
REF=$REPO/opt/results/full_enzyme
rc=0
$C --label "regression: nominal probe vs opt/results/full_enzyme ($NSTEPS_REG steps)" --test "$O/opt_nominal.probe.csv" --ref "$REF/nominal_probe.csv" --nrows "$NSTEPS_REG" --tol 1e-8 || rc=1
$C --label "regression: d1 sensitivities vs opt/results/full_enzyme ($NSTEPS_REG steps)" --test "$O/opt_d1.sens.csv" --ref "$REF/d1_sens.csv" --nrows "$NSTEPS_REG" --floor 1e-3 --tol 1e-6 || rc=1
$C --label "regression: d1 run's nominal probe == nominal run's" --test "$O/opt_d1.probe.csv" --ref "$O/opt_nominal.probe.csv" --tol 1e-10 || rc=1
$C --label "regression: residual_stress driver == opt driver (nominal)" --test "$O/rs_nominal.probe.csv" --ref "$O/opt_nominal.probe.csv" --tol 1e-10 || rc=1
$C --label "regression: residual_stress driver == opt driver (d1)" --test "$O/rs_d1.sens.csv" --ref "$O/opt_d1.sens.csv" --floor 1e-3 --tol 1e-8 || rc=1
$C --label "regression: d1 vs results/deriv_first_order (src/ lineage)" --test "$O/opt_d1.sens.csv" --ref "$REPO/results/deriv_first_order/d1_sens.csv" --nrows "$NSTEPS_REG" --floor 1e-3 --tol 1e-6 || rc=1
# INFO (suspected IC inconsistency, see tests/README.md): row 1 is active from t = 0, so the tie slaves (bridge bottom)
# keep T_ambient while their masters (substrate top) start at T_sub_bc; the first step then absorbs the jump.
python3 - "$REPO/tests/tools" "$O/opt_nominal.probe.csv" <<'PY' | while IFS= read -r l; do summary "$l"; done
import sys; sys.path.insert(0, sys.argv[1]); import tcsv
h, d = tcsv.load(sys.argv[2]); s = d[:, h.index('substrate_top')]
print(f"INFO  regression: substrate_top {353.0:.1f} K at t0 -> {s[0]:.4f} K after step 1 (jump {s[0]-353.0:+.4f} K), "
      f"then {s[1]-s[0]:+.2e} K/step -- a one-step jump >> later rate points at the tie-slave initial condition")
PY
exit $rc
