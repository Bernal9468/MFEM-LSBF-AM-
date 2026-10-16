#!/bin/bash
#SBATCH --job-name=t03_enz_vs_oti
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=tests/logs/%x_%j.out
# 03 -- first-order Enzyme vs OTI (two independent AD implementations of the same discrete equations), plus the
# sensitivity linear-solver option. Same window/probe as 02.
#   enz_fdsrc vs oti1     both with FD laser-source blocks -> must agree to round-off, all 20 parameters
#   enz vs oti1           non-source parameters to round-off; source parameters (Rx..TL) only to the FD truncation
#                         of the OTI build's source blocks (Enzyme differentiates the recorded source exactly)
#   enz_gmres vs enz      SENS_DIRECT=0 (GMRES + stale PARDISO preconditioner) vs the direct exact solve
#   probes                the sensitivity pass must not alter the nominal solution
set -uo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/tests/config.sh"
O=$OUT_ROOT/03_enzyme_vs_oti; mkdir -p "$O"
for b in "$ENZ_OPT" "$OTI1_OPT"; do [ -x "$b" ] || { summary "FAIL  enz-vs-oti: missing $b (run 00_build)"; exit 1; }; done
TF=$(tf_after "$NSTEPS_FD")
export NEWTON_LS=pardiso IN_DIR
cat > "$O/jobs.txt" <<EOF
oti1||-dorder 1|$OTI1_OPT
enz||-dorder 1
enz_fdsrc|SRC_FD_LEGACY=1|-dorder 1
enz_gmres|SENS_DIRECT=0|-dorder 1
EOF
bash "$REPO/tests/tools/pack.sh" "$O/jobs.txt" "$O" "$ENZ_OPT" "$COMMON $PROBE_FD -tf $TF $FD_NEWTON" "${SLOT_CORES_OTI:-10}"

C="python3 $REPO/tests/tools/compare_runs.py --summary $SUMMARY --metric nrmse --floor 1e-3"
SRC='^dT_d(Rx|Ry|Rz|X0|Vx|TL)_'
NONSRC='^dT_d(k|cp|rho|Lfus|Ts|Tliq|hc|P|Eta|hp|kp|Rhop)_'
SEED='^dT_d(T0|Tw)_'     # enter through IC / Dirichlet seeds; COMPARISON_TABLE.txt shows 6e-3 % / 0.34 % Enzyme-OTI gaps
rc=0
$C --label "enz-vs-oti: Enzyme (FD source) == OTI, all parameters" --test "$O/enz_fdsrc.sens.csv" --ref "$O/oti1.sens.csv" --tol 1e-6 || rc=1
$C --label "enz-vs-oti: Enzyme == OTI, material/BC/power parameters" --test "$O/enz.sens.csv" --ref "$O/oti1.sens.csv" --cols "$NONSRC" --tol 1e-6 || rc=1
$C --label "enz-vs-oti: Enzyme == OTI, IC/BC-seeded T0, Tw" --test "$O/enz.sens.csv" --ref "$O/oti1.sens.csv" --cols "$SEED" --tol 1e-6 || rc=1
$C --label "enz-vs-oti: Enzyme AD source vs OTI FD source (Rx..TL)" --test "$O/enz.sens.csv" --ref "$O/oti1.sens.csv" --cols "$SRC" --tol 1e-2 || rc=1
$C --label "enz-vs-oti: GMRES sensitivity solve == direct" --test "$O/enz_gmres.sens.csv" --ref "$O/enz.sens.csv" --tol 1e-4 || rc=1
$C --label "enz-vs-oti: nominal probe, OTI run == Enzyme run" --test "$O/oti1.probe.csv" --ref "$O/enz.probe.csv" --metric maxrel --tol 1e-10 || rc=1
exit $rc
