#!/bin/bash
# run_all.sh -- submit the thermal verification suite to SLURM (run from anywhere; jobs run from the repo root).
#
#   tests/run_all.sh                 build + all tests (00 -> 01,02,03,04 in parallel after a successful build)
#   tests/run_all.sh 02 03           only these tests (uses the binaries frozen by an earlier 00_build)
#   tests/run_all.sh --report        print tests/out/SUMMARY.txt
# Environment overrides (see tests/config.sh): PARTITION, NSTEPS_FD, FD_HS, FD_PARAMS, FD_TOL, FD_NEWTON, SLOT_CORES,
# NSTEPS_REG, NSTEPS_RST, NSPLIT_RST, TEST_OUT_ROOT.  Example: FD_PARAMS=k,cp,Tw NSTEPS_FD=80 tests/run_all.sh 02
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export REPO
OUT_ROOT=${TEST_OUT_ROOT:-$REPO/tests/out}
if [ "${1:-}" = "--report" ]; then cat "$OUT_ROOT/SUMMARY.txt"; exit 0; fi
mkdir -p tests/logs "$OUT_ROOT"
SB=(sbatch --parsable --export=ALL)
[ -n "${PARTITION:-}" ] && SB+=(--partition="$PARTITION")

declare -A JOB=([01]=tests/jobs/01_regression.slurm [02]=tests/jobs/02_fd_first_order.slurm
                [03]=tests/jobs/03_enzyme_vs_oti.slurm [04]=tests/jobs/04_restart.slurm)
if [ $# -eq 0 ]; then
  sel=(01 02 03 04)
  b=$("${SB[@]}" tests/jobs/00_build.slurm)
  echo "00_build             job $b"
  dep=(--dependency=afterok:$b)
  { echo; echo "=== suite submitted $(date '+%F %T'), commit $(git rev-parse --short HEAD 2>/dev/null || echo ?)"; } >> "$OUT_ROOT/SUMMARY.txt"
else
  sel=("$@"); dep=()
fi
for k in "${sel[@]}"; do
  [ -n "${JOB[$k]:-}" ] || { echo "unknown test '$k' (choose from ${!JOB[*]})"; exit 1; }
  j=$("${SB[@]}" ${dep[@]+"${dep[@]}"} "${JOB[$k]}")
  printf '%-20s job %s\n' "$(basename "${JOB[$k]}" .slurm)" "$j"
done
echo "logs: tests/logs/   results: $OUT_ROOT/   summary: tests/run_all.sh --report"
