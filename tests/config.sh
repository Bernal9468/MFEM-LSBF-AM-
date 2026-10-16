# tests/config.sh -- shared settings for tests/jobs/*.slurm (sourced; every value can be overridden from the
# environment, e.g. `NSTEPS_FD=80 tests/run_all.sh`).
#
# Submit through tests/run_all.sh from the repository root: SLURM copies job scripts to a spool directory, so the
# jobs locate the repository through REPO (exported by run_all.sh) or SLURM_SUBMIT_DIR.

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ ! -f "$REPO/tests/config.sh" ]; then
  echo "tests/config.sh: REPO=$REPO is not the repository root (submit with tests/run_all.sh from the repo root)" >&2
  exit 2
fi
IN_DIR=$REPO/inputs                                    # the driver reads full_bridge.mesh etc. relative to here
OUT_ROOT=${TEST_OUT_ROOT:-$REPO/tests/out}             # all test output (git-ignored)
BIN_DIR=$OUT_ROOT/bin                                  # binaries frozen by 00_build
SUMMARY=$OUT_ROOT/SUMMARY.txt                          # one PASS/FAIL line per check, all jobs
mkdir -p "$OUT_ROOT" "$BIN_DIR"

# frozen binaries (00_build copies them here; every later job uses these copies, never the source trees)
ENZ_OPT=$BIN_DIR/fbke_enzyme_opt                       # opt/src            -DUSE_ENZYME_AD  (production thermal driver)
OTI1_OPT=$BIN_DIR/fbko_opt_order1                      # opt/src            OTI first order  (independent AD reference)
ENZ_RS=$BIN_DIR/fbke_enzyme_rs                         # residual_stress/src (opt + cool-down/snapshots)

# production flags of opt/jobs/full_enzyme.slurm (the run that produced opt/results/full_enzyme), minus probe / -tf
TS=1.2522
DT=1e-6
COMMON="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -pf 1 \
-spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt $DT -ts $TS -powderstart -zshift 0.18"
PROBE_REF="-probex 2.5 -probey 0.0"                   # probe of the stored reference runs
PROBE_FD="-probex ${FD_PROBEX:-2.05} -probey 0.0"     # the laser passes this probe inside the short windows

# window lengths [steps of DT]
NSTEPS_REG=${NSTEPS_REG:-200}      # 01 regression vs stored full-window results
NSTEPS_FD=${NSTEPS_FD:-140}        # 02 finite differences and 03 Enzyme vs OTI
NSTEPS_RST=${NSTEPS_RST:-600}      # 04 restart (split at NSPLIT_RST)
NSPLIT_RST=${NSPLIT_RST:-300}

# finite differences: relative steps (two -> Richardson extrapolation + an h-convergence check), tight Newton
FD_HS=${FD_HS:-5e-3,1e-3}
FD_PARAMS=${FD_PARAMS:-all}
FD_TOL=${FD_TOL:-1e-2}             # pass if NRMSE(AD, FD) <= 1 %
FD_NEWTON=${FD_NEWTON:-"-ntolr 1e-10 -ntola 1e-7"}    # driver defaults are 1e-6 / 1e-3 (too loose for FD)
SLOT_CORES=${SLOT_CORES:-8}        # cores per packed run

export OMP_PROC_BIND=close OMP_PLACES=cores

tf_after() { python3 -c "print(f'{$TS + $1 * $DT:.8f}')"; }   # end time after N steps

# append a line to the shared summary (and echo it)
summary() { echo "$*"; echo "$*" >> "$SUMMARY"; }
