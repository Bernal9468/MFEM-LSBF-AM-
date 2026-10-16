#!/bin/bash
#SBATCH --job-name=t00_build
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=tests/logs/%x_%j.out
# 00 -- build the thermal drivers, run the kernel unit tests, freeze the binaries for jobs 01-04.
#   * opt/src: fbke_enzyme (Enzyme, production) + fbko_opt_order1 (OTI first order, independent AD reference)
#   * residual_stress/src: fbke_enzyme (opt + cool-down/snapshot options)
#   * tests/unit: thermal Enzyme kernels vs hyper-dual reference (PASS/FAIL, gates nothing: later jobs still run)
#   * residual_stress/mech/src: existing mechanical kernel tests (informational, output saved)
# A build failure fails the job (and, through afterok, cancels 01-04).
set -uo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/tests/config.sh"
U=$OUT_ROOT/unit; mkdir -p "$U"
fail() { summary "FAIL  build: $*"; exit 1; }

echo "== opt/src"
( cd "$REPO/opt/src" && make -j8 fbke_enzyme fbko_opt_order1 ) > "$U/build_opt.log" 2>&1 || fail "opt/src (see $U/build_opt.log)"
echo "== residual_stress/src"
( cd "$REPO/residual_stress/src" && make -j8 fbke_enzyme ) > "$U/build_rs.log" 2>&1 || fail "residual_stress/src (see $U/build_rs.log)"
cp "$REPO/opt/src/fbke_enzyme" "$ENZ_OPT" && cp "$REPO/opt/src/fbko_opt_order1" "$OTI1_OPT" \
  && cp "$REPO/residual_stress/src/fbke_enzyme" "$ENZ_RS" || fail "copying binaries"
{ echo "commit $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  git -C "$REPO" status --short -- opt/src residual_stress/src 2>/dev/null | sed 's/^/dirty /'
  date; hostname; } > "$BIN_DIR/VERSION"
summary "PASS  build: opt/src (Enzyme + OTI order 1), residual_stress/src; binaries in $BIN_DIR"

echo "== thermal kernel unit test"
( cd "$REPO/tests/unit" && make test ) > "$U/test_fb_kernels.txt" 2>&1
rc=$?
cat "$U/test_fb_kernels.txt"
if [ $rc -eq 0 ]; then summary "PASS  unit: thermal Enzyme kernels vs hyper-dual reference ($(grep -c '^PASS' "$U/test_fb_kernels.txt") checks)"
else summary "FAIL  unit: thermal Enzyme kernels -- $(grep -c '^FAIL' "$U/test_fb_kernels.txt") failing checks (see $U/test_fb_kernels.txt)"; fi

echo "== mechanical kernel tests (informational)"
( cd "$REPO/residual_stress/mech/src" && make -f Makefile.test test_kernel test_second && ./test_kernel && ./test_second 3000 ) \
  > "$U/mech_kernel.txt" 2>&1 \
  && summary "INFO  unit: mechanical kernel tests ran (errors printed in $U/mech_kernel.txt; no pass/fail thresholds)" \
  || summary "FAIL  unit: mechanical kernel tests did not build/run (see $U/mech_kernel.txt)"
exit 0
