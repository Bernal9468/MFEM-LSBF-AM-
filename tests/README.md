# Thermal verification suite (HPC)

Tests for the thermal driver's nominal response and first-order sensitivities: the Enzyme build of `opt/src` and the
`residual_stress/src` copy built from it. Everything runs through SLURM from the repository root.

```bash
tests/run_all.sh            # 00 build, then 01-04 in parallel once the build succeeds
tests/run_all.sh 02 03      # only some tests (reuses the binaries frozen by an earlier 00)
tests/run_all.sh --report   # print tests/out/SUMMARY.txt (one PASS/FAIL/INFO line per check)
```

Before the first run, check the partitions in the `#SBATCH` headers (`compute1,compute2`) or set `PARTITION=...`. Also
check the toolchain paths in `opt/src/Makefile` and `tests/unit/Makefile`: MFEM, MKL, HYPRE, the OTI library, clang 21
and the Enzyme plugin. They are the paths the existing jobs use.

## The jobs

| job | node time | what it checks | pass criterion |
|---|---|---|---|
| `00_build` | 8 cores, ~30 min | builds `opt/src` (Enzyme + OTI order 1) and `residual_stress/src`, freezes them in `tests/out/bin/`; runs the kernel unit test and the existing mechanical kernel tests | build OK; unit test exit 0 |
| `01_regression` | 1 node, ~1 h | nominal probe and all first-order sensitivities over the first 200 steps of the production run against `opt/results/full_enzyme` (same driver); `residual_stress` driver = `opt` driver; a `-dorder 1` run gives the same nominal solution as `-dorder 0` | max error / max reference ≤ 1e-8 (T), 1e-6 (dT/ds); driver-vs-driver 1e-10 / 1e-8 |
| `02_fd_first_order` | 1 node, ~2-3 h | **all 20 wired parameters** against central finite differences of the nominal run, two step sizes + Richardson extrapolation, 140-step window in which the laser passes the probe (x = 2.05, melt band reached) | NRMSE(AD, FD) ≤ 1 % over the window |
| `03_enzyme_vs_oti` | 1 node, ~40 min | Enzyme against the OTI build (two independent AD implementations); direct against GMRES sensitivity solve | round-off (NRMSE ≤ 1e-6) except the laser-source parameters, where the OTI build uses finite differences (≤ 1 %) |
| `04_restart` | 1 node, ~40 min | a run stopped at step 300 and resumed from its checkpoint against a continuous 600-step run through the melt (the 2026-10-09 restart bug) | max rel diff ≤ 1e-10 (T), 1e-8 (dT/ds); full row count; `RESUMED from` in the log |

The runtimes are estimates from the per-step costs in `README_IMPLEMENTATION.md`. Jobs 01-03 pack several independent
runs onto one exclusive node (`tests/tools/pack.sh`, `SLOT_CORES` cores each). If a job is killed, resubmit it:
finished runs are skipped.

## Unit test of the Enzyme kernels (`tests/unit`)

`test_fb_kernels.cpp` calls every exported entry point of `opt/src/fb_enzyme_kernels.h`:

- `fb_dk1`, `fb_dq1`, `fb_nodal1` for all 26 directions with history seeding;
- the `fb2_*` second-order curves;
- `fb_local_hess` (gradient, Hessian, symmetry, and consistency with `fb_nodal1`);
- the laser-source kernel `ak_src_*` in point and Goldak mode, including energy conservation.

The reference is **independent of the kernels**. `ref_model.hpp` copies the physics from the *driver's* forward IN625
functions and differentiates it with hyper-dual numbers, an operator-overloading AD. That gives exact first, second and
mixed derivatives with no finite-difference step to tune, so the tolerance is 1e-11. A wrong seed, slot, sign, a missing
chain-rule term, or a material table that drifted from the driver fails by many orders of magnitude. The harness was
checked by injecting six such bugs into a stand-in, and each was caught.

```bash
cd tests/unit && make test       # cluster: real Enzyme kernels (ENZ_CLANGXX / ENZYME_PLUGIN / MFEM_DIR overridable)
cd tests/unit && make selftest   # any machine: hyper-dual stand-in, checks only that the harness builds and runs
```

## How the finite-difference check is set up (and why)

- **Perturbations use the driver's own flags**, defined once in `tests/tools/fd_params.py` (`-kscale`, `-tsolidus`,
  `-tambc`, `-gaf/-gar`, `-xshift`, `-tshift`, …), with the nominal values the driver actually uses. C2K = 273, so
  T_ambient = 313 K and T_sub = 353 K, not 313.15/353.15 as in the old `validate_fd4.slurm`.
- **hc is perturbed with hp held fixed.** The driver sets hp = `HP_OVER_HC` · h_conv, so a run with `-hconv` alone
  measures dT/dhc + dT/dhp. The test scales `-hpratio` by 1/(1+h) at the same time. The old validation scripts didn't,
  which is probably why hc never appeared among the FD-verified parameters.
- **Tight Newton tolerance and the direct solver for every run** (`FD_NEWTON="-ntolr 1e-10 -ntola 1e-7"`,
  `NEWTON_LS=pardiso`). The AD derivative is exact for the converged discrete solution, so the finite difference must
  differentiate the same solution, not Newton noise. A 10-step preflight stops the job early if Newton can't reach the
  tolerance. In that case loosen `FD_NEWTON`.
- **Two step sizes** (`FD_HS=5e-3,1e-3`). The report gives the error for each h, the Richardson-extrapolated value, and
  an "FD conv" column (difference between the two FD estimates). When the AD is right, the FD error drops about 25× from
  h = 5e-3 to 1e-3. A disagreement where the FD itself has not converged is reported as INCONCLUSIVE, not FAIL.
- **Parameters that are exactly zero in the window** (e.g. Ry ~ 1e-10) are reported as ZERO. The report also says whether
  the probe reached the melt band, i.e. whether Lfus/Ts/Tliq/rho were actually exercised.

Outputs: `tests/out/02_fd_first_order/fd_report.txt`, plus `fd_vs_ad.csv` (time series of AD and FD per parameter, for
plots).

## Expected / known outcomes

- **Tw** has a documented 13-34 % disagreement with FD (`README_IMPLEMENTATION.md`). The FD job is expected to show it
  (marked in the report). In `COMPARISON_TABLE.txt` Enzyme and OTI also differ for T0 (6e-3 %) and Tw (0.34 %), far more
  than for the other non-source parameters. Job 03 compares these two separately so a failure points at the IC/BC
  seeding.
- **Ts**: the finite difference also moves the powder-to-bulk switch (`ever_melted` is set at T ≥ T_solidus). That is a
  discrete effect the AD holds fixed, so a small Ts gap near melt onset is a modelling caveat, not an AD bug.
- **Vx, TL**: AD and FD use the same definition. Only the laser *position* is re-timed; the step-hold power schedule and
  the travel direction stay on the nominal segment. A physical lead-time shift would also move the power on/off times,
  which neither includes.
- **Suspected initial-condition inconsistency (reported as INFO by job 01).** Row 1 is born at t = 0, so the tie is
  active from the first step. The code that sets the tie slaves (bridge bottom) to the interpolated master temperature
  runs only when an activation event happens (`full_bridge_activation_tier1_oti.cpp`, "Row1 just activated: seed slave
  dofs"), so the slaves start at T_ambient = 313 K while their masters (substrate top) start at 353 K. In the stored
  run, `substrate_top` drops 2.2 K in the first 1 µs step and then changes by only ~3.5e-4 K/step. That pattern fits a
  one-time energy sink at the interface. The fix is probably to apply the same slave seeding after the initial
  condition, but that changes the nominal results and needs a decision.

## Configuration

All settings are in `tests/config.sh` and can be overridden from the environment, for example:

```bash
FD_PARAMS=Tw,T0,hc NSTEPS_FD=80 tests/run_all.sh 02    # a quick look at three parameters
FD_HS=1e-2,2e-3 FD_TOL=5e-3 tests/run_all.sh 02        # other steps / tolerance
TEST_OUT_ROOT=/scratch/$USER/fbtests tests/run_all.sh   # output elsewhere
```

`tests/out/` and `tests/logs/` are git-ignored.
