# full_bridge_20_param_single_enzyme

Enzyme forward-mode (AD) version of `../full_bridge_20_param_single_layer` (OTI): single-layer NIST AM-Bench bridge,
20 wired parameters (k, cp, rho, Lfus, Ts, Tliq, hc, T0, P, Eta, Rx, Ry, Rz, X0, Vx, TL, Tw, hp, kp, Rhop), same mesh, same
window (t = 1.2522 .. 1.2566, 4400 steps of 1e-6), same flags.  Three stages:

| stage | flag | content |
|---|---|---|
| nominal | `-dorder 0` | temperature field / probe, no sensitivities |
| first order | `-dorder 1` | dT/dp for all 20 parameters |
| second order | `-dorder 2` + `SENS_MIXED=1` | **full Hessian** d2T/dp_i dp_j: 20 pure + 190 mixed (cross) terms |

(`-dorder 2` without `SENS_MIXED` gives the diagonal only, like the OTI reference.)

## What changed w.r.t. the OTI driver (`src/`)
* `fb_enzyme_kernels.{h,cpp}` (new, clang + `ClangEnzyme-21.so`): plain-double kernels differentiated with Enzyme forward mode.
  * first order: `fb_dk1` (blended k, d/dS_K, d/dS_KP), `fb_dq1` (boundary flux, d/dhc, d/dT0, d/dhp), `fb_nodal1`
    (capacitive + latent term with the powder/bulk rho blend, all 20 directions incl. the propagated history).
  * second order: `fb2_*` (reference-faithful diagonal, no powder blend) and `fb_local_hess` (nested forward-over-forward
    gradient + full Hessian of each blended kernel w.r.t. its own inputs -> every pure and mixed term).
  * `ak_src_*`: Enzyme derivatives of the recorded-choice laser-source kernel `src_kernel.hpp`.
* `full_bridge_activation_tier1_oti.cpp`: blocks (a),(b),(c) and (a2),(b2),(c2) call the kernels under `-DUSE_ENZYME_AD`
  (OTI code kept in the `#else` branches).  Laser-shape/position parameters (Rx, Ry, Rz, X0, Vx, TL) are now AD too: `BuildSource`
  records its discrete choices (`SrcTape`) and the kernel is differentiated (no finite differences; `SRC_FD_LEGACY=1` restores the
  reference's FD blocks).
* Full Hessian (`SENS_MIXED=1`): local Hessians at every quadrature point / boundary point / node, mixed explicit source terms
  (P*Eta, P*q, Eta*q, q_a*q_b), mixed history `prev_der2m`, mixed fields in the checkpoint, probe Hessian written to
  `<sens csv>.hess.csv` (210 columns, i <= j).
* `SENS_DIRECT` (default on in the Enzyme build): the exact corrected sensitivity Jacobian is factorised once per step and every
  right-hand side is one direct PARDISO solve (no GMRES); `SENS_DIRECT=0` restores the reference's GMRES path.
* `FB_NOBLEND2=1`: drop the powder blends in the second-order kernels (reproduces the reference OTI diagonal blocks, see below).

## Build / run
```
sbatch jobs/build.slurm          # kernels (clang+Enzyme plugin) + driver (MFEM g++) -> src/fbke_enzyme
sbatch jobs/run_full.slurm       # nominal + first order + full Hessian, one 40-core node, checkpointed (resubmit to resume)
python3 jobs/make_report.py      # de-duplicates restarted CSVs, compares with the OTI reference, writes REPORT.txt + figures/
```
Outputs: `results/nominal/nominal_probe.csv`, `results/deriv_first_order/d1_sens.csv`,
`results/deriv_second_order/d2_sens.csv` (20 first-order + 20 pure second-order columns) and
`results/deriv_second_order/d2_sens.csv.hess.csv` (all 210 entries d2T/dp_i dp_j, i <= j, row-1 probe).
Derivatives are w.r.t. the *relative* scale of each parameter (p -> p(1+s)), except X0 (absolute shift in mm).

## Validation (`jobs/validate_*.slurm`, `jobs/check_*.slurm`, `jobs/cmp_*.py`; results in `results/validate_short*`)
Window: 140 steps, probe x = 2.05 mm (the laser passes it), all runs packed in one node job.
1. **Enzyme vs OTI reference** (reference formulation + FD source, `SRC_FD_LEGACY=1 FB_REFERENCE2=1`): all first- and second-order
   diagonal columns agree to <= 1.4e-10 % (probe x = 2.5, 200 steps).
2. **First order vs central FD** (h = 0.5 %): cp, P, kp, rho_p, hp, Rx, Rz, X0: 1e-4 .. 3e-3 relative.
3. **Cross terms vs 4-point central FD**: (cp,P) (cp,rx) (cp,rz) (cp,x0) (cp,kp) (cp,rho_p) (P,rx) (P,rz) (P,x0) (P,kp) (rx,rz)
   (x0,kp) (kp,rho_p) ... agree to 1e-4 .. 2e-3.  Pairs involving T0 or x0,x0 differ by a few % (the piecewise-linear k(T)/cp(T)
   tables and the element-crossing of the source make the 2nd difference ill-defined), entries with |value| < 1e-4 are below the
   finite-difference noise floor (probe printed with 12 digits).
4. **Exact identities** (hold to 1e-16): d2(P,Eta) = dT/dP + d2(P,P);  d2(P,P) = d2(Eta,Eta);  d2(cp,rho_p) = dT/dcp + d2(cp,cp)
   while nothing has melted.
5. Direct vs GMRES solves: <= 5e-3 % (tiny entries).

## Two problems found in the OTI reference (not changed there)
1. **Second order drops the powder blend.**  Blocks (a2)(b2)(c2) use the bulk k / rho / hc while the forward model and the first
   order use the powder-blended ones.  The history term then multiplies d2T_n by M_bulk/dt but the Jacobian carries M_powder/dt, so
   the second-order fields are amplified by rho/rho_p = 1.67 every step: d2T/dcp2 = 2.7e3, 1.7e12, 1.5e23, 3.8e27 at steps 10/50/100/140
   in the reference, 9.2e2 (smooth, matches FD) in the blend-consistent version.  The Enzyme build uses the blend-consistent Hessian
   everywhere (`FB_REFERENCE2=1` restores the reference blocks to reproduce it).
2. **Tw (substrate bottom temperature) misses the initial condition.**  The substrate starts at T_sub_bc, so its IC depends on Tw, but
   only the Dirichlet rows were seeded: reference dT/dTw = 1e-107 vs FD 9e-3 at the probe.  Fixed here with an IC seed.  A residual
   disagreement with FD remains (13 % at step 26, 34 % at step 44; absolute size ~1e-2 K per unit scale) - not resolved.

## Cost (this node type, 16-24 threads)
Reference OTI: first order ~8 s/step, diagonal second order ~36 s/step (44 h for 4400 steps).  Enzyme: nominal 1.3 s/step (16 thr),
first order ~3.4 s/step, diagonal second order ~8 s/step, **full Hessian (210 fields) ~11 s/step** (24 cores, direct solves).

## Restart bug (found 2026-10-09)
Checkpoint/restart did not save `ever_melted` (the powder -> bulk flag of every node) nor the laser's `last_dir`.  A resumed run therefore treats
nodes that melted and cooled below the solidus as powder again: the nominal temperature itself drifts (1254 K vs 1146 K at step 4000) and
all sensitivities after the restart point are wrong.  The first full-Hessian run was resumed at step 1250 and is kept as
`results/deriv_second_order_INVALID_restart_bug`.  Fixed here (both fields are now in the checkpoint); `jobs/restart_test.slurm` shows a
killed-and-resumed run reproduces the continuous one (probe: identical, sensitivities: 1e-14).  The same omission exists in the original
OTI driver, so any reference run that was resumed from a checkpoint is affected after its restart point.
