// ============================================================
//  BRIDGE MULTILAYER: real NIST AMB2018-01 bridge geometry (truncated to
//  4 element rows, Z<=0.8mm), matching Abaqus job 772124
//  (thermal_4layer.odb) exactly -- same mesh, same coarse one-increment-
//  per-cycle philosophy, same material, same tie-constrained substrate.
//
//  Reuses xtrack_multilayer.cpp's nonlinear residual/Jacobian/Newton
//  machinery (IN625, JacobianOp, FachinottiResidualOp,
//  FachinottiLinearSolver, DampedNewtonSolver) UNCHANGED. New pieces:
//   - TieMap / ReducedResidualOp / ReducedLinearSolver (tie_constraint.hpp):
//     exact tie-constraint elimination, validated in isolation.
//   - SpatialHashSource: unstructured-mesh point-location (bounding-box
//     hash + MFEM's own InverseElementTransformation for the local solve)
//     replacing DirectBoxSubdivSource's structured-grid lookup, and a
//     "Concentrated" (point, not ellipsoidal-Goldak) deposition model
//     matching the bridge deck's own ABQ_AM_MovingHeatSource type.
//   - TableLaserParams: table-driven (linearly interpolated) toolpath
//     from es_bridge_laser_4layer.inp, replacing the closed-form linear-
//     segment LaserParams.
//   - Row-based birth times derived from the roller schedule.
//   - Celsius->Kelvin conversion at every bridge-deck-native input value
//     (ambient=40C, substrate BC=80C) -- physics stays in Kelvin
//     throughout, matching the rest of this project.
// ============================================================
#include "mfem.hpp"
#include "tie_constraint.hpp"
#include "pardiso_direct.hpp"
#include "src_kernel.hpp"                  // differentiable laser-source kernel (recorded-choice AD for rx,ry,rz,x0,Vx,tL)
#ifdef USE_ENZYME_AD
#include "fb_enzyme_kernels.h"
#endif
#include "query.hpp"          // v7: any-order derivative query (from full_bridge_directional_user/anyorder)
#ifdef USE_HYPRE_AMG
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD ((MPI_Comm)0)   // sequential hypre: communicator is a dummy int
#endif
#endif
#include "oti_type_generic30.hpp"
#include <chrono>
#include <memory>
#include <cstdio>
#include <omp.h>
#include <fstream>
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <iomanip>
#include <sstream>
#include <algorithm>

// ============================================================================
//  full_bridge port (2026-09-12): merges the full-bridge-specific
//  parallelization/checkpoint infra from full_bridge_activation_tier1.cpp
//  (the no-OTI, 552,822-element production driver) onto this file's own
//  validated Kokkos+OTI (A2/A3) sensitivity machinery. See
//  thermal_kokkos_oti/ for the merge notes. Key rule followed throughout:
//  wherever the two ancestors' own perf passes overlapped, the version that
//  is safe for the sensitivity/Jacobian code (matches this file's own
//  2026-09-11/09-12 correctness fixes -- Finalize(fix_empty_rows=false),
//  the Mdiag-based AssembleFull(), etc.) was kept/adapted; the full-bridge
//  ancestor's version was taken as-is only where the affected code has NO
//  OTI/sensitivity involvement at all.
// ============================================================================

// A2 perf pass (2026-09-12): Kokkos::parallel_for over the nodal
// sensitivity block, matching the reference single-layer driver's own
// pattern (OTI_driver_reference_bridge_local_oti_30param_kokkos.cpp).
// oti_type_generic30.hpp's own OTI_FUNCTION macro only pulls in the real
// <Kokkos_Core.hpp> when -DOTI_ENABLE_KOKKOS is passed (degrading every
// OTI method to plain inline otherwise) -- this file follows the SAME
// optional-dependency philosophy for Kokkos::parallel_for/KOKKOS_LAMBDA
// themselves: a real Kokkos backend when enabled, a trivial serial-loop
// shim otherwise, so the file always builds and runs identically (just
// without the parallel speedup) with no Kokkos install at all.
#ifdef OTI_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#else
namespace Kokkos {
    template <typename F>
    inline void parallel_for(const char *, int n, F &&f) { for (int i = 0; i < n; i++) f(i); }
}
#define KOKKOS_LAMBDA [=]
#endif

using namespace std;
using namespace mfem;

const double C2K = 273.0; // bridge deck's own convention: ABSOLUTE ZERO=-273

// A2 toy-problem FD validation: multiplicative scale overrides on the
// FORWARD (double-precision) material path, matching S_K/S_CP's own
// convention exactly (k_eff=k(T)*scale). Default 1.0 -- unused elsewhere.
// Purpose: run the forward solve at kscale=1+/-h (or cpscale=1+/-h) and
// central-difference the probe temperature, to check against dT/d(s_K)
// (resp. s_CP) already reported by the -dorder 1 OTI pass -- a toy check
// on the NEW parts of this driver (fully-implicit-k sensitivity + tie-
// reduction interaction), not a re-validation of the OTI/Kokkos machinery
// itself (already cross-validated vs Abaqus in the single-layer driver).
static double g_kscale = 1.0, g_cpscale = 1.0;
static const std::vector<double> *g_ever_melted = nullptr;   // v8: set once in main(); nullptr = pure bulk everywhere
// Melt-regime FD verification (2026-09-12): same toy-FD convention as
// g_kscale/g_cpscale above, extended to rho and L_fusion so dT/drho and
// dT/dLfus (both structurally zero pre-melt, so never independently FD-
// checked before) can be verified once T crosses T_solidus. rho_dev()
// applies g_rhoscale internally (mirrors k()/cp_dev()); L_fusion is a
// plain constant used directly at 3 call sites, so g_Lscale is applied at
// each of those instead of wrapping it in a function.
static double g_rhoscale = 1.0, g_Lscale = 1.0;
// P3 (NW=10) FD verification: P and eta are pure multiplicative scale
// factors on the ALREADY-ASSEMBLED nodal source field f_src (see block
// (d)'s own comment) -- no existing CLI flag scales the whole laser
// deposition globally, so these toys apply directly to f_src right after
// BuildSource(), mirroring g_kscale/g_cpscale's own convention.
static double g_pscale = 1.0, g_etascale = 1.0;
// P5 (NW=16) FD verification: x0 is a pure rigid x-shift of the toolpath
// (RHS[LI_X0] = dfdx directly, the reference's own comment calls this
// "exact"). Applying a constant offset directly inside
// TableLaserParams::position() gives the most direct possible FD test:
// -xshift IS x0's own physical definition, not an indirect proxy for it.
// -yshift/-zshift exist purely to independently validate
// BuildSourcePosGrad's dfdy/dfdz components (which Vx/tL's own chain-rule
// combination also depends on), since y/z aren't reachable via any other
// wired parameter.
static double g_xshift = 0.0, g_yshift = 0.0, g_zshift = 0.0;
// Vx/tL diagnostic hooks (2026-09-12), NOT a valid end-to-end FD check for
// this toolpath: g_vxscale/g_tshift perturb the time argument INSIDE
// TableLaserParams::position() only, so BuildSource's per-segment energy
// (laser.p[i]) and travel direction (laser.x[i+1]-laser.x[i], both indexed
// by the UNTRANSFORMED bracket(t0)/bracket(t1) done outside position())
// stay tied to the nominal segment i while the deposited xL/yL/zL jumps to
// whatever segment the TRANSFORMED time re-brackets to. es_bridge_laser_
// 4layer.inp has raw segments as short as ~0.0005 time-units (rapid
// weld-pulse on/off), far shorter than any h that avoids this, so a full-
// trajectory FD via these hooks measures segment-bracket mismatch noise,
// not the true dT/dVx or dT/dTL derivative (confirmed: rd swung
// erratically from 3.5 to 200+ with no smooth h-dependence, unlike the
// clean <0.001% x0 result from the SAME gx/gy/gz these both depend on).
// Doing this properly would require transforming t0/t1 in BuildSource's
// OWN bracket calls too -- out of scope here, and risks the already-
// validated k..rz parameters sharing that function. Left in place only
// for future debugging; do not use them as a validation claim for Vx/tL.
static double g_vxscale = 0.0, g_tshift = 0.0;
// Vx is a time scaling about g_tpivot (= window start -ts), the SAME definition used by the AD source kernel
// (src_kernel.hpp) and by block (f)'s closed-form chain rule, so -vxscale/-tshift FD verify it exactly.
static double g_tpivot = 0.0;
#ifdef USE_ENZYME_AD
static const bool g_src_fd_legacy = (getenv("SRC_FD_LEGACY") != nullptr);   // old FD source derivatives (comparison only)
#else
static const bool g_src_fd_legacy = true;                                   // OTI build keeps the FD source blocks
#endif

// =========================================================================
//  IN625 material -- byte-identical to xtrack_multilayer.cpp (Kelvin)
// =========================================================================
namespace IN625 {
// Corrected 2026-07-15: every breakpoint in this namespace (T_solidus,
// T_liquidus, and every k(T)/cp(T) table temperature) previously used a
// standard +273.15 C->K conversion (294.15, 311.15, ...), inconsistent
// with C2K=273.0 used everywhere else in this file (T_ambient,
// T_substrate_bc) -- which itself matches the deck's own explicit
// convention: "*PHYSICAL CONSTANTS, ABSOLUTE ZERO=-273" (not the standard
// -273.15). Found while cross-checking the deck's raw *Conductivity/
// *Specific Heat/*Latent Heat values against this file during the
// plateau-gap investigation. Every breakpoint below is now T_C + 273.0,
// matching the deck's own convention exactly (a small, systematic
// ~0.15K offset previously existed in every table breakpoint location).
// NOT const (2026-09-12): melt-regime FD validation needs to artificially
// lower these breakpoints via CLI override (-tsolidus/-tliquidus), since
// bridge_4layer.mesh does not resolve a real melt pool at nominal power
// (a single laser pass only reaches ~550K here vs the real 1563K -- a
// mesh-resolution gap flagged separately in PLAN.md, not a code bug). This
// only isolates/validates the phase-change SENSITIVITY MATH (dT/dLfus
// turn-on, post-melt dT/dcp != dT/drho) at a reachable crossing point; the
// real 1563/1623K values stay the default for all non-test runs.
static double T_solidus  = 1563.0;
static double T_liquidus = 1623.0;
// Corrected 2026-07-14 to exactly match this bridge deck's own *MATERIAL,
// NAME=IN625 block in thermal.inp (272E9 in the deck's mJ/(tonne.K)-style
// convention) -- was 2.9e11, a ~6.6% mismatch found while auditing the
// property tables during the plateau-gap investigation.
static const double L_fusion   = 2.72e11;
static double lerp_dev(double T, double T0, double V0, double T1, double V1) {
    return V0 + (T - T0) / (T1 - T0) * (V1 - V0);
}
static double cp_dev(double T) {
    // Corrected 2026-07-14: matched through 1144.0K already, but was
    // missing this deck's own last two *Specific Heat points (982C/
    // 1255.0K, 1093C/1366.0K) and invented values up to T_solidus/
    // T_liquidus not present in the deck; Abaqus holds the last tabulated
    // value (6.7e8 @ 1366.0K) constant beyond the table's end.
    double v;
    if (T <= 294.0)       v = 4.1e8;
    else if (T < 366.0)   v = lerp_dev(T,294.0,4.1e8,366.0,4.27e8);
    else if (T < 477.0)   v = lerp_dev(T,366.0,4.27e8,477.0,4.56e8);
    else if (T < 589.0)   v = lerp_dev(T,477.0,4.56e8,589.0,4.81e8);
    else if (T < 700.0)   v = lerp_dev(T,589.0,4.81e8,700.0,5.11e8);
    else if (T < 811.0)   v = lerp_dev(T,700.0,5.11e8,811.0,5.36e8);
    else if (T < 922.0)   v = lerp_dev(T,811.0,5.36e8,922.0,5.65e8);
    else if (T < 1033.0)  v = lerp_dev(T,922.0,5.65e8,1033.0,5.90e8);
    else if (T < 1144.0)  v = lerp_dev(T,1033.0,5.90e8,1144.0,6.20e8);
    else if (T < 1255.0)  v = lerp_dev(T,1144.0,6.20e8,1255.0,6.45e8);
    else if (T < 1366.0)  v = lerp_dev(T,1255.0,6.45e8,1366.0,6.70e8);
    else                  v = 6.70e8;
    return v * g_cpscale;
}
// d(cp)/dT (2026-07-15): exact derivative of the piecewise-linear cp(T)
// table above -- a.e. constant, zero outside the tabulated range (constant
// extrapolation). Needed for the capacitance term's own missing Jacobian
// chain-rule piece: d/dT_i[Mc(T_i)/dt*(T_i-Tn_i)] = Mc(T_i)/dt +
// (dMc/dT_i)*(T_i-Tn_i)/dt -- the forward solver's Jacobian only ever used
// the first term. This exact term (dMc/dT*(T-Tn)) was already identified
// as a real, non-negligible (~13-16%) correction in this project's earlier
// Figure-12/OTI sensitivity work -- see project memory -- but was never
// applied to THIS forward solver's own Jacobian.
static double cp_prime(double T) {
    double v;
    if (T <= 294.0)        v = 0.0;
    else if (T < 366.0)    v = (4.27e8-4.1e8)/(366.0-294.0);
    else if (T < 477.0)    v = (4.56e8-4.27e8)/(477.0-366.0);
    else if (T < 589.0)    v = (4.81e8-4.56e8)/(589.0-477.0);
    else if (T < 700.0)    v = (5.11e8-4.81e8)/(700.0-589.0);
    else if (T < 811.0)    v = (5.36e8-5.11e8)/(811.0-700.0);
    else if (T < 922.0)    v = (5.65e8-5.36e8)/(922.0-811.0);
    else if (T < 1033.0)   v = (5.90e8-5.65e8)/(1033.0-922.0);
    else if (T < 1144.0)   v = (6.20e8-5.90e8)/(1144.0-1033.0);
    else if (T < 1255.0)   v = (6.45e8-6.20e8)/(1255.0-1144.0);
    else if (T < 1366.0)   v = (6.70e8-6.45e8)/(1366.0-1255.0);
    else                   v = 0.0;
    return v * g_cpscale;
}
static double rho_dev(double T) {
    // Corrected 2026-07-14: this bridge deck's own *MATERIAL, NAME=IN625
    // block defines *Density with a SINGLE value (8.44e-09), no
    // temperature column at all -- i.e. Abaqus treats it as T-independent
    // for this deck, unlike the T-dependent table previously here (sourced
    // from a different, generic IN625 reference used elsewhere in the
    // project, but not matching THIS deck's own material definition).
    (void)T;
    return 8.44e-9 * g_rhoscale;
}
static double rho_cp_dev(double T) { return rho_dev(T) * cp_dev(T); }
// Abaqus *LATENT HEAT, SMOOTH internal-energy fraction:
// f=xi^3(10-15xi+6xi^2), xi=(T-Ts)/(Tl-Ts).
static double f_L_dev(double T) {
    if (T <= T_solidus)  return 0.0;
    if (T >= T_liquidus) return 1.0;
    double xi=(T-T_solidus)/(T_liquidus-T_solidus);
    return xi*xi*xi*(10.0-15.0*xi+6.0*xi*xi);
}
static double df_L_dT_dev(double T) {
    if (T <= T_solidus || T >= T_liquidus) return 0.0;
    double DT=T_liquidus-T_solidus;
    double xi=(T-T_solidus)/DT;
    double omx=1.0-xi;
    return 30.0*xi*xi*omx*omx/DT;
}
// v8 powder/bulk material-state constants (2026-10-02, user request). NOT digitized from the paper's own
// Figure 3 (not numerically extractable from the PDF text layer); literature-typical values for a loose
// gas-atomized metal powder bed: effective conductivity via point-contact conduction is commonly modeled
// as a fraction of the bulk solid's own k(T) curve (shape unchanged, magnitude reduced); apparent/tap
// density of virgin powder is commonly ~55-65% of the bulk solid density. hp/hc ratio taken directly from
// the paper's own Table 1 (hp=8, hc=80 -> 0.1), applied to THIS bridge deck's own validated h_conv.
static double KP_FRACTION = 0.2, RHOP_FRACTION = 0.6, HP_OVER_HC = 0.1;   // runtime-overridable (-kpfrac/-rhopfrac/-hpratio) for FD checks of kp/rho_p/hp
static double k(double T) {
    // Corrected 2026-07-14 to exactly match this bridge deck's own
    // *Conductivity table in thermal.inp -- the previous table was
    // missing two of the deck's own data points (10.1 @ 311.0K,
    // 25.2 @ 1255.0K) and invented an unsupported ramp to k=30 above
    // 1563.0K; the deck's table actually stops at 1255.0K/25.2, and
    // Abaqus's default (no EXTRAPOLATION option given) holds the last
    // value constant beyond the table's end, not ramping further.
    static const double Ta[11] = {294.0,311.0,366.0,477.0,589.0,700.0,
                                    811.0,922.0,1033.0,1144.0,1255.0};
    static const double Va[11] = {9.8,10.1,10.8,12.5,14.1,15.7,
                                    17.5,19.0,20.8,22.8,25.2};
    double v;
    if (T <= Ta[0]) v = Va[0];
    else if (T >= Ta[10]) v = Va[10];
    else { int i = 0; while (i < 9 && Ta[i+1] < T) i++; v = Va[i] + (T-Ta[i])/(Ta[i+1]-Ta[i])*(Va[i+1]-Va[i]); }
    return v * g_kscale;
}
// dk/dT (2026-09-11, A2 sensitivity-solve correction): exact derivative of
// the piecewise-linear k(T) table above -- a.e. constant, zero outside the
// tabulated range (constant extrapolation), mirroring cp_prime's own
// pattern. NOT used by the forward Newton solve's own JacobianOp (that
// omission is this file's own documented, intentional chord approximation
// -- see JacobianOp's "dk/dT term tried and REMOVED" comment; harmless
// there since Newton self-corrects regardless of the linearization used to
// reach convergence). It DOES matter for the sensitivity solve: unlike the
// forward problem, J*dT/ds=RHS uses J as the literal governing operator of
// the answer, not just a search direction -- an internal FD/Taylor-
// remainder check (2026-09-11) found the sensitivity solve using the same
// chord-approximate J gave dT/dk off by ~9x from a direct finite-difference
// reference, confirming this term is NOT negligible for sensitivity
// accuracy even though it is for Newton's own convergence rate.
static double kp_dev(double T) { return k(T) * KP_FRACTION * (1.0 / g_kscale); }  // k(T) already carries g_kscale; undo it, kp has its own scale
static double rhop_dev(double T) { (void)T; return rho_dev(T) * RHOP_FRACTION * (1.0 / g_rhoscale); }  // same: rho_dev carries g_rhoscale
static double k_prime(double T) {
    static const double Ta[11] = {294.0,311.0,366.0,477.0,589.0,700.0,
                                    811.0,922.0,1033.0,1144.0,1255.0};
    static const double Va[11] = {9.8,10.1,10.8,12.5,14.1,15.7,
                                    17.5,19.0,20.8,22.8,25.2};
    double v;
    if (T <= Ta[0] || T >= Ta[10]) v = 0.0;
    else { int i = 0; while (i < 9 && Ta[i+1] < T) i++; v = (Va[i+1]-Va[i])/(Ta[i+1]-Ta[i]); }
    return v * g_kscale;
}

// =========================================================================
// A2: OTI-typed material twins for the NW=4 sensitivity directions
// (k, cp, rho, Lfus). S_K/S_CP/S_RHO/S_L are UNIT (1.0) multiplicative
// scale factors -- k_oti(T)=k(T)*S_K etc, matching the validated single-
// layer driver's own convention (bridge_local_oti_30param_kokkos.cpp).
// Table VALUES here are byte-identical to this file's own IN625::k/cp_dev/
// rho_dev/f_L_dev above (same physical deck, verified by direct
// comparison against the reference driver's own k_Ta/k_Va/cp table --
// same IN625 bridge deck, same numbers).
// =========================================================================
static OTI S_K, S_CP, S_RHO, S_L;   // seeded in main(): S_K on basis 1, ... S_L on basis 4
static OTI S_KP, S_RHOP, S_HP;   // v8: powder conductivity/density, part-cake convection coefficient
// P2 (NW=8, 2026-09-12): Ts/Tl/hc/T0, ported from the reference driver's
// own P2 staging (OTI_driver_reference_bridge_local_oti_30param_kokkos.cpp).
// SAME "unit-scale multiplier" convention as S_K/S_CP/S_RHO/S_L above --
// none of these are seeded directly on themselves; T_solidus_oti =
// OTI(T_solidus)*S_TS etc, so the nominal value is unchanged but every OTI
// expression built from it carries d/d(Ts,Tl,hc,T0). S_HC/S_T0 are used in
// the NEW "(b) boundary" sensitivity block below; S_TS/S_TLIQ in f_L_oti.
static OTI S_TS, S_TLIQ, S_HC, S_T0;
static OTI T_solidus_oti, T_liquidus_oti;   // set in main() before the time loop

static OTI pwlin_oti(const double *Ta, const double *Va, int n, const OTI &T) {
    double Tr = T.real();
    if (Tr <= Ta[0]) return OTI(Va[0]);
    if (Tr >= Ta[n-1]) return OTI(Va[n-1]);
    int i = 0; while (i < n-2 && Ta[i+1] < Tr) ++i;
    return Va[i] + (T-Ta[i])/(Ta[i+1]-Ta[i])*(Va[i+1]-Va[i]);
}
static OTI kp_oti(const OTI &T) {
    static const double Ta[11] = {294.0,311.0,366.0,477.0,589.0,700.0,
                                    811.0,922.0,1033.0,1144.0,1255.0};
    static const double Va[11] = {9.8,10.1,10.8,12.5,14.1,15.7,
                                    17.5,19.0,20.8,22.8,25.2};
    return pwlin_oti(Ta,Va,11,T) * OTI(KP_FRACTION) * S_KP;
}
static OTI k_oti(const OTI &T) {
    static const double Ta[11] = {294.0,311.0,366.0,477.0,589.0,700.0,
                                    811.0,922.0,1033.0,1144.0,1255.0};
    static const double Va[11] = {9.8,10.1,10.8,12.5,14.1,15.7,
                                    17.5,19.0,20.8,22.8,25.2};
    return pwlin_oti(Ta,Va,11,T) * S_K;
}
static OTI cp_oti(const OTI &T) {
    static const double Ta[11] = {294.,366.,477.,589.,700.,811.,922.,1033.,1144.,1255.,1366.};
    static const double Va[11] = {4.1e8,4.27e8,4.56e8,4.81e8,5.11e8,5.36e8,5.65e8,5.90e8,6.20e8,6.45e8,6.70e8};
    return pwlin_oti(Ta,Va,11,T) * S_CP;
}
static OTI rho_oti(const OTI &T) { (void)T; return OTI(8.44e-9) * S_RHO; }
static OTI rhop_oti(const OTI &T) { (void)T; return OTI(8.44e-9 * RHOP_FRACTION) * S_RHOP; }
static OTI f_L_oti(const OTI &T) {
    // Clamp on the REAL value only (matches f_L_dev's own branch structure
    // and the reference driver's own f_L_oti -- a smoothstep evaluated
    // outside [Ts,Tl] isn't what Abaqus does). Inside the open interval,
    // xi uses the OTI-typed T_solidus_oti/T_liquidus_oti (P2, 2026-09-12)
    // so d/d(Ts,Tl) flows through via ordinary OTI arithmetic.
    double Tr = T.real();
    if (Tr <= T_solidus)  return OTI(0.0);
    if (Tr >= T_liquidus) return OTI(1.0);
    OTI xi = (T - T_solidus_oti) / (T_liquidus_oti - T_solidus_oti);
    return xi*xi*xi*(10.0 - 15.0*xi + 6.0*xi*xi);
}
static OTI Lfusion_oti() { return OTI(L_fusion) * S_L; }

} // namespace IN625

// A2 sensitivity directions (compact numbering, 1-indexed OTI basis via WB(p)).
// P1 (NW=4, refactor-only, byte-exact nominal path) DONE + FD-validated.
// P2 (NW=8): +Ts,Tl,hc,T0, DONE + FD-validated.
// P3 (NW=10): +P,eta (laser power, absorption efficiency), DONE + FD-validated.
// P4 (NW=13, 2026-09-12): +rx,ry,rz (Goldak ellipsoid shape). All matching
// the reference single-layer driver's own P1..P5 staging exactly. Unlike
// P/eta (pure multiplicative scale, closed-form), rx/ry/rz RESHAPE the
// heat source -- genuinely nonlinear in f_src -- so the reference uses
// CENTRAL FINITE DIFFERENCE on the plain double-precision BuildSource
// (not an OTI-typed source geometry) to get d(f_src)/d(rx,ry,rz): see
// block (d) [P/eta] then the new source-shape FD block below. rx moves
// goldak_af AND goldak_ar together (front+rear travel-axis size); ry
// moves goldak_b alone (transverse); rz moves goldak_c alone
// (penetration) -- goldak_af/ar's own front/rear ASYMMETRY is never
// independently differentiated (matches the reference's own documented,
// intentional omission). No IC seed (fresh per-step forcing, like P/eta).
// P5 (NW=16, 2026-09-12): +x0,Vx,tL (toolpath rigid x-shift, scan-speed
// fractional scale, lead-time shift). A THIRD distinct mechanism: these
// TRANSLATE the deposition point, and FD across a re-run FindElement
// would inject element-crossing discontinuities, so the reference uses a
// frozen-element ANALYTIC POSITION GRADIENT (SpatialHashSource::
// BuildSourcePosGrad, d(f_src)/d(deposition x/y/z)) plus a closed-form
// chain rule: d(pos)/d(x0)=(1,0,0); d(pos)/d(Vx)=elapsed_time*velocity
// (elapsed_time measured from t_start, the run's own window start);
// d(pos)/d(tL)=-velocity (a delay = a negative along-path shift). No IC
// seed, purely additive to RHS from the source term (same family as
// blocks (d)/(e) above) -- see the new block right after rx/ry/rz.
static constexpr int NW = 26;
enum WLocal { LI_K = 0, LI_CP = 1, LI_RHO = 2, LI_LFUS = 3,
              LI_TS = 4, LI_TLIQ = 5, LI_HC = 6, LI_T0 = 7,
              LI_P = 8, LI_ETA = 9, LI_RX = 10, LI_RY = 11, LI_RZ = 12,
              LI_X0 = 13, LI_VX = 14, LI_TL = 15,
              // v8 (2026-10-02, user request): the paper's (Rincon-Tabares et al. 2025, Table 1) remaining
              // 10 "non-structural" parameters our bridge never wired. Tw is wired below (build-platform
              // Dirichlet BC value, same mechanism as T0's IC seed). hp/kp/rho_p/bp/lp/wp/b/l/w are NOT yet
              // wired -- the slots exist so dTds[LI_KP] etc. stay correctly 0 (nothing seeds them) until
              // implemented; see the driver README for the staged plan.
              LI_TW = 16, LI_HP = 17, LI_KP = 18, LI_RHOP = 19,
              LI_BP = 20, LI_LP = 21, LI_WP = 22, LI_B = 23, LI_L = 24, LI_W = 25 };
static constexpr int WB(int p) { return p + 1; }

// ---- full second-order Hessian (SENS_MIXED=1, needs -dorder 2 and the Enzyme build) ----
// Adds every mixed d2T/dp_i dp_j (i<j) among the NACT wired parameters (LI_K .. LI_RHOP; y0/Vy/lL/T0p and
// bp..w are structurally zero) to the pure ones. All local second derivatives come from Enzyme nested
// forward mode (fb_local_hess / ak_src_hess); the SAME exact Jacobian solves every field.
static const bool g_sens_mixed = (std::getenv("SENS_MIXED") != nullptr);
// FB_NOBLEND2=1: second-order kernels drop the powder blends (mf=Mv=1), reproducing the reference OTI diagonal blocks
static const bool g_noblend2 = (std::getenv("FB_NOBLEND2") != nullptr);
// OTI branch of the second-order blocks: OTI_BLEND2_OFF=1 restores the reference's blend-less (a2)(b2)(c2)
static const bool g_oti_blend2_off = (std::getenv("OTI_BLEND2_OFF") != nullptr);
static constexpr int NACT = 20;
static constexpr int NPAIR = NACT * (NACT - 1) / 2;
static int g_pi[NPAIR], g_pj[NPAIR];
static int g_pair_index[NW][NW];
static void init_pairs() { int k = 0; for (int i = 0; i < NACT; i++) for (int j = i + 1; j < NACT; j++) { g_pi[k] = i; g_pj[k] = j; g_pair_index[i][j] = g_pair_index[j][i] = k; k++; } }

class ActiveTDepKCoefficient : public Coefficient {
    const GridFunction *T_gf;
    const vector<char> *attr_active;
public:
    ActiveTDepKCoefficient(const GridFunction &gf, const vector<char> &active)
        : T_gf(&gf), attr_active(&active) {}
    double Eval(ElementTransformation &T, const IntegrationPoint &ip) override {
        if (!(*attr_active)[T.Attribute]) return 0.0;
        return IN625::k(T_gf->GetValue(T, ip));
    }
};

// =========================================================================
//  Table-driven laser toolpath (linear interpolation between waypoints),
//  replacing the closed-form segment-based LaserParams -- the bridge's
//  own toolpath is a real point-to-point event series, not simple
//  straight scan lines.
// =========================================================================
struct TableLaserParams {
    vector<double> t, x, y, z, p;
    double absorptivity = 0.45; // bridge deck's own value
    void Load(const string &path) {
        ifstream f(path);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            double tt,xx,yy,zz,pp; char c;
            istringstream ss(line);
            ss >> tt >> c >> xx >> c >> yy >> c >> zz >> c >> pp;
            t.push_back(tt); x.push_back(xx); y.push_back(yy); z.push_back(zz); p.push_back(pp);
        }
    }
    int bracket(double tt) const {
        if (t.empty() || tt <= t.front()) return 0;
        if (tt >= t.back()) return (int)t.size()-2;
        int lo=0, hi=(int)t.size()-1;
        while (hi-lo>1) { int mid=(lo+hi)/2; if (t[mid]<=tt) lo=mid; else hi=mid; }
        return lo;
    }
    void position(double tt, double &xL, double &yL, double &zL) const {
        tt = g_tpivot + (tt - g_tpivot)*(1.0+g_vxscale) - g_tshift;   // Vx/tL FD verification (pivot = window start)
        int i = bracket(tt);
        double f = (t[i+1]>t[i]) ? (tt-t[i])/(t[i+1]-t[i]) : 0.0;
        f = max(0.0, min(1.0, f));
        xL = x[i] + f*(x[i+1]-x[i]);
        yL = y[i] + f*(y[i+1]-y[i]);
        zL = z[i] + f*(z[i+1]-z[i]);
        xL += g_xshift; yL += g_yshift; zL += g_zshift;   // P5 FD verification
    }
    // ZERO-ORDER HOLD (2026-07-14, confirmed via Abaqus's own "Point
    // Toolpath-Mesh Intersection" documentation): "The field defined for
    // a point Xn remains constant over the segment connecting Xn and
    // Xn+1" -- i.e. the power/state field is a STEP function, held
    // constant at p[i] for the WHOLE segment until the next table row,
    // NOT linearly interpolated toward p[i+1] (unlike position, which the
    // same documentation confirms IS linearly interpolated: "the tool
    // travels at a constant velocity over a segment"). This was
    // previously (wrongly) linearly interpolated here, which systematically
    // OVERESTIMATED energy across every on/off transition -- a brief
    // "off" table row was treated as a momentary linear dip back up to
    // the next row's value, instead of a full-width hold at zero for the
    // entire gap until that next row.
    double power_mW(double tt) const {
        int i = bracket(tt);
        return p[i];
    }
    bool is_active(double tt) const { return power_mW(tt) > 1.0; }
    double t_final() const { return t.empty() ? 0.0 : t.back(); }
};

// =========================================================================
//  Spatial-hash point-location ("Concentrated" source: deposit onto the
//  containing element's nodes via its own shape functions at the exact
//  point, using MFEM's InverseElementTransformation for the local solve).
// =========================================================================
// recorded discrete choices of one BuildSource call (input of the AD source kernel)
struct SrcTape { std::vector<SrcSub> subs; std::vector<SrcCell> cells; double t0 = 0, t1 = 0; };

class SpatialHashSource {
    Mesh *mesh; FiniteElementSpace *fes;
    double cell;
    unordered_map<long, vector<int>> grid;
    vector<array<double,3>> lo, hi;
    const vector<char> *attr_active;
    int nseg;
public:
    // Spatial energy-distribution kernel (2026-07-16, diagnostic A/B test):
    // 0 (default) = shape-function-weighted deposition at the exact toolpath
    // point, the mathematically standard way to convert a point load into
    // nodal-equivalent FE loads. 1 = uniform deposition across the
    // containing element's own DOFs (1/n_dof each), ignoring the point's
    // exact position within the element -- a plausible alternative to
    // Abaqus's own (undocumented in detail) internal Toolpath-Mesh
    // Intersection Module kernel. Testing whether the RESULT is sensitive
    // to this choice, since the temporal semantics (zero-order-hold power,
    // linear position) are already confirmed to match Abaqus's own
    // documentation, but the exact spatial smearing algorithm is not.
    // 2 = Goldak double-ellipsoidal volumetric source (2026-07-15): the
    // deck's own *DFLUX/ABQ_AM_MovingHeatSource keyword is confirmed
    // "Concentrated" (point), NOT Goldak -- so this is a deliberate
    // diagnostic departure from a literal Abaqus replication, requested to
    // test whether spreading energy over a small ellipsoidal melt-pool
    // volume (elongated along the travel direction, asymmetric front/rear)
    // changes the zigzag spatial-error pattern found in the row1 full-field
    // comparison, as an alternative explanation to the exact-point/element
    // deposition already tested via modes 0/1.
    int spread_mode = 0;
    double goldak_af = 0.3, goldak_ar = 0.6, goldak_b = 0.3, goldak_c = 0.3; // mm, CLI-tunable
    double goldak_ff = 0.6, goldak_fr = 1.4; // front/rear energy fraction, ff+fr=2 (classic Goldak split)
    int goldak_subdiv = 3;
    double goldak_cutoff = 3.0;
    mutable array<double,3> last_dir = {1.0, 0.0, 0.0}; // fallback travel direction for degenerate (zero-length) segments
    mutable long seam_skip_count = 0, seam_total_count = 0; // debug counters, BuildSourcePosGrad's near_seam guard
    SpatialHashSource(Mesh *m, FiniteElementSpace *f, const vector<char> &active, int nseg_, double cellsize=1.0)
        : mesh(m), fes(f), cell(cellsize), attr_active(&active), nseg(nseg_)
    {
        int ne = mesh->GetNE();
        lo.resize(ne); hi.resize(ne);
        Array<int> verts;
        for (int e = 0; e < ne; e++) {
            mesh->GetElementVertices(e, verts);
            double l[3]={1e30,1e30,1e30}, h[3]={-1e30,-1e30,-1e30};
            for (int k = 0; k < verts.Size(); k++) {
                const double *c = mesh->GetVertex(verts[k]);
                for (int d=0;d<3;d++){ l[d]=min(l[d],c[d]); h[d]=max(h[d],c[d]); }
            }
            for (int d=0;d<3;d++){ lo[e][d]=l[d]; hi[e][d]=h[d]; }
            long cxlo=(long)floor(l[0]/cell), cxhi=(long)floor(h[0]/cell);
            long cylo=(long)floor(l[1]/cell), cyhi=(long)floor(h[1]/cell);
            long czlo=(long)floor(l[2]/cell), czhi=(long)floor(h[2]/cell);
            for (long cx=cxlo; cx<=cxhi; cx++)
            for (long cy=cylo; cy<=cyhi; cy++)
            for (long cz=czlo; cz<=czhi; cz++)
                grid[(cx*73856093L)^(cy*19349663L)^(cz*83492791L)].push_back(e);
        }
    }
    int FindElement(double x,double y,double z) const {
        long cx=(long)floor(x/cell), cy=(long)floor(y/cell), cz=(long)floor(z/cell);
        auto it = grid.find((cx*73856093L)^(cy*19349663L)^(cz*83492791L));
        if (it != grid.end()) {
            for (int e : it->second) {
                if (!(*attr_active)[mesh->GetAttribute(e)]) continue;
                if (x>=lo[e][0]&&x<=hi[e][0]&&y>=lo[e][1]&&y<=hi[e][1]&&z>=lo[e][2]&&z<=hi[e][2])
                    return e;
            }
        }
        return FindElementFallback(x,y,z);
    }
    // Robust fallback: search a widening ring of neighboring hash cells
    // for the nearest active element by centroid distance, so a toolpath
    // sample that doesn't land inside any element's exact AABB (hash-cell
    // gap, or floating-point edge case at a shared boundary) still
    // deposits its energy somewhere reasonable instead of being silently
    // dropped.
    int FindElementFallback(double x,double y,double z) const {
        long cx=(long)floor(x/cell), cy=(long)floor(y/cell), cz=(long)floor(z/cell);
        int best_e = -1; double best_d2 = 1e30;
        for (int ring = 0; ring <= 3; ring++) {
            for (long dx=-ring; dx<=ring; dx++)
            for (long dy=-ring; dy<=ring; dy++)
            for (long dz=-ring; dz<=ring; dz++) {
                if (max({std::abs(dx),std::abs(dy),std::abs(dz)}) != ring) continue;
                auto it = grid.find(((cx+dx)*73856093L)^((cy+dy)*19349663L)^((cz+dz)*83492791L));
                if (it == grid.end()) continue;
                for (int e : it->second) {
                    if (!(*attr_active)[mesh->GetAttribute(e)]) continue;
                    double ecx=0.5*(lo[e][0]+hi[e][0]), ecy=0.5*(lo[e][1]+hi[e][1]), ecz=0.5*(lo[e][2]+hi[e][2]);
                    double d2=(x-ecx)*(x-ecx)+(y-ecy)*(y-ecy)+(z-ecz)*(z-ecz);
                    if (d2 < best_d2) { best_d2 = d2; best_e = e; }
                }
            }
            if (best_e >= 0) return best_e;
        }
        return best_e;
    }
    // Collect every active element whose bounding box overlaps a cube of
    // half-width R centered at (x,y,z). The Goldak load is then integrated
    // as a true volumetric FE load over these elements.
    void GatherNearbyActiveElements(double x, double y, double z, double R,
                                    unordered_set<int> &elements_out) const {
        elements_out.clear();
        long ring = (long)ceil(R/cell) + 1;
        long cx=(long)floor(x/cell), cy=(long)floor(y/cell), cz=(long)floor(z/cell);
        for (long dx=-ring; dx<=ring; dx++)
        for (long dy=-ring; dy<=ring; dy++)
        for (long dz=-ring; dz<=ring; dz++) {
            auto it = grid.find(((cx+dx)*73856093L)^((cy+dy)*19349663L)^((cz+dz)*83492791L));
            if (it == grid.end()) continue;
            for (int e : it->second) {
                if (elements_out.count(e)) continue;
                if (!(*attr_active)[mesh->GetAttribute(e)]) continue;
                if (x-R>hi[e][0]||x+R<lo[e][0]||y-R>hi[e][1]||y+R<lo[e][1]||z-R>hi[e][2]||z+R<lo[e][2]) continue;
                elements_out.insert(e);
            }
        }
    }
    // Deposit total absorbed power (mW) at time window [t_end-dt, t_end],
    // as a point/"Concentrated" source via shape functions.
    //
    // EXACT breakpoint integration with WITHIN-SEGMENT sub-sampling
    // (2026-07-14, fourth attempt). Two earlier attempts assumed LINEAR
    // interpolation of the power field between table rows -- wrong, per
    // Abaqus's own "Point Toolpath-Mesh Intersection" documentation:
    // "The field defined for a point Xn remains constant over the segment
    // connecting Xn and Xn+1" (zero-order hold; position, by contrast, IS
    // linearly interpolated -- "constant velocity... over a segment").
    // Fixing that closed the ENERGY-MAGNITUDE gap (confirmed: per-sweep
    // average power is now flat/stable, matching the observed temperature
    // plateau's mean level). But depositing each segment's entire energy
    // at a single midpoint still left a spatial artifact: the same
    // documentation states that for each element the toolpath crosses,
    // Abaqus computes the exact entry/exit points and TIMES of that
    // intersection -- i.e. a segment's constant power is integrated
    // continuously along its length, split across every element it
    // actually passes through (proportional to time spent in each), not
    // dumped at one representative point. A single raw segment can span
    // several elements (segments cover mm-scale travel; elements are
    // ~0.24mm), so point-depositing concentrated energy unevenly
    // sweep-to-sweep depending on where each segment's midpoint happened
    // to land -- producing a real, observed sawtooth in the probe history
    // that Abaqus's own smooth reference curve does not show. Rather than
    // implement exact line-element intersection geometry, this
    // approximates the same effect by finely sub-sampling each segment in
    // time (constant velocity -> even spatial sub-sampling too) and
    // depositing an equal fractional share of that segment's (exact,
    // already-correct) total energy at each sub-sample -- the total
    // energy per segment is unchanged, only how it's spatially spread.
    void RecordCell(SrcTape *tape, SrcSub &rec, int e, double xi0, double xi1, double xi2, const Array<int> &dofs,
                    int ix, int iy, int iz, int const_shape) const {
        SrcCell c; c.ix = ix; c.iy = iy; c.iz = iz; c.const_shape = const_shape;
        c.xi0[0] = xi0; c.xi0[1] = xi1; c.xi0[2] = xi2;
        Array<int> ev; mesh->GetElementVertices(e, ev);
        MFEM_VERIFY(ev.Size() == 8 && dofs.Size() == 8, "source AD kernel expects trilinear hexes");
        for (int a = 0; a < 8; a++) {
            const double *X = mesh->GetVertex(ev[a]);
            for (int d = 0; d < 3; d++) c.X[a*3+d] = X[d];
            c.dofs[a] = (dofs[a] >= 0) ? dofs[a] : (-1 - dofs[a]);
        }
        tape->cells.push_back(c); rec.ncell++;
    }
    void BuildSource(const TableLaserParams &laser, double t_end, double dt, Vector &f, SrcTape *tape = nullptr) const {
        f = 0.0;
        double t0 = max(0.0, t_end - dt), t1 = t_end;
        if (tape) { tape->subs.clear(); tape->cells.clear(); tape->t0 = t0; tape->t1 = t1; }
        if (t1 <= t0) return;
        Vector shape; Array<int> vdofs;
        int i0 = laser.bracket(t0);
        int i1 = laser.bracket(t1);
        int nT = (int)laser.t.size();
        const int n_sub = std::max(1, nseg); // sub-samples per raw segment, controlled by -nseg
        unordered_set<int> nearby_elements;
        for (int i = i0; i <= i1; i++) {
            double ta = max(t0, laser.t[i]);
            double t_next = (i+1 < nT) ? laser.t[i+1] : t1;
            double tb = min(t1, t_next);
            if (tb <= ta) continue;
            double energy_mJ = laser.absorptivity * laser.p[i] * (tb-ta); // step-hold: constant at p[i] for the whole segment
            if (energy_mJ <= 0.0) continue;
            double energy_sub_mJ = energy_mJ / n_sub;
            double sub_dt = (tb-ta)/n_sub;
            // Constant velocity over a segment (per Abaqus docs), so the
            // travel direction is a per-segment constant, not per-sub-sample.
            if (spread_mode == 2) {
                double ddx = laser.x[i+1<nT?i+1:i] - laser.x[i];
                double ddy = laser.y[i+1<nT?i+1:i] - laser.y[i];
                double ddz = laser.z[i+1<nT?i+1:i] - laser.z[i];
                double dn = sqrt(ddx*ddx+ddy*ddy+ddz*ddz);
                if (dn > 1e-9) last_dir = {ddx/dn, ddy/dn, ddz/dn};
            }
        for (int s = 0; s < n_sub; s++) {
            double tm = ta + (s+0.5)*sub_dt;
            double xL,yL,zL; laser.position(tm,xL,yL,zL);
            SrcSub rec{};
            if (tape) {   // frozen position segment, exactly as position() brackets it
                rec.mode = spread_mode; rec.tm = tm; rec.t_s = g_tpivot; rec.Esub = energy_sub_mJ;
                rec.first_cell = (int)tape->cells.size(); rec.ncell = 0;
                const double tt = g_tpivot + (tm - g_tpivot)*(1.0+g_vxscale) - g_tshift;
                const int ii = laser.bracket(tt);
                const bool lin = laser.t[ii+1] > laser.t[ii];
                double fr_ = lin ? (tt - laser.t[ii])/(laser.t[ii+1]-laser.t[ii]) : 0.0;
                rec.seg_linear = (lin && fr_ > 0.0 && fr_ < 1.0) ? 1 : 0;
                rec.ts0 = laser.t[ii]; rec.ts1 = lin ? laser.t[ii+1] : laser.t[ii] + 1.0;
                const double fc = max(0.0, min(1.0, fr_));
                rec.xa[0] = laser.x[ii]; rec.xa[1] = laser.y[ii]; rec.xa[2] = laser.z[ii];
                rec.xb[0] = laser.x[ii+1]; rec.xb[1] = laser.y[ii+1]; rec.xb[2] = laser.z[ii+1];
                if (!rec.seg_linear) for (int d = 0; d < 3; d++) rec.xa[d] = rec.xa[d] + fc*(rec.xb[d]-rec.xa[d]);
                { const double sh[3] = {g_xshift, g_yshift, g_zshift};   // rigid toolpath shift (-xshift/-yshift/-zshift), as in position()
                  for (int d = 0; d < 3; d++) { rec.xa[d] += sh[d]; rec.xb[d] += sh[d]; } }
                rec.af = goldak_af; rec.ar = goldak_ar; rec.b = goldak_b; rec.c = goldak_c;
                rec.ff = goldak_ff; rec.fr = goldak_fr; rec.box = goldak_cutoff; rec.nq = max(1, goldak_subdiv);
            }
            if (spread_mode == 2) {
                // Abaqus Goldak convention: e1 follows the toolpath and the
                // Advanced-table vector (0,0,-1) defines e3 into the build.
                // The deck's BoxSizeFactor=3 box is divided into the stated
                // SubDivX/Y/Z=3 cells. A toolpath subsegment starts at each
                // cell centroid, Goldak is evaluated there, and its energy
                // is deposited with the containing element's shape functions.
                // Normalize over the complete box before discarding centroids
                // outside active material (Conserve total energy=False).
                array<double,3> e1=last_dir;
                double e1n=sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]);
                if(e1n>1e-12) for(int d=0;d<3;d++) e1[d]/=e1n;
                else e1={1.0,0.0,0.0};
                array<double,3> e3={0.0,0.0,-1.0};
                double proj=e3[0]*e1[0]+e3[1]*e1[1]+e3[2]*e1[2];
                for(int d=0;d<3;d++) e3[d]-=proj*e1[d];
                double e3n=sqrt(e3[0]*e3[0]+e3[1]*e3[1]+e3[2]*e3[2]);
                if(e3n<1e-12) e3={0.0,-1.0,0.0};
                else for(int d=0;d<3;d++) e3[d]/=e3n;
                array<double,3> e2={e3[1]*e1[2]-e3[2]*e1[1],
                                     e3[2]*e1[0]-e3[0]*e1[2],
                                     e3[0]*e1[1]-e3[1]*e1[0]};
                double e2n=sqrt(e2[0]*e2[0]+e2[1]*e2[1]+e2[2]*e2[2]);
                for(int d=0;d<3;d++) e2[d]/=e2n;
                e3={e1[1]*e2[2]-e1[2]*e2[1],
                    e1[2]*e2[0]-e1[0]*e2[2],
                    e1[0]*e2[1]-e1[1]*e2[0]};
                if (tape) for (int d = 0; d < 3; d++) { rec.e1[d] = e1[d]; rec.e2[d] = e2[d]; rec.e3[d] = e3[d]; }

                struct GoldakCell { double lx,ly,lz,w; int ix,iy,iz; };
                vector<GoldakCell> cells;
                const int nq=max(1,goldak_subdiv);
                const double box_factor=goldak_cutoff;
                const double dx=box_factor*(goldak_af+goldak_ar)/nq;
                const double dy=2.0*box_factor*goldak_b/nq;
                const double dz=box_factor*goldak_c/nq;
                double full_weight=0.0;
                for(int ix=0;ix<nq;ix++) for(int iy=0;iy<nq;iy++)
                for(int iz=0;iz<nq;iz++) {
                    double lx=-box_factor*goldak_ar+(ix+0.5)*dx;
                    double ly=-box_factor*goldak_b +(iy+0.5)*dy;
                    double lz=(iz+0.5)*dz;
                    const bool front=(lx>=0.0);
                    const double ax=front?goldak_af:goldak_ar;
                    const double frac=front?goldak_ff:goldak_fr;
                    double raw=frac/ax*exp(-3.0*(lx*lx/(ax*ax)+
                                                   ly*ly/(goldak_b*goldak_b)+
                                                   lz*lz/(goldak_c*goldak_c)));
                    cells.push_back({lx,ly,lz,raw,ix,iy,iz});
                    full_weight+=raw;
                }
                Vector qshape,pt(3);
                Array<int> qdofs;
                for(const auto &cell_q:cells) {
                    double dE=energy_sub_mJ*(cell_q.w/full_weight);
                    double px=xL+cell_q.lx*e1[0]+cell_q.ly*e2[0]+cell_q.lz*e3[0];
                    double py=yL+cell_q.lx*e1[1]+cell_q.ly*e2[1]+cell_q.lz*e3[1];
                    double pz=zL+cell_q.lx*e1[2]+cell_q.ly*e2[2]+cell_q.lz*e3[2];
                    int e=FindElement(px,py,pz);
                    if(e<0 || px<lo[e][0] || px>hi[e][0] ||
                              py<lo[e][1] || py>hi[e][1] ||
                              pz<lo[e][2] || pz>hi[e][2]) continue;
                    IsoparametricTransformation Tr;
                    mesh->GetElementTransformation(e,&Tr);
                    InverseElementTransformation inv_tr(&Tr);
                    IntegrationPoint qip;
                    pt[0]=px; pt[1]=py; pt[2]=pz;
                    int res=inv_tr.Transform(pt,qip);
                    if(res!=InverseElementTransformation::Inside &&
                       res!=InverseElementTransformation::Unknown) continue;
                    const FiniteElement &qfe=*fes->GetFE(e);
                    qshape.SetSize(qfe.GetDof());
                    qfe.CalcShape(qip,qshape);
                    fes->GetElementDofs(e,qdofs);
                    for(int a=0;a<qdofs.Size();a++) {
                        int dof=(qdofs[a]>=0)?qdofs[a]:(-1-qdofs[a]);
                        f[dof]+=dE*qshape[a];
                    }
                    if (tape) RecordCell(tape, rec, e, qip.x, qip.y, qip.z, qdofs, cell_q.ix, cell_q.iy, cell_q.iz, 0);
                }
                if (tape && rec.ncell > 0) tape->subs.push_back(rec);
                continue;
            }
            int e = FindElement(xL,yL,zL);
            if (e < 0) continue;
            IsoparametricTransformation Tr;
            mesh->GetElementTransformation(e, &Tr);
            InverseElementTransformation inv_tr(&Tr);
            IntegrationPoint ip;
            Vector pt(3); pt(0)=xL; pt(1)=yL; pt(2)=zL;
            int res = inv_tr.Transform(pt, ip);
            if (res != InverseElementTransformation::Inside &&
                res != InverseElementTransformation::Unknown) {
                ip = Geometries.GetCenter(fes->GetFE(e)->GetGeomType());
            }
            const FiniteElement &fe = *fes->GetFE(e);
            shape.SetSize(fe.GetDof());
            fe.CalcShape(ip, shape);
            fes->GetElementDofs(e, vdofs);
            if (tape && spread_mode == 0) {
                const bool inside = (res == InverseElementTransformation::Inside || res == InverseElementTransformation::Unknown);
                RecordCell(tape, rec, e, ip.x, ip.y, ip.z, vdofs, 0, 0, 0, inside ? 0 : 1);
                tape->subs.push_back(rec);
            }
            if (spread_mode == 1) {
                double share = 1.0 / shape.Size();
                for (int a = 0; a < shape.Size(); a++)
                    f[vdofs[a]] += energy_sub_mJ * share;
            } else {
                for (int a = 0; a < shape.Size(); a++)
                    f[vdofs[a]] += energy_sub_mJ * shape(a);
            }
        } // end sub-sample loop
        } // end segment loop
        f /= (t1 - t0); // accumulated energy (mJ) -> time-averaged power (mW)
    }

    // NOT CALLED from the production sensitivity path any more (2026-09-12):
    // x0/Vx/tL now use plain central FD on BuildSource instead (see the
    // "(f) toolpath position/timing sensitivity" block below) because this
    // frozen-element approach has a genuine, root-caused ~9% relative L2
    // error at the substrate/row1 tie interface (FindElement's inclusive
    // bounds test lets two independently-meshed regions' elements both
    // claim a point on their shared z=0 face; resolution then falls to
    // hash-bucket insertion order, not physics). Left in place only as a
    // reference for the failure mode below; do not wire this back in
    // without re-solving that tie-interface ambiguity first.
    //
    // Original design intent, frozen-element analytic position gradient of
    // f_src, d(f_src)/d(deposition point x/y/z), ported from the reference
    // driver's own BuildSourcePosGrad -- used for x0/Vx/tL (pure
    // translations of the deposition point). FD across a re-run
    // FindElement would inject element-crossing discontinuities, so this
    // instead re-runs the SAME nominal (unperturbed) Goldak-cell/element
    // lookup as BuildSource's own spread_mode==2 branch, but accumulates
    // dE*physical-gradient(shape_a) instead of dE*shape_a -- a normal FEM
    // spatial-gradient evaluation at the frozen nominal element/reference-
    // point, not a finite difference of anything. Only meaningful when
    // spread_mode==2 (Goldak); early-returns zero otherwise, matching
    // BuildSourcePosGrad's own physical scope (rx/ry/rz's central-FD block
    // above is likewise a no-op outside Goldak mode).
    void BuildSourcePosGrad(const TableLaserParams &laser, double t_end, double dt,
                            Vector &dfdx, Vector &dfdy, Vector &dfdz) const {
        dfdx = 0.0; dfdy = 0.0; dfdz = 0.0;
        dfdx = 0.0; dfdy = 0.0; dfdz = 0.0;
        double t0 = max(0.0, t_end - dt), t1 = t_end;
        if (t1 <= t0 || spread_mode != 2) return;
        int i0 = laser.bracket(t0), i1 = laser.bracket(t1), nT = (int)laser.t.size();
        const int n_sub = std::max(1, nseg);
        for (int i = i0; i <= i1; i++) {
            double ta = max(t0, laser.t[i]);
            double t_next = (i+1 < nT) ? laser.t[i+1] : t1;
            double tb = min(t1, t_next);
            if (tb <= ta) continue;
            double energy_mJ = laser.absorptivity * laser.p[i] * (tb-ta);
            if (energy_mJ <= 0.0) continue;
            double energy_sub_mJ = energy_mJ / n_sub;
            double sub_dt = (tb-ta)/n_sub;
            double ddx = laser.x[i+1<nT?i+1:i] - laser.x[i];
            double ddy = laser.y[i+1<nT?i+1:i] - laser.y[i];
            double ddz = laser.z[i+1<nT?i+1:i] - laser.z[i];
            double dn = sqrt(ddx*ddx+ddy*ddy+ddz*ddz);
            if (dn > 1e-9) last_dir = {ddx/dn, ddy/dn, ddz/dn};
            for (int s = 0; s < n_sub; s++) {
                double tm = ta + (s+0.5)*sub_dt;
                double xL,yL,zL; laser.position(tm,xL,yL,zL);
                array<double,3> e1=last_dir;
                double e1n=sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]);
                if(e1n>1e-12) for(int d=0;d<3;d++) e1[d]/=e1n;
                else e1={1.0,0.0,0.0};
                array<double,3> e3={0.0,0.0,-1.0};
                double proj=e3[0]*e1[0]+e3[1]*e1[1]+e3[2]*e1[2];
                for(int d=0;d<3;d++) e3[d]-=proj*e1[d];
                double e3n=sqrt(e3[0]*e3[0]+e3[1]*e3[1]+e3[2]*e3[2]);
                if(e3n<1e-12) e3={0.0,-1.0,0.0};
                else for(int d=0;d<3;d++) e3[d]/=e3n;
                array<double,3> e2={e3[1]*e1[2]-e3[2]*e1[1],
                                     e3[2]*e1[0]-e3[0]*e1[2],
                                     e3[0]*e1[1]-e3[1]*e1[0]};
                double e2n=sqrt(e2[0]*e2[0]+e2[1]*e2[1]+e2[2]*e2[2]);
                for(int d=0;d<3;d++) e2[d]/=e2n;
                e3={e1[1]*e2[2]-e1[2]*e2[1],
                    e1[2]*e2[0]-e1[0]*e2[2],
                    e1[0]*e2[1]-e1[1]*e2[0]};

                struct GoldakCell { double lx,ly,lz,w; };
                vector<GoldakCell> cells;
                const int nq=max(1,goldak_subdiv);
                const double box_factor=goldak_cutoff;
                const double dx=box_factor*(goldak_af+goldak_ar)/nq;
                const double dy=2.0*box_factor*goldak_b/nq;
                const double dz=box_factor*goldak_c/nq;
                double full_weight=0.0;
                for(int ix=0;ix<nq;ix++) for(int iy=0;iy<nq;iy++)
                for(int iz=0;iz<nq;iz++) {
                    double lx=-box_factor*goldak_ar+(ix+0.5)*dx;
                    double ly=-box_factor*goldak_b +(iy+0.5)*dy;
                    double lz=(iz+0.5)*dz;
                    const bool front=(lx>=0.0);
                    const double ax=front?goldak_af:goldak_ar;
                    const double frac=front?goldak_ff:goldak_fr;
                    double raw=frac/ax*exp(-3.0*(lx*lx/(ax*ax)+
                                                   ly*ly/(goldak_b*goldak_b)+
                                                   lz*lz/(goldak_c*goldak_c)));
                    cells.push_back({lx,ly,lz,raw});
                    full_weight+=raw;
                }
                DenseMatrix dshape, invJ(3,3), gphys;
                Array<int> qdofs;
                Vector pt(3);
                for(const auto &cell_q:cells) {
                    double dE=energy_sub_mJ*(cell_q.w/full_weight);
                    double px=xL+cell_q.lx*e1[0]+cell_q.ly*e2[0]+cell_q.lz*e3[0];
                    double py=yL+cell_q.lx*e1[1]+cell_q.ly*e2[1]+cell_q.lz*e3[1];
                    double pz=zL+cell_q.lx*e1[2]+cell_q.ly*e2[2]+cell_q.lz*e3[2];
                    int e=FindElement(px,py,pz);
                    if(e<0 || px<lo[e][0] || px>hi[e][0] ||
                              py<lo[e][1] || py>hi[e][1] ||
                              pz<lo[e][2] || pz>hi[e][2]) continue;
                    // KNOWN LIMITATION (2026-09-12, not yet resolved): near
                    // the substrate/row1 tie interface, FindElement can
                    // genuinely resolve to a different element for two
                    // positions differing by as little as 1e-6mm (confirmed
                    // h-independent down to 1e-6, ruling out ordinary FD/
                    // quadrature noise) -- root-caused via a direct FD
                    // isolation test (jobs 810932-950) to ~8% relative L2
                    // error in dfdx/dfdy/dfdz, ~89% of it concentrated at
                    // dofs within a coincident-node band near a layer
                    // boundary. Two guard attempts were tried and REVERTED:
                    // (1) skip cells whose DEPOSITION POINT sits near a
                    // layer-boundary z -- never fired (skip=0/386640), since
                    // the Goldak box can deposit deep into row1 while still
                    // landing in an element that itself touches z=0 by
                    // construction; (2) skip cells whose FOUND ELEMENT has
                    // any vertex near a layer-boundary z -- WAY too
                    // aggressive (skip=257760/386640, ~67%) because row1 is
                    // only one element thick in z, so EVERY row1 element
                    // touches both z=0 and z=0.2 -- this zeroed dfdx almost
                    // entirely (2.9e-6 vs the correct ~212). Left unguarded
                    // for now: the underlying mechanism is correct away from
                    // the seam (dT/dRx, dT/dRz FD-validated <0.0001%; dT/dX0
                    // shows the ~8%-driven error). x0/Vx/tL should be treated
                    // as NOT YET FULLY VALIDATED pending a properly targeted
                    // fix -- does not affect NW=13 (k through rz) at all.
                    IsoparametricTransformation Tr;
                    mesh->GetElementTransformation(e,&Tr);
                    InverseElementTransformation inv_tr(&Tr);
                    IntegrationPoint qip;
                    pt[0]=px; pt[1]=py; pt[2]=pz;
                    int res=inv_tr.Transform(pt,qip);
                    // Fixed (2026-09-12): must match THIS file's own
                    // BuildSource() convention (skip the point) exactly --
                    // the reference driver's own BuildSourcePosGrad instead
                    // falls back to the element center here, which silently
                    // diverged from what BuildSource(spread_mode==2) above
                    // actually does at the same failure case, corrupting
                    // dfdx/dfdy/dfdz at every point that hits this branch
                    // (found via a direct raw-FD isolation check: ~9%
                    // relative L2 mismatch against BuildSource's own output).
                    if(res!=InverseElementTransformation::Inside &&
                       res!=InverseElementTransformation::Unknown) continue;
                    const FiniteElement &qfe=*fes->GetFE(e);
                    Tr.SetIntPoint(&qip);
                    CalcInverse(Tr.Jacobian(), invJ);
                    dshape.SetSize(qfe.GetDof(),3);
                    qfe.CalcDShape(qip,dshape);
                    gphys.SetSize(qfe.GetDof(),3);
                    mfem::Mult(dshape, invJ, gphys);   // physical gradients [ndof x 3]
                    fes->GetElementDofs(e,qdofs);
                    for(int a=0;a<qdofs.Size();a++) {
                        int dof=(qdofs[a]>=0)?qdofs[a]:(-1-qdofs[a]);
                        dfdx[dof]+=dE*gphys(a,0);
                        dfdy[dof]+=dE*gphys(a,1);
                        dfdz[dof]+=dE*gphys(a,2);
                    }
                }
            }
        }
        double inv = 1.0/(t1-t0);
        dfdx *= inv; dfdy *= inv; dfdz *= inv;
    }
};

// =========================================================================
//  JacobianOp / FachinottiResidualOp / FachinottiLinearSolver /
//  DampedNewtonSolver -- byte-identical to xtrack_multilayer.cpp
// =========================================================================
// CONSISTENT boundary quadrature record (2026-07-14): Abaqus's own theory
// manual ("Uncoupled heat transfer analysis theory") gives the film/
// radiation Jacobian contribution as a standard Galerkin surface integral
// INT_S N^N (dq/dtheta) N^M dS -- i.e. NOT lumped. Only the volumetric
// capacitance term is documented as lumped ("integration stations located
// at the corners of the element ... Jacobian term associated with the
// internal energy rate is diagonal"). The earlier "consistent boundary
// mass matrix" experiment (reverted) was NOT a faithful implementation of
// this: it evaluated the T^4 nonlinearity at each node first, then smeared
// those nodal values through a mass matrix (B*g(T_node)), which is NOT the
// same as evaluating q(T) at the true quadrature point using the
// INTERPOLATED temperature field (q is nonlinear, so nodal-eval-then-smear
// != quadrature-point-eval). One BdryQP record per (face, quadrature
// point): dof/shp list the local element dofs and shape values touching
// that point (matches el.GetDof(), e.g. 8 for a trilinear hex -- entries
// for dofs on the opposite face are exactly 0 and simply contribute
// nothing), w is the quadrature weight * face Jacobian.
struct BdryQP {
    vector<int> dof;
    vector<double> shp;
    double w;
    int kind; // 1=substrate exterior, 2=bridge exterior, 3=active-inactive interior face
};

class JacobianOp : public Operator {
    // kpa is a REFERENCE-TO-POINTER (2026-07-14): FachinottiResidualOp
    // reassembles the diffusion matrix fresh at every Newton call (see
    // UpdateConductivity below), so this must always dereference to
    // whichever BilinearForm was most recently built, not a fixed object.
    // Jbdry_mat is the same pattern, for the (also fully re-assembled every
    // Newton call) consistent boundary Jacobian -- see UpdateBoundary.
    //
    // dk/dT term (Kprime_mat) tried and REMOVED (2026-07-15): implemented,
    // tested correctly (matches theory manual, non-symmetric, needed
    // GMRES), but confirmed via controlled A/B testing to have negligible
    // effect on Newton's convergence rate for this problem's temperature
    // range (iteration-by-iteration residuals nearly identical with vs.
    // without it) -- the "missing derivative" hypothesis for the slow
    // (linear, not quadratic) convergence after switching to the
    // consistent boundary Jacobian was disproven. Removed to avoid its
    // real per-Newton-iteration assembly cost (a hand-written element loop
    // over ~148K active elements) for no measured benefit; reverted the
    // linear solver back to CG accordingly (see FachinottiLinearSolver).
    BilinearForm *&kpa;
    SparseMatrix *&Jbdry_mat;
    const Vector &Meff_over_dt;
    const Array<int> &ess_tdof_list;
    mutable Vector Kd, JbX;
public:
    JacobianOp(BilinearForm *&k, SparseMatrix *&jb, const Vector &m,
               const Array<int> &ess, int n)
        : Operator(n), kpa(k), Jbdry_mat(jb), Meff_over_dt(m),
          ess_tdof_list(ess), Kd(n), JbX(n)
    { Kd.UseDevice(true); JbX.UseDevice(true); }
    void Mult(const Vector &x, Vector &y) const override {
        kpa->Mult(x, Kd);
        Jbdry_mat->Mult(x, JbX);
        const double *kd = Kd.Read();
        const double *jb = JbX.Read();
        const double *xp = x.Read();
        const double *md = Meff_over_dt.Read();
        double *yp       = y.Write();
        int _n = Height();
        mfem::forall(_n, [=] MFEM_HOST_DEVICE(int i){
            yp[i] = kd[i] + jb[i] + md[i]*xp[i];
        });
        const int *ess = ess_tdof_list.Read();
        int ness = ess_tdof_list.Size();
        double *yp2 = y.ReadWrite();
        const double *xp2 = x.Read();
        mfem::forall(ness, [=] MFEM_HOST_DEVICE(int k){
            int i = ess[k];
            yp2[i] = xp2[i];
        });
    }

    // TIER-1 (direct solver) accessor: assemble the full-space Jacobian as a
    // single SparseMatrix -- kpa->SpMat() + *Jbdry_mat + diag(Meff_over_dt),
    // with essential rows/cols reduced to identity. Matches Mult() exactly.
    // Caller owns the returned matrix.
    SparseMatrix* AssembleFull() const {
        // FIX (2026-09-11): used to do `S->Add(i,i,md[i])` directly on
        // `Add(kpa->SpMat(), *Jbdry_mat)`'s result -- relied on Jbdry_mat's
        // old fix_empty_rows=true Finalize() to guarantee EVERY row had at
        // least a diagonal placeholder to update. Now that Jbdry_mat is
        // correctly Finalize(0, false) (see UpdateBoundary -- fix_empty_rows
        // was corrupting the exact sensitivity Jacobian at ordinary active
        // interior dofs), a genuinely-inactive dof whose kpa row is ALSO
        // narrowed (RebuildConductivityFreshForSensitivity's plain
        // Assemble() only touches active elements, unlike the forward
        // solve's own Assemble(0)-based kpa_dynamic which spans all
        // elements) can have BOTH inputs empty at that row -> `S->Add`
        // throws "Could not find entry" (SparseMatrix is finalized, can't
        // insert new entries). Fix: fold Meff_over_dt in via a proper
        // mfem::Add() with its own always-present diagonal pattern, instead
        // of a post-hoc in-place Add into a matrix that might not have that
        // entry at all.
        SparseMatrix Mdiag(Height(), Height());
        {
            const double *md0 = Meff_over_dt.HostRead();
            for (int i = 0; i < Height(); i++) Mdiag.Add(i, i, md0[i]);
            Mdiag.Finalize(0, false);   // skip_zeros=0: keep every diagonal entry, even 0.0
        }
        SparseMatrix *KplusJb = Add(kpa->SpMat(), *Jbdry_mat);
        SparseMatrix *S = Add(*KplusJb, Mdiag);
        delete KplusJb;
        // Not-yet-active DOFs assemble to an all-zero row AND all-zero
        // column (no active element/boundary/capacitance term references
        // them, by construction) -> EliminateRowCol(i)'s row+column zeroing
        // is therefore a no-op here; all that is actually needed is to pin
        // the diagonal to 1 so the direct solver sees a nonsingular row.
        // Ported from full_bridge_activation_tier1.cpp's own AssembleFull()
        // perf fix (2026-09-10), applied on top of this file's later
        // (2026-09-11) Mdiag-based construction of S above: Mdiag's
        // Finalize(0,false) guarantees the (i,i) slot always EXISTS in S
        // (even when numerically 0), so S->Add(i,i,delta) below is an
        // in-place update of an already-present entry, not an insertion --
        // safe to parallelize (each i touches a distinct existing slot) and
        // O(1) instead of EliminateRowCol's O(row_nnz), which matters when
        // 500k+ DOFs are still inactive early in a 63-layer build. Their
        // residual is ~0 and main() overwrites inactive nodes with
        // T_ambient after the solve, so this does not perturb the physics.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Height(); i++) {
            double d = (*S)(i, i);
            if (std::fabs(d) < 1e-30) S->Add(i, i, 1.0 - d);
        }
        for (int k = 0; k < ess_tdof_list.Size(); k++)
            S->EliminateRowCol(ess_tdof_list[k]);   // default DIAG_ONE, zero row+col
        return S;
    }
    const Array<int>& EssList() const { return ess_tdof_list; }
};

static inline double now_ms();   // defined further below
// ---------------------------------------------------------------------------
// PERF (2026-09-23): persistent conductivity assembler. FachinottiResidualOp is constructed INSIDE the
// per-step loop, and its first UpdateConductivity() used to rebuild the spanning CSR pattern (serial MFEM
// Assemble(0)) AND a nested per-quadrature-point geometry cache (serial, one heap object per qp) EVERY
// step -- measured 10.4 s of a 45.6 s step at 553k elements, while the refill itself is ~0.1 s per call.
// Here the pattern + a flat geometry/position cache are built ONCE (geometry loop OpenMP-parallel) and
// shared by every residual-op instance; each call only zeroes and re-scatters values. Same arithmetic per
// element as the old refill (symmetric half computed and mirrored: s(a,c)==s(c,a) bit-for-bit).
// Set KPERSIST_OFF=1 to use the original per-step path.
// ---------------------------------------------------------------------------
struct PersistentK {
    BilinearForm *form = nullptr;
    bool built = false, ok = false;
    int ne = 0, nd = 0, nq = 0;
    std::vector<int> vd, pos, attr;
    std::vector<double> w, g, sh;
    // v8: when set, RefillInto blends k(T)/kp(T) per quadrature point by the shape-interpolated melt
    // fraction (1 = fully bulk/consolidated, 0 = still powder). nullptr (default) = pure bulk k(T),
    // byte-identical to every pre-existing run. This is the ONE shared assembly routine behind the
    // forward Newton's own K, RebuildConductivityFreshForSensitivity, and CorrectedJacobianOp's fast
    // path (see their own call sites) -- one edit here keeps all three consistent by construction.
    const double *ever_melted_ptr = nullptr;
    void Build(FiniteElementSpace &fes) {
        built = true;
        double _t0 = now_ms();
        Mesh &mesh = *fes.GetMesh();
        ne = mesh.GetNE();
        ConstantCoefficient zero(0.0);
        form = new BilinearForm(&fes);
        form->AddDomainIntegrator(new DiffusionIntegrator(zero));
        form->Assemble(0);              // skip_zeros=0: pattern spans every element (all birth events)
        form->Finalize(0);
        form->SpMat().Finalize(0, true);
        const FiniteElement &fe0 = *fes.GetFE(0);
        IsoparametricTransformation T0; fes.GetElementTransformation(0, &T0);
        nd = fe0.GetDof();
        const int order0 = 2 * T0.OrderGrad(&fe0);
        const IntegrationRule &ir = IntRules.Get(fe0.GetGeomType(), order0);
        nq = ir.GetNPoints();
        ok = (nd == 8) && (fe0.GetDim() == 3) && nq > 0;
        if (!ok) { std::cout << "  [persistent K] not order-1 hex -> disabled\n"; return; }
        vd.resize((size_t)ne * nd); pos.assign((size_t)ne * nd * nd, -1); attr.resize(ne);
        w.resize((size_t)ne * nq); g.resize((size_t)ne * nq * nd * 3); sh.resize((size_t)nq * nd);
        { Vector shape(nd); for (int q = 0; q < nq; q++) { fe0.CalcShape(ir.IntPoint(q), shape); for (int a = 0; a < nd; a++) sh[(size_t)q*nd+a] = shape(a); } }
        SparseMatrix &A = form->SpMat();
        const int *I = A.GetI(), *J = A.GetJ();
        int bad = 0;
        #pragma omp parallel reduction(|:bad)
        {
            IsoparametricTransformation Tr; DenseMatrix dshape(nd, 3), gsh(nd, 3); Vector shape(nd); Array<int> vdl;
            #pragma omp for schedule(static)
            for (int e = 0; e < ne; e++) {
                const FiniteElement &el = *fes.GetFE(e);
                fes.GetElementTransformation(e, &Tr);
                fes.GetElementDofs(e, vdl);
                if (vdl.Size() != nd || el.GetDof() != nd || 2 * Tr.OrderGrad(&el) != order0) { bad |= 1; continue; }
                attr[e] = mesh.GetAttribute(e);
                for (int a = 0; a < nd; a++) vd[(size_t)e*nd + a] = vdl[a];
                for (int q = 0; q < nq; q++) {
                    const IntegrationPoint &ip = ir.IntPoint(q);
                    el.CalcShape(ip, shape);
                    for (int a = 0; a < nd; a++) if (shape(a) != sh[(size_t)q*nd+a]) bad |= 2;
                    el.CalcDShape(ip, dshape);
                    Tr.SetIntPoint(&ip);
                    w[(size_t)e*nq + q] = ip.weight * Tr.Weight();
                    mfem::Mult(dshape, Tr.InverseJacobian(), gsh);
                    double *gp = &g[(((size_t)e*nq + q) * nd) * 3];
                    for (int a = 0; a < nd; a++) for (int d = 0; d < 3; d++) gp[a*3 + d] = gsh(a, d);
                }
                for (int a = 0; a < nd; a++) {
                    const int r = vdl[a];
                    for (int c = 0; c < nd; c++) {
                        const int col = vdl[c];
                        for (int k = I[r]; k < I[r+1]; k++) if (J[k] == col) { pos[((size_t)e*nd + a)*nd + c] = k; break; }
                    }
                }
            }
        }
        ok = (bad == 0);
        std::cout << "  [persistent K] built once in " << (now_ms() - _t0) / 1000.0 << " s ("
                  << (ok ? "ON" : "DISABLED: mesh not uniform") << ", ne=" << ne << ", nq=" << nq << ")\n";
    }
    void Refill(const double *xp, const std::vector<char> &active) {
        SparseMatrix &A = form->SpMat();
        RefillInto(xp, active, A.GetData(), A.NumNonZeroElems());
    }
    // values of K(xp) on the shared spanning pattern, written into caller-owned storage Ad[nnz]
    void RefillInto(const double *xp, const std::vector<char> &active, double *Ad, const int nnz) {
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < nnz; k++) Ad[k] = 0.0;
        const int nd_ = nd, nq_ = nq;
        #pragma omp parallel for schedule(dynamic, 128)
        for (int e = 0; e < ne; e++) {
            if (!active[attr[e]]) continue;
            const int *vde = &vd[(size_t)e * nd_];
            double Tloc[8], elmat[64];
            for (int a = 0; a < nd_; a++) Tloc[a] = xp[vde[a]];
            for (int i = 0; i < 64; i++) elmat[i] = 0.0;
            double Mloc[8];
            if (ever_melted_ptr) for (int a = 0; a < nd_; a++) Mloc[a] = ever_melted_ptr[vde[a]];
            for (int q = 0; q < nq_; q++) {
                const double *gq = &g[(((size_t)e * nq_ + q) * nd_) * 3];
                const double *shq = &sh[(size_t)q * nd_];
                double Tq = 0.0;
                for (int a = 0; a < nd_; a++) Tq += shq[a] * Tloc[a];
                double k_eff;
                if (ever_melted_ptr) {
                    double mf = 0.0;
                    for (int a = 0; a < nd_; a++) mf += shq[a] * Mloc[a];
                    k_eff = mf * IN625::k(Tq) + (1.0 - mf) * IN625::kp_dev(Tq);
                } else {
                    k_eff = IN625::k(Tq);
                }
                const double kw = w[(size_t)e * nq_ + q] * k_eff;
                for (int a = 0; a < nd_; a++)
                    for (int c = a; c < nd_; c++) {
                        double sd = 0.0;
                        for (int d = 0; d < 3; d++) sd += gq[a*3+d] * gq[c*3+d];
                        elmat[a*8 + c] += kw * sd;
                    }
            }
            const int *pe = &pos[(size_t)e * nd_ * nd_];
            for (int a = 0; a < nd_; a++)
                for (int c = 0; c < nd_; c++) {
                    const double v = (c >= a) ? elmat[a*8 + c] : elmat[c*8 + a];
                    if (v != 0.0) {
                        #pragma omp atomic
                        Ad[pe[a*nd_ + c]] += v;
                    }
                }
        }
    }
};
static PersistentK g_pk;

// =========================================================================
// A2 sensitivity-solve correction (2026-09-11): adds the dk/dT_next term
// JacobianOp deliberately omits (see JacobianOp's own "dk/dT term tried and
// REMOVED" comment -- fine for Newton's own convergence rate, NOT fine for
// sensitivity accuracy, per the internal FD check that found dT/dk off by
// ~9x without it). Matrix-free: caches (shape, per-qp coefficient) ONCE at
// construction (Tnext frozen at the converged u_iter for this step), then
// Mult() adds the correction action on top of the base (approximate)
// JacobianOp. Non-symmetric in general (grad(Tnext) direction breaks CG's
// SPD requirement) -- solved via GMRES, preconditioned by the existing
// approximate solver (CG/PARDISO on the base JacobianOp), NOT used for the
// forward Newton solve itself (that stays exactly as validated).
class CorrectedJacobianOp : public Operator {
    // A2 fix (2026-09-11): does NOT reuse the forward Newton solve's own
    // kpa_dynamic (built via BilinearForm::Assemble()) at all. Extensive
    // diagnosis proved kpa_dynamic's own assembled diagonal is off by ~1.0
    // at certain rows, REGARDLESS of whether it's built via the repeated
    // "SpMat()=0;Assemble(0)" reuse cycle OR a single fresh plain
    // Assemble() -- i.e. the defect is in BilinearForm::Assemble() itself
    // for this mesh/coefficient, not in how many times it's called. The
    // ONLY method proven to give the correct value (matches direct FD of
    // the residual to <0.1%, verified element-by-element) is manual
    // per-element DiffusionIntegrator::AssembleElementMatrix + scatter-add
    // via SparseMatrix::AddSubMatrix, entirely bypassing BilinearForm. This
    // class now owns that manually-assembled domain matrix directly,
    // reusing the ALREADY-independently-verified Jbdry_mat (boundary,
    // proven correct/negligible here) and Meff_over_dt (nodal, proven
    // correct/matches neighbors) from the forward solve.
    mutable SparseMatrix ManualKpa;
    // REFERENCE-TO-POINTER (fixed 2026-09-11, was a dangling raw-pointer
    // copy -- see GetJbdryMat()'s own comment): always dereferences to
    // whichever SparseMatrix UpdateBoundary() most recently built, even if
    // that happens after this object's construction (e.g. under
    // -jaccheck/-rhscheck diagnostics that call residual.Mult() again).
    SparseMatrix *&Jbdry_mat;
    const Vector &Meff_over_dt;
    const Array<int> &ess_tdof_list;
    struct ElemCache {
        Array<int> vdofs;
        std::vector<Vector> shape;      // [qp](dof)
        std::vector<std::vector<double>> rowcoef; // [qp][dof] = k'(Tq)*w*(gshape_a . gradTnext_q)
    };
    std::vector<ElemCache> cache;
    mutable Vector tmp;
    // PERF (2026-09-23): parallel fast path. Values of K(Tnext) come from the persistent flat cache
    // (PersistentK::RefillInto) on the shared CSR pattern; the dk/dT correction cache is flat and
    // built in parallel; Mult() uses an OpenMP CSR SpMV (same per-row summation order as MFEM's) and an
    // element-parallel correction with one atomic add per node. CORRJ_LEGACY=1 restores the original.
    bool fast = false;
    std::vector<double> kvals;                       // declared before ManualKpaP: destroyed after it
    mutable std::unique_ptr<SparseMatrix> ManualKpaP;
    std::vector<int> f_act; std::vector<char> f_nz; std::vector<double> f_rc;
    const SparseMatrix &KM() const { return ManualKpaP ? *ManualKpaP : ManualKpa; }
public:
    CorrectedJacobianOp(FiniteElementSpace &fes, Mesh &mesh,
                         const vector<char> &attr_active, const Vector &Tnext,
                         SparseMatrix *&jbdry, const Vector &meff,
                         const Array<int> &ess, int n)
        : Operator(n), ManualKpa(n, n), Jbdry_mat(jbdry), Meff_over_dt(meff),
          ess_tdof_list(ess), tmp(n)
    {
        tmp.UseDevice(true);
        {
            static const bool corrj_legacy = (getenv("CORRJ_LEGACY") != nullptr);
            if (!corrj_legacy) {
                if (!g_pk.built) g_pk.Build(fes);
                if (g_pk.ok) {
                    const int ne = g_pk.ne, nd = g_pk.nd, nq = g_pk.nq;
                    SparseMatrix &Kp = g_pk.form->SpMat();
                    const int nnz = Kp.NumNonZeroElems();
                    kvals.assign((size_t)nnz, 0.0);
                    const double *Tp = Tnext.HostRead();
                    g_pk.RefillInto(Tp, attr_active, kvals.data(), nnz);
                    ManualKpaP.reset(new SparseMatrix(Kp.GetI(), Kp.GetJ(), kvals.data(), n, n, false, false, false));
                    for (int e = 0; e < ne; e++) if (attr_active[g_pk.attr[e]]) f_act.push_back(e);
                    const int nact = (int)f_act.size();
                    f_nz.assign((size_t)nact, 0); f_rc.assign((size_t)nact * nq * nd, 0.0);
                    #pragma omp parallel for schedule(dynamic, 128)
                    for (int i = 0; i < nact; i++) {
                        const int e = f_act[i];
                        const int *vde = &g_pk.vd[(size_t)e * nd];
                        double Tloc[8];
                        for (int a = 0; a < nd; a++) Tloc[a] = Tp[vde[a]];
                        double *rc = &f_rc[(size_t)i * nq * nd];
                        bool any = false;
                        for (int q = 0; q < nq; q++) {
                            const double *gq = &g_pk.g[(((size_t)e * nq + q) * nd) * 3];
                            const double *shq = &g_pk.sh[(size_t)q * nd];
                            const double w = g_pk.w[(size_t)e * nq + q];
                            double Tq = 0.0;
                            for (int a = 0; a < nd; a++) Tq += shq[a] * Tloc[a];
                            const double kp_q = IN625::k_prime(Tq);
                            if (kp_q != 0.0) {
                                double gT[3] = {0.0, 0.0, 0.0};
                                for (int a = 0; a < nd; a++) for (int d = 0; d < 3; d++) gT[d] += gq[a*3+d] * Tloc[a];
                                for (int a = 0; a < nd; a++) {
                                    double gdot = 0.0;
                                    for (int d = 0; d < 3; d++) gdot += gq[a*3+d] * gT[d];
                                    rc[q * nd + a] = kp_q * w * gdot;
                                    if (rc[q * nd + a] != 0.0) any = true;
                                }
                            }
                        }
                        f_nz[i] = any ? 1 : 0;
                    }
                    fast = true;
                    return;
                }
            }
        }
        // ---- manually assemble the domain stiffness matrix, element by
        // element, bypassing BilinearForm::Assemble() entirely ----
        {
            GridFunction Tnext_gf(&fes); Tnext_gf = Tnext;
            ActiveTDepKCoefficient k_coeff_manual(Tnext_gf, attr_active);
            DiffusionIntegrator diff_manual(k_coeff_manual);
            Array<int> evdofs;
            DenseMatrix elmat;
            for (int e = 0; e < mesh.GetNE(); e++) {
                if (!attr_active[mesh.GetAttribute(e)]) continue;
                const FiniteElement &fe = *fes.GetFE(e);
                ElementTransformation *Trans = fes.GetElementTransformation(e);
                diff_manual.AssembleElementMatrix(fe, *Trans, elmat);
                fes.GetElementDofs(e, evdofs);
                ManualKpa.AddSubMatrix(evdofs, evdofs, elmat, 0);
            }
            ManualKpa.Finalize(0, true);
            // MARKER (2026-09-11): definitive check that this rewritten
            // code is actually compiled/running, and that ManualKpa itself
            // holds the corrected value at the known problem dof (41928,
            // true value ~7.17-7.18, kpa_dynamic's own buggy value ~8.17).
            if (n > 41928) {
                std::cout << "  [MANUAL_KPA_MARKER_V2] ManualKpa(41928,41928)="
                          << ManualKpa(41928,41928) << "  (expect ~7.17-7.18 if fix works, ~8.17 if not)\n"
                          << std::flush;
            }
        }
        const double *Tp = Tnext.HostRead();
        DenseMatrix dshape, gshape_e;
        Vector shp;
        for (int e = 0; e < mesh.GetNE(); e++) {
            if (!attr_active[mesh.GetAttribute(e)]) continue;
            const FiniteElement &el = *fes.GetFE(e);
            IsoparametricTransformation Trans;
            fes.GetElementTransformation(e, &Trans);
            int dof = el.GetDof(), dm = el.GetDim();
            ElemCache ec;
            fes.GetElementDofs(e, ec.vdofs);
            dshape.SetSize(dof, dm); gshape_e.SetSize(dof, dm); shp.SetSize(dof);
            Vector Tloc(dof);
            for (int a = 0; a < dof; a++) Tloc(a) = Tp[ec.vdofs[a]];
            const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), 2*Trans.OrderGrad(&el));
            int nq = ir.GetNPoints();
            ec.shape.resize(nq); ec.rowcoef.resize(nq);
            bool any_nonzero = false;
            for (int q = 0; q < nq; q++) {
                const IntegrationPoint &ip = ir.IntPoint(q);
                el.CalcShape(ip, shp);
                el.CalcDShape(ip, dshape);
                Trans.SetIntPoint(&ip);
                double w = ip.weight * Trans.Weight();
                mfem::Mult(dshape, Trans.InverseJacobian(), gshape_e);  // qualified: this class also has a member Mult()
                double Tq = 0.0;
                for (int a = 0; a < dof; a++) Tq += shp(a) * Tloc(a);
                double kp_q = IN625::k_prime(Tq);
                ec.shape[q] = shp;
                ec.rowcoef[q].resize(dof, 0.0);
                if (kp_q != 0.0) {
                    double gT[3] = {0,0,0};
                    for (int a = 0; a < dof; a++)
                        for (int d = 0; d < dm; d++) gT[d] += gshape_e(a,d) * Tloc(a);
                    for (int a = 0; a < dof; a++) {
                        double gdot = 0.0;
                        for (int d = 0; d < dm; d++) gdot += gshape_e(a,d)*gT[d];
                        ec.rowcoef[q][a] = kp_q * w * gdot;
                        if (ec.rowcoef[q][a] != 0.0) any_nonzero = true;
                    }
                }
            }
            if (any_nonzero) cache.push_back(std::move(ec));
        }
    }
    // v7 (2026-09-24): this operator as an explicit matrix on the persistent all-element pattern (g_pk), for the
    // AO_SOLVER=direct / exactpc paths. Built from the SAME cached data Mult() applies (kvals, the dk/dT correction
    // f_rc on g_pk's quadrature, Jbdry, Meff, identity essential rows with their columns kept), so it equals Mult()
    // up to summation order. The pattern spans all elements, so it is constant across layer births.
    bool IsFast() const { return fast; }
    const SparseMatrix &Pattern() const { return KM(); }
    void AssembleExact(std::vector<double> &D) const {
        MFEM_VERIFY(fast, "AssembleExact needs the persistent-K fast path");
        const SparseMatrix &Kf = KM();
        const int *I = Kf.GetI(), *J = Kf.GetJ();
        const int nnz = Kf.NumNonZeroElems(), n = Height();
        D.resize((size_t)nnz);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < nnz; k++) D[k] = kvals[k];
        const int nact = (int)f_act.size(), nd = g_pk.nd, nq = g_pk.nq;
        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < nact; i++) {
            if (!f_nz[i]) continue;
            const int e = f_act[i];
            const double *rc = &f_rc[(size_t)i * nq * nd];
            double M[64] = {0};
            for (int q = 0; q < nq; q++) {
                const double *shq = &g_pk.sh[(size_t)q * nd];
                for (int a = 0; a < nd; a++) {
                    const double r = rc[q * nd + a];
                    if (r == 0.0) continue;
                    for (int b = 0; b < nd; b++) M[a*nd + b] += r * shq[b];
                }
            }
            const int *pe = &g_pk.pos[(size_t)e * nd * nd];
            for (int k = 0; k < nd*nd; k++) if (M[k] != 0.0) {
                #pragma omp atomic
                D[pe[k]] += M[k];
            }
        }
        const SparseMatrix &Jb = *Jbdry_mat;
        const int *bI = Jb.GetI(), *bJ = Jb.GetJ(); const double *bD = Jb.GetData();
        const double *mo = Meff_over_dt.HostRead();
        int missing = 0;
        #pragma omp parallel for schedule(dynamic, 1024) reduction(+:missing)
        for (int r = 0; r < n; r++) {
            for (int kb = bI[r]; kb < bI[r+1]; kb++) {
                int k = I[r]; while (k < I[r+1] && J[k] != bJ[kb]) k++;
                if (k == I[r+1]) { if (bD[kb] != 0.0) missing++; continue; }
                D[k] += bD[kb];
            }
            int k = I[r]; while (k < I[r+1] && J[k] != r) k++;
            if (k == I[r+1]) { if (mo[r] != 0.0) missing++; } else D[k] += mo[r];
        }
        MFEM_VERIFY(missing == 0, "AssembleExact: " << missing << " Jbdry/Meff entries outside the element pattern");
        for (int q = 0; q < ess_tdof_list.Size(); q++) {
            const int r = ess_tdof_list[q];
            for (int k = I[r]; k < I[r+1]; k++) D[k] = (J[k] == r) ? 1.0 : 0.0;
        }
    }
    void Mult(const Vector &x, Vector &y) const override {
        // Domain (manually assembled, verified correct) + boundary (reused,
        // independently verified negligible/correct here) + nodal
        // capacitance diagonal (reused, verified matches neighbors) --
        // matches JacobianOp::Mult()'s own kd+jb+md*x structure exactly,
        // just with a trustworthy domain matrix instead of kpa_dynamic.
        if (fast) {
            const SparseMatrix &Kf = KM();
            const int *Ip = Kf.GetI(), *Jp = Kf.GetJ(); const double *Kv = kvals.data();
            const double *xs = x.HostRead(); double *ys = y.HostWrite(); const int nr = Height();
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < nr; r++) {
                double d = 0.0;
                for (int k = Ip[r]; k < Ip[r+1]; k++) d += Kv[k] * xs[Jp[k]];
                ys[r] = d;
            }
        } else {
            ManualKpa.Mult(x, y);
        }
        Jbdry_mat->Mult(x, tmp);
        const double *xp = x.HostRead();
        const double *tp = tmp.HostRead();
        const double *mo = Meff_over_dt.HostRead();
        double *yp = y.HostReadWrite();
        int _n = Height();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < _n; i++) yp[i] += tp[i] + mo[i]*xp[i];
        // dk/dT correction (unchanged from before).
        if (fast) {
            const int nact = (int)f_act.size(), nd = g_pk.nd, nq = g_pk.nq;
            #pragma omp parallel for schedule(dynamic, 128)
            for (int i = 0; i < nact; i++) {
                if (!f_nz[i]) continue;
                const int e = f_act[i];
                const int *vde = &g_pk.vd[(size_t)e * nd];
                const double *rc = &f_rc[(size_t)i * nq * nd];
                double xl[8], yl[8];
                for (int a = 0; a < nd; a++) { xl[a] = xp[vde[a]]; yl[a] = 0.0; }
                for (int q = 0; q < nq; q++) {
                    const double *shq = &g_pk.sh[(size_t)q * nd];
                    double xq = 0.0;
                    for (int a = 0; a < nd; a++) xq += shq[a] * xl[a];
                    if (xq == 0.0) continue;
                    for (int a = 0; a < nd; a++) yl[a] += rc[q * nd + a] * xq;
                }
                for (int a = 0; a < nd; a++) {
                    if (yl[a] != 0.0) {
                        #pragma omp atomic
                        yp[vde[a]] += yl[a];
                    }
                }
            }
        } else
        for (const auto &ec : cache) {
            int dof = ec.vdofs.Size();
            for (size_t q = 0; q < ec.shape.size(); q++) {
                double xq = 0.0;
                for (int a = 0; a < dof; a++) xq += ec.shape[q](a) * xp[ec.vdofs[a]];
                if (xq == 0.0) continue;
                for (int a = 0; a < dof; a++)
                    yp[ec.vdofs[a]] += ec.rowcoef[q][a] * xq;
            }
        }
        // Essential rows (BC + not-yet-active dofs) are identity, matching
        // JacobianOp::Mult()'s own convention.
        const int *essp = ess_tdof_list.Read();
        int ness = ess_tdof_list.Size();
        for (int k = 0; k < ness; k++) yp[essp[k]] = xp[essp[k]];
    }
    // DEBUG (2026-09-11): break down Mult()'s row-`row` output into each
    // additive term separately, to find which one introduces the ~1.0
    // excess absent from ManualKpa's own diagonal entry.
    void DebugBreakdown(int row, const Vector &x) const {
        Vector y_manual(Height());
        KM().Mult(x, y_manual);
        Vector y_jbdry(Height());
        Jbdry_mat->Mult(x, y_jbdry);
        const double *xp = x.HostRead();
        double meff_term = Meff_over_dt.HostRead()[row] * xp[row];
        double corr_term = 0.0;
        for (const auto &ec : cache) {
            int dof = ec.vdofs.Size();
            bool touches_row = false;
            for (int a = 0; a < dof; a++) if (ec.vdofs[a] == row) touches_row = true;
            if (!touches_row) continue;
            for (size_t q = 0; q < ec.shape.size(); q++) {
                double xq = 0.0;
                for (int a = 0; a < dof; a++) xq += ec.shape[q](a) * xp[ec.vdofs[a]];
                if (xq == 0.0) continue;
                for (int a = 0; a < dof; a++)
                    if (ec.vdofs[a] == row) corr_term += ec.rowcoef[q][a] * xq;
            }
        }
        bool is_ess = false;
        {
            const int *essp = ess_tdof_list.Read();
            int ness = ess_tdof_list.Size();
            for (int k = 0; k < ness; k++) if (essp[k] == row) is_ess = true;
        }
        double manual_entry = KM()(row, row);
        cout << "  [DEBUG_BREAKDOWN] row=" << row
             << "  ManualKpa.Mult()[row]=" << y_manual[row]
             << "  ManualKpa(row,row) direct=" << manual_entry
             << "  Jbdry_mat.Mult()[row]=" << y_jbdry[row]
             << "  Meff_over_dt term=" << meff_term
             << "  dk/dT corr term=" << corr_term
             << "  is_essential=" << is_ess
             << "  SUM=" << (y_manual[row] + y_jbdry[row] + meff_term + corr_term)
             << std::flush << "\n";
    }
};

// PERF #6: the damped-Newton scaling search evaluates ||R(x + c*dx)|| for a
// handful of trial c per Newton iteration. The conductivity matrix k(T) is
// the expensive part of that residual (full 161k-element assembly) and
// Picard-lagging it across the trial evals -- frozen at the Newton iterate,
// refreshed once per real iteration -- is a negligible accuracy change but
// removes ~3-4 of every ~4-5 full assemblies. Set by DampedNewtonSolver
// around its trial loop.
static bool g_freeze_conductivity = false;

// -------- per-step component wall timers (full-bridge port, 2026-09-12;
//          single-writer -- parallelism is INSIDE each timed region, so
//          plain += is safe). Ported from full_bridge_activation_tier1.cpp
//          unchanged; g_tie_ms is tie_constraint.hpp's own extern. --------
struct StepTimers {
    double t_kasm = 0, t_bdry = 0, t_cap = 0, t_tie = 0, t_pard = 0, t_rap = 0, t_other = 0;
    long   n_kasm = 0;
    void reset() { *this = StepTimers(); }
    void report(double step_wall_ms) const {
        double acc = t_kasm+t_bdry+t_cap+t_tie+t_pard+t_rap;
        printf("  [timers ms] kasm=%.0f(x%ld) bdry=%.0f cap=%.0f tie=%.0f "
               "pardiso=%.0f rap=%.0f | sum=%.0f step=%.0f other=%.0f\n",
               t_kasm, n_kasm, t_bdry, t_cap, t_tie, t_pard, t_rap,
               acc, step_wall_ms, step_wall_ms - acc);
        fflush(stdout);
    }
};
static StepTimers g_tm;
double g_tie_ms = 0;   // defn for tie_constraint.hpp's extern
static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
// PROFILE (2026-09-23): finer timers, timing only -- no behaviour change.
static double g_kbuild_ms = 0, g_krefill_ms = 0, g_kfresh_ms = 0;   // K: first-call build / refill / fresh rebuild
static double g_tk[10] = {0}, g_tk_last = 0;                        // sensitivity-section marks
static inline void tk_mark(int k) { double n = now_ms(); if (k > 0) g_tk[k] += n - g_tk_last; g_tk_last = n; }

// -------------------------------------------------------------------------
//  Row-parallel sparse C = A*B (Gustavson / sparse-accumulator), ported
//  verbatim from full_bridge_activation_tier1.cpp's own ParSpGEMM. Each
//  output row is written by exactly one thread -> no races, no atomics.
//  Columns come out sorted per row (PARDISO-ready). Numerically equals
//  mfem::Mult(A,B) up to the within-row summation order (~1e-15 relative),
//  far below PARDISO's own pivot perturbation. Used below (PardisoReduced-
//  Solver::Mult) to replace the two serial mfem::Mult() calls in the
//  P^T J P reduced-system assembly -- operates ONLY on plain-double
//  SparseMatrix objects (the full-space Jacobian Jf, the tie prolongation
//  P, and their products), never on OTI-typed data, so this is a pure
//  perf transplant with no sensitivity-math interaction at all.
//
//  PERSISTENT per-thread scratch (mark/acc, sized to the max column count
//  ever seen): allocated + zeroed ONCE, then each output row undoes only
//  its own marks so the buffers stay clean between rows AND between calls
//  at no re-init cost.
// -------------------------------------------------------------------------
class ParSpGEMM {
    int nthreads;
    mutable std::vector<std::vector<int>>    mark;   // [tid][nc], all -1 at rest
    mutable std::vector<std::vector<double>> acc;    // [tid][nc], garbage-safe (gated by mark)
    mutable int nc_alloc = 0;
    void ensure(int nc) const {
        if (nc <= nc_alloc) return;
        for (int t = 0; t < nthreads; t++) { mark[t].assign(nc, -1); acc[t].assign(nc, 0.0); }
        nc_alloc = nc;
    }
public:
    ParSpGEMM() : nthreads(std::max(1, omp_get_max_threads())),
                  mark(nthreads), acc(nthreads) {}

    mfem::SparseMatrix* mult(const mfem::SparseMatrix &A,
                             const mfem::SparseMatrix &B) const
    {
        const int nr = A.Height();
        const int nc = B.Width();
        const int    *Ai = A.GetI(); const int *Aj = A.GetJ(); const double *Ax = A.GetData();
        const int    *Bi = B.GetI(); const int *Bj = B.GetJ(); const double *Bx = B.GetData();
        ensure(nc);

        int *Ci = new int[nr + 1];
        Ci[0] = 0;

        // pass 1: count distinct columns per output row
        #pragma omp parallel
        {
            int *mk = mark[omp_get_thread_num()].data();
            std::vector<int> touched; touched.reserve(64);
            #pragma omp for schedule(dynamic, 256)
            for (int i = 0; i < nr; i++) {
                touched.clear();
                for (int p = Ai[i]; p < Ai[i + 1]; p++) {
                    const int k = Aj[p];
                    for (int q = Bi[k]; q < Bi[k + 1]; q++) {
                        const int j = Bj[q];
                        if (mk[j] != i) { mk[j] = i; touched.push_back(j); }
                    }
                }
                Ci[i + 1] = (int)touched.size();
                for (int j : touched) mk[j] = -1;        // undo -> buffer clean
            }
        }
        for (int i = 0; i < nr; i++) Ci[i + 1] += Ci[i];

        const int nnz = Ci[nr];
        int    *Cj = new int[nnz > 0 ? nnz : 1];
        double *Cx = new double[nnz > 0 ? nnz : 1];

        // pass 2: sparse-accumulate each row, sort its columns, emit
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            int    *mk = mark[tid].data();
            double *ac = acc[tid].data();
            std::vector<int> cols; cols.reserve(64);
            #pragma omp for schedule(dynamic, 256)
            for (int i = 0; i < nr; i++) {
                cols.clear();
                for (int p = Ai[i]; p < Ai[i + 1]; p++) {
                    const int k = Aj[p];
                    const double aik = Ax[p];
                    for (int q = Bi[k]; q < Bi[k + 1]; q++) {
                        const int j = Bj[q];
                        const double v = aik * Bx[q];
                        if (mk[j] != i) { mk[j] = i; ac[j] = v; cols.push_back(j); }
                        else ac[j] += v;
                    }
                }
                std::sort(cols.begin(), cols.end());
                const int base = Ci[i];
                for (size_t t = 0; t < cols.size(); t++) {
                    Cj[base + (int)t] = cols[t];
                    Cx[base + (int)t] = ac[cols[t]];
                }
                for (int j : cols) mk[j] = -1;           // undo -> buffer clean
            }
        }

        return new mfem::SparseMatrix(Ci, Cj, Cx, nr, nc);
    }
};


class FachinottiResidualOp : public Operator {
    FiniteElementSpace &fes;
    const vector<char> &attr_active_ref;
    mutable BilinearForm *kpa_dynamic = nullptr;
    mutable ActiveTDepKCoefficient *k_coeff = nullptr;   // persistent; reads x_gf
    mutable bool owns_k = true;   // false while kpa_dynamic points at the shared PersistentK form
    mutable SparseMatrix *Jbdry_mat = nullptr;
    // A2 perf pass, full-bridge port (2026-09-12): persistent boundary-
    // Jacobian sparsity (bdry_qps is fixed for this object's lifetime) +
    // per-qp flat CSR positions, ported from full_bridge_activation_tier1
    // .cpp's own UpdateBoundary parallelization -- built ONCE, then
    // UpdateBoundary below only zeroes+re-scatters the numeric values via
    // OpenMP + atomics. Jb IS Jbdry_mat once built (Jbdry_mat is kept as a
    // separate pointer purely so JacobianOp's own reference-to-pointer
    // member keeps working unchanged).
    mutable SparseMatrix *Jb = nullptr;
    mutable std::vector<int> bpos, boff;
    mutable GridFunction x_gf;
    // A2 perf pass (2026-09-12): OpenMP + geometry/position cache for
    // kpa_dynamic's numeric refill, matching ParConductivity's pattern in
    // the sibling full_bridge_activation_tier1.cpp driver -- built ONCE per
    // FachinottiResidualOp instance (alongside the existing one-time
    // pattern-laying Assemble(0) call below), then Refresh (called ~4x per
    // Newton iteration -- the identified bottleneck: "96% was
    // FachinottiResidualOp rebuilding the 161k-element conductivity
    // BilinearForm") does its OWN OpenMP element loop + atomic scatter
    // directly into kpa_dynamic->SpMat()'s raw data, bypassing MFEM's
    // serial BilinearForm::Assemble() entirely after the first call. No
    // MFEM FiniteElement/ElementTransformation calls in the hot loop (all
    // precomputed here), so this is safe without an MFEM_THREAD_SAFE build.
    mutable vector<Array<int>> kpa_vdofs;
    mutable vector<vector<Vector>> kpa_shape;
    mutable vector<vector<DenseMatrix>> kpa_gshape;
    mutable vector<vector<double>> kpa_weight;
    mutable vector<vector<int>> kpa_epos;   // [e][a*nd+c] -> flat CSR data index
    // PERF: skip the 3 Update* passes when the Newton solver re-queries the
    // SAME iterate (it calls Mult(x) then GetGradient(x) back-to-back, and
    // the line search re-probes). last_update_x holds the state the current
    // kpa_dynamic/Jbdry_mat/Meff were assembled at.
    mutable Vector last_update_x;
    mutable bool has_update = false;
    mutable long n_asm = 0, n_skip = 0;
    const vector<BdryQP> &bdry_qps;
    const Vector &u_n, &L_n, &V_node, &f_src;
    Vector &M_cp_lump, &M_L_diag, &L_now, &Meff_over_dt;
    Vector &Rbdry;
    double dt_cur, T_solidus, T_liquidus, DT_mushy;
    double h_conv, emissivity, sigma_sb, T_ambient;
    double substrate_loss_scale, bridge_loss_scale, interface_loss_scale;
    const Array<int> &ess_tdof_list;
    mutable JacobianOp J;
public:
    // FULLY IMPLICIT conductivity (2026-07-14): a quick analytical energy
    // check found row1's own average absorbed laser power (~24.4W) is
    // ~70x larger than what its own surface (convection+radiation) can
    // dissipate at its plateau temperature (~0.35W) -- meaning conduction
    // into the substrate through the tie, not row1's own surface, is
    // overwhelmingly the dominant heat-removal path during active
    // heating. k(T) was still Picard-lagged (frozen at the previous
    // step's u_gf) even after fixing the boundary term, and the earlier
    // dt=52-vs-dt=5.2 controlled test showed a residual ~3.3K bias
    // consistent with this. Reassembling the diffusion matrix at the
    // CURRENT Newton iterate (not the frozen previous state) makes the
    // RESIDUAL exact/self-consistent in k(T) at convergence, matching
    // Abaqus's own UNSYMM=YES fully-coupled treatment -- same principle
    // as the boundary-radiation fix, just applied to the (non-diagonal)
    // stiffness matrix instead of a diagonal nodal term.
    //
    // dk/dT Jacobian term tried and REMOVED (2026-07-15): implemented as a
    // hand-written per-element quadrature loop (Abaqus's theory manual
    // documents this as a real, non-symmetric Jacobian contribution for a
    // T-dependent conductivity, included whenever UNSYMM=YES as this deck
    // sets) -- but controlled A/B testing showed it has negligible effect
    // on Newton's convergence rate for this problem's temperature range
    // (iteration-by-iteration residuals nearly identical with vs. without
    // it), disproving the hypothesis that it explained the slow (linear,
    // not quadratic) convergence seen after switching to the consistent
    // boundary Jacobian. Removed to avoid its real per-Newton-iteration
    // assembly cost (a loop over ~148K active elements) for no measured
    // benefit -- see JacobianOp's comment for the fuller writeup.
    void UpdateConductivity(const Vector &x) const {
        double _t_kasm = now_ms();
        const bool _first_k = !kpa_dynamic;
        x_gf = x;
        x_gf.HostReadWrite();
        {
            static const bool use_pk = (getenv("KPERSIST_OFF") == nullptr);
            if (use_pk) {
                if (!g_pk.built) g_pk.Build(fes);
                if (g_pk.ok) {
                    if (kpa_dynamic != g_pk.form) {   // first call of this step, or re-attach after a fresh rebuild
                        if (owns_k) { delete kpa_dynamic; delete k_coeff; }
                        kpa_dynamic = g_pk.form; owns_k = false; k_coeff = nullptr;
                    }
                    g_pk.Refill(x_gf.HostRead(), attr_active_ref);
                    g_tm.t_kasm += now_ms() - _t_kasm;  g_tm.n_kasm++;
                    (_first_k ? g_kbuild_ms : g_krefill_ms) += now_ms() - _t_kasm;
                    return;
                }
            }
        }
        if (!kpa_dynamic) {
            // first call: build the form + CSR pattern once. The pattern
            // spans ALL elements (inactive ones scatter structural zeros
            // via the zero coefficient), so it is stable across birth
            // events -- later calls only refresh the numeric values.
            k_coeff = new ActiveTDepKCoefficient(x_gf, attr_active_ref);
            kpa_dynamic = new BilinearForm(&fes);
            kpa_dynamic->AddDomainIntegrator(new DiffusionIntegrator(*k_coeff));
            kpa_dynamic->Assemble(0);            // skip_zeros=0: every element
                                                 // scatters its slots -> pattern
                                                 // covers not-yet-active layers
            kpa_dynamic->Finalize(0);
            kpa_dynamic->SpMat().Finalize(0, true);

            // Build the geometry + CSR-position cache ONCE, right after the
            // pattern exists -- every later call reuses this, doing its own
            // OpenMP element loop directly into SpMat().GetData() instead of
            // MFEM's serial BilinearForm::Assemble().
            {
                Mesh &mesh_ = *fes.GetMesh();
                int ne = mesh_.GetNE();
                kpa_vdofs.resize(ne); kpa_shape.resize(ne); kpa_gshape.resize(ne);
                kpa_weight.resize(ne); kpa_epos.resize(ne);
                SparseMatrix &A = kpa_dynamic->SpMat();
                const int *I = A.GetI(), *J = A.GetJ();
                for (int e = 0; e < ne; e++) {
                    const FiniteElement &el = *fes.GetFE(e);
                    IsoparametricTransformation Trans;
                    fes.GetElementTransformation(e, &Trans);
                    int dof = el.GetDof(), dm = el.GetDim();
                    fes.GetElementDofs(e, kpa_vdofs[e]);
                    const Array<int> &vd = kpa_vdofs[e];
                    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), 2*Trans.OrderGrad(&el));
                    int nq = ir.GetNPoints();
                    kpa_shape[e].resize(nq); kpa_gshape[e].resize(nq); kpa_weight[e].resize(nq);
                    DenseMatrix dshape(dof, dm);
                    for (int q = 0; q < nq; q++) {
                        const IntegrationPoint &ip = ir.IntPoint(q);
                        kpa_shape[e][q].SetSize(dof);
                        el.CalcShape(ip, kpa_shape[e][q]);
                        el.CalcDShape(ip, dshape);
                        Trans.SetIntPoint(&ip);
                        kpa_weight[e][q] = ip.weight * Trans.Weight();
                        kpa_gshape[e][q].SetSize(dof, dm);
                        mfem::Mult(dshape, Trans.InverseJacobian(), kpa_gshape[e][q]);
                    }
                    kpa_epos[e].assign((size_t)dof*dof, -1);
                    for (int a = 0; a < dof; a++) {
                        int r = vd[a];
                        for (int c = 0; c < dof; c++) {
                            int col = vd[c];
                            for (int k = I[r]; k < I[r+1]; k++)
                                if (J[k] == col) { kpa_epos[e][a*dof + c] = k; break; }
                        }
                    }
                }
            }
        } else {
            // OpenMP refill (2026-09-12): replaces
            // "SpMat()=0.0; Assemble(0)" (MFEM's own serial element loop)
            // with a direct parallel scatter into the SAME (already-laid-
            // down, unchanged) sparsity pattern, using the geometry/
            // position cache built above. Byte-matches DiffusionIntegrator's
            // own adjugate-form element matrix up to the non-associativity
            // of the atomic float add (~1e-14 rel) -- same tolerance the
            // sibling full_bridge_activation_tier1.cpp driver's own
            // ParConductivity documents for the identical technique.
            SparseMatrix &A = kpa_dynamic->SpMat();
            double *Ad = A.GetData();
            const int nnz = A.NumNonZeroElems();
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < nnz; k++) Ad[k] = 0.0;

            const double *xp = x_gf.HostRead();
            const vector<char> &attr_active_ = attr_active_ref;
            Mesh &mesh_ = *fes.GetMesh();
            const int ne = (int)kpa_vdofs.size();
            #pragma omp parallel for schedule(dynamic, 64)
            for (int e = 0; e < ne; e++) {
                if (!attr_active_[mesh_.GetAttribute(e)]) continue;
                const Array<int> &vd = kpa_vdofs[e];
                int dof = vd.Size();
                Vector Tloc(dof);
                for (int a = 0; a < dof; a++) Tloc(a) = xp[vd[a]];
                int nq = (int)kpa_weight[e].size();
                vector<double> elmat((size_t)dof*dof, 0.0);
                for (int q = 0; q < nq; q++) {
                    const Vector &shape = kpa_shape[e][q];
                    const DenseMatrix &gshape_e = kpa_gshape[e][q];
                    double w = kpa_weight[e][q];
                    int dm = gshape_e.Width();
                    double Tq = 0.0;
                    for (int a = 0; a < dof; a++) Tq += shape(a) * Tloc(a);
                    double kw = w * IN625::k(Tq);
                    for (int a = 0; a < dof; a++)
                        for (int c = 0; c < dof; c++) {
                            double s = 0.0;
                            for (int d = 0; d < dm; d++) s += gshape_e(a,d)*gshape_e(c,d);
                            elmat[a*dof + c] += kw * s;
                        }
                }
                const vector<int> &pos = kpa_epos[e];
                for (int a = 0; a < dof; a++)
                    for (int c = 0; c < dof; c++) {
                        double v = elmat[a*dof + c];
                        if (v != 0.0) {
                            #pragma omp atomic
                            Ad[pos[a*dof + c]] += v;
                        }
                    }
            }
        }
        g_tm.t_kasm += now_ms() - _t_kasm;  g_tm.n_kasm++;
        (_first_k ? g_kbuild_ms : g_krefill_ms) += now_ms() - _t_kasm;
    }
    // one guarded refresh of all state-dependent operators; skipped when the
    // Newton solver re-queries the same iterate.
    void EnsureUpdated(const Vector &x) const {
        if (has_update && x.Size() == last_update_x.Size()) {
            double d = 0.0;
            const double *a = x.HostRead(), *b = last_update_x.HostRead();
            for (int i = 0; i < x.Size(); i++) { double e = a[i]-b[i]; d += e*e; }
            if (d == 0.0) { ++n_skip; return; }
        }
        UpdateCapacitance(x);
        UpdateBoundary(x);
        if (g_freeze_conductivity && kpa_dynamic) {
            // Picard-lagged: reuse k(T) from the last full update. State is
            // now partial -> don't let a later same-x query skip.
            has_update = false;
            ++n_asm_light;
        } else {
            UpdateConductivity(x);
            last_update_x = x;
            has_update = true;
            ++n_asm;
        }
    }
    mutable long n_asm_light = 0;
    // LUMPED capacitance/latent-heat terms only (2026-07-14 split): Abaqus's
    // own theory manual confirms first-order elements use corner-integrated
    // (diagonal) heat capacitance -- this part of the nodal update stays
    // lumped, matching that. The boundary film/radiation term used to live
    // here too (as a lumped nodal Rb/Jb pair); it has moved to
    // UpdateBoundary below, which is NOT lumped (see BdryQP comment above).
    void UpdateCapacitance(const Vector &Tvec) const {
        double _t = now_ms();
        const double *Tp = Tvec.Read();
        const double *Tn = u_n.Read();
        const double *Vp = V_node.Read();
        double *Mc = M_cp_lump.Write();
        double *Ml = M_L_diag.Write();
        double *Lp = L_now.Write();
        double *Mo = Meff_over_dt.Write();
        const double _Ts = T_solidus;
        const double _Tl = T_liquidus;
        const double _DT = DT_mushy;
        const double _idt = 1.0 / dt_cur;
        int n = Height();
        // Ported from full_bridge_activation_tier1.cpp (2026-09-10): plain
        // double nodal arrays only (Tvec is the Newton iterate, never an
        // OTI-typed quantity here -- sensitivity RHS terms are built
        // separately in the A2/A3 blocks below from the converged nominal
        // state), so this is a safe drop-in OpenMP parallelization of what
        // was a device-portable-but-CPU-serial mfem::forall.
        const double *Mv = g_ever_melted ? g_ever_melted->data() : nullptr;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double Ti = Tp[i];
            // v8: density blends powder/bulk at this node (lumped term, no quadrature interpolation
            // needed); cp has no powder variant per the paper's own Table 1, stays bulk cp_dev always.
            double rho_i = Mv ? (Mv[i] * IN625::rho_dev(Ti) + (1.0 - Mv[i]) * IN625::rhop_dev(Ti))
                              : IN625::rho_dev(Ti);
            Mc[i] = Vp[i] * rho_i * IN625::cp_dev(Ti);
            Ml[i] = Vp[i] * rho_i * (IN625::L_fusion * g_Lscale) *
                    IN625::df_L_dT_dev(Ti);
            Lp[i] = Vp[i] * rho_i * (IN625::L_fusion * g_Lscale) * IN625::f_L_dev(Ti);
            // dMc/dT_i*(T_i-Tn_i)/dt: the capacitance term's own missing
            // chain-rule Jacobian piece (see IN625::cp_prime comment).
            double dMc_dT = Vp[i] * rho_i * IN625::cp_prime(Ti);
            Mo[i] = (Mc[i] + Ml[i]) * _idt + dMc_dT * (Ti - Tn[i]) * _idt;
        }
        g_tm.t_cap += now_ms() - _t;
    }
    // CONSISTENT (non-lumped) boundary radiation (2026-07-14): per the
    // Abaqus theory manual, the film/radiation Jacobian contribution is a
    // standard Galerkin surface integral, not a lumped diagonal. Evaluated
    // fresh at the CURRENT Newton iterate x (fully implicit, matching
    // Abaqus's own UNSYMM=YES treatment) using true quadrature-point
    // interpolation of x (not per-node evaluation), so the T^4
    // nonlinearity is handled consistently with how Abaqus assembles it.
    // Rebuilds Rbdry and Jbdry_mat from scratch every Newton call -- cheap,
    // since bdry_qps only has O(1e3) quadrature points.
    // Full-bridge port (2026-09-12): rewritten from a "delete+new the whole
    // SparseMatrix, then Jbdry_mat->Add(...) into a list-of-lists, then
    // Finalize" pattern (fine for a 4-layer mesh's O(1e3) boundary QPs) to a
    // persistent-pattern + OpenMP + atomic-scatter refill, ported from
    // full_bridge_activation_tier1.cpp's own UpdateBoundary -- needed on the
    // 552k-element full-bridge mesh where the boundary QP count scales with
    // the whole exterior+interface surface. CRITICAL: the persistent
    // pattern below is laid down with Finalize(0, false) (fix_empty_rows=
    // false), NOT the ancestor's Finalize(0, true) -- the ROOT-CAUSE FIX
    // comment this replaces documents that fix_empty_rows=true silently
    // stamps a spurious diagonal 1.0 on every row with zero boundary
    // contribution, corrupting the exact Jacobian AssembleFull() needs for
    // the sensitivity solve. The full-bridge ancestor's own UpdateBoundary
    // predates that fix (still used Finalize(0,true) for its one-time
    // pattern build) -- adopting its parallel STRUCTURE here while keeping
    // this file's (false) fix is exactly the "OTI-safe pattern wins"
    // resolution called for when the two ancestors' perf passes conflict.
    void UpdateBoundary(const Vector &x) const {
        double _t = now_ms();
        const int n = Height();
        const int nqp = (int)bdry_qps.size();
        const double *xp = x.HostRead();
        double *Rb = Rbdry.HostWrite();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) Rb[i] = 0.0;

        if (!Jb) {
            // build the boundary-Jacobian sparsity ONCE for this residual-op
            // instance (bdry_qps is fixed for its lifetime), then only refill.
            Jb = new SparseMatrix(n, n);
            for (const BdryQP &qp : bdry_qps) {
                int nd = (int)qp.dof.size();
                for (int a = 0; a < nd; a++)
                    for (int b = 0; b < nd; b++) Jb->Add(qp.dof[a], qp.dof[b], 0.0);
            }
            Jb->Finalize(0, false);   // fix_empty_rows=false -- see comment above
            const int *I = Jb->GetI(), *Jc = Jb->GetJ();
            boff.resize(nqp + 1); boff[0] = 0;
            for (int q = 0; q < nqp; q++) {
                int nd = (int)bdry_qps[q].dof.size();
                boff[q+1] = boff[q] + nd*nd;
            }
            bpos.assign(boff[nqp], -1);
            #pragma omp parallel for schedule(dynamic, 512)
            for (int q = 0; q < nqp; q++) {
                const BdryQP &qp = bdry_qps[q];
                int nd = (int)qp.dof.size();
                for (int a = 0; a < nd; a++) {
                    int r = qp.dof[a];
                    for (int b = 0; b < nd; b++) {
                        int col = qp.dof[b];
                        for (int k = I[r]; k < I[r+1]; k++)
                            if (Jc[k] == col) { bpos[boff[q] + a*nd + b] = k; break; }
                    }
                }
            }
        } else {
            double *d = Jb->GetData(); int nnz = Jb->NumNonZeroElems();
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < nnz; k++) d[k] = 0.0;
        }
        double *Jd = Jb->GetData();
        const double _hconv = h_conv, _hp = h_conv * IN625::HP_OVER_HC, _emis = emissivity, _sigma = sigma_sb, _Ta = T_ambient;
        const double _Ta2 = _Ta*_Ta;
        const double *Mv_b = g_ever_melted ? g_ever_melted->data() : nullptr;
        #pragma omp parallel for schedule(dynamic, 512)
        for (int q = 0; q < nqp; q++) {
            const BdryQP &qp = bdry_qps[q];
            double qscale = (qp.kind == 1) ? substrate_loss_scale
                          : ((qp.kind == 2) ? bridge_loss_scale : interface_loss_scale);
            int nd = (int)qp.dof.size();
            double Tq = 0.0;
            for (int a = 0; a < nd; a++) Tq += qp.shp[a] * xp[qp.dof[a]];
            // v8: part-cake (hp) vs argon (hc) convection, blended by this face's own melt fraction --
            // same physical split as the domain k/kp blend, applied to the film coefficient only
            // (radiation/emissivity has no powder variant in the paper's own Table 1).
            double mf_b = 1.0;
            if (Mv_b) { mf_b = 0.0; for (int a = 0; a < nd; a++) mf_b += qp.shp[a] * Mv_b[qp.dof[a]]; }
            const double hconv_eff = mf_b * _hconv + (1.0 - mf_b) * _hp;
            double Tq3 = Tq*Tq*Tq;
            double qval  = hconv_eff*(Tq - _Ta) + _emis*_sigma*(Tq3*Tq - _Ta2*_Ta2);
            double qprime = hconv_eff + 4.0*_emis*_sigma*Tq3;
            const int *bp = &bpos[boff[q]];
            for (int a = 0; a < nd; a++) {
                double rr = qscale * qval * qp.shp[a] * qp.w;
                #pragma omp atomic
                Rb[qp.dof[a]] += rr;
                for (int b = 0; b < nd; b++) {
                    double jj = qscale * qprime * qp.shp[a] * qp.shp[b] * qp.w;
                    #pragma omp atomic
                    Jd[bp[a*nd + b]] += jj;
                }
            }
        }
        Jbdry_mat = Jb;
        g_tm.t_bdry += now_ms() - _t;
    }
    FachinottiResidualOp(FiniteElementSpace &fespace_, const vector<char> &attr_active_,
                         const vector<BdryQP> &qps,
                         const Vector &un, const Vector &ln,
                         const Vector &vnode, Vector &mcp, Vector &mld,
                         Vector &lnow, Vector &meff,
                         Vector &rbdry,
                         const Vector &fsrc,
                         double dt, double Ts, double Tl, double dTmush,
                         double hconv, double emis, double sigma, double Tamb,
                         double subscale, double bridgescale, double ifscale,
                         const Array<int> &ess, int n)
        : Operator(n), fes(fespace_), attr_active_ref(attr_active_), x_gf(&fespace_),
          bdry_qps(qps), u_n(un), L_n(ln), V_node(vnode),
          f_src(fsrc), M_cp_lump(mcp), M_L_diag(mld),
          L_now(lnow), Meff_over_dt(meff), Rbdry(rbdry),
          dt_cur(dt), T_solidus(Ts), T_liquidus(Tl), DT_mushy(dTmush),
          h_conv(hconv), emissivity(emis), sigma_sb(sigma), T_ambient(Tamb),
          substrate_loss_scale(subscale), bridge_loss_scale(bridgescale), interface_loss_scale(ifscale),
          ess_tdof_list(ess), J(kpa_dynamic, Jbdry_mat, meff, ess, n)
    { UpdateConductivity(u_n); }
    // Jb IS Jbdry_mat once UpdateBoundary has run once (Jbdry_mat = Jb, a
    // plain pointer alias) -- delete only Jb to avoid a double-free.
    ~FachinottiResidualOp() { if (owns_k) { delete kpa_dynamic; delete k_coeff; } delete Jb; }
    void Mult(const Vector &x, Vector &r) const override {
        EnsureUpdated(x);
        kpa_dynamic->Mult(x, r);
        const double *r1 = r.Read();
        const double *rb = Rbdry.Read();
        const double *mc = M_cp_lump.Read();
        const double *lnow = L_now.Read();
        const double *ln = L_n.Read();
        const double *xp = x.Read();
        const double *un = u_n.Read();
        const double *fs = f_src.Read();
        double *rp = r.Write();
        double _idt = 1.0 / dt_cur;
        int n = Height();
        // Ported from full_bridge_activation_tier1.cpp (2026-09-10): plain
        // double residual combine, each i only touches its own index (no
        // cross-index dependence) and no OTI-typed data is involved here
        // (the A2/A3 sensitivity RHS vectors are assembled separately) --
        // safe drop-in OpenMP parallelization of the CPU-serial mfem::forall.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            rp[i] = r1[i] + rb[i]
                  + mc[i] * _idt * (xp[i] - un[i])
                  + _idt * (lnow[i] - ln[i])
                  - fs[i];
        const int *ess = ess_tdof_list.Read();
        int ness = ess_tdof_list.Size();
        double *rp2 = r.ReadWrite();
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < ness; k++) rp2[ess[k]] = 0.0;
    }
    Operator &GetGradient(const Vector &x) const override {
        EnsureUpdated(x);
        return J;
    }
    void ReportAsmStats() const {
        std::cout << "  [resid-op] full k-assemblies=" << n_asm
                  << "  light (k-frozen)=" << n_asm_light
                  << "  skipped (same-x)=" << n_skip << std::endl;
    }

    // A2 fix (2026-09-11): the forward Newton solve's own kpa_dynamic is
    // rebuilt ~70x/step via a "keep sparsity pattern, SpMat()=0, Assemble(0)
    // again" reuse cycle (see UpdateConductivity above) -- efficient for
    // driving Newton, which never needs this matrix to be exact. Found via
    // extensive diagnosis (RHS proven exact on the full vector; Jacobian
    // proven exact on 3 independent test vectors including the actual
    // solution; GMRES's true residual proven tiny; element-local
    // AssembleElementMatrix on each of a problem row's own 4 elements,
    // summed independently, reproduces the true FD value to <0.1% -- but
    // kpa_dynamic's OWN reused-and-reassembled matrix is off by ~1.0 at
    // that exact row) that this reuse cycle leaves a stale/duplicated
    // contribution at at least one row, invisible to Newton's own
    // convergence but fatal to a sensitivity solve that needs J to be the
    // true ∂r/∂T. Fix: for the sensitivity solve ONLY, throw away
    // kpa_dynamic's reused state and rebuild it completely fresh (plain
    // Assemble(), no skip_zeros=0 pattern-reuse dance) at the converged
    // u_iter -- matching exactly what the diagnostic proved correct.
    // Harmless to the forward solve: this runs strictly AFTER Newton has
    // already converged for this step, and next step's UpdateConductivity
    // rebuilds from scratch again regardless of what's left here.
    void RebuildConductivityFreshForSensitivity(const Vector &x) const {
        const double _tf0 = now_ms();
        x_gf = x;
        x_gf.HostReadWrite();
        // v6: the fresh rebuild was a SERIAL MFEM assembly (18-20 s/step on the 2x mesh). The PersistentK refill computes
        // the same K(x) (values equal up to summation order; the pattern spans all elements, so there are no empty rows
        // and no fix_empty_rows padding). This matrix only feeds the PARDISO preconditioner of the sensitivity GMRES.
        // KFRESH_LEGACY=1 restores the serial rebuild.
        static const bool kfresh_legacy = (std::getenv("KFRESH_LEGACY") != nullptr);
        if (!kfresh_legacy && g_pk.built && g_pk.ok) {
            if (owns_k) { delete kpa_dynamic; delete k_coeff; }
            kpa_dynamic = g_pk.form; owns_k = false; k_coeff = nullptr;
            g_pk.Refill(x_gf.HostRead(), attr_active_ref);
            g_kfresh_ms += now_ms() - _tf0;
            return;
        }
        if (owns_k) { delete kpa_dynamic; delete k_coeff; }
        owns_k = true;
        k_coeff = new ActiveTDepKCoefficient(x_gf, attr_active_ref);
        kpa_dynamic = new BilinearForm(&fes);
        kpa_dynamic->AddDomainIntegrator(new DiffusionIntegrator(*k_coeff));
        kpa_dynamic->Assemble();   // plain assemble -- no pattern-reuse cycle
        kpa_dynamic->Finalize();
        kpa_dynamic->SpMat().Finalize(0, true);
        g_kfresh_ms += now_ms() - _tf0;
    }

    // Accessor for the sensitivity solve's own manually-assembled Jacobian
    // (needs Jbdry_mat, which is otherwise private).
    // Returns a REFERENCE TO THE POINTER member (not a copy) -- matches the
    // kpa/Jbdry_mat pattern JacobianOp already uses (see its own comment
    // above). A plain-value return here was the bug (2026-09-11): callers
    // holding onto that returned pointer would dangle the moment
    // UpdateBoundary() next delete+new's the real object; binding a
    // reference instead means callers always see the CURRENT object.
    SparseMatrix*& GetJbdryMat() const { return Jbdry_mat; }
    const vector<char>& GetAttrActive() const { return attr_active_ref; }
};

class FachinottiLinearSolver : public Solver {
    // Reverted to CG (2026-07-15): the dk/dT Jacobian term that required
    // GMRES (non-symmetric) was removed -- see JacobianOp's comment. The
    // remaining Jacobian (kpa_dynamic + Jbdry_mat + diag(Meff_over_dt)) is
    // symmetric, so CG is valid again and cheaper per iteration than GMRES.
    CGSolver cg;
    const Vector &Kd_pa, &Kb_diag, &Meff_over_dt;
    const Array<int> &ess_tdof_list;
    mutable Vector Jdiag;
    mutable OperatorJacobiSmoother *jac = nullptr;
    const Operator *A = nullptr;
public:
    FachinottiLinearSolver(const Vector &kd, const Vector &kb, const Vector &meff,
                           const Array<int> &ess, int n)
        : Solver(n), Kd_pa(kd), Kb_diag(kb), Meff_over_dt(meff),
          ess_tdof_list(ess), Jdiag(n)
    {
        Jdiag.UseDevice(true);
        cg.iterative_mode = false;
        cg.SetRelTol(1e-8);
        cg.SetAbsTol(0.0);
        cg.SetMaxIter(2000);
        cg.SetPrintLevel(-1);
    }
    ~FachinottiLinearSolver() override { delete jac; }
    void SetOperator(const Operator &op) override { A = &op; cg.SetOperator(op); }
    void Mult(const Vector &b, Vector &x) const override {
        MFEM_VERIFY(A != nullptr, "FachinottiLinearSolver operator is not set.");
        const double *kd = Kd_pa.Read();
        const double *kb = Kb_diag.Read();
        const double *mo = Meff_over_dt.Read();
        double *jd = Jdiag.Write();
        int n = Height();
        mfem::forall(n, [=] MFEM_HOST_DEVICE(int i){ jd[i] = kd[i] + kb[i] + mo[i]; });
        {
            const int *ess = ess_tdof_list.Read();
            int ness = ess_tdof_list.Size();
            double *jd2 = Jdiag.ReadWrite();
            mfem::forall(ness, [=] MFEM_HOST_DEVICE(int k){ jd2[ess[k]] = 1.0; });
        }
        delete jac;
        jac = new OperatorJacobiSmoother(Jdiag, ess_tdof_list, 1.0);
        const_cast<CGSolver&>(cg).SetPreconditioner(*jac);
        const_cast<CGSolver&>(cg).Mult(b, x);
    }
};

// ============================================================
//  TIER-1: direct (MKL PARDISO) reduced-space linear solver.
//  Drop-in replacement for ReducedLinearSolver (Jacobi-CG). Each Newton
//  iteration it assembles the full-space Jacobian from the JacobianOp
//  (reached through the ReducedOpWrapper it is handed), forms the reduced
//  system Jr = P^T J_full P, and factorizes it. Refactorize() reuses the
//  symbolic analysis while the sparsity pattern holds (between activation
//  events) and auto-falls-back to a full Factorize() when the pattern
//  grows. Single RHS here; SolveMulti() is the A2 multi-parameter path.
// ============================================================
class PardisoReducedSolver : public mfem::Solver {
    const TieMap &tie;
    mfem::SparseMatrix *P;                 // ndof_full x ndof_free, owned
    mutable sensei::PardisoDirectSolver pard;
    mutable const JacobianOp *jac = nullptr;
    mutable std::unique_ptr<mfem::SparseMatrix> Jr;
    // A2 perf pass (2026-09-12): lazy refactorization. Mult() used to
    // rebuild+refactorize Jr on EVERY call -- correct but wasteful for the
    // sensitivity solve, where SetOperator() is called ONCE before the
    // whole NW-parameter GMRES loop (the Jacobian never changes across all
    // 4 parameters' entire GMRES solves), yet every GMRES iteration was
    // paying a full ~500ms+ refactorization for an UNCHANGED matrix. Fix:
    // SetOperator() marks the factorization stale; Mult() only rebuilds/
    // refactorizes when stale, then reuses it. This is automatically safe
    // for the FORWARD Newton solve too -- MFEM's NewtonSolver calls
    // SetOperator() again every iteration (the Jacobian genuinely changes
    // there), so it keeps refactorizing every iteration exactly as before;
    // only the sensitivity-solve use site (SetOperator() called once,
    // Mult() called many times) actually benefits.
    mutable bool need_refactor = true;
    // Full-bridge port (2026-09-12): row-parallel Gustavson SpGEMM
    // (ParSpGEMM, defined above) replaces the two serial mfem::Mult() calls
    // below -- the last serial hotspot in the reduced-system assembly on
    // the 552k-element mesh. Operates purely on plain-double SparseMatrix
    // objects (Jf/P/Pt/AP/Jr never carry OTI-typed data -- the sensitivity
    // solve reuses this SAME reduced Jacobian via IFT, it does not rebuild
    // it with dual numbers), so there is no OTI-safety concern here, unlike
    // UpdateBoundary/UpdateConductivity above. rap_checked/rap_fallback
    // implement a ONE-TIME self-verification against the trusted
    // mfem::Mult() path the first time the (now lazy) refactor actually
    // runs; a permanent fallback to the serial path if it ever disagrees.
    mutable bool rap_checked = false;
    mutable bool rap_fallback = false;
    ParSpGEMM spgemm;
public:
    explicit PardisoReducedSolver(const TieMap &t)
        : mfem::Solver(t.ndof_free), tie(t), P(t.BuildProlongation()) {}
    ~PardisoReducedSolver() override { delete P; }

    void SetOperator(const mfem::Operator &op) override {
        auto *rw = dynamic_cast<const ReducedOpWrapper*>(&op);
        MFEM_VERIFY(rw, "PardisoReducedSolver expects a ReducedOpWrapper");
        jac = dynamic_cast<const JacobianOp*>(&rw->full_op);
        MFEM_VERIFY(jac, "PardisoReducedSolver expects a JacobianOp inside the wrapper");
        need_refactor = true;
    }

    mutable int call_count = 0;
    void Mult(const mfem::Vector &b, mfem::Vector &x) const override {
        MFEM_VERIFY(jac, "PardisoReducedSolver operator is not set");
        auto ms = [](auto a, auto b){ return std::chrono::duration_cast<std::chrono::milliseconds>(b-a).count(); };
        bool did_refactor = need_refactor;
        if (need_refactor) {
            auto t0 = std::chrono::steady_clock::now();
            std::unique_ptr<mfem::SparseMatrix> Jf(jac->AssembleFull());   // ndof_full^2
            std::unique_ptr<mfem::SparseMatrix> Pt(mfem::Transpose(*P));   // free x full
            if (rap_fallback) {
                std::unique_ptr<mfem::SparseMatrix> AP(mfem::Mult(*Jf, *P));
                Jr.reset(mfem::Mult(*Pt, *AP));
                Jr->SortColumnIndices();
            } else {
                std::unique_ptr<mfem::SparseMatrix> AP(spgemm.mult(*Jf, *P));   // full x free
                Jr.reset(spgemm.mult(*Pt, *AP));   // P^T J_full P : free x free (cols pre-sorted)
            }
            // one-time cross-check of the parallel triple product vs
            // mfem::Mult, the first time this (lazy) block ever runs;
            // permanently fall back to the serial path if it disagrees.
            if (!rap_checked && !rap_fallback) {
                std::unique_ptr<mfem::SparseMatrix> AP0(mfem::Mult(*Jf, *P));
                std::unique_ptr<mfem::SparseMatrix> Jr0(mfem::Mult(*Pt, *AP0));
                Jr0->SortColumnIndices();
                double num = 0.0, den = 0.0;
                for (int i = 0; i < Jr->Height(); i++) {
                    mfem::Vector r1v, r0v; mfem::Array<int> c1, c0;
                    Jr->GetRow(i, c1, r1v); Jr0->GetRow(i, c0, r0v);
                    for (int t = 0; t < r0v.Size(); t++) den = std::max(den, std::fabs(r0v[t]));
                    if (c1.Size() == c0.Size())
                        for (int t = 0; t < r1v.Size(); t++) num = std::max(num, std::fabs(r1v[t] - r0v[t]));
                    else num = 1e30;
                }
                double rel = (den > 0 ? num / den : 0.0);
                std::cout << "  [rap-check] parallel vs mfem::Mult  max|dJr|=" << num
                          << "  max|Jr|=" << den << "  rel=" << rel << std::endl;
                if (rel > 1e-9) {
                    rap_fallback = true;
                    Jr.reset(Jr0.release());   // use the trusted serial result this step
                    std::cout << "  [rap-check] MISMATCH -> permanent serial fallback for P^T J P"
                              << std::endl;
                }
                rap_checked = true;
            }
            auto t1 = std::chrono::steady_clock::now();
            pard.Refactorize(*Jr);                 // Factorize() on first call / pattern change
            auto t2 = std::chrono::steady_clock::now();
            need_refactor = false;
            g_tm.t_rap  += std::chrono::duration<double, std::milli>(t1 - t0).count();
            g_tm.t_pard += std::chrono::duration<double, std::milli>(t2 - t1).count();
            if (call_count < 3 || call_count % 10 == 0)
                std::cout << "  [pardiso call " << call_count << "] RAP=" << ms(t0,t1)
                          << "ms  factor=" << ms(t1,t2)
                          << "ms  nnz(Jr)=" << Jr->NumNonZeroElems() << std::endl;
        }
        auto t2 = std::chrono::steady_clock::now();
        pard.Solve(b, x);
        auto t3 = std::chrono::steady_clock::now();
        g_tm.t_pard += std::chrono::duration<double, std::milli>(t3 - t2).count();
        if (call_count < 3 || call_count % 10 == 0)
            std::cout << "  [pardiso call " << call_count << "] solve=" << ms(t2,t3) << "ms"
                      << (did_refactor ? "" : " (reused factorization)") << std::endl;
        ++call_count;
    }

    // A2 hook: many sensitivity RHS against the same factorization.
    void SolveMulti(const std::vector<const mfem::Vector*> &B,
                    std::vector<mfem::Vector*> &X) const {
        pard.SolveMulti(B, X);
    }
};

// ============================================================
//  v7 (2026-09-24): exact reduced sensitivity matrix Jr = P^T J_corr P factorised with nonsymmetric PARDISO
//  (mtype 11). J_corr is the explicit form of CorrectedJacobianOp (AssembleExact) on the fixed all-element pattern,
//  so PARDISO's analysis is reused between births (exact-zero entries of unborn layers are dropped, as in v3).
//  AO_SOLVER=direct (default): factorise every step, solve each field with that exact factorisation (no GMRES;
//                     single-RHS solves, AO_BATCH=1 for one batched call per order).
//  AO_SOLVER=exactpc: the same factorisation reused across steps as the GMRES preconditioner (refresh policy as for
//                     AO_SOLVER=gmres, whose preconditioner is PARDISO on the base Jacobian K + Jbdry + M).
//  Approach follows bridge_case_enzyme/activation_4layer (SENS_DIRECT / SENS_FASTJAC).
// ============================================================
class AoExactReducedSolver : public mfem::Solver {
    const TieMap &tie;
    std::unique_ptr<mfem::SparseMatrix> P, Pt;
    ParSpGEMM spgemm;
    // AO_PARD_MTYPE: 11 = real nonsymmetric (default), 1 = real structurally symmetric (the element pattern is symmetric).
    static MKL_INT PardMtype() { const char *e = std::getenv("AO_PARD_MTYPE"); return e ? (MKL_INT)std::atoi(e) : 11; }
    mutable sensei::PardisoDirectSolver pard{PardMtype()};
    std::vector<double> D;
    std::unique_ptr<mfem::SparseMatrix> Jr;
    mutable std::vector<mfem::Vector> bf, xf;
public:
    double t_asm = 0, t_rap = 0, t_fact = 0, t_solve = 0;
    long n_fact = 0, n_rhs = 0;
    explicit AoExactReducedSolver(const TieMap &t)
        : mfem::Solver(t.ndof_free), tie(t), P(t.BuildProlongation()), Pt(mfem::Transpose(*P)) {}
    void SetOperator(const mfem::Operator &) override {}
    void Factor(const CorrectedJacobianOp &Jc) {
        auto t0 = std::chrono::steady_clock::now();
        Jc.AssembleExact(D);
        const mfem::SparseMatrix &pat = Jc.Pattern();
        mfem::SparseMatrix Jf(const_cast<int*>(pat.GetI()), const_cast<int*>(pat.GetJ()), D.data(),
                              pat.Height(), pat.Width(), false, false, false);
        auto t1 = std::chrono::steady_clock::now();
        std::unique_ptr<mfem::SparseMatrix> AP(spgemm.mult(Jf, *P));
        Jr.reset(spgemm.mult(*Pt, *AP));
        auto t2 = std::chrono::steady_clock::now();
        pard.Refactorize(*Jr);
        auto t3 = std::chrono::steady_clock::now();
        t_asm += std::chrono::duration<double>(t1 - t0).count();
        t_rap += std::chrono::duration<double>(t2 - t1).count();
        t_fact += std::chrono::duration<double>(t3 - t2).count();
        ++n_fact;
    }
    long PerturbedPivots() const { return (long)pard.PerturbedPivots(); }
    const mfem::SparseMatrix &ReducedMatrix() const { return *Jr; }
    // reduced-space preconditioner / solve (free dofs)
    void Mult(const mfem::Vector &b, mfem::Vector &x) const override {
        auto t0 = std::chrono::steady_clock::now();
        pard.Solve(b, x);
        const_cast<AoExactReducedSolver*>(this)->t_solve += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    // full-space right-hand sides -> full-space solutions, one batched PARDISO call
    void SolveFull(const std::vector<const mfem::Vector*> &B, std::vector<mfem::Vector*> &X) {
        auto t0 = std::chrono::steady_clock::now();
        const int m = (int)B.size();
        if ((int)bf.size() < m) { bf.resize(m); xf.resize(m); }
        std::vector<const mfem::Vector*> Bp; std::vector<mfem::Vector*> Xp;
        for (int k = 0; k < m; k++) {
            bf[k].SetSize(tie.ndof_free); xf[k].SetSize(tie.ndof_free);
            tie.Contract(*B[k], bf[k]); Bp.push_back(&bf[k]); Xp.push_back(&xf[k]);
        }
        // Measured (553k mesh, 16 threads): PARDISO's multi-RHS phase 33 for mtype 11 costs ~0.2 s per RHS vs ~0.03 s
        // for single-RHS calls, so the default loops over single solves. AO_BATCH=1 uses the batched call.
        static const bool batch = std::getenv("AO_BATCH") && std::string(std::getenv("AO_BATCH")) == "1";
        if (batch) pard.SolveMulti(Bp, Xp);
        else for (int k = 0; k < m; k++) pard.Solve(bf[k], xf[k]);
        for (int k = 0; k < m; k++) tie.Expand(xf[k], *X[k]);
        t_solve += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        n_rhs += m;
    }
};

#ifdef USE_HYPRE_AMG
// ============================================================
//  v5 MIXED SOLVER: BoomerAMG-preconditioned CG for the NEWTON linear solves (large systems only), PARDISO kept for the
//  sensitivity solve (its exact factorisation is reused for all 16 RHS; AMG there costs ~7x more GMRES iterations).
//  hypre C API, sequential + OpenMP build; smoother l1-Jacobi (relax 18) -- the default hybrid GS is INDEFINITE on the
//  tie-reduced 2x-refined system (measured). The AMG hierarchy is REUSED across calls and rebuilt only when CG needs
//  more than AMG_REUSE_FACTOR x the iterations it needed right after the last setup, after AMG_MAX_AGE calls, or when
//  CG fails; if CG still fails on a fresh hierarchy the call falls back to PARDISO. The CG operator is always the
//  current matrix, so reuse only affects the preconditioner, never the converged answer (up to NEWTON_AMG_TOL).
// ============================================================
static HYPRE_Int amg_noop_setup(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector) { return 0; }
struct AmgStats { long calls = 0, setups = 0, iters = 0, fallbacks = 0; double t_build = 0, t_setup = 0, t_solve = 0; void reset() { *this = AmgStats(); } };
static AmgStats g_amg;
class HypreAMGReducedSolver : public mfem::Solver {
    const TieMap &tie;
    mfem::SparseMatrix *P;
    mutable const JacobianOp *jac = nullptr;
    mutable const mfem::Operator *op_ = nullptr;
    mutable bool need_matrix = true, need_setup = true;
    mutable ParSpGEMM spgemm;
    mutable HYPRE_IJMatrix Aij_cur = nullptr, Aij_h = nullptr;   // current matrix (CG operator) / matrix the hierarchy was built on
    mutable HYPRE_ParCSRMatrix A_cur = nullptr, A_h = nullptr;
    mutable HYPRE_IJVector bij = nullptr, xij = nullptr; mutable HYPRE_ParVector bv = nullptr, xv = nullptr;
    mutable HYPRE_Solver amg = nullptr, pcg = nullptr;
    mutable std::vector<HYPRE_Int> rows;
    mutable int baseline_its = 0, age = 0;
    mutable std::unique_ptr<PardisoReducedSolver> fallback;
    const double tol, reuse_factor; const int max_age;
    void destroy_matrix(HYPRE_IJMatrix &m) const { if (m) { HYPRE_IJMatrixDestroy(m); m = nullptr; } }
    void destroy_solvers() const {
        if (pcg) { HYPRE_ParCSRPCGDestroy(pcg); pcg = nullptr; }
        if (amg) { HYPRE_BoomerAMGDestroy(amg); amg = nullptr; }
    }
    void build_matrix() const {
        const double t0 = now_ms();
        const HYPRE_Int n = (HYPRE_Int)Height();
        std::unique_ptr<mfem::SparseMatrix> Jf(jac->AssembleFull());
        std::unique_ptr<mfem::SparseMatrix> Pt(mfem::Transpose(*P));
        std::unique_ptr<mfem::SparseMatrix> AP(spgemm.mult(*Jf, *P));
        std::unique_ptr<mfem::SparseMatrix> Jr(spgemm.mult(*Pt, *AP));
        const int *I = Jr->GetI(), *J = Jr->GetJ(); const double *D = Jr->GetData();
        std::vector<HYPRE_Int> nc(n, 0), cols; std::vector<double> vals; cols.reserve(Jr->NumNonZeroElems()); vals.reserve(Jr->NumNonZeroElems());
        if ((HYPRE_Int)rows.size() != n) { rows.resize(n); for (HYPRE_Int i = 0; i < n; i++) rows[i] = i; }
        for (HYPRE_Int i = 0; i < n; i++) for (int k = I[i]; k < I[i+1]; k++) if (J[k] == i || D[k] != 0.0) { cols.push_back(J[k]); vals.push_back(D[k]); ++nc[i]; }
        if (Aij_cur && Aij_cur != Aij_h) destroy_matrix(Aij_cur);   // never destroy the matrix the live hierarchy was built on
        Aij_cur = nullptr;
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, n - 1, 0, n - 1, &Aij_cur); HYPRE_IJMatrixSetObjectType(Aij_cur, HYPRE_PARCSR); HYPRE_IJMatrixInitialize(Aij_cur);
        HYPRE_IJMatrixSetValues(Aij_cur, n, nc.data(), rows.data(), cols.data(), vals.data()); HYPRE_IJMatrixAssemble(Aij_cur);
        HYPRE_IJMatrixGetObject(Aij_cur, (void **)&A_cur);
        if (!bij) {
            for (HYPRE_IJVector *v : {&bij, &xij}) { HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n - 1, v); HYPRE_IJVectorSetObjectType(*v, HYPRE_PARCSR); HYPRE_IJVectorInitialize(*v); HYPRE_IJVectorAssemble(*v); }
            HYPRE_IJVectorGetObject(bij, (void **)&bv); HYPRE_IJVectorGetObject(xij, (void **)&xv);
        }
        g_amg.t_build += now_ms() - t0;
    }
    void setup_hierarchy() const {
        const double t0 = now_ms();
        destroy_solvers();
        if (Aij_h && Aij_h != Aij_cur) destroy_matrix(Aij_h);
        Aij_h = Aij_cur; A_h = A_cur;
        HYPRE_BoomerAMGCreate(&amg); HYPRE_BoomerAMGSetPrintLevel(amg, 0); HYPRE_BoomerAMGSetCoarsenType(amg, 10); HYPRE_BoomerAMGSetInterpType(amg, 6);
        HYPRE_BoomerAMGSetPMaxElmts(amg, 4); HYPRE_BoomerAMGSetStrongThreshold(amg, 0.25); HYPRE_BoomerAMGSetRelaxType(amg, 18);
        HYPRE_BoomerAMGSetMaxIter(amg, 1); HYPRE_BoomerAMGSetTol(amg, 0.0);
        HYPRE_BoomerAMGSetup(amg, A_h, bv, xv);
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg); HYPRE_PCGSetTol(pcg, tol); HYPRE_PCGSetTwoNorm(pcg, 1); HYPRE_PCGSetMaxIter(pcg, 1000);
        HYPRE_PCGSetPrecond(pcg, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn)amg_noop_setup, amg);
        need_setup = false; age = 0; baseline_its = 0; ++g_amg.setups;
        g_amg.t_setup += now_ms() - t0;
    }
    bool pcg_solve(const mfem::Vector &b, int &its) const {
        const double t0 = now_ms(); const HYPRE_Int n = (HYPRE_Int)Height();
        HYPRE_IJVectorSetValues(bij, n, rows.data(), b.HostRead());
        std::vector<double> z(n, 0.0); HYPRE_IJVectorSetValues(xij, n, rows.data(), z.data());
        HYPRE_ParCSRPCGSetup(pcg, A_cur, bv, xv);          // binds the CURRENT matrix; preconditioner setup is a no-op
        HYPRE_ParCSRPCGSolve(pcg, A_cur, bv, xv);
        HYPRE_Int it, conv; HYPRE_PCGGetNumIterations(pcg, &it); HYPRE_PCGGetConverged(pcg, &conv);
        its = (int)it; g_amg.iters += it; g_amg.t_solve += now_ms() - t0;
        return conv != 0;
    }
public:
    HypreAMGReducedSolver(const TieMap &t, double tol_)
        : mfem::Solver(t.ndof_free), tie(t), P(t.BuildProlongation()), tol(tol_),
          reuse_factor(std::getenv("AMG_REUSE_FACTOR") ? atof(std::getenv("AMG_REUSE_FACTOR")) : 1.5),
          max_age(std::getenv("AMG_MAX_AGE") ? atoi(std::getenv("AMG_MAX_AGE")) : 20) {}
    ~HypreAMGReducedSolver() override {
        destroy_solvers();
        if (Aij_cur && Aij_cur != Aij_h) destroy_matrix(Aij_cur);
        destroy_matrix(Aij_h);
        if (bij) HYPRE_IJVectorDestroy(bij); if (xij) HYPRE_IJVectorDestroy(xij);
        delete P;
    }
    void SetOperator(const mfem::Operator &op) override {
        auto *rw = dynamic_cast<const ReducedOpWrapper*>(&op);
        MFEM_VERIFY(rw, "HypreAMGReducedSolver expects a ReducedOpWrapper");
        jac = dynamic_cast<const JacobianOp*>(&rw->full_op);
        MFEM_VERIFY(jac, "HypreAMGReducedSolver expects a JacobianOp inside the wrapper");
        op_ = &op; need_matrix = true;
    }
    void Mult(const mfem::Vector &b, mfem::Vector &x) const override {
        const double t0 = now_ms(); ++g_amg.calls;
        if (need_matrix) { build_matrix(); need_matrix = false; }
        if (need_setup || !amg || age >= max_age) setup_hierarchy();
        const bool fresh = (age == 0);
        int its = 0; bool ok = pcg_solve(b, its);
        if (!ok && !fresh) { setup_hierarchy(); ok = pcg_solve(b, its); }          // stale hierarchy: rebuild once
        if (ok) {
            const HYPRE_Int n = (HYPRE_Int)Height(); x.SetSize(n); HYPRE_IJVectorGetValues(xij, n, rows.data(), x.HostWrite());
            if (age == 0) baseline_its = its;
            else if (its > reuse_factor * std::max(baseline_its, 1)) need_setup = true;   // rebuild before the next call
            ++age;
        } else {                                                                    // fresh hierarchy failed too: PARDISO
            std::cout << "  [amg] CG+AMG did not converge (" << its << " its) -> PARDISO fallback for this call" << std::endl;
            if (!fallback) fallback.reset(new PardisoReducedSolver(tie));
            fallback->SetOperator(*op_); fallback->Mult(b, x); ++g_amg.fallbacks; need_setup = true;
        }
        g_tm.t_pard += now_ms() - t0;
    }
};
#endif

class DampedNewtonSolver : public NewtonSolver {
public:
    // OVER-RELAXATION line search (2026-07-14): the consistent boundary
    // Jacobian (see BdryQP/UpdateBoundary above) is exact for its own
    // term, but the conductivity term (kpa_dynamic/UpdateConductivity)
    // still uses a documented chord approximation missing a dk/dT term.
    // Diagnosed via a controlled debug run: CG solves each linear system
    // to ~1e-14 relative accuracy every time (not the bottleneck), and the
    // correction direction is always accepted at c_scale=1 (never
    // backtracks) -- yet the outer residual only shrinks ~2-3% per full
    // Newton step instead of quadratically, a textbook symptom of a
    // systematically UNDER-SCALED correction from an approximate
    // Jacobian (the true Jacobian is "stiffer" than the chord
    // approximation suggests). A plain 20-Newton-iteration budget only
    // reached ||r||=14.2 (need ~0.017); extrapolating the observed
    // geometric ratio implied ~350+ iterations needed -- impractical
    // across 43 timesteps. Fix: once c_scale=1 already improves the
    // residual, keep DOUBLING (over-relaxing) as long as it keeps
    // improving, instead of only ever backtracking from 1. This costs a
    // handful of extra residual evaluations (cheap Mult() calls) per
    // Newton iteration but converts the slow linear crawl into a much
    // faster process by directly exploiting the observed under-correction.
    real_t ComputeScalingFactor(const Vector &x, const Vector &b) const override {
        double norm0 = Norm(r); // r = F(x) - b at the current (pre-step) x
        Vector x_trial(x.Size()), r_trial(x.Size());
        // PERF #6: Picard-lag k(T) across the trial residual evaluations.
        struct FreezeGuard {
            FreezeGuard()  { g_freeze_conductivity = true;  }
            ~FreezeGuard() { g_freeze_conductivity = false; }
        } _fg;
        auto eval_norm = [&](double c_scale) {
            add(x, -c_scale, c, x_trial);
            oper->Mult(x_trial, r_trial);
            if (b.Size() == x.Size()) r_trial -= b;
            return Norm(r_trial);
        };
        double n1 = eval_norm(1.0);
        if (n1 >= norm0) {
            double c_scale = 1.0;
            for (int bt = 0; bt < 20; bt++) {
                c_scale *= 0.5;
                if (eval_norm(c_scale) < norm0) return c_scale;
            }
            return c_scale; // smallest tried; still better than nothing
        }
        double best_c = 1.0, best_n = n1, c_scale = 1.0;
        for (int fw = 0; fw < 14; fw++) {
            c_scale *= 2.0;
            double nt = eval_norm(c_scale);
            if (nt < best_n) { best_n = nt; best_c = c_scale; }
            else break;
        }
        return best_c;
    }
};

// RAII guard: Kokkos::initialize()/finalize() must bracket all Kokkos use.
// No-op when built without OTI_ENABLE_KOKKOS (matches reference driver's
// own KokkosGuard, OTI_driver_reference_bridge_local_oti_30param_kokkos.cpp).
struct KokkosGuard {
#ifdef OTI_ENABLE_KOKKOS
    KokkosGuard(int argc, char **argv) { Kokkos::initialize(argc, argv); }
    ~KokkosGuard() { Kokkos::finalize(); }
#else
    KokkosGuard(int, char **) {}
#endif
};

// A2 perf pass (2026-09-12): geometry cache for sensitivity block (a),
// matching the reference single-layer driver's own "dom_cache1" (see its
// comment above its block (a): "Geometry ... comes entirely from
// dom_cache1, built once before the time loop ... Only the per-step-
// varying pieces (gather Tn/prevder, evaluate k_oti, scatter-add) are
// redone here; no CalcDShape/SetIntPoint/InverseJacobian left in the hot
// loop at all."). Built ONCE over the WHOLE mesh (mesh geometry is static;
// element activation only changes which elements CONTRIBUTE, gated by
// attr_active[] inside the hot loop, exactly like ParConductivity's own
// pattern in full_bridge_activation_tier1.cpp). Because the hot loop then
// touches only plain arrays/Vectors (no MFEM FiniteElement/
// ElementTransformation calls), it can be OpenMP-parallelized without
// needing an MFEM_THREAD_SAFE build -- only this ONE-TIME, serial cache-
// building pass calls MFEM's own (non-thread-safe) geometry routines.
struct DomSensCache {
    int ne = 0;
    vector<Array<int>> vdofs;          // [e]
    vector<vector<Vector>> shape;      // [e][q]
    vector<vector<DenseMatrix>> gshape; // [e][q] physical (post-InverseJacobian) gradients
    vector<vector<double>> weight;      // [e][q]
    void Build(FiniteElementSpace &fespace, Mesh &mesh) {
        ne = mesh.GetNE();
        vdofs.resize(ne); shape.resize(ne); gshape.resize(ne); weight.resize(ne);
        for (int e = 0; e < ne; e++) {
            const FiniteElement &el = *fespace.GetFE(e);
            IsoparametricTransformation Trans;
            fespace.GetElementTransformation(e, &Trans);
            int dof = el.GetDof(), dm = el.GetDim();
            fespace.GetElementDofs(e, vdofs[e]);
            const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), 2*Trans.OrderGrad(&el));
            int nq = ir.GetNPoints();
            shape[e].resize(nq); gshape[e].resize(nq); weight[e].resize(nq);
            DenseMatrix dshape(dof, dm);
            for (int q = 0; q < nq; q++) {
                const IntegrationPoint &ip = ir.IntPoint(q);
                shape[e][q].SetSize(dof);
                el.CalcShape(ip, shape[e][q]);
                el.CalcDShape(ip, dshape);
                Trans.SetIntPoint(&ip);
                weight[e][q] = ip.weight * Trans.Weight();
                gshape[e][q].SetSize(dof, dm);
                Mult(dshape, Trans.InverseJacobian(), gshape[e][q]);
            }
        }
    }
};

int main(int argc, char *argv[])
{
    KokkosGuard kokkos_guard(argc, argv);
    Device device("cpu"); device.Print();

    // Full-bridge port (2026-09-12): defaults point at the production
    // 552,822-element mesh + full tie/toolpath assets (symlinked into
    // thermal_kokkos_oti/inputs/, same convention as thermal/inputs/) --
    // all three remain overridable via -m/-tie/-laserfile.
    const char *mesh_file = "full_bridge.mesh";
    const char *tie_file = "tie_weights_full.txt";
    const char *laser_file = "es_bridge_laser.inp";
    // Configurable (2026-07-15): was hardcoded, so two instances run from
    // the same directory at once silently corrupted each other's file via
    // concurrent writes (caught via a real collision between the
    // production accuracy run and a ParaView smoke test). Now defaults to
    // the same name for backward compatibility, but can be overridden.
    const char *probe_file = "probe_bridge.csv";
    // Full-bridge port: checkpoint/restart, ported from
    // full_bridge_activation_tier1.cpp (needed for a 63-layer, multi-day
    // build that a single job's time limit can't cover in one shot).
    const char *ckpt_file = "";      // checkpoint output path
    const char *restart_file = "";   // checkpoint to resume from
    int ckpt_every = 0;              // write checkpoint every N steps
    const char *lin_solver_kind = "pardiso";   // "pardiso" (Tier-1 direct) | "cg" (Jacobi-CG)
    double fixed_dt = 52.0;   // native bridge time units, matching thermal_4layer.odb's cadence
    double t_final  = 2200.0;
    int nseg = 200;
    double emissivity = 0.45, sigma_sb = 5.670373e-11, h_conv = 0.018;
    double loss_scale = 1.0;
    double substrate_loss_scale = 1.0, bridge_loss_scale = 1.0, interface_loss_scale = 1.0;
    bool abaqus_ffs_substrate = false;
    bool abaqus_time_grid = false;
    double first_time_point = 52.78;
    double t_start = 0.0;   // window start; IC (preheated plate) is valid for
                            // any t before the laser turns on (~1.25)
    double T_ambient_C = 40.0, T_substrate_bc_C = 80.0;
    int newt_max = 120;
    double newt_tol_rel = 1e-6, newt_tol_abs = 1e-3;
    int print_freq = 1;
    bool do_paraview = false;
    const char *pv_dir = "paraview_bridge";
    int pv_every = 1;
    int spread_mode = 0;
    double goldak_af = 0.3, goldak_ar = 0.6, goldak_b = 0.3, goldak_c = 0.3;
    double goldak_ff = 0.6, goldak_fr = 1.4;
    int goldak_subdiv = 3;
    double goldak_cutoff = 3.0;
    int derivative_order = 0;   // A2: 0=nominal only (byte-identical to Tier1), 1=first-order OTI sensitivities (NW=4: k,cp,rho,Lfus)
    const char *sens_csv = "sens_bridge.csv";
    // ---- v7 any-order query (see query.hpp) ----
    const char *deriv_query  = "";        // -deriv "k^2*cp ; k^4 ; P*Eta"
    const char *dir_query    = "";        // -dir "v1=k:1,cp:0.5@4"
    const char *probes_query = "row1";    // -probes row1,row2,sub
    bool raw_coeff = false, do_selfcheck = false;
    double gmres_tol = 1e-12;
    double kscale_cli = 1.0, cpscale_cli = 1.0;  // toy-problem FD validation (see g_kscale/g_cpscale comment)
    double rhoscale_cli = 1.0, Lfusscale_cli = 1.0; // melt-regime FD validation (see g_rhoscale/g_Lscale comment)
    double pscale_cli = 1.0, etascale_cli = 1.0; // P3 FD validation (see g_pscale/g_etascale comment)
    double xshift_cli = 0.0, yshift_cli = 0.0, zshift_cli = 0.0; // P5 FD validation (see g_xshift comment)
    double vxscale_cli = 0.0, tshift_cli = 0.0; // P5 Vx/tL FD validation (see g_vxscale comment)
    double probex_override = -1e30, probey_override = -1e30; // melt-regime test: override row-probe (x,y)
    double tsolidus_cli = IN625::T_solidus, tliquidus_cli = IN625::T_liquidus; // melt-regime test: override breakpoints
    // v8 (2026-10-02): newly-activated bridge rows default to ALREADY-CONSOLIDATED (bulk k/rho/hc),
    // matching every existing validated run -- the coarse dt=52 build-timeline model never resolves the
    // melt transient, so "already bulk at birth" is that model's own correct simplification, not a bug.
    // -powderstart makes newly-activated rows start as raw powder instead (kp/rho_p/hp until they
    // locally reach solidus for the first time): use ONLY for a deliberately melt-resolving run (fine
    // dt + refined mesh) that means to capture a row's own first-pass fusion event.
    bool start_powder_cli = false;
    // v8 (2026-10-02): the substrate is active EVERYWHERE from t=0 regardless of how many bridge rows
    // are active (confirmed: "Tie: ... 3607928 free DOFs" on the full refined mesh with only row1 born
    // -- essentially the whole mesh). A melt event from a tight, microsecond-scale source cannot
    // physically affect anything more than a few mm away within a few ms, so for a LOCAL melt-resolving
    // test it is safe and enormously cheaper to also pin every dof outside a small box around the probe
    // to ambient, regardless of its own row/substrate activation state. 0 (default) = off, every dof
    // behaves exactly as before. Centered on -probex/-probey (NOT the toolpath centroid, which isn't
    // computed yet at the point this is first used) -- pass both explicitly together with -actwin.
    double act_window_cli = 0.0;
    bool rhs_check = false;  // 2026-09-11 debug: FD-check the RHS[] itself directly against the residual formula, isolating RHS-building from the solve/Jacobian step

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
    args.AddOption(&tie_file, "-tie", "--tie-file", "Tie-weights file (full-bridge port; default tie_weights_full.txt).");
    args.AddOption(&laser_file, "-laserfile", "--laser-file", "Toolpath table file override, for Vx/tL synthetic-toolpath validation (default es_bridge_laser.inp).");
    args.AddOption(&fixed_dt, "-fdt", "--fixed-dt", "Fixed time step [native time units].");
    args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
    args.AddOption(&nseg, "-nseg", "--nseg", "Source sub-samples per raw toolpath segment.");
    args.AddOption(&h_conv, "-hconv", "--h-conv", "Film coefficient [mW/mm^2/K].");
    args.AddOption(&emissivity, "-emis", "--emissivity", "Surface emissivity.");
    args.AddOption(&sigma_sb, "-sigma", "--sigma-sb", "Stefan-Boltzmann constant [mW/mm^2/K^4].");
    args.AddOption(&loss_scale, "-lscale", "--loss-scale", "Scale all film and radiation losses for sensitivity tests.");
    args.AddOption(&T_ambient_C, "-tambc", "--t-ambient-c", "Ambient/initial temperature [C] (default 40) -- P2 dT/dT0 FD check.");
    args.AddOption(&substrate_loss_scale, "-subscale", "--substrate-loss-scale", "Additional scale for substrate exterior film/radiation.");
    args.AddOption(&bridge_loss_scale, "-bridgescale", "--bridge-loss-scale", "Additional scale for bridge exterior film/radiation.");
    args.AddOption(&interface_loss_scale, "-ifscale", "--interface-loss-scale", "Additional scale for active/inactive interface film/radiation.");
    args.AddOption(&abaqus_ffs_substrate, "-abqffs-sub", "--abaqus-ffs-substrate", "-no-abqffs-sub", "--no-abaqus-ffs-substrate",
                   "Apply Abaqus FFS-equivalent effective cooling on substrate exterior surfaces.");
    args.AddOption(&abaqus_time_grid, "-abqgrid", "--abaqus-time-grid", "-no-abqgrid", "--no-abaqus-time-grid",
                   "Clip time steps to Abaqus output time points after the initial 52-unit increment.");
    args.AddOption(&first_time_point, "-ftp", "--first-time-point", "First Abaqus output time point [native time units].");
    args.AddOption(&t_start, "-ts", "--t-start", "Window start time [native time units] (default 0).");
    args.AddOption(&ckpt_file, "-ckpt", "--checkpoint-file",
                   "Checkpoint path: written every -ckpt-every steps (and at exit).");
    args.AddOption(&ckpt_every, "-ckpt-every", "--checkpoint-every",
                   "Write the checkpoint every N accepted steps (0 = only at exit).");
    args.AddOption(&restart_file, "-restart", "--restart-file",
                   "Resume from this checkpoint (overrides -ts; probe/sens CSVs are appended).");
    args.AddOption(&print_freq, "-pf", "--print-freq", "Print every N steps.");
    args.AddOption(&do_paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                   "Write a ParaView time-series (temperature + element-activation fields).");
    args.AddOption(&pv_every, "-pv-every", "--paraview-every", "Write a ParaView frame every N steps (default 1).");
    args.AddOption(&pv_dir, "-pvd", "--paraview-dir", "ParaView output directory.");
    args.AddOption(&probe_file, "-pcsv", "--probe-csv", "Probe output CSV filename.");
    args.AddOption(&newt_tol_abs, "-ntola", "--newton-tol-abs", "Newton absolute residual tolerance.");
    args.AddOption(&newt_tol_rel, "-ntolr", "--newton-tol-rel", "Newton relative residual tolerance.");
    args.AddOption(&spread_mode, "-spread", "--spread-mode",
                   "Source spatial distribution: 0=shape-function-weighted (default), 1=uniform over containing element, 2=Goldak double-ellipsoidal.");
    args.AddOption(&newt_max, "-nmax", "--newton-max-iter", "Newton max iterations.");
    args.AddOption(&lin_solver_kind, "-solver", "--linear-solver", "Reduced linear solver: pardiso (Tier-1 direct) | cg (Jacobi-CG).");
    args.AddOption(&goldak_af, "-gaf", "--goldak-af", "Goldak front semi-axis [mm] (spread_mode=2).");
    args.AddOption(&goldak_ar, "-gar", "--goldak-ar", "Goldak rear semi-axis [mm] (spread_mode=2).");
    args.AddOption(&goldak_b, "-gb", "--goldak-b", "Goldak width semi-axis [mm] (spread_mode=2).");
    args.AddOption(&goldak_c, "-gc", "--goldak-c", "Goldak depth semi-axis [mm] (spread_mode=2).");
    args.AddOption(&goldak_ff, "-gff", "--goldak-ff", "Goldak front energy fraction (spread_mode=2, ff+fr=2).");
    args.AddOption(&goldak_fr, "-gfr", "--goldak-fr", "Goldak rear energy fraction (spread_mode=2, ff+fr=2).");
    args.AddOption(&goldak_subdiv, "-gsub", "--goldak-subdiv",
                   "Per-element Goldak quadrature subdivisions per direction (Abaqus deck uses 3).");
    args.AddOption(&goldak_cutoff, "-gcut", "--goldak-cutoff",
                   "Goldak numerical integration cutoff in semi-axis units.");
    args.AddOption(&derivative_order, "-dorder", "--derivative-order",
                   "A2/A3 OTI sensitivity order: 0 (default, nominal only) | 1 (first-order, all NW params) | 2 (also second-order diagonal Hessian d^2T/dp^2, no cross terms).");
    args.AddOption(&sens_csv, "-scsv", "--sens-csv", "Sensitivity probe output CSV filename (A2/A3, or the any-order output with -deriv/-dir).");
    args.AddOption(&deriv_query, "-deriv", "--deriv", "Any-order derivative terms, ';'-separated products of name[^power], e.g. \"k^2*cp ; k^4\". Names: k cp rho Lfus Ts Tliq hc T0 P Eta.");
    args.AddOption(&dir_query, "-dir", "--dir", "Any-order directional derivatives: name=param:weight,...@maxorder; e.g. \"v1=k:1,cp:0.5@4\".");
    args.AddOption(&probes_query, "-probes", "--probes", "Any-order output probes: row1..rowN, sub (default row1).");
    args.AddOption(&raw_coeff, "-rawcoeff", "--raw-coeff", "-no-rawcoeff", "--no-raw-coeff", "Any-order: output normalized Taylor coefficients instead of partial derivatives.");
    args.AddOption(&do_selfcheck, "-selfcheck", "--self-check", "-no-selfcheck", "--no-self-check", "Any-order: re-evaluate the residual after the solves and print max |R| per order.");
    args.AddOption(&gmres_tol, "-gmrestol", "--gmres-tol", "Any-order GMRES relative tolerance (default 1e-12).");
    args.AddOption(&T_substrate_bc_C, "-tsub", "--t-substrate-bc", "v8 FD check: substrate-bottom Dirichlet BC temperature [C] (default 80.0). Also the real Tw parameter's nominal value.");
    args.AddOption(&IN625::KP_FRACTION, "-kpfrac", "--kp-fraction", "FD check of kp: powder conductivity as a fraction of the bulk k(T) (default 0.2).");
    args.AddOption(&IN625::RHOP_FRACTION, "-rhopfrac", "--rhop-fraction", "FD check of rho_p: powder density as a fraction of the bulk density (default 0.6).");
    args.AddOption(&IN625::HP_OVER_HC, "-hpratio", "--hp-over-hc", "FD check of hp: part-cake film coefficient / h_conv (default 0.1).");
    args.AddOption(&kscale_cli, "-kscale", "--k-scale", "Toy-problem FD check: multiplicative scale on k(T) (default 1.0).");
    args.AddOption(&cpscale_cli, "-cpscale", "--cp-scale", "Toy-problem FD check: multiplicative scale on cp(T) (default 1.0).");
    args.AddOption(&rhoscale_cli, "-rhoscale", "--rho-scale", "Melt-regime FD check: multiplicative scale on rho (default 1.0).");
    args.AddOption(&Lfusscale_cli, "-lfusscale", "--lfus-scale", "Melt-regime FD check: multiplicative scale on L_fusion (default 1.0). NOT to be confused with -lscale/-ifscale (boundary loss scales).");
    args.AddOption(&pscale_cli, "-pscale", "--p-scale", "P3 FD check: multiplicative scale on laser power (default 1.0).");
    args.AddOption(&etascale_cli, "-etascale", "--eta-scale", "P3 FD check: multiplicative scale on absorption efficiency (default 1.0).");
    args.AddOption(&xshift_cli, "-xshift", "--x-shift", "P5 FD check: rigid x-shift of the toolpath [mm] (default 0.0) -- x0's own definition.");
    args.AddOption(&yshift_cli, "-yshift", "--y-shift", "P5 FD check: rigid y-shift of the toolpath [mm] (default 0.0) -- isolates BuildSourcePosGrad's dfdy.");
    args.AddOption(&zshift_cli, "-zshift", "--z-shift", "P5 FD check: rigid z-shift of the toolpath [mm] (default 0.0) -- isolates BuildSourcePosGrad's dfdz.");
    args.AddOption(&vxscale_cli, "-vxscale", "--vx-scale", "Debug only, NOT a valid Vx FD check on this toolpath (see g_vxscale comment): fractional toolpath speed scale (default 0.0).");
    args.AddOption(&tshift_cli, "-tshift", "--t-shift", "Debug only, NOT a valid tL FD check on this toolpath (see g_tshift comment): constant toolpath lead-time delay (default 0.0).");
    args.AddOption(&probex_override, "-probex", "--probe-x", "Override row-probe x (default: toolpath centroid).");
    args.AddOption(&probey_override, "-probey", "--probe-y", "Override row-probe y (default: toolpath centroid).");
    args.AddOption(&tsolidus_cli, "-tsolidus", "--t-solidus", "Melt-regime test: override T_solidus [K] (default 1563).");
    args.AddOption(&start_powder_cli, "-powderstart", "--powder-start", "-no-powderstart", "--no-powder-start",
                   "v8: newly-activated rows start as raw powder (kp/rho_p/hp) instead of already-bulk; default off (matches every pre-existing validated run).");
    args.AddOption(&act_window_cli, "-actwin", "--activation-window", "v8: half-width [mm] of a local box around -probex/-probey; dofs outside it are pinned to ambient regardless of row/substrate activation (default 0 = off, full mesh active as before). For a local melt-resolving test only.");
    args.AddOption(&tliquidus_cli, "-tliquidus", "--t-liquidus", "Melt-regime test: override T_liquidus [K] (default 1623).");
    args.AddOption(&rhs_check, "-rhscheck", "--rhs-check", "-no-rhscheck", "--no-rhs-check",
                   "Debug: FD-check RHS[] directly against the residual formula (isolates RHS-building from the solve).");
    bool jac_check = false;
    args.AddOption(&jac_check, "-jaccheck", "--jac-check", "-no-jaccheck", "--no-jac-check",
                   "Debug: FD-check J*delta directly against the residual formula (isolates the Jacobian operator).");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(cout); return 1; }
    MFEM_VERIFY(!(act_window_cli > 0.0 && probex_override < -1e29),
                "-actwin requires -probex/-probey to be set explicitly (it centers the window on them, not the toolpath centroid, which isn't computed yet at this point)");
#if OTI_ORDER_VAL < 2 && !defined(USE_ENZYME_AD)
    MFEM_VERIFY(derivative_order <= 1, "built with OTI_ORDER=1 (first-order only); use the order-2 build for -dorder 2");
#endif
    args.PrintOptions(cout);

    // ---- any-order query ----
    aoq::Query qry = aoq::parse_query(deriv_query, dir_query);
    if (!qry.ok()) { cerr << "ERROR: " << qry.error << endl; return 1; }
    const bool aoq_on = !qry.empty();
    unique_ptr<aoq::AnyOrderEval> ev;
    vector<unique_ptr<aoq::AnyOrderEval>> ev_lo;   // v7: truncated evaluators for the lower graded orders
    vector<double> Wpad;
    if (aoq_on) {
        ev = aoq::make_eval(qry.Mrun, qry.Nrun);
        if (!ev) {
            cerr << "ERROR: no compiled evaluator shape with M>=" << qry.Mrun << ", N>=" << qry.Nrun << ". Available:";
            for (auto &sh : aoq::registered_shapes()) cerr << " " << sh.first << "x" << sh.second;
            cerr << "\nRebuild with SHAPES=\"... " << qry.Mrun << "x" << qry.Nrun << "\" (see Makefile.anyorder)." << endl;
            return 1;
        }
        if (!ev->self_test()) { cerr << "ERROR: multi-index table self-test failed" << endl; return 1; }
        const int Msh = ev->M();
        Wpad.assign((size_t)aoq::NP * Msh, 0.0);
        for (int p = 0; p < aoq::NP; p++)
            for (int j = 0; j < qry.Mrun; j++) Wpad[(size_t)p * Msh + j] = qry.W[(size_t)p * qry.Mrun + j];
        cout << "  [ANYORDER] query: " << qry.Mrun << " basis column(s), max order " << qry.Nrun
             << " -> evaluator shape M=" << Msh << " N=" << ev->N() << " (" << ev->NC() << " coefficients)\n";
    }

    if (aoq_on) {
        MFEM_VERIFY(derivative_order == 0, "-deriv/-dir (any-order mode) replaces the A2/A3 path: run with -dorder 0");
        MFEM_VERIFY(!(restart_file && restart_file[0]), "any-order mode does not support -restart yet (Taylor history is not checkpointed)");
    }

    g_kscale = kscale_cli;
    g_cpscale = cpscale_cli;
    g_rhoscale = rhoscale_cli;
    g_Lscale = Lfusscale_cli;
    g_pscale = pscale_cli;
    g_etascale = etascale_cli;
    g_xshift = xshift_cli; g_yshift = yshift_cli; g_zshift = zshift_cli;
    g_vxscale = vxscale_cli; g_tshift = tshift_cli; g_tpivot = t_start;
    IN625::T_solidus = tsolidus_cli;
    IN625::T_liquidus = tliquidus_cli;

    if (aoq_on) {
        aoq::ParamSpec ps;
        for (int p = 0; p < aoq::NP; p++) ps.base[p] = 1.0;
        ps.base[aoq::P_K] = g_kscale; ps.base[aoq::P_CP] = g_cpscale;
        ps.base[aoq::P_RHO] = g_rhoscale; ps.base[aoq::P_LFUS] = g_Lscale;
        ps.W = Wpad;
        ev->set_params(ps);
        // v7 perf: graded order n only needs residual coefficients of order n, which depend only on inputs of order
        // <= n. Evaluate call n with the smallest compiled Jet<M,n'> (n' >= n, same M) whose flat coefficient layout is
        // a prefix of the full shape's (checked), reading/writing the same [c*ndof] buffers. AO_TRUNC=0 disables.
        const bool ao_trunc = !(std::getenv("AO_TRUNC") && std::string(std::getenv("AO_TRUNC")) == "0");
        ev_lo.resize(ev->N() + 1);
        for (int n = 1; ao_trunc && n <= ev->N(); n++) {
            auto sub = aoq::make_eval(ev->M(), n);
            if (!sub || sub->M() != ev->M() || sub->N() >= ev->N() || !sub->self_test()) continue;
            bool prefix = sub->order_offset(n + 1) == ev->order_offset(n + 1);
            for (int c = 0; prefix && c < sub->NC(); c++) prefix = (sub->alpha_of(c) == ev->alpha_of(c));
            if (!prefix) { cout << "  [ANYORDER] truncated shape " << sub->M() << "x" << sub->N() << " layout mismatch -> full shape for order " << n << "\n"; continue; }
            sub->set_params(ps);
            cout << "  [ANYORDER] order-" << n << " residual uses shape " << sub->M() << "x" << sub->N() << " (" << sub->NC() << " coefficients)\n";
            ev_lo[n] = std::move(sub);
        }
    }

    // A2: unit (1.0) material/boundary scale factors. Each only carries a
    // nonzero first-order seed on its own basis when derivative_order>=1 --
    // at derivative_order=0 these stay plain 1.0 with no seed, so
    // IN625::k_oti etc. are never even called and the nominal path is
    // untouched. P2 (NW=8): extends the same array/loop pattern to
    // S_TS/S_TLIQ/S_HC/S_T0, matching the reference driver's own seeding
    // loop exactly.
    IN625::S_K = OTI(1.0); IN625::S_CP = OTI(1.0); IN625::S_RHO = OTI(1.0); IN625::S_L = OTI(1.0);
    IN625::S_TS = OTI(1.0); IN625::S_TLIQ = OTI(1.0); IN625::S_HC = OTI(1.0); IN625::S_T0 = OTI(1.0);
    if (derivative_order >= 1) {
        // Fixed size 8, NOT NW: P3's P/eta (LI_P=8, LI_ETA=9) are
        // deliberately NOT in this array (see block (d)'s own comment --
        // they're pure multiplicative scales on f_src, fully decoupled
        // from k_oti/cp_oti/f_L_oti/the boundary term). Looping to NW here
        // (bug found 2026-09-12: SIGSEGV, S[8]/S[9] value-initialized to
        // nullptr then dereferenced) would try to seed nonexistent slots.
        OTI *S[8] = {&IN625::S_K, &IN625::S_CP, &IN625::S_RHO, &IN625::S_L,
                     &IN625::S_TS, &IN625::S_TLIQ, &IN625::S_HC, &IN625::S_T0};
        for (int p = 0; p < 8 && p < NW; p++) S[p]->set_e(WB(p), 1.0);
    }
    // v8 (2026-10-02): kp/rho_p/hp -- same unit-scale convention, non-contiguous basis indices (LI_KP=18,
    // LI_RHOP=19, LI_HP=17), seeded individually rather than via the fixed S[8] array above.
    IN625::S_KP = OTI(1.0); IN625::S_RHOP = OTI(1.0); IN625::S_HP = OTI(1.0);
    if (derivative_order >= 1 && NW > LI_W) {
        IN625::S_KP.set_e(WB(LI_KP), 1.0);
        IN625::S_RHOP.set_e(WB(LI_RHOP), 1.0);
        IN625::S_HP.set_e(WB(LI_HP), 1.0);
    }
    IN625::T_solidus_oti  = OTI(IN625::T_solidus)  * IN625::S_TS;
    IN625::T_liquidus_oti = OTI(IN625::T_liquidus) * IN625::S_TLIQ;

    if (abaqus_ffs_substrate) substrate_loss_scale *= 2.0;
    h_conv *= loss_scale;
    emissivity *= loss_scale;
    cout << "  Effective boundary loss scale=" << loss_scale
         << " h=" << h_conv << " emissivity=" << emissivity
         << " category scales: substrate=" << substrate_loss_scale
         << " bridge=" << bridge_loss_scale
         << " interface=" << interface_loss_scale << "\n";

    double T_ambient = T_ambient_C + C2K;
    double T_sub_bc  = T_substrate_bc_C + C2K;

    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    int max_attr = mesh.attributes.Max(); // 1=substrate, 2-5=bridge rows
    int n_rows = max_attr - 1;
    cout << "  Elements: " << mesh.GetNE() << "  Vertices: " << mesh.GetNV()
         << "  max attribute: " << max_attr << "\n";

    H1_FECollection fec(1, dim);
    FiniteElementSpace fespace(&mesh, &fec);
    int ndof = fespace.GetTrueVSize();
    cout << "  DOFs (full space): " << ndof << "\n";

    // ParaView visualization support (2026-07-15): the mesh itself never
    // changes -- element "activation" (birth) is purely a physics on/off
    // switch (see ActiveTDepKCoefficient etc.), not real mesh modification.
    // Left alone, that means every layer would be visible in ParaView from
    // frame 0 onward, just sitting at ambient temperature, which doesn't
    // look like a believable AM build video. Fix: export a second,
    // per-element piecewise-constant field ("Active", DG0/L2) alongside
    // temperature; in ParaView, a Threshold filter on Active>=1 hides
    // not-yet-deposited elements, so layers visually appear exactly when
    // they're born.
    L2_FECollection fec_l2(0, dim);
    FiniteElementSpace fespace_l2(&mesh, &fec_l2);
    GridFunction active_gf(&fespace_l2);

    TieMap tie;
    tie.Load(tie_file, ndof);
    // PERF: persistent Pardiso reduced solvers (symbolic analysis + RAP scratch reused across steps; Refactorize()
    // verifies the pattern and falls back to a full Factorize() when it changes, e.g. at activation events).
    // Declared after `tie` so they are destroyed before it. PARD_PERSIST_OFF=1 restores per-step construction.
    std::unique_ptr<PardisoReducedSolver> pardiso_persist, pardiso_sens_persist;
    const bool pard_persist_on = (getenv("PARD_PERSIST_OFF") == nullptr);
    // SENS_DIRECT (default ON in the Enzyme build; SENS_DIRECT=0 restores GMRES + stale PARDISO preconditioner): the exact corrected
    // sensitivity Jacobian is assembled and factorised ONCE per step (AoExactReducedSolver), then every first/second-order and
    // mixed right-hand side is one direct PARDISO solve -- no GMRES, ~8x cheaper once there are 200+ right-hand sides.
#ifdef USE_ENZYME_AD
    const bool sens_direct_on = !(getenv("SENS_DIRECT") && std::string(getenv("SENS_DIRECT")) == "0");
#else
    const bool sens_direct_on = false;
#endif
    cout << "  Tie: " << tie.n_slave << " slave DOFs, " << tie.ndof_free << " free DOFs\n";

    // FIX (2026-07-14): the substrate's top footprint (90x15mm) is much
    // LARGER than the bridge's own footprint (75x5mm, ~28% of the area) --
    // most of the substrate's own top surface is NOT under the bridge at
    // all and should be a perfectly normal exposed free surface (losing
    // heat via convection/radiation), not part of the tie. A blanket
    // "exclude every Z~0 face" rule in assemble_free_surface wrongly
    // excluded ~72% of the substrate's true exposed area from cooling,
    // letting heat build up well above the substrate's own fixed BC value
    // (found via a real, measured discrepancy: MFEM's substrate-top probe
    // ran at 370K vs Abaqus's 354.6K, EXCEEDING its own 353K boundary
    // condition -- physically implausible given the free surface's much
    // higher thermal resistance than the substrate's own conduction, and
    // traced directly to this over-exclusion). Only genuinely-tied
    // substrate vertices (masters referenced by some tie constraint)
    // should be excluded; the bridge's own bottom (all slave dofs) is
    // always excluded, unconditionally.
    vector<bool> is_tied_master(ndof, false);
    for (int s = 0; s < tie.n_slave; s++)
        for (int k = 0; k < 4; k++) is_tied_master[tie.masters[s][k]] = true;

    TableLaserParams laser;
    laser.Load(laser_file);
    cout << "  Laser toolpath: " << laser.t.size() << " waypoints, t_final=" << laser.t_final() << "\n";

    // Row birth times (CORRECTED 2026-07-14, via empirical extraction from
    // the reference odb -- see extract_full_field.py / bridge_zlevel_
    // activation.csv): each row activates at the START of its own build
    // cycle, not the END. Confirmed by the reference's own field data:
    // row-top Z levels first depart from ambient at t=52.78, 520.78,
    // 1040.78, 1560.78 -- exactly 520 time units apart (10 roller sweeps *
    // 52 units/sweep), starting at row1's cycle START (t~0), not its
    // completion (t=468, the original -- and wrong -- formula below).
    // Root cause: "Full" activation with zero volume-fraction thresholds
    // means the WHOLE row-element block activates as soon as the
    // deposition front makes ANY contact with its Z range, not once fully
    // swept through it.
    vector<double> birth_time(max_attr + 1);
    birth_time[1] = -1.0; // substrate: always active
    for (int r = 1; r <= n_rows; r++) birth_time[1+r] = 520.0 * (r - 1);
    cout << "  Row birth times: ";
    for (int r = 1; r <= n_rows; r++) cout << "row" << r << "=" << birth_time[1+r] << "  ";
    cout << "\n";

    vector<char> attr_active(max_attr + 1);
    auto set_active = [&](double t) {
        attr_active[1] = 1;
        for (int r = 1; r <= n_rows; r++) attr_active[1+r] = (t >= birth_time[1+r]) ? 1 : 0;
    };
    set_active(0.0);

    // ---- Dirichlet BC (substrate bottom, global Zmin) + activation-masking
    // essential dofs, combined into one ess_tdof_list (both are simple
    // "T=fixed value" constraints in FULL space) ----
    double zmin = 1e30;
    for (int v = 0; v < mesh.GetNV(); v++) zmin = min(zmin, mesh.GetVertex(v)[2]);
    Array<int> bc_dofs;
    for (int v = 0; v < mesh.GetNV(); v++)
        if (mesh.GetVertex(v)[2] < zmin + 1e-3) bc_dofs.Append(v);
    cout << "  Substrate-bottom Dirichlet BC nodes: " << bc_dofs.Size()
         << " (T=" << T_sub_bc << "K)\n";

    Array<int> ess_tdof_list;
    vector<bool> node_active(ndof);
    // v8 (2026-10-02): "has this node's temperature ever reached solidus" ratchet (monotonic, like
    // node_active's own activation-is-permanent property), driving the powder/bulk material-state
    // blend (kp/rho_p vs k/rho, hp vs hc) at LI_KP/LI_RHOP/LI_HP. Updated once per accepted step from
    // the converged u_iter, right after Newton convergence (see "ever_melted[i] = 1.0" below). Stored
    // as double (not bool) so it can be interpolated at quadrature points via ordinary shape functions,
    // giving a smooth (not discontinuous) melt-front blend, consistent with f_L's own smoothstep.
    vector<double> ever_melted(ndof, start_powder_cli ? 0.0 : 1.0);
    g_pk.ever_melted_ptr = ever_melted.data();   // safe: ever_melted outlives every g_pk.RefillInto call below
    g_ever_melted = &ever_melted;
    vector<bool> is_bc(ndof, false);
    for (int i = 0; i < bc_dofs.Size(); i++) is_bc[bc_dofs[i]] = true;
    // The tie only makes physical sense once row1 (which owns the slave/
    // BOTTOM_SURF dofs) actually exists -- before that, those dofs are
    // just ordinary not-yet-born nodes like anything else, and must be
    // excluded from tie treatment or they conflict with the plain
    // essential-dof (inactive->ambient) clamping (found via a real Newton
    // non-convergence at t=0: the full-space residual forced r[slave]=0
    // AND expected x[slave]=T_ambient via ess_tdof_list, while the tie's
    // Expand() simultaneously overwrote x[slave] with an interpolated
    // hot-substrate value -- two contradictory constraints on the same
    // DOFs). rebuild_ess_tdofs(tie_active) excludes slave dofs from
    // essential-clamping exactly when the tie is governing them instead.
    auto rebuild_ess_tdofs = [&](bool tie_active) {
        fill(node_active.begin(), node_active.end(), false);
        Array<int> vdofs, everts;
        for (int e = 0; e < mesh.GetNE(); e++) {
            if (!attr_active[mesh.GetAttribute(e)]) continue;
            if (act_window_cli > 0.0) {
                // v8: element centroid (x,y) must fall within the local activation window, in addition
                // to the usual attribute-based activation check above.
                mesh.GetElementVertices(e, everts);
                double cx = 0.0, cy = 0.0;
                for (int v = 0; v < everts.Size(); v++) {
                    const double *p = mesh.GetVertex(everts[v]);
                    cx += p[0]; cy += p[1];
                }
                cx /= everts.Size(); cy /= everts.Size();
                if (std::fabs(cx - probex_override) > act_window_cli || std::fabs(cy - probey_override) > act_window_cli)
                    continue;
            }
            fespace.GetElementDofs(e, vdofs);
            for (int d : vdofs) node_active[d] = true;
        }
        ess_tdof_list.SetSize(0);
        for (int i = 0; i < ndof; i++) {
            if (tie_active && tie.is_slave[i]) continue;
            if (!node_active[i] || is_bc[i]) ess_tdof_list.Append(i);
        }
    };
    bool tie_active = (attr_active[2] != 0); // row1's attribute is 2
    rebuild_ess_tdofs(tie_active);

    GridFunction u_gf(&fespace);
    u_gf = T_ambient; // bridge's own initial condition
    for (int e = 0; e < mesh.GetNE(); e++)
        if (mesh.GetAttribute(e) == 1) {
            Array<int> vdofs; fespace.GetElementDofs(e, vdofs);
            for (int d : vdofs) u_gf[d] = T_sub_bc; // substrate's own initial condition
        }
    for (int i = 0; i < ndof; i++) if (!node_active[i]) u_gf[i] = T_ambient;
    for (int i = 0; i < bc_dofs.Size(); i++) u_gf[bc_dofs[i]] = T_sub_bc;
    // v8: the substrate/build-platform is pre-existing solid, never freshly-deposited powder -- mark it
    // "consolidated" (bulk k/rho/hc) from t=0, mirroring the IC loop above exactly. Only bridge rows
    // (attr 2+, laser-deposited) start at ever_melted=0 (powder) and ratchet to 1 once locally melted.
    for (int e = 0; e < mesh.GetNE(); e++)
        if (mesh.GetAttribute(e) == 1) {
            Array<int> vdofs; fespace.GetElementDofs(e, vdofs);
            for (int d : vdofs) ever_melted[d] = 1.0;
        }

    // Probes: one per row's own top surface (z=0.2,0.4,0.6,0.8), near the
    // toolpath centroid -- matches the layer1_top..layer4_top convention
    // used everywhere else in this project for MFEM-vs-Abaqus comparisons.
    double px=0, py=0, cnt=0;
    for (size_t i = 0; i < laser.x.size(); i++) { px+=laser.x[i]; py+=laser.y[i]; cnt++; }
    px/=cnt; py/=cnt;
    // Melt-regime test override (2026-09-12): the toolpath-centroid probe
    // above sits far from where the laser actually IS at small t (its
    // first waypoint is near x=0-5, but the centroid over the WHOLE build
    // is ~x=35) -- a fine-dt melt test needs a probe actually under the
    // laser's early path, not wherever the build ends up centered.
    if (probex_override > -1e29) { px = probex_override; py = probey_override; }
    vector<int> probe_dof(n_rows, -1);
    for (int r = 1; r <= n_rows; r++) {
        // Full-bridge port (2026-09-10): snap the nominal layer-top z
        // (0.2*r slightly overshoots the real mesh top for the highest
        // rows -- 63 activation layers span the 12.5 mm build, not 12.6)
        // to the nearest actual mesh z-plane, so EVERY row gets a valid
        // probe dof instead of -1 -> 0.0.
        double row_top_z = 0.2 * r;
        double snap_z = row_top_z, snap_dz = 1e30;
        for (int v = 0; v < mesh.GetNV(); v++) {
            double dz = fabs(mesh.GetVertex(v)[2] - row_top_z);
            if (dz < snap_dz) { snap_dz = dz; snap_z = mesh.GetVertex(v)[2]; }
        }
        double best_d2 = 1e30;
        for (int v = 0; v < mesh.GetNV(); v++) {
            const double *c = mesh.GetVertex(v);
            if (fabs(c[2] - snap_z) > 1e-6) continue;
            double d = (c[0]-px)*(c[0]-px) + (c[1]-py)*(c[1]-py);
            if (d < best_d2) { best_d2 = d; probe_dof[r-1] = v; }
        }
        cout << "  Probe row" << r << "_top dof=" << probe_dof[r-1] << " at "
             << (probe_dof[r-1]>=0 ? mesh.GetVertex(probe_dof[r-1])[0] : 0) << ","
             << (probe_dof[r-1]>=0 ? mesh.GetVertex(probe_dof[r-1])[1] : 0)
             << "," << row_top_z << "\n";
    }

    // DIAGNOSTIC probe (2026-07-14): substrate's own top surface (Z~0,
    // a MASTER dof, not a slave), right at the tie interface, near the
    // same XY as the bridge probes. Unlike the bridge-row probes, this
    // location has NO direct dependence on the laser source model (only
    // on the fixed substrate BC + tie conduction + free-surface loss) --
    // isolates whether the observed MFEM-vs-Abaqus plateau gap originates
    // from the source deposition model or from the tie/BC/free-surface
    // treatment instead.
    int sub_probe_dof = -1; { double best_d2 = 1e30;
        for (int v = 0; v < mesh.GetNV(); v++) {
            const double *c = mesh.GetVertex(v);
            if (fabs(c[2]) > 1e-3) continue;
            double d = (c[0]-px)*(c[0]-px) + (c[1]-py)*(c[1]-py);
            if (d < best_d2) { best_d2 = d; sub_probe_dof = v; }
        }
    }
    cout << "  DIAGNOSTIC substrate-top probe dof=" << sub_probe_dof << " at "
         << (sub_probe_dof>=0 ? mesh.GetVertex(sub_probe_dof)[0] : 0) << ","
         << (sub_probe_dof>=0 ? mesh.GetVertex(sub_probe_dof)[1] : 0) << ",0\n";

    SpatialHashSource source(&mesh, &fespace, attr_active, nseg);
    source.spread_mode = spread_mode;
    source.goldak_af = goldak_af; source.goldak_ar = goldak_ar;
    source.goldak_b = goldak_b;   source.goldak_c = goldak_c;
    source.goldak_ff = goldak_ff; source.goldak_fr = goldak_fr;
    source.goldak_subdiv = goldak_subdiv;
    source.goldak_cutoff = goldak_cutoff;

    Vector V_node(ndof), M_cp_lump(ndof), M_L_diag(ndof);
    Vector L_now(ndof), L_n(ndof), Meff_over_dt(ndof);
    Vector f_src(ndof);
    Vector u_n(ndof), u_iter(ndof);
    Vector Kd_pa(ndof), Kb_diag(ndof);
    Vector A_bdry(ndof), Rbdry_vec(ndof);
    vector<BdryQP> bdry_qps;
    for (auto v : {&V_node,&M_cp_lump,&M_L_diag,&L_now,&L_n,&Meff_over_dt,
                   &f_src,&u_n,&u_iter,&Kd_pa,&Kb_diag,
                   &A_bdry,&Rbdry_vec}) v->UseDevice(true);

    Vector k_by_attr(max_attr);
    PWConstCoefficient active_coeff(k_by_attr);
    auto rebuild_V_node = [&]() {
        for (int i = 0; i < max_attr; i++) k_by_attr(i) = attr_active[i+1] ? 1.0 : 0.0;
        active_coeff.UpdateConstants(k_by_attr);
        BilinearForm mf(&fespace);
        mf.AddDomainIntegrator(new MassIntegrator(active_coeff));
        mf.Assemble();
        mf.SpMat().Finalize(0, true);
        SparseMatrix &Mm = mf.SpMat();
        double *p = V_node.HostWrite();
        for (int i = 0; i < ndof; i++) {
            const int *ci = Mm.GetI(); const double *cd = Mm.GetData();
            double s = 0; for (int j = ci[i]; j < ci[i+1]; j++) s += cd[j];
            p[i] = s;
        }
    };
    rebuild_V_node();

    auto eval_L_from_T = [&](Vector &Tvec, Vector &Lout){
        const double *Tp = Tvec.Read();
        const double *Vp = V_node.Read();
        double *Lp = Lout.Write();
        const double *Mp = ever_melted.data();   // v8: same powder/bulk density blend as UpdateCapacitance
        mfem::forall(ndof,[=] MFEM_HOST_DEVICE(int i){
            double Ti = Tp[i];
            double rho_i = Mp[i] * IN625::rho_dev(Ti) + (1.0 - Mp[i]) * IN625::rhop_dev(Ti);
            Lp[i] = Vp[i] * rho_i * (IN625::L_fusion * g_Lscale) * IN625::f_L_dev(Ti);
        });
    };

    // ---- Free-surface CONSISTENT (non-lumped) BOUNDARY MASS MATRIX, on
    // exterior boundary faces of currently-active elements AND on interior
    // faces straddling an active/inactive element pair (e.g. row1's own
    // top while row2 hasn't been deposited yet -- row1-row2 is a
    // topologically INTERIOR mesh face, since the bridge's own rows share
    // nodes internally and are split only by attribute, so it never
    // appears in mesh.GetNBE() regardless of activation state; found
    // missing ~375mm^2 of row1's own top area), EXCLUDING the tie
    // interface (Z~0, either the bridge's own bottom or substrate top
    // genuinely beneath the bridge's footprint -- the substrate's much
    // larger 90x15mm footprint vs. the bridge's 75x5mm means most of the
    // substrate's own Z~0 top is NOT under the bridge and must stay a
    // normal free surface, not part of the tie).
    //
    // A_bdry_out remains a LUMPED nodal area weight, but is now used only
    // as a rough preconditioner estimate (Kb_diag) -- the actual boundary
    // residual/Jacobian used by the Newton solve is the CONSISTENT
    // quadrature-point record list qps_out (BdryQP), built here alongside
    // A_bdry_out from the exact same face loops (same geometry, same
    // exclusions), consumed by FachinottiResidualOp::UpdateBoundary. See
    // the BdryQP comment above JacobianOp for why: Abaqus's own theory
    // manual documents the film/radiation Jacobian as a standard (non-
    // lumped) Galerkin surface integral; an earlier lumped-nodal version of
    // this term, and separately a flawed "nodal-eval-then-smear" consistent
    // attempt, were both tried and are documented in git history.
    auto assemble_free_surface_area = [&](Vector &A_bdry_out, vector<BdryQP> &qps_out) {
        A_bdry_out = 0.0;
        qps_out.clear();
        Vector shape;
        Array<int> vdofs, bverts;
        double diag_total_area = 0.0, diag_bridge_area = 0.0, diag_sub_area = 0.0;
        int diag_n_faces = 0, diag_n_skipped_inactive = 0, diag_n_skipped_tie = 0, diag_n_skipped_noFT = 0;
        for (int be = 0; be < mesh.GetNBE(); be++) {
            int elem, info;
            mesh.GetBdrElementAdjacentElement(be, elem, info);
            if (!attr_active[mesh.GetAttribute(elem)]) { diag_n_skipped_inactive++; continue; }
            mesh.GetBdrElementVertices(be, bverts);
            bool at_zzero = true;
            for (int k = 0; k < bverts.Size(); k++)
                if (fabs(mesh.GetVertex(bverts[k])[2]) > 1e-3) { at_zzero = false; break; }
            bool at_tie_interface = false;
            if (at_zzero) {
                if (mesh.GetAttribute(elem) != 1) {
                    at_tie_interface = true; // bridge's own bottom
                } else {
                    at_tie_interface = true;
                    for (int k = 0; k < bverts.Size(); k++)
                        if (!is_tied_master[bverts[k]]) { at_tie_interface = false; break; }
                }
            }
            if (at_tie_interface) { diag_n_skipped_tie++; continue; }
            FaceElementTransformations *FT = mesh.GetBdrFaceTransformations(be);
            if (!FT) { diag_n_skipped_noFT++; continue; }
            diag_n_faces++;
            {
                const IntegrationRule &ir0 = IntRules.Get(FT->GetGeometryType(), 4);
                double a = 0.0;
                for (int q = 0; q < ir0.GetNPoints(); q++) {
                    FT->SetAllIntPoints(&ir0.IntPoint(q));
                    a += ir0.IntPoint(q).weight * FT->Weight();
                }
                diag_total_area += a;
                if (mesh.GetAttribute(elem) == 1) diag_sub_area += a; else diag_bridge_area += a;
            }
            const FiniteElement &el = *fespace.GetFE(elem);
            shape.SetSize(el.GetDof());
            fespace.GetElementDofs(elem, vdofs);
            const IntegrationRule &ir = IntRules.Get(FT->GetGeometryType(), 2*el.GetOrder()+2);
            for (int q = 0; q < ir.GetNPoints(); q++) {
                const IntegrationPoint &ip = ir.IntPoint(q);
                FT->SetAllIntPoints(&ip);
                double w = ip.weight * FT->Weight();
                IntegrationPoint eip; FT->Loc1.Transform(ip, eip);
                el.CalcShape(eip, shape);
                BdryQP rec; rec.w = w; rec.kind = (mesh.GetAttribute(elem) == 1) ? 1 : 2;
                for (int a = 0; a < shape.Size(); a++) {
                    A_bdry_out[vdofs[a]] += shape(a) * shape(a) * w;
                    if (shape(a) != 0.0) { rec.dof.push_back(vdofs[a]); rec.shp.push_back(shape(a)); }
                }
                qps_out.push_back(std::move(rec));
            }
        }
        int diag_n_int_faces = 0; double diag_int_area = 0.0;
        for (int f = 0; f < mesh.GetNumFaces(); f++) {
            if (!mesh.FaceIsInterior(f)) continue;
            int el1, el2;
            mesh.GetFaceElements(f, &el1, &el2);
            if (el1 < 0 || el2 < 0) continue;
            bool active1 = attr_active[mesh.GetAttribute(el1)];
            bool active2 = attr_active[mesh.GetAttribute(el2)];
            if (active1 == active2) continue; // both active: true interior; both inactive: irrelevant
            int active_elem = active1 ? el1 : el2;

            FaceElementTransformations *FT = mesh.GetInteriorFaceTransformations(f);
            if (!FT) continue;
            bool use1 = (FT->Elem1No == active_elem);
            IntegrationPointTransformation &Loc = use1 ? FT->Loc1 : FT->Loc2;

            diag_n_int_faces++;
            {
                const IntegrationRule &ir0 = IntRules.Get(FT->GetGeometryType(), 4);
                double a = 0.0;
                for (int q = 0; q < ir0.GetNPoints(); q++) {
                    FT->SetAllIntPoints(&ir0.IntPoint(q));
                    a += ir0.IntPoint(q).weight * FT->Weight();
                }
                diag_int_area += a;
            }
            const FiniteElement &el = *fespace.GetFE(active_elem);
            shape.SetSize(el.GetDof());
            fespace.GetElementDofs(active_elem, vdofs);
            const IntegrationRule &ir = IntRules.Get(FT->GetGeometryType(), 2*el.GetOrder()+2);
            for (int q = 0; q < ir.GetNPoints(); q++) {
                const IntegrationPoint &ip = ir.IntPoint(q);
                FT->SetAllIntPoints(&ip);
                double w = ip.weight * FT->Weight();
                IntegrationPoint eip; Loc.Transform(ip, eip);
                el.CalcShape(eip, shape);
                BdryQP rec; rec.w = w; rec.kind = 3;
                for (int a = 0; a < shape.Size(); a++) {
                    A_bdry_out[vdofs[a]] += shape(a) * shape(a) * w;
                    if (shape(a) != 0.0) { rec.dof.push_back(vdofs[a]); rec.shp.push_back(shape(a)); }
                }
                qps_out.push_back(std::move(rec));
            }
        }
        cout << "  [diag free-surface] faces=" << diag_n_faces
             << " (skipped: inactive=" << diag_n_skipped_inactive
             << " tie=" << diag_n_skipped_tie << " noFT=" << diag_n_skipped_noFT << ")"
             << "  area: total=" << diag_total_area
             << " bridge=" << diag_bridge_area << " substrate=" << diag_sub_area << " mm^2"
             << "  [interior] faces=" << diag_n_int_faces << " area=" << diag_int_area << " mm^2"
             << "  [effective scaled area] substrate=" << diag_sub_area*substrate_loss_scale
             << " bridge=" << diag_bridge_area*bridge_loss_scale
             << " interface=" << diag_int_area*interface_loss_scale << " mm^2\n";
    };

    BilinearForm *k_pa_diff=nullptr;
    double last_Tmax_asm=-1e30;
    int n_asm=0, n_rejected=0, step=0, n_newt_tot=0;

    // A2: per-dof first-order sensitivities dT/d(k,cp,rho,Lfus,Ts,Tl,hc,T0),
    // propagated step to step. Full space (pre-tie-reduction); zero-
    // initialized, which is exactly correct at t=0 for every P1/P2 param
    // EXCEPT T0 (=T_ambient itself as a parameter -- the IC directly equals
    // it, so d(IC)/d(T0)=1 at every dof that starts at T_ambient, seeded
    // explicitly below) and at every later birth event for all params
    // (freshly-activated material starts at T_ambient, but T0's OWN
    // sensitivity there is already correctly propagated forward from t=0,
    // not re-seeded -- only the initial t=0 seed is special).
    Vector prev_der[NW], RHS[NW], dTds[NW];
    for (int p = 0; p < NW; p++) {
        prev_der[p].SetSize(ndof); prev_der[p] = 0.0; prev_der[p].UseDevice(true);
        RHS[p].SetSize(ndof); RHS[p].UseDevice(true);
        dTds[p].SetSize(ndof); dTds[p] = 0.0; dTds[p].UseDevice(true);
    }
    // A3 (2026-09-12): second-order (diagonal Hessian, d^2T/ds_p^2 only, no
    // cross terms d^2T/ds_p ds_q -- matches the reference driver's own
    // documented scope) sensitivity state, mirroring prev_der/RHS/dTds
    // exactly. prev_der2 default-initializes to 0.0 for every parameter
    // including T0: d(IC)/d(T0)=T_ambient is a LINEAR IC shift (see the
    // prev_der[LI_T0] seed below), so its second derivative is exactly 0
    // everywhere -- no special second-order IC seed needed, unlike first
    // order's own T0 seed.
    Vector prev_der2[NW], RHS2[NW], dT2ds[NW];
    for (int p = 0; p < NW; p++) {
        prev_der2[p].SetSize(ndof); prev_der2[p] = 0.0; prev_der2[p].UseDevice(true);
        RHS2[p].SetSize(ndof); RHS2[p].UseDevice(true);
        dT2ds[p].SetSize(ndof); dT2ds[p] = 0.0; dT2ds[p].UseDevice(true);
    }
    // SENS_MIXED: mixed second derivatives d2T/dp_i dp_j (i<j among the NACT wired parameters)
    std::vector<Vector> prev_der2m, RHSm, dT2m;
    const bool mixed_on_run = g_sens_mixed && derivative_order >= 2;
    // Blend-consistent second order (Enzyme local Hessians of the SAME powder-blended kernels the first order and the forward
    // model use).  The reference OTI blocks (a2)/(b2)/(c2) drop the powder blends (bulk k, bulk rho), so their history term
    // multiplies d2T_n by M_bulk/dt while the Jacobian carries M_powder/dt -> the second-order fields grow like (rho/rho_p)^n
    // = 1.67^n in powder (d2T/dcp2 reaches 1e27 within 140 steps).  FB_REFERENCE2=1 restores the reference blocks.
#ifdef USE_ENZYME_AD
    const bool use_hess = derivative_order >= 2 && !std::getenv("FB_REFERENCE2");
#else
    const bool use_hess = false;
#endif
    if (g_sens_mixed) {
        MFEM_VERIFY(derivative_order >= 2, "SENS_MIXED=1 needs -dorder 2");
#ifndef USE_ENZYME_AD
        MFEM_ABORT("SENS_MIXED=1 needs the Enzyme build (-DUSE_ENZYME_AD)");
#endif
        MFEM_VERIFY(!g_src_fd_legacy, "SENS_MIXED=1 needs the AD source path (unset SRC_FD_LEGACY)");
        init_pairs();
        prev_der2m.assign(NPAIR, Vector(ndof)); RHSm.assign(NPAIR, Vector(ndof)); dT2m.assign(NPAIR, Vector(ndof));
        for (int k = 0; k < NPAIR; k++) { prev_der2m[k] = 0.0; RHSm[k] = 0.0; dT2m[k] = 0.0; }
        cout << "  [SENS_MIXED] full Hessian: " << NACT << " pure + " << NPAIR << " mixed second-order fields"
             << (g_noblend2 ? " (FB_NOBLEND2: powder blends dropped at second order)" : "") << "\n";
    }
    // P2 T0 initial-condition seed (2026-09-12): T(x,0)=T_ambient*(1+s_T0)
    // ONLY at dofs that actually start at T_ambient. BUG FOUND (via
    // -rhscheck at the substrate probe: analytic RHS=1.37 vs direct-FD=0,
    // an outright wrong-dof-set error, not a magnitude issue): the FIRST
    // version of this seed applied T_ambient to every non-ess_tdof_list
    // dof, wrongly INCLUDING substrate dofs -- but substrate's own IC is
    // T_sub_bc (see the u_gf IC-setting loop above, "for attr==1 elements:
    // u_gf[d]=T_sub_bc"), entirely unrelated to T0/T_ambient, so those
    // dofs need a ZERO seed, not T_ambient. That wrong seed fed a spurious
    // nonzero dT/dT0 into block (c)'s nodal term at every substrate dof,
    // which then leaked into row1's own dTds[T0] too (~10% error there)
    // via the tie-coupled linear solve, even though row1 itself (a real
    // bridge dof) has the correct T_ambient IC. Fix: explicitly zero the
    // seed at substrate (attr==1) dofs, mirroring the IC-setting loop's
    // own attribute check exactly, THEN apply the ess_tdof_list zeroing
    // (not-yet-active/BC dofs) on top.
    if (NW > LI_T0) {
        double *pdT0 = prev_der[LI_T0].HostWrite();
        for (int i = 0; i < ndof; i++) pdT0[i] = T_ambient;
        for (int e = 0; e < mesh.GetNE(); e++)
            if (mesh.GetAttribute(e) == 1) {
                Array<int> vdofs; fespace.GetElementDofs(e, vdofs);
                for (int d : vdofs) pdT0[d] = 0.0;
            }
        for (int q = 0; q < ess_tdof_list.Size(); q++) pdT0[ess_tdof_list[q]] = 0.0;
    }
    // Tw initial-condition seed (bridge_20_param_single_enzyme, 2026-10-08).  The substrate starts at T_sub_bc (see the u_gf IC loop:
    // attr==1 elements), so T(x,0) there is Tw_nominal*(1+s_Tw), exactly like T0's IC seed but on the OTHER dof set.  The reference only
    // injects T_sub_bc at the Dirichlet rows each step, which misses the IC dependence (FD: dT/dTw = 9e-3 at the probe, reference AD:
    // 1e-107).  Not-yet-active / non-Dirichlet essential dofs stay unseeded.
    if (NW > LI_TW) {
        double *pdTw = prev_der[LI_TW].HostWrite();
        for (int i = 0; i < ndof; i++) pdTw[i] = 0.0;
        for (int e = 0; e < mesh.GetNE(); e++)
            if (mesh.GetAttribute(e) == 1) {
                Array<int> vdofs; fespace.GetElementDofs(e, vdofs);
                for (int d : vdofs) pdTw[d] = T_sub_bc;
            }
        std::vector<char> is_bc(ndof, 0);
        for (int i = 0; i < bc_dofs.Size(); i++) is_bc[bc_dofs[i]] = 1;
        for (int q = 0; q < ess_tdof_list.Size(); q++) if (!is_bc[ess_tdof_list[q]]) pdTw[ess_tdof_list[q]] = 0.0;
    }
    // =====================================================================
    // ANY-ORDER sensitivity state.  Fields are stored as NORMALIZED Taylor
    // coefficients in the evaluator's flat graded layout, coefficient-major
    // [c*ndof + i]; Hist holds the previous step's fields (c>=1), Tfield the
    // current step's.  Derivatives are w.r.t. the relative scale s_p of each
    // parameter (p -> p*(1+s_p)).
    // =====================================================================
    vector<double> Tfield, Hist, Rbuf;
    vector<int> cE;                       // flat index of order-1 coefficient of each basis column
    vector<char> inD;                     // downward closure of the requested multi-indices
    vector<int> req_c;                    // flat index of each requested multi-index
    int NmaxD = 0;
    vector<pair<string,int>> out_probes;  // (label, dof)
    ofstream aosens;   // any-order CSV (v6 keeps its own fsens for -dorder output)
    // Initial-condition seed: T(x,0) = T_ambient*(1+s_T0) at every dof that
    // starts at T_ambient (not substrate, whose IC is T_sub_bc).  The seed is
    // re-applied every step at NOT-YET-ACTIVE dofs, so newborn nodes (rows
    // 2-4) start from T_ambient*(1+s_T0) too (the order-2 driver zeroed it).
    auto seed_dof = [&](int i) {
        const int Msh = ev->M(), NCc = ev->NC();
        for (int c = 1; c < NCc; c++) Hist[(size_t)c * ndof + i] = 0.0;
        for (int j = 0; j < Msh; j++) Hist[(size_t)cE[j] * ndof + i] = T_ambient * Wpad[(size_t)aoq::P_T0 * Msh + j];
    };
    if (aoq_on) {
        const int NCc = ev->NC(), Msh = ev->M();
        Tfield.assign((size_t)NCc * ndof, 0.0);
        Hist.assign((size_t)NCc * ndof, 0.0);
        Rbuf.assign((size_t)NCc * ndof, 0.0);
        cE.resize(Msh);
        for (int j = 0; j < Msh; j++) { vector<int> a(Msh, 0); a[j] = 1; cE[j] = ev->index_of(a); }
        inD.assign(NCc, 0);
        for (auto &a : qry.alphas) {
            vector<int> ap(a); ap.resize(Msh, 0);
            int c = ev->index_of(ap);
            if (c < 0) { cerr << "ERROR: requested derivative exceeds evaluator order" << endl; return 1; }
            req_c.push_back(c);
            for (int c2 = 0; c2 < NCc; c2++) {
                vector<int> b = ev->alpha_of(c2);
                bool le = true;
                for (int j = 0; j < Msh; j++) if (b[j] > ap[j]) { le = false; break; }
                if (le) inD[c2] = 1;
            }
        }
        inD[0] = 0;
        int nsolve = 0;
        for (int c = 1; c < NCc; c++) if (inD[c]) { NmaxD = max(NmaxD, ev->order_of(c)); nsolve++; }
        cout << "  [ANYORDER] " << req_c.size() << " requested derivative(s); solving " << nsolve
             << " coefficient field(s) per step (downward closure), max order " << NmaxD << "\n";

        for (const string &tok : aoq::q_split(probes_query, ',')) {
            if (tok.empty()) continue;
            if (tok == "sub") out_probes.push_back({tok, sub_probe_dof});
            else if (tok.rfind("row", 0) == 0) {
                int r = atoi(tok.c_str() + 3);
                if (r < 1 || r > n_rows) { cerr << "ERROR: bad probe '" << tok << "'" << endl; return 1; }
                out_probes.push_back({tok, probe_dof[r - 1]});
            } else { cerr << "ERROR: bad probe '" << tok << "' (row1..row" << n_rows << ", sub)" << endl; return 1; }
        }

        // IC seed (order-2 driver's rule) + not-yet-active dofs (new).
        vector<char> is_sub(ndof, 0), ess0(ndof, 0);
        for (int e = 0; e < mesh.GetNE(); e++)
            if (mesh.GetAttribute(e) == 1) {
                Array<int> vdofs; fespace.GetElementDofs(e, vdofs);
                for (int d : vdofs) is_sub[d] = 1;
            }
        for (int q = 0; q < ess_tdof_list.Size(); q++) ess0[ess_tdof_list[q]] = 1;
        for (int i = 0; i < ndof; i++) if (!is_sub[i] && !ess0[i]) seed_dof(i);
        for (int i = 0; i < ndof; i++) if (!node_active[i] && !is_bc[i]) seed_dof(i);

        aosens.open(sens_csv);
        aosens << "time_native";
        for (auto &pr : out_probes) for (auto &nm : qry.alpha_name) aosens << "," << nm << "_" << pr.first;
        aosens << "\n";
        {
            ofstream meta(string(sens_csv) + ".meta");
            meta << "# any-order sensitivity output\n"
                 << "convention: derivatives w.r.t. relative scale s_p, parameter p -> base_p*(1+s_p); "
                 << (raw_coeff ? "values are NORMALIZED Taylor coefficients" : "values are partial derivatives (alpha! * coefficient)") << "\n"
                 << "evaluator_shape: M=" << Msh << " N=" << ev->N() << " coeffs=" << NCc << "\n"
                 << "deriv_query: " << deriv_query << "\ndir_query: " << dir_query << "\nprobes: " << probes_query << "\n"
                 << "basis_columns:";
            for (auto &n : qry.col_name) meta << " " << n;
            meta << "\nW (rows = params k cp rho Lfus Ts Tliq hc T0 P Eta, cols = basis columns):\n";
            for (int p = 0; p < aoq::NP; p++) {
                meta << "  " << aoq::PARAM_NAMES[p] << ":";
                for (int j = 0; j < qry.Mrun; j++) meta << " " << qry.W[(size_t)p * qry.Mrun + j];
                meta << "\n";
            }
            meta << "requested:";
            for (auto &n : qry.alpha_name) meta << " " << n;
            meta << "\nregularity: k,cp piecewise-linear (breakpoints e.g. 366 K), f_L quintic (d3f/dT3 jumps at Ts/Tliq): order>=3 across those is not smooth\n";
        }
    }
    // v7: persistent-factor reuse state for the any-order solves
    int ao_age = 0, ao_max_its = 0; bool ao_force = true, ao_refreshed = true;
    Solver *ao_pre = nullptr; const Operator *ao_rJ_base = nullptr;
    std::unique_ptr<AoExactReducedSolver> ao_ex;       // v7: AO_SOLVER=direct / exactpc
    double ao_t_res = 0, ao_t_solve = 0, ao_t_setup = 0;
    u_n = u_gf;
    eval_L_from_T(u_n, L_n);
    double t = t_start;

    // -------- checkpoint restore (full-bridge port, 2026-09-12) --------
    // Layout: int ndof, step, dorder; double t; then u_gf/u_n/L_n[ndof];
    // then, if dorder>=1, prev_der[0..NW-1][ndof]; if dorder>=2, also
    // prev_der2[0..NW-1][ndof]. EXTENDS full_bridge_activation_tier1.cpp's
    // own checkpoint format (which only ever had u_gf/u_n/L_n, since that
    // driver has no sensitivity state) -- required here: this driver's own
    // A2/A3 first/second-order sensitivities (prev_der/prev_der2) are
    // propagated step-to-step exactly like u_n/L_n, so a restart that only
    // restored the forward temperature state would silently reset every
    // sensitivity to 0 at the resume point and corrupt dT/ds for the rest
    // of the build -- a resume-time analogue of the OTI-safety concerns
    // called out elsewhere in this merge, so treated with the same care.
    // Done BEFORE the fsens/fprobe CSV setup below so both can open in
    // append mode (skip the header) when resuming.
    bool resumed = false;
    if (restart_file && restart_file[0]) {
        ifstream cf(restart_file, ios::binary);
        MFEM_VERIFY(cf, "cannot open restart file " << restart_file);
        int cn=0, cstep=0, cdorder=0; double ct=0.0;
        cf.read((char*)&cn,sizeof(int)); cf.read((char*)&cstep,sizeof(int));
        cf.read((char*)&cdorder,sizeof(int));
        cf.read((char*)&ct,sizeof(double));
        MFEM_VERIFY(cn == ndof, "restart ndof " << cn << " != mesh ndof " << ndof);
        MFEM_VERIFY(cdorder == derivative_order,
                    "restart checkpoint was written with -dorder " << cdorder
                    << " but this run requested -dorder " << derivative_order
                    << " -- cannot safely resume a sensitivity run at a different order.");
        cf.read((char*)u_gf.HostReadWrite(), (size_t)ndof*sizeof(double));
        cf.read((char*)u_n.HostReadWrite(),  (size_t)ndof*sizeof(double));
        cf.read((char*)L_n.HostReadWrite(),  (size_t)ndof*sizeof(double));
        if (cdorder >= 1)
            for (int p = 0; p < NW; p++)
                cf.read((char*)prev_der[p].HostReadWrite(), (size_t)ndof*sizeof(double));
        if (cdorder >= 2)
            for (int p = 0; p < NW; p++)
                cf.read((char*)prev_der2[p].HostReadWrite(), (size_t)ndof*sizeof(double));
        if (cdorder >= 2 && g_sens_mixed)
            for (int k = 0; k < NPAIR; k++)
                cf.read((char*)prev_der2m[k].HostReadWrite(), (size_t)ndof*sizeof(double));
        // history that the forward model carries besides T: powder->bulk flag per node and the laser's last travel direction
        // (bridge_20_param_single_enzyme fix 2026-10-09: without these a restarted run diverges from the continuous one)
        cf.read((char*)ever_melted.data(), (size_t)ndof*sizeof(double));
        cf.read((char*)source.last_dir.data(), 3*sizeof(double));
        t = ct; step = cstep; resumed = true;
        cout << "  RESUMED from " << restart_file << ": t=" << t
             << "  step=" << step << "  dorder=" << cdorder << "\n";
    }
    auto write_ckpt = [&](){
        if (!(ckpt_file && ckpt_file[0])) return;
        string tmp = string(ckpt_file) + ".tmp";
        ofstream cf(tmp, ios::binary);
        cf.write((char*)&ndof,sizeof(int)); cf.write((char*)&step,sizeof(int));
        cf.write((char*)&derivative_order,sizeof(int));
        cf.write((char*)&t,sizeof(double));
        cf.write((char*)u_gf.HostRead(), (size_t)ndof*sizeof(double));
        cf.write((char*)u_n.HostRead(),  (size_t)ndof*sizeof(double));
        cf.write((char*)L_n.HostRead(),  (size_t)ndof*sizeof(double));
        if (derivative_order >= 1)
            for (int p = 0; p < NW; p++)
                cf.write((char*)prev_der[p].HostRead(), (size_t)ndof*sizeof(double));
        if (derivative_order >= 2)
            for (int p = 0; p < NW; p++)
                cf.write((char*)prev_der2[p].HostRead(), (size_t)ndof*sizeof(double));
        if (derivative_order >= 2 && g_sens_mixed)
            for (int k = 0; k < NPAIR; k++)
                cf.write((char*)prev_der2m[k].HostRead(), (size_t)ndof*sizeof(double));
        cf.write((char*)ever_melted.data(), (size_t)ndof*sizeof(double));
        cf.write((char*)source.last_dir.data(), 3*sizeof(double));
        cf.close();
        rename(tmp.c_str(), ckpt_file);   // atomic
    };

    ofstream fsens, fhess;
    static const char *kParamName[NACT] = {"k","cp","rho","Lfus","Ts","Tliq","hc","T0","P","Eta","Rx","Ry","Rz","X0","Vx","TL","Tw","hp","kp","Rhop"};
    if (mixed_on_run) {   // full Hessian d2T/dp_i dp_j at the row-1 probe: all NACT*(NACT+1)/2 entries, i<=j
        fhess.open(string(sens_csv) + ".hess.csv", resumed ? (ios::out|ios::app) : ios::out);
        if (!resumed) {
            fhess << "time_native";
            for (int i = 0; i < NACT; i++) for (int j = i; j < NACT; j++) fhess << ",d2T_d" << kParamName[i] << "_d" << kParamName[j];
            fhess << "\n";
        }
        fhess << setprecision(15) << scientific;
    }
    if (derivative_order >= 1) {
        fsens.open(sens_csv, resumed ? (ios::out|ios::app) : ios::out);
        if (!resumed) {
        // dT_dk at EVERY row probe (to watch a newly-activated row's own
        // sensitivity turn on from its zero-initialized value right at its
        // birth event -- the untested "seed to 0 at birth" path); dT_dcp/
        // dT_drho/dT_dLfus at row1 only (phase-change/melt regime check,
        // not birth-specific).
        fsens << "time_native";
        for (int r = 0; r < n_rows; r++) fsens << ",dT_dk_row" << (r+1);
        fsens << ",dT_dcp_row1,dT_drho_row1,dT_dLfus_row1";
        if (NW > LI_T0) fsens << ",dT_dTs_row1,dT_dTliq_row1,dT_dhc_row1,dT_dT0_row1";
        if (NW > LI_ETA) fsens << ",dT_dP_row1,dT_dEta_row1";
        if (NW > LI_RX) fsens << ",dT_dRx_row1,dT_dRy_row1,dT_dRz_row1";
        if (NW > LI_X0) fsens << ",dT_dX0_row1,dT_dVx_row1,dT_dTL_row1";
        // 2026-10-02 (user request): 4 extra always-zero columns completing the paper's (Rincon-Tabares et al.
        // 2025, Table 1) 20 "non-structural" parameters. y0/Vy/lL/T0p are NOT wired on this bridge: the toolpath
        // is a precomputed row-based waypoint table (es_bridge_laser.inp) with motion only along x within a row
        // and discrete y-hops BETWEEN rows (no continuous y0/Vy degree of freedom to differentiate against), no
        // separate dwell-length state (lL), and a single ambient/IC temperature (no separate preheat T0p). These
        // are written as exact 0.0 every step, same as this codebase's other 30-param drivers document slots
        // 21-30 (powder-bed/build-platform geometry) as structurally zero -- NOT a placeholder bug, a documented
        // fact about this geometry's own toolpath representation. Making them real would need new sensitivity
        // code against a different toolpath parametrization, not a compile-time change.
        if (NW > LI_X0) fsens << ",dT_dY0_row1,dT_dVy_row1,dT_dlL_row1,dT_dT0p_row1";
        // v8 (2026-10-02): the paper's remaining 10 non-structural parameters. Tw is real (wired above);
        // hp/kp/rho_p/bp/lp/wp/b/l/w are not yet implemented (see the LI_* enum comment) -- their columns
        // report dTds directly (not a literal "0" like the 4 above), so they start reading exactly 0 now
        // and will read real values with no header/output change once each is wired.
        if (NW > LI_W) fsens << ",dT_dTw_row1,dT_dhp_row1,dT_dkp_row1,dT_dRhop_row1,"
                                 "dT_dBp_row1,dT_dLp_row1,dT_dWp_row1,dT_dB_row1,dT_dL_row1,dT_dW_row1";
        // A3 (2026-09-12): second-order (diagonal Hessian) columns, row1
        // only, same parameter order as the first-order block above.
        if (derivative_order >= 2) {
            for (int r = 0; r < n_rows; r++) fsens << ",d2T_dk2_row" << (r+1);
            fsens << ",d2T_dcp2_row1,d2T_drho2_row1,d2T_dLfus2_row1";
            if (NW > LI_T0) fsens << ",d2T_dTs2_row1,d2T_dTliq2_row1,d2T_dhc2_row1,d2T_dT02_row1";
            if (NW > LI_ETA) fsens << ",d2T_dP2_row1,d2T_dEta2_row1";
            if (NW > LI_RX) fsens << ",d2T_dRx2_row1,d2T_dRy2_row1,d2T_dRz2_row1";
            if (NW > LI_X0) fsens << ",d2T_dX02_row1,d2T_dVx2_row1,d2T_dTL2_row1";
            if (NW > LI_X0) fsens << ",d2T_dY02_row1,d2T_dVy2_row1,d2T_dlL2_row1,d2T_dT0p2_row1";
            if (NW > LI_W) fsens << ",d2T_dTw2_row1,d2T_dhp2_row1,d2T_dkp2_row1,d2T_dRhop2_row1,"
                                      "d2T_dBp2_row1,d2T_dLp2_row1,d2T_dWp2_row1,d2T_dB2_row1,d2T_dL2_row1,d2T_dW2_row1";
        }
        fsens << "\n";
        }
    }

    auto t_wall_start=chrono::high_resolution_clock::now();

    ofstream fprobe(probe_file, resumed ? (ios::out|ios::app) : ios::out);
    if (!resumed) {
        fprobe << "time_native";
        for (int r = 1; r <= n_rows; r++) fprobe << ",layer" << r << "_top";
        fprobe << ",T_max,substrate_top\n";
    }

    // ParaView time series (see the fespace_l2/active_gf setup above for why
    // a second field is needed): Active[e] = 1 if element e's row is
    // currently deposited, 0 otherwise. Updated and saved once per step
    // (plus one initial frame at t=0) so a ParaView Threshold on Active>=1
    // shows the part visually growing layer by layer as the animation plays.
    auto update_active_gf = [&]() {
        for (int e = 0; e < mesh.GetNE(); e++)
            active_gf[e] = attr_active[mesh.GetAttribute(e)] ? 1.0 : 0.0;
    };
    ParaViewDataCollection *pvdc = nullptr;
    if (do_paraview) {
        pvdc = new ParaViewDataCollection("bridge_multilayer", &mesh);
        pvdc->SetPrefixPath(pv_dir);
        pvdc->SetDataFormat(VTKFormat::BINARY32);
        pvdc->RegisterField("Temperature_K", &u_gf);
        pvdc->RegisterField("Active", &active_gf);
        if (!resumed) {
            update_active_gf();
            pvdc->SetCycle(0);
            pvdc->SetTime(0.0);
            pvdc->Save();
        }
        cout << "  ParaView: writing time series to " << pv_dir << "/\n";
    }

    DomSensCache dom_cache;
    dom_cache.Build(fespace, mesh);

    // Flat, MFEM-free copy of the domain geometry for the any-order evaluator.
    aoq::EvalGeom egeom;
    if (aoq_on) {
        egeom.ndof = ndof; egeom.ne = mesh.GetNE();
        egeom.attr.resize(egeom.ne); egeom.dim.resize(egeom.ne);
        egeom.e_dof_off.assign(egeom.ne + 1, 0); egeom.e_q_off.assign(egeom.ne + 1, 0);
        egeom.shape_off.assign(egeom.ne, 0); egeom.gshape_off.assign(egeom.ne, 0);
        int dofo = 0, qo = 0;
        for (int e = 0; e < egeom.ne; e++) {
            const Array<int> &vd = dom_cache.vdofs[e];
            int dof = vd.Size(), nq = (int)dom_cache.weight[e].size();
            int dm = dom_cache.gshape[e][0].Width();
            egeom.attr[e] = mesh.GetAttribute(e); egeom.dim[e] = dm;
            egeom.e_dof_off[e] = dofo; egeom.e_q_off[e] = qo;
            egeom.shape_off[e] = (long)egeom.shape.size();
            egeom.gshape_off[e] = (long)egeom.gshape.size();
            for (int a = 0; a < dof; a++) egeom.vdofs.push_back(vd[a]);
            for (int q = 0; q < nq; q++) {
                for (int a = 0; a < dof; a++) egeom.shape.push_back(dom_cache.shape[e][q](a));
                for (int a = 0; a < dof; a++)
                    for (int d = 0; d < dm; d++) egeom.gshape.push_back(dom_cache.gshape[e][q](a, d));
                egeom.weight.push_back(dom_cache.weight[e][q]);
            }
            dofo += dof; qo += nq;
        }
        egeom.e_dof_off[egeom.ne] = dofo; egeom.e_q_off[egeom.ne] = qo;
        ev->set_geometry(&egeom);
        for (auto &sub : ev_lo) if (sub) sub->set_geometry(&egeom);
    }

    cout << "\n=== Starting BRIDGE multilayer Fachinotti time integration ===\n\n";

    const double DT_mushy = IN625::T_liquidus - IN625::T_solidus;

    while (t < t_final - 1e-9) {
        auto _step_t0 = std::chrono::steady_clock::now();
        g_tm.reset(); g_tie_ms = 0.0;
        double dt_try = min(fixed_dt, t_final - t);
        if (abaqus_time_grid) {
            double next_tp = first_time_point;
            if (t >= first_time_point - 1e-9) {
                double k = ceil((t - first_time_point + 1e-9)/fixed_dt);
                next_tp = first_time_point + k*fixed_dt;
                if (next_tp <= t + 1e-9) next_tp += fixed_dt;
            }
            if (next_tp < t_final + 1e-9) dt_try = min(dt_try, next_tp - t);
        }
        // Make every row-birth time an exact increment boundary. The Abaqus
        // output grid is offset by 0.78, so this permits the newly active row
        // to heat over [birth,birth+0.78] before the matching output sample.
        for (int r = 1; r <= n_rows; r++) {
            double tbirth = birth_time[1+r];
            if (tbirth > t + 1e-9 && tbirth < t + dt_try - 1e-9)
                dt_try = tbirth - t;
        }
        step++;

        u_n = u_gf;
        eval_L_from_T(u_n, L_n);

        vector<char> prev_active = attr_active;
        // Use the active set that exists at the START of this increment.
        // A layer born exactly at t+dt is not physically present during the
        // preceding increment; it should first participate in the next step.
        set_active(t);
        bool activation_changed = (attr_active != prev_active);
        tie_active = (attr_active[2] != 0);
        if (activation_changed) {
            const vector<bool> was_active = node_active;
            rebuild_ess_tdofs(tie_active);
            // FIX 2026-09-24 (found by the FD check in bridge_case_enzyme/activation_4layer): newly-born
            // dofs enter this step at u_n = T_ambient = T_ambient_nominal*S_T0, so their dT_n/dT0 =
            // T_ambient, not 0 (every other parameter is correctly 0 there). Before the fix they entered
            // with 0: ~1.5% dT/dT0 error right after each birth, decaying over later steps.
            // Pre-fix source: <this file>.bak_pre_birthT0fix_20260924
            if (derivative_order >= 1 && NW > LI_T0) {
                double *pdT0 = prev_der[LI_T0].HostReadWrite();
                for (int i = 0; i < ndof; i++)
                    if (node_active[i] && !was_active[i] && !is_bc[i] && !(tie_active && tie.is_slave[i]))
                        pdT0[i] = T_ambient;
            }
            rebuild_V_node();
            for (int i = 0; i < ndof; i++) if (!node_active[i]) { u_gf[i] = T_ambient; u_n[i] = T_ambient; }
            for (int i = 0; i < bc_dofs.Size(); i++) { u_gf[bc_dofs[i]] = T_sub_bc; u_n[bc_dofs[i]] = T_sub_bc; }
            if (tie_active) {
                // Row1 just activated: seed slave dofs from the tie
                // interpolation immediately (otherwise they'd sit at
                // T_ambient, an inconsistent starting point for the
                // reduced-space solve's initial guess).
                Vector tmp_free(tie.ndof_free), tmp_full(ndof);
                for (int j = 0; j < tie.ndof_free; j++) tmp_free[j] = u_gf[tie.full_of_free[j]];
                tie.Expand(tmp_free, tmp_full);
                for (int s = 0; s < tie.n_slave; s++) { u_gf[tie.slave_of[s]] = tmp_full[tie.slave_of[s]]; u_n[tie.slave_of[s]] = tmp_full[tie.slave_of[s]]; }
            }
            cout << "  [row activation] t=" << t << "  tie_active=" << tie_active << "\n";
        }

        double Tmax_n = u_gf.Max();
        // Full-bridge port (2026-09-12), CAREFULLY NOT a verbatim copy of
        // full_bridge_activation_tier1.cpp's own "need_cg_precond && ..."
        // fix: that ancestor gates its ENTIRE do_asm block (including the
        // assemble_free_surface_area(...) call that builds bdry_qps --
        // the actual consistent boundary-radiation term FachinottiResidualOp
        // uses, via ANY solver) behind "lin_solver_kind != pardiso". Since
        // do_asm's own "|| true" makes it solver-kind-constant for the
        // whole run, that would mean bdry_qps is NEVER populated at all
        // when running pardiso (the default) -- silently zeroing the whole
        // film/radiation term for every step, not just skipping the
        // CG-only preconditioner data. Judgment call: keep the real
        // perf win (skip the O(552k-element) k_pa_diff BilinearForm
        // assembly, which truly only feeds Kd_pa/Kb_diag for the Jacobi-CG
        // preconditioner) but ALWAYS still call assemble_free_surface_area
        // (an O(surface elements) sweep, not O(volume elements)) so bdry_qps
        // stays correct for every solver, matching this file's own
        // validated physics.
        bool need_cg_precond = (string(lin_solver_kind) != "pardiso");
        bool do_asm = (k_pa_diff==nullptr) || activation_changed || true; // coarse mode: reassemble every step (few steps total)

        if (do_asm) {
            n_asm++;
            last_Tmax_asm = Tmax_n;
            u_gf.HostRead();
            assemble_free_surface_area(A_bdry, bdry_qps);   // always: builds the real bdry_qps physics
            if (need_cg_precond) {
                delete k_pa_diff;
                ActiveTDepKCoefficient k_coeff(u_gf, attr_active);
                k_pa_diff = new BilinearForm(&fespace);
                k_pa_diff->AddDomainIntegrator(new DiffusionIntegrator(k_coeff));
                k_pa_diff->Assemble();
                k_pa_diff->SpMat().Finalize(0, true);
                k_pa_diff->SpMat().GetDiag(Kd_pa);
                // Rough preconditioner estimate for Kb_diag: only feeds the
                // Jacobi-CG preconditioner (FachinottiLinearSolver /
                // ReducedLinearSolver), not the actual residual/Jacobian, so it
                // doesn't need to track the Newton iterate exactly -- just the
                // right order of magnitude, using the pre-step temperature.
                const double *up = u_gf.HostRead();
                const double *ab = A_bdry.HostRead();
                double *kb = Kb_diag.HostWrite();
                for (int i = 0; i < ndof; i++) {
                    double Ti = up[i];
                    kb[i] = ab[i] * (h_conv + 4.0*emissivity*sigma_sb*Ti*Ti*Ti);
                }
            }
        }

        if (getenv("LASTDIR_TRACE")) {
            printf("  [LASTDIR] pre-call last_dir=(%.6f,%.6f,%.6f) seam_skip=%ld seam_total=%ld goldak=(af=%.9f,ar=%.9f,b=%.9f,c=%.9f) ff=%.9f fr=%.9f cutoff=%.9f subdiv=%d spread=%d nseg=%d\n",
                   source.last_dir[0], source.last_dir[1], source.last_dir[2], source.seam_skip_count, source.seam_total_count,
                   source.goldak_af, source.goldak_ar, source.goldak_b, source.goldak_c,
                   source.goldak_ff, source.goldak_fr, source.goldak_cutoff, source.goldak_subdiv, source.spread_mode, nseg);
            double tend = t + dt_try;
            long long traw, draw, tendraw;
            memcpy(&traw, &t, 8); memcpy(&draw, &dt_try, 8); memcpy(&tendraw, &tend, 8);
            printf("  [HIBITS] t=%.17g (0x%016llx) dt_try=%.17g (0x%016llx) t_end=%.17g (0x%016llx) i0=%d i1=%d\n",
                   t, traw, dt_try, draw, tend, tendraw, laser.bracket(max(0.0,tend-dt_try)), laser.bracket(tend));
        }
        source.BuildSource(laser, t + dt_try, dt_try, f_src);
        f_src *= (g_pscale * g_etascale);   // P3 toy-FD: pure multiplicative scale
        for (int i = 0; i < ndof; i++) if (!node_active[i]) f_src[i] = 0.0;
        if (getenv("FSRC_TRACE")) {
            int fi = atoi(getenv("FSRC_TRACE"));
            printf("  [FSRC] i=%d t=%.9f dt=%.9e f_src=%.9e node_active=%d\n", fi, t, dt_try, f_src[fi], (int)node_active[fi]);
        }

        u_iter = u_n;

        FachinottiResidualOp residual(fespace, attr_active, bdry_qps, u_n, L_n, V_node,
                                      M_cp_lump, M_L_diag, L_now, Meff_over_dt,
                                      Rbdry_vec, f_src, dt_try,
                                      IN625::T_solidus, IN625::T_liquidus, DT_mushy,
                                      h_conv, emissivity, sigma_sb, T_ambient,
                                      substrate_loss_scale, bridge_loss_scale, interface_loss_scale,
                                      ess_tdof_list, ndof);
        DampedNewtonSolver newton;
        newton.iterative_mode = true;
        newton.SetRelTol(newt_tol_rel);
        newton.SetAbsTol(newt_tol_abs);
        newton.SetMaxIter(newt_max);
        newton.SetPrintLevel(-1);

        bool converged; int newt_iters_done;
        if (!tie_active) {
            // Row1 (and hence the tie) doesn't exist yet: plain full-space
            // solve, identical in structure to xtrack_multilayer.cpp.
            FachinottiLinearSolver lin_solver(Kd_pa, Kb_diag, Meff_over_dt, ess_tdof_list, ndof);
            newton.SetOperator(residual);
            newton.SetSolver(lin_solver);
            Vector zero;
            newton.Mult(zero, u_iter);
            for (int i = 0; i < ndof; i++) if (!node_active[i]) u_iter[i] = T_ambient;
            for (int i = 0; i < bc_dofs.Size(); i++) u_iter[bc_dofs[i]] = T_sub_bc;
        } else {
            // Row1 active: reduced (tie-eliminated) space.
            ReducedResidualOp reduced_residual(residual, tie);
            ReducedLinearSolver cg_solver(Kd_pa, Kb_diag, Meff_over_dt, tie);
            std::unique_ptr<PardisoReducedSolver> pardiso_local;
            if (pard_persist_on) { if (!pardiso_persist) pardiso_persist.reset(new PardisoReducedSolver(tie)); }
            else pardiso_local.reset(new PardisoReducedSolver(tie));
            PardisoReducedSolver &pardiso_solver = pard_persist_on ? *pardiso_persist : *pardiso_local;
            Solver *reduced_solver = (string(lin_solver_kind) == "pardiso")
                                   ? static_cast<Solver*>(&pardiso_solver)
                                   : static_cast<Solver*>(&cg_solver);
#ifdef USE_HYPRE_AMG
            // v5: NEWTON_LS = auto (default: AMG when the reduced system has >= AMG_MIN_DOFS unknowns) | amg | pardiso
            static const std::string newton_ls = std::getenv("NEWTON_LS") ? std::getenv("NEWTON_LS") : "auto";
            static const long amg_min_dofs = std::getenv("AMG_MIN_DOFS") ? atol(std::getenv("AMG_MIN_DOFS")) : 1500000;
            static const double newton_amg_tol = std::getenv("NEWTON_AMG_TOL") ? atof(std::getenv("NEWTON_AMG_TOL")) : 1e-10;
            static HypreAMGReducedSolver *newton_amg_solver = nullptr;
            const bool use_amg = (string(lin_solver_kind) == "pardiso") &&
                                 (newton_ls == "amg" || (newton_ls == "auto" && (long)tie.ndof_free >= amg_min_dofs));
            if (use_amg) {
                if (!newton_amg_solver) {
                    newton_amg_solver = new HypreAMGReducedSolver(tie, newton_amg_tol);
                    cout << "  [solver] Newton linear solves: CG+BoomerAMG (tol " << newton_amg_tol << ", " << tie.ndof_free
                         << " unknowns); sensitivity: PARDISO" << endl;
                }
                reduced_solver = newton_amg_solver;
            }
#endif
            newton.SetOperator(reduced_residual);
            newton.SetSolver(*reduced_solver);
            Vector u_iter_free(tie.ndof_free);
            for (int j = 0; j < tie.ndof_free; j++) u_iter_free[j] = u_iter[tie.full_of_free[j]];
            Vector zero;
            newton.Mult(zero, u_iter_free);
            tie.Expand(u_iter_free, u_iter);
            for (int i = 0; i < ndof; i++) if (!node_active[i]) u_iter[i] = T_ambient;
            for (int i = 0; i < bc_dofs.Size(); i++) u_iter[bc_dofs[i]] = T_sub_bc;
        }
        converged = newton.GetConverged();
        newt_iters_done = newton.GetNumIterations();
        n_newt_tot += newt_iters_done;
        residual.ReportAsmStats();
        cout << "  [newton diag] final_norm=" << newton.GetFinalNorm()
             << " init_norm=" << newton.GetInitialNorm() << "\n";

        if (!converged) {
            cerr << "ERROR: Newton failed to converge at t=" << t << ", dt=" << dt_try
                 << " after " << newt_iters_done << " iterations. Aborting.\n";
            return 1;
        }

        // v8 (2026-10-02): ratchet ever_melted from this step's accepted state. Monotonic (never reset),
        // same convention as node_active's own permanence -- once a point has reached solidus it stays
        // "consolidated" (bulk k/rho/hc) for the rest of the run, matching the paper's own USFLD
        // powder-to-bulk transition (one-way, triggered by the point's own thermal history).
        {
            const double *up = u_iter.HostRead();
            for (int i = 0; i < ndof; i++) if (up[i] >= IN625::T_solidus) ever_melted[i] = 1.0;
        }

        // =================================================================
        // A2: first-order OTI sensitivity pass (NW=4: k,cp,rho,Lfus).
        // Runs once per accepted step, after Newton convergence, on the SAME
        // converged Jacobian Newton just solved with (implicit-function-
        // theorem: J*dT/ds = -dR/ds|_{T_next fixed}). This driver's k(T) is
        // FULLY IMPLICIT (evaluated at the current Newton iterate, unlike
        // the single-layer reference driver's Picard-lagged k(Tn) -- see
        // FachinottiResidualOp::UpdateConductivity's own comment) -- so
        // T_next is held FIXED/UNSEEDED when building the RHS: its own
        // ds/dp dependence is exactly what the J-solve recovers on the LHS,
        // seeding it here would double-count. Only Tn (a genuinely already-
        // known, previous-step quantity) is seeded with prev_der. Boundary
        // (h_conv/emissivity/T_ambient) and source (laser) terms have zero
        // EXPLICIT dependence on k/cp/rho/Lfus at this NW, so only the
        // domain (a) and nodal (c) blocks are non-trivial -- confirmed
        // against the reference single-layer driver's own block structure.
        // =================================================================
        if (aoq_on) {
            auto t_aoq0 = std::chrono::steady_clock::now();
            const int NCc = ev->NC();
            Operator &J_full_sens = residual.GetGradient(u_iter);
            CorrectedJacobianOp J_corr_sens(fespace, mesh, attr_active, u_iter,
                                            residual.GetJbdryMat(), Meff_over_dt,
                                            ess_tdof_list, ndof);
            static const std::string ao_solver = std::getenv("AO_SOLVER") ? std::getenv("AO_SOLVER") : "direct";   // v7 default: direct (30-layer campaign)
            MFEM_VERIFY(ao_solver == "gmres" || ao_solver == "direct" || ao_solver == "exactpc", "AO_SOLVER must be gmres|direct|exactpc");
            const bool ao_exact = tie_active && ao_solver != "gmres" && J_corr_sens.IsFast();
            const bool ao_direct = ao_exact && ao_solver == "direct";
            if (ao_solver != "gmres" && !ao_exact) cout << "  [ANYORDER] AO_SOLVER=" << ao_solver << " unavailable here (no tie / no fast path) -> gmres\n";
            if (!ao_exact) residual.RebuildConductivityFreshForSensitivity(u_iter);   // feeds only the K-based preconditioner

            unique_ptr<FachinottiLinearSolver> ls_full;
            unique_ptr<ReducedOpWrapper> rJ_base, rJ_corr;
            unique_ptr<ReducedLinearSolver> cg_red;
            unique_ptr<PardisoReducedSolver> pardiso_red;
            unique_ptr<GMRESSolver> gm(new GMRESSolver);
            Vector rhs_free, sol_free;
            if (!tie_active) {
                ls_full.reset(new FachinottiLinearSolver(Kd_pa, Kb_diag, Meff_over_dt, ess_tdof_list, ndof));
                ls_full->SetOperator(J_full_sens);
                gm->SetOperator(J_corr_sens);
                gm->SetPreconditioner(*ls_full);
            } else {
                rJ_base.reset(new ReducedOpWrapper(J_full_sens, tie));
                rJ_corr.reset(new ReducedOpWrapper(J_corr_sens, tie));
                cg_red.reset(new ReducedLinearSolver(Kd_pa, Kb_diag, Meff_over_dt, tie));
                // v7: persistent sensitivity PARDISO factorisation (shared with v6's -dorder path), reused across steps;
                // refreshed at layer births, after AO_MAX_AGE reuses, when the previous step's GMRES needed more than
                // AO_REFAC_ITS iterations, or immediately if GMRES fails (see solve_full).
                if (!ao_exact && !pardiso_sens_persist) pardiso_sens_persist.reset(new PardisoReducedSolver(tie));
                if (ao_exact && !ao_ex) ao_ex.reset(new AoExactReducedSolver(tie));
                Solver *pre = ao_exact ? static_cast<Solver*>(ao_ex.get())
                            : (string(lin_solver_kind) == "pardiso")
                            ? static_cast<Solver*>(pardiso_sens_persist.get()) : static_cast<Solver*>(cg_red.get());
                static const int ao_max_age = std::getenv("AO_MAX_AGE") ? atoi(std::getenv("AO_MAX_AGE")) : 10;
                static const int ao_refac_its = std::getenv("AO_REFAC_ITS") ? atoi(std::getenv("AO_REFAC_ITS")) : 8;
                static const bool ao_reuse = !(std::getenv("AO_REUSE") && std::string(std::getenv("AO_REUSE")) == "0");
                ao_pre = pre; ao_rJ_base = rJ_base.get();
                ao_refreshed = ao_direct || !ao_reuse || ao_force || activation_changed || ao_age >= ao_max_age
                             || (!ao_exact && string(lin_solver_kind) != "pardiso");
                if (ao_refreshed) {
                    if (ao_exact) ao_ex->Factor(J_corr_sens); else pre->SetOperator(*rJ_base);
                    ao_age = 0; ao_force = false;
                } else ++ao_age;
                static bool ao_mv_checked = false;
                if (ao_exact && !ao_mv_checked) {
                    // one-time check: explicit exact matrix vs CorrectedJacobianOp::Mult on a random vector
                    ao_mv_checked = true;
                    std::vector<double> Dchk; J_corr_sens.AssembleExact(Dchk);
                    const SparseMatrix &pat = J_corr_sens.Pattern();
                    SparseMatrix Jchk(const_cast<int*>(pat.GetI()), const_cast<int*>(pat.GetJ()), Dchk.data(),
                                      pat.Height(), pat.Width(), false, false, false);
                    Vector xr(ndof), y1(ndof), y2(ndof); xr.Randomize(12345);
                    J_corr_sens.Mult(xr, y1); Jchk.Mult(xr, y2);
                    double dmax = 0, ymax = 0;
                    for (int i = 0; i < ndof; i++) { dmax = max(dmax, fabs(y1[i] - y2[i])); ymax = max(ymax, fabs(y1[i])); }
                    cout << "  [ANYORDER] exact-matrix matvec check vs CorrectedJacobianOp: max|dy|/max|y| = "
                         << dmax / max(ymax, 1e-300) << "  (nnz " << pat.NumNonZeroElems() << ", reduced nnz "
                         << ao_ex->ReducedMatrix().NumNonZeroElems() << ", perturbed pivots " << ao_ex->PerturbedPivots() << ")\n";
                }
                (void)ao_refac_its;
                gm->SetOperator(*rJ_corr);
                gm->SetPreconditioner(*pre);
                rhs_free.SetSize(tie.ndof_free); sol_free.SetSize(tie.ndof_free);
            }
            gm->SetRelTol(gmres_tol); gm->SetAbsTol(0.0);
            gm->SetMaxIter(300); gm->SetKDim(60); gm->SetPrintLevel(-1);

            int n_bad = 0; long gm_its = 0; int n_solves = 0; ao_max_its = 0;
            auto solve_full = [&](const Vector &rhs, Vector &sol) {
                if (!tie_active) {
                    sol = 0.0; gm->Mult(rhs, sol);
                } else {
                    tie.Contract(rhs, rhs_free);
                    sol_free = 0.0;
                    gm->Mult(rhs_free, sol_free);
                    tie.Expand(sol_free, sol);
                }
                if (!gm->GetConverged() && tie_active && !ao_refreshed) {      // stale factorisation failed: refresh + redo
                    if (ao_exact) ao_ex->Factor(J_corr_sens); else ao_pre->SetOperator(*ao_rJ_base);
                    ao_refreshed = true; ao_age = 0;
                    tie.Contract(rhs, rhs_free); sol_free = 0.0; gm->Mult(rhs_free, sol_free); tie.Expand(sol_free, sol);
                }
                if (!gm->GetConverged()) n_bad++;
                ao_max_its = std::max(ao_max_its, gm->GetNumIterations());
                gm_its += gm->GetNumIterations(); n_solves++;
            };

            // ---- inputs for the evaluator
            std::fill(Tfield.begin(), Tfield.end(), 0.0);
            {
                const double *up = u_iter.HostRead(), *unp = u_n.HostRead();
                for (int i = 0; i < ndof; i++) { Tfield[i] = up[i]; Hist[i] = unp[i]; }
            }
            aoq::StepInputs si;
            si.dt = dt_try; si.h_conv = h_conv; si.emis = emissivity; si.sigma = sigma_sb; si.T_ambient = T_ambient;
            si.T_solidus = IN625::T_solidus; si.T_liquidus = IN625::T_liquidus;
            si.T_solidus = IN625::T_solidus; si.T_liquidus = IN625::T_liquidus;
            si.qscale[0] = substrate_loss_scale; si.qscale[1] = bridge_loss_scale; si.qscale[2] = interface_loss_scale;
            vector<char> essmask(ndof, 0);
            for (int q = 0; q < ess_tdof_list.Size(); q++) essmask[ess_tdof_list[q]] = 1;
            si.attr_active = attr_active.data();
            si.V_node = V_node.HostRead(); si.f_src = f_src.HostRead(); si.ess = essmask.data();
            aoq::EvalBoundary bd;
            bd.off.push_back(0);
            for (const BdryQP &qp : bdry_qps) {
                for (size_t a = 0; a < qp.dof.size(); a++) { bd.dof.push_back(qp.dof[a]); bd.shp.push_back(qp.shp[a]); }
                bd.off.push_back((int)bd.dof.size()); bd.w.push_back(qp.w); bd.kind.push_back(qp.kind);
            }

            // ---- graded solve: order n forcing needs orders < n
            vector<double> rhs_max(NmaxD + 1, 0.0);
            Vector rhs(ndof), sol(ndof);
            const double t_setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_aoq0).count();
            double t_res_ms = 0.0, t_sol_ms = 0.0;
            static const bool ao_direct_check = (std::getenv("AO_DIRECT_CHECK") != nullptr);
            double ao_dcheck = 0.0;
            for (int n = 1; n <= NmaxD; n++) {
                auto tr0 = std::chrono::steady_clock::now();
                const aoq::AnyOrderEval *evn = (n < (int)ev_lo.size() && ev_lo[n]) ? ev_lo[n].get() : ev.get();
                evn->residual(si, bd, Tfield.data(), Hist.data(), Rbuf.data());
                auto tr1 = std::chrono::steady_clock::now();
                t_res_ms += std::chrono::duration<double, std::milli>(tr1 - tr0).count();
                std::vector<char> is_ess_row(ndof, 0);
                for (int q = 0; q < ess_tdof_list.Size(); q++) is_ess_row[ess_tdof_list[q]] = 1;
                if (ao_direct) {
                    // all requested fields of order n in one batched solve of the exact reduced system
                    std::vector<int> cs; std::vector<Vector> Bv, Xv;
                    for (int c = ev->order_offset(n); c < ev->order_offset(n + 1); c++) {
                        if (!inD[c]) continue;
                        const double *R = &Rbuf[(size_t)c * ndof];
                        double mx = 0.0;
                        for (int i = 0; i < ndof; i++) mx = max(mx, fabs(R[i]));
                        rhs_max[n] = max(rhs_max[n], mx);
                        if (mx == 0.0) continue;
                        cs.push_back(c);
                    }
                    Bv.assign(cs.size(), Vector(ndof)); Xv.assign(cs.size(), Vector(ndof));
                    std::vector<const Vector*> Bp; std::vector<Vector*> Xp;
                    for (size_t a = 0; a < cs.size(); a++) {
                        const double *R = &Rbuf[(size_t)cs[a] * ndof];
                        double *bp = Bv[a].HostWrite();
                        if (n == 1 && getenv("RHSDUMP_DIR")) {
                            static int ao_rdstep = 0;
                            std::ofstream of(std::string(getenv("RHSDUMP_DIR")) + "/aorhs_" + std::to_string(ao_rdstep++) + ".bin", std::ios::binary);
                            of.write(reinterpret_cast<const char*>(R), sizeof(double) * ndof);
                        }
                        for (int i = 0; i < ndof; i++) bp[i] = is_ess_row[i] ? 0.0 : -R[i];
                        Bp.push_back(&Bv[a]); Xp.push_back(&Xv[a]);
                    }
                    if (!cs.empty()) ao_ex->SolveFull(Bp, Xp);
                    n_solves += (int)cs.size();
                    for (size_t a = 0; a < cs.size(); a++) {
                        if (ao_direct_check) {       // true reduced residual |P^T(J x - b)| / |P^T b|
                            Vector y(ndof), yr(tie.ndof_free), br(tie.ndof_free);
                            J_corr_sens.Mult(Xv[a], y); y -= Bv[a];
                            tie.Contract(y, yr); tie.Contract(Bv[a], br);
                            ao_dcheck = max(ao_dcheck, yr.Normlinf() / max(br.Normlinf(), 1e-300));
                        }
                        const double *sp = Xv[a].HostRead();
                        double *dst = &Tfield[(size_t)cs[a] * ndof];
                        for (int i = 0; i < ndof; i++) dst[i] = sp[i];
                    }
                    t_sol_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr1).count();
                    continue;
                }
                for (int c = ev->order_offset(n); c < ev->order_offset(n + 1); c++) {
                    if (!inD[c]) continue;
                    double *rp = rhs.HostWrite();
                    const double *R = &Rbuf[(size_t)c * ndof];
                    double mx = 0.0;
                    for (int i = 0; i < ndof; i++) { rp[i] = is_ess_row[i] ? 0.0 : -R[i]; if (!is_ess_row[i]) mx = max(mx, fabs(R[i])); }
                    rhs_max[n] = max(rhs_max[n], mx);
                    if (mx == 0.0) continue;
                    solve_full(rhs, sol);
                    const double *sp = sol.HostRead();
                    double *dst = &Tfield[(size_t)c * ndof];
                    for (int i = 0; i < ndof; i++) dst[i] = sp[i];
                }
                t_sol_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr1).count();
            }
            if (do_selfcheck) {
                ev->residual(si, bd, Tfield.data(), Hist.data(), Rbuf.data());
                std::vector<char> is_ess_row(ndof, 0);
                for (int q = 0; q < ess_tdof_list.Size(); q++) is_ess_row[ess_tdof_list[q]] = 1;
                if (getenv("AO_LOCATE")) {
                    const double *R0 = &Rbuf[(size_t)ev->order_offset(0) * ndof];
                    Vector Tv(ndof), Rd(ndof);
                    for (int i = 0; i < ndof; i++) Tv(i) = Tfield[i];
                    residual.Mult(Tv, Rd);
                    double dmax = 0.0, dd = 0.0, dR = 0.0; int imax = -1;
                    for (int i = 0; i < ndof; i++) {
                        const double dif = fabs(Rd(i) - R0[i]);
                        if (dif > dmax) { dmax = dif; imax = i; }
                        dd = max(dd, fabs(Rd(i))); dR = max(dR, fabs(R0[i]));
                    }
                    cout << "  [LOCATE-DRV] max|R_driver|=" << dd << " max|R_ev|=" << dR << " max|R_drv-R_ev|=" << dmax << " at dof " << imax << "\n";
                    std::vector<std::pair<double,int>> top;
                    for (int i = 0; i < ndof; i++) top.push_back({fabs(R0[i]), i});
                    std::partial_sort(top.begin(), top.begin() + 8, top.end(), [](auto &x, auto &y){ return x.first > y.first; });
                    for (int q = 0; q < 8; q++) {
                        int i = top[q].second; const double *v = mesh.GetVertex(i);
                        cout << "  [LOCATE] dof " << i << " |R0|=" << top[q].first << " R0=" << R0[i] << " x=(" << v[0] << "," << v[1] << "," << v[2]
                             << ") ess=" << (int)is_ess_row[i] << " T=" << Tfield[i] << "\n";
                    }
                }
                for (int n = 0; n <= NmaxD; n++) {
                    // With the tie only P^T R (the contracted residual) vanishes.
                    double mx = 0.0;
                    Vector vfull(ndof), vfree(tie_active ? tie.ndof_free : 0);
                    for (int c = ev->order_offset(n); c < ev->order_offset(n + 1); c++)
                        if (n == 0 || inD[c]) {
                            double *vp = vfull.HostWrite();
                            for (int i = 0; i < ndof; i++) vp[i] = is_ess_row[i] ? 0.0 : Rbuf[(size_t)c * ndof + i];
                            if (tie_active) {
                                tie.Contract(vfull, vfree);
                                const double *fp = vfree.HostRead();
                                for (int i = 0; i < vfree.Size(); i++) mx = max(mx, fabs(fp[i]));
                            } else {
                                for (int i = 0; i < ndof; i++) mx = max(mx, fabs(vp[i]));
                            }
                        }
                    cout << "  [SELFCHECK] order " << n << ": max|R| after solve = " << mx
                         << (n > 0 ? "   (forcing max before = " : "") << (n > 0 ? to_string(rhs_max[n]) : string())
                         << (n > 0 ? ")" : "") << "\n";
                }
            }

            // ---- output
            aosens << scientific << setprecision(17) << (t + dt_try);
            for (auto &pr : out_probes)
                for (size_t a = 0; a < req_c.size(); a++) {
                    int c = req_c[a];
                    double v = (pr.second >= 0) ? Tfield[(size_t)c * ndof + pr.second] : 0.0;
                    if (!raw_coeff) v *= ev->fact(c);
                    aosens << "," << v;
                }
            aosens << "\n";
            aosens.flush();

            // ---- roll history; not-yet-active dofs restart from the IC seed
            if (const char *dd = getenv("DUMP_DIR")) {
                static int astep = 0;
                if (ev && ev->M() >= 1 && ev->N() >= 1) {
                    std::string fn = std::string(dd) + "/ao_" + std::to_string(astep++) + ".bin";
                    std::ofstream of(fn, std::ios::binary);
                    of.write(reinterpret_cast<const char*>(&Tfield[(size_t)ev->order_offset(1) * ndof]), sizeof(double) * ndof);
                }
            }
            for (size_t k = (size_t)ndof; k < Tfield.size(); k++) Hist[k] = Tfield[k];
            for (int i = 0; i < ndof; i++) if (!node_active[i]) seed_dof(i);
            { static const int rf = std::getenv("AO_REFAC_ITS") ? atoi(std::getenv("AO_REFAC_ITS")) : 8; if (ao_max_its > rf) ao_force = true; }
            ao_t_res += t_res_ms; ao_t_solve += t_sol_ms; ao_t_setup += t_setup_ms;
            cout << "  [ANYORDER] factor=" << (ao_refreshed ? "refreshed" : "reused") << " solves=" << n_solves << " gmres_its=" << gm_its
                 << (n_bad ? "  NOT-CONVERGED=" + to_string(n_bad) : string(""))
                 << "  setup=" << (long)t_setup_ms << "ms res=" << (long)t_res_ms << "ms solve=" << (long)t_sol_ms << "ms"
                 << [&]{ std::ostringstream o; o << std::scientific << std::setprecision(2);
                         if (ao_direct_check && ao_direct) o << "  direct_relres=" << ao_dcheck;
                         if (ao_exact) o << std::fixed << "  exact_cum[asm/rap/fact/solve]=" << ao_ex->t_asm << "/" << ao_ex->t_rap
                                         << "/" << ao_ex->t_fact << "/" << ao_ex->t_solve << "s nfact=" << ao_ex->n_fact
                                         << " pivpert=" << ao_ex->PerturbedPivots();
                         return o.str(); }()
                 << "  wall=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t_aoq0).count() << "ms\n" << flush;
            (void)NCc;
        }

        if (derivative_order >= 1) {
            tk_mark(0);
            for (int p = 0; p < NW; p++) RHS[p] = 0.0;
            // A3 (2026-09-12): d^2 f_src/ds_p^2 for rx/ry/rz, filled in
            // block (e) below (reusing its already-computed fp/fm/f0, no
            // extra BuildSource calls), consumed in block (e2) after the
            // first-order solve. x0/Vx/tL's own second derivative is left
            // at exactly 0 -- see block (e2)'s comment.
            Vector src2_fd_rx(ndof), src2_fd_ry(ndof), src2_fd_rz(ndof);
            src2_fd_rx = 0.0; src2_fd_ry = 0.0; src2_fd_rz = 0.0;
            Vector src2_x0, src2_vx, src2_tl;                   // AD source: pure second derivatives of x0, Vx, tL
            std::vector<Vector> srcD1keep, srcHkeep;            // SENS_MIXED: explicit df/dq (q=rx..tL) and d2f/dq_a dq_b (a<b)

            // ---- (a) domain diffusion: d[k(T_next)*T_next]/ds_k ----
            // (only k has any domain-diffusion dependence; cp/rho/Lfus don't
            // appear in the diffusion operator, so RHS_dom stays 0 for them)
            // OpenMP + geometry cache (2026-09-12, A2 perf pass): matches the
            // reference driver's own block (a) exactly -- see DomSensCache's
            // comment above. Geometry (vdofs/shape/gshape/weight) comes
            // entirely from dom_cache, built once before the time loop; only
            // the per-step-varying pieces (gather T_next, evaluate k_oti,
            // scatter-add) happen here, with NO MFEM FiniteElement/
            // ElementTransformation calls in the hot loop -- safe to thread
            // without an MFEM_THREAD_SAFE build.
            {
                double *rk = RHS[LI_K].HostReadWrite();
                double *rkp = (NW > LI_W) ? RHS[LI_KP].HostReadWrite() : nullptr;
                const double *up = u_iter.HostRead();
                const double *Mv = g_ever_melted ? g_ever_melted->data() : nullptr;
                const int NE_a = dom_cache.ne;
                #pragma omp parallel for schedule(dynamic, 64)
                for (int e = 0; e < NE_a; e++) {
                    if (!attr_active[mesh.GetAttribute(e)]) continue;
                    const Array<int> &vdofs = dom_cache.vdofs[e];
                    int dof = vdofs.Size();
                    Vector Tnext_loc(dof);
                    for (int a = 0; a < dof; a++) Tnext_loc(a) = up[vdofs[a]];
                    int nq = (int)dom_cache.weight[e].size();
                    for (int q = 0; q < nq; q++) {
                        const Vector &shape = dom_cache.shape[e][q];
                        const DenseMatrix &gshape_e = dom_cache.gshape[e][q];
                        double w = dom_cache.weight[e][q];
                        int dm = gshape_e.Width();
                        double Tnext_q = 0.0;
                        for (int a = 0; a < dof; a++) Tnext_q += shape(a) * Tnext_loc(a);
                        double gT[3] = {0,0,0};
                        for (int a = 0; a < dof; a++)
                            for (int d = 0; d < dm; d++) gT[d] += gshape_e(a,d) * Tnext_loc(a);
                        // v8: blend k_oti/kp_oti by the shape-interpolated melt fraction, exactly
                        // mirroring PersistentK::RefillInto's own blend (same quadrature rule here via
                        // dom_cache, so this stays consistent with the actual assembled K).
                        double mf = 1.0;
                        if (Mv) { mf = 0.0; for (int a = 0; a < dof; a++) mf += shape(a) * Mv[vdofs[a]]; }
#ifdef USE_ENZYME_AD
                        double dk_dk, dk_dkp;
                        fb_dk1(Tnext_q, mf, IN625::KP_FRACTION, &dk_dk, &dk_dkp);
                        if (!rkp) dk_dkp = 0.0;
#else
                        OTI k_q = OTI(mf) * IN625::k_oti(OTI(Tnext_q)) + OTI(1.0 - mf) * IN625::kp_oti(OTI(Tnext_q));
                        double dk_dk = k_q.get_e(WB(LI_K));
                        double dk_dkp = rkp ? k_q.get_e(WB(LI_KP)) : 0.0;
#endif
                        if (dk_dk == 0.0 && dk_dkp == 0.0) continue;
                        for (int a = 0; a < dof; a++) {
                            double gdot = 0.0;
                            for (int d = 0; d < dm; d++) gdot += gshape_e(a,d)*gT[d];
                            if (getenv("RHS_TRACE") && vdofs[a] == atoi(getenv("RHS_TRACE")) && dk_dk != 0.0) {
                                #pragma omp critical
                                cout << "  [RHS_TRACE] elem " << e << " attr " << mesh.GetAttribute(e) << " q " << q << " dk " << dk_dk
                                     << " gdot " << gdot << " w " << w << " contrib " << -dk_dk*gdot*w << " Tq " << Tnext_q << "\n";
                            }
                            if (dk_dk != 0.0) {
                                #pragma omp atomic
                                rk[vdofs[a]] -= dk_dk * gdot * w;
                            }
                            if (dk_dkp != 0.0) {
                                #pragma omp atomic
                                rkp[vdofs[a]] -= dk_dkp * gdot * w;
                            }
                        }
                    }
                }
            }

            tk_mark(1);
            // ---- (b) boundary: d[qval(T_next)]/ds_{hc,T0} ----
            // P2 (NW=8, 2026-09-12). Unlike the reference driver (Picard-
            // lagged k(Tn), so its own block (b) linearizes radiation at a
            // history-carrying Tn_oti), THIS driver's boundary term is
            // FULLY IMPLICIT -- UpdateBoundary's own qval is evaluated
            // entirely at the current iterate Tq, with no T_n dependence
            // at all. So T_next is held FIXED/UNSEEDED here too, exactly
            // matching block (a)'s own convention (see its comment): only
            // hc and T0 have explicit dependence in qval, so RHS_bnd stays
            // 0 for every other parameter. Driven directly off the
            // existing bdry_qps list (no separate geometry cache needed --
            // it's already a flat per-quadrature-point record).
            {
                double *rhc = RHS[LI_HC].HostReadWrite();
                double *rt0 = RHS[LI_T0].HostReadWrite();
                double *rhp = (NW > LI_W) ? RHS[LI_HP].HostReadWrite() : nullptr;
                const double *up = u_iter.HostRead();
                const double *Mv_c = g_ever_melted ? g_ever_melted->data() : nullptr;
                const int nqp = (int)bdry_qps.size();
                #pragma omp parallel for schedule(dynamic, 512)
                for (int qi = 0; qi < nqp; qi++) {
                    const BdryQP &qp = bdry_qps[qi];
                    double qscale = (qp.kind == 1) ? substrate_loss_scale
                                  : (qp.kind == 2) ? bridge_loss_scale
                                                    : interface_loss_scale;
                    int nd = (int)qp.dof.size();
                    double Tq = 0.0;
                    for (int a = 0; a < nd; a++) Tq += qp.shp[a] * up[qp.dof[a]];
#ifdef USE_ENZYME_AD
                    double mf_c = 1.0;
                    if (Mv_c) { mf_c = 0.0; for (int a = 0; a < nd; a++) mf_c += qp.shp[a] * Mv_c[qp.dof[a]]; }
                    double dq_dhc, dq_dt0, dq_dhp;
                    fb_dq1(Tq, mf_c, qscale * h_conv, qscale * h_conv * IN625::HP_OVER_HC,
                           qscale * emissivity * sigma_sb, T_ambient, &dq_dhc, &dq_dt0, &dq_dhp);
                    if (!rhp) dq_dhp = 0.0;
#else
                    OTI Tq_oti(Tq);   // UNSEEDED: held fixed, matches block (a)
                    // v8: same hc/hp melt-fraction blend as the real UpdateBoundary path above.
                    double mf_c = 1.0;
                    if (Mv_c) { mf_c = 0.0; for (int a = 0; a < nd; a++) mf_c += qp.shp[a] * Mv_c[qp.dof[a]]; }
                    OTI hconv_oti = OTI(mf_c) * (OTI(qscale * h_conv) * IN625::S_HC)
                                  + OTI(1.0 - mf_c) * (OTI(qscale * h_conv * IN625::HP_OVER_HC) * IN625::S_HP);
                    OTI Tinf_oti  = OTI(T_ambient) * IN625::S_T0;
                    OTI qval_oti = hconv_oti * (Tq_oti - Tinf_oti)
                                 + OTI(qscale * emissivity * sigma_sb)
                                   * (Tq_oti*Tq_oti*Tq_oti*Tq_oti - Tinf_oti*Tinf_oti*Tinf_oti*Tinf_oti);
                    double dq_dhc = qval_oti.get_e(WB(LI_HC));
                    double dq_dt0 = qval_oti.get_e(WB(LI_T0));
                    double dq_dhp = rhp ? qval_oti.get_e(WB(LI_HP)) : 0.0;
#endif
                    if (rhs_check) {
                        for (int a = 0; a < nd; a++) {
                            if (qp.dof[a] != sub_probe_dof) continue;
                            #pragma omp critical
                            std::cout << "  [BNDQ_SUBPROBE] kind=" << qp.kind << " qscale=" << qscale
                                      << " Tq=" << Tq << " shp=" << qp.shp[a] << " w=" << qp.w
                                      << " dq_dhc=" << dq_dhc << " dq_dt0=" << dq_dt0
                                      << " contrib_t0=" << (-dq_dt0*qp.shp[a]*qp.w) << "\n" << std::flush;
                        }
                    }
                    if (dq_dhc == 0.0 && dq_dt0 == 0.0 && dq_dhp == 0.0) continue;
                    for (int a = 0; a < nd; a++) {
                        double contrib = qp.shp[a] * qp.w;
                        if (dq_dhc != 0.0) {
                            #pragma omp atomic
                            rhc[qp.dof[a]] -= dq_dhc * contrib;
                        }
                        if (dq_dt0 != 0.0) {
                            #pragma omp atomic
                            rt0[qp.dof[a]] -= dq_dt0 * contrib;
                        }
                        if (dq_dhp != 0.0) {
                            #pragma omp atomic
                            rhp[qp.dof[a]] -= dq_dhp * contrib;
                        }
                    }
                }
            }

            tk_mark(2);
            // ---- (c) nodal capacitive+latent term ----
            // Kokkos::parallel_for (2026-09-12, A2 perf pass): matches the
            // reference single-layer driver's own "(c) nodal" block exactly
            // -- iteration i touches only index i of every array (read and
            // write), an embarrassingly-parallel per-dof OTI evaluation, no
            // atomics/races possible. Degrades to the identical serial loop
            // when built without -DOTI_ENABLE_KOKKOS (see Kokkos shim above).
            {
                const double *Tnext_p = u_iter.HostRead();
                const double *Tn_p = u_n.HostRead();
                const double *Vp = V_node.HostRead();
                const double *pderv[NW];
                for (int p = 0; p < NW; p++) pderv[p] = prev_der[p].HostRead();
                double *rp[NW];
                for (int p = 0; p < NW; p++) rp[p] = RHS[p].HostReadWrite();
                double idt_sens = 1.0 / dt_try;
                const double *Mv_d = ever_melted.data();   // v8: generic-p loop below picks up d/d(s_RHOP) for free
#ifdef USE_ENZYME_AD
                int slot_map[NW];
                for (int p = 0; p < NW; p++) slot_map[p] = -1;
                slot_map[LI_K]=0; slot_map[LI_CP]=1; slot_map[LI_RHO]=2; slot_map[LI_LFUS]=3; slot_map[LI_TS]=4;
                slot_map[LI_TLIQ]=5; slot_map[LI_HC]=6; slot_map[LI_T0]=7; slot_map[LI_KP]=8; slot_map[LI_RHOP]=9; slot_map[LI_HP]=10;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < ndof; i++) {
                    double pd_i[NW], rhs_i[NW];
                    for (int p = 0; p < NW; p++) pd_i[p] = pderv[p][i];
                    fb_nodal1(Tnext_p[i], Tn_p[i], pd_i, NW, slot_map, Mv_d[i], IN625::RHOP_FRACTION,
                              Vp[i], idt_sens, IN625::T_solidus, IN625::T_liquidus, rhs_i);
                    for (int p = 0; p < NW; p++) rp[p][i] += rhs_i[p];
                }
#else
                Kokkos::parallel_for("block_c_nodal_sens", ndof, KOKKOS_LAMBDA(int i) {
                    OTI Tnext_i(Tnext_p[i]);            // UNSEEDED: held fixed for d/ds|_{T_next fixed}
                    OTI Tn_i(Tn_p[i]);
                    for (int p = 0; p < NW; p++) Tn_i.set_e(WB(p), pderv[p][i]);  // Tn: known, seeded with propagated history
                    OTI Vi(Vp[i]);
                    // v8: rho blends powder/bulk at this node (current ever_melted[i]; cp has no powder
                    // variant, stays bulk). dT/d(s_RHOP) then falls out of the EXISTING generic p-loop
                    // below with no other change, since rhop_oti carries its own S_RHOP seed.
                    OTI rho_i = OTI(Mv_d[i]) * IN625::rho_oti(Tnext_i) + OTI(1.0 - Mv_d[i]) * IN625::rhop_oti(Tnext_i);
                    OTI Mc_i = Vi * rho_i * IN625::cp_oti(Tnext_i);
                    OTI L_i  = Vi * rho_i * IN625::Lfusion_oti() * IN625::f_L_oti(Tnext_i);
                    OTI Ln_i = Vi * rho_i * IN625::Lfusion_oti() * IN625::f_L_oti(Tn_i);
                    OTI nodal = Mc_i * OTI(idt_sens) * (Tnext_i - Tn_i) + (L_i - Ln_i) * OTI(idt_sens);
                    if (getenv("BLOCKC_TRACE") && i == atoi(getenv("BLOCKC_TRACE"))) {
                        printf("  [BLOCKC] i=%d Tnext=%.6f Tn=%.6f Mc_e=%.6e Tn_e=%.6e nodal_e=%.6e idt=%.6e Mc_real=%.6e L_e=%.6e Ln_e=%.6e\n",
                               i, Tnext_i.real(), Tn_i.real(), Mc_i.get_e(WB(0)), Tn_i.get_e(WB(0)), nodal.get_e(WB(0)), idt_sens, Mc_i.real(), L_i.get_e(WB(0)), Ln_i.get_e(WB(0)));
                    }
                    for (int p = 0; p < NW; p++) rp[p][i] += -nodal.get_e(WB(p));
                });
#endif
            }

            tk_mark(3);
            // ---- (d) laser magnitude: P (LI_P), eta (LI_ETA) ----
            // P3 (NW=10, 2026-09-12), ported from the reference driver's own
            // block (d) exactly. f_src is temperature-independent and enters
            // the residual purely additively (R_i = ... - f_src_i); P and
            // eta are both multiplicative on the deposited energy, so
            // f_src(s_P,s_eta) = f_src_nominal*(1+s_P)*(1+s_eta), giving
            // RHS_P = RHS_eta = -dR/ds = +f_src_nominal directly -- no
            // per-quadrature-point loop, no OTI evaluation of the Goldak/
            // spatial-hash source geometry needed at all (unlike a future
            // P4/P5 toolpath-shape extension, which WOULD be nonlinear in
            // f_src). f_src is already activation-masked (node_active check
            // above, before this block runs), so inactive dofs automatically
            // get zero contribution here too.
            if (NW > LI_ETA) {
                const double *fs = f_src.HostRead();
                double *rP = RHS[LI_P].HostReadWrite();
                double *rEta = RHS[LI_ETA].HostReadWrite();
                for (int i = 0; i < ndof; i++) { rP[i] += fs[i]; rEta[i] += fs[i]; }
            }

            tk_mark(4);
            // ---- (e) Goldak-shape source sensitivity: rx, ry, rz ----
            // P4 (NW=13, 2026-09-12), ported from the reference driver's own
            // block (e) exactly. Unlike (d), rx/ry/rz RESHAPE the Goldak
            // ellipsoid -- genuinely nonlinear in f_src -- so this uses
            // CENTRAL FD directly on the plain double-precision BuildSource
            // (save -> mutate goldak_af/ar/b/c -> call -> restore, twice per
            // param), not an OTI-typed source geometry. h=H_RXYZ is
            // FRACTIONAL/multiplicative (param*(1+-h)), matching every other
            // block's "unit scale" convention, NOT an absolute mm offset.
            // Sign convention: R_i = ... - f_src_i, so RHS_p=-dR/ds=+df_src/ds
            // -- a plain "+=", no extra negation (same as block (d)).
            // rx moves goldak_af AND goldak_ar together (front+rear travel-
            // axis size, the reference's own explicit choice -- their
            // individual asymmetry is intentionally not a separate
            // direction); ry moves goldak_b alone; rz moves goldak_c alone.
            // CRITICAL: fp/fm must get the SAME g_pscale*g_etascale scale
            // and node_active masking that the nominal f_src already got
            // above, or this derivative would be inconsistent with it.
            // ---- (e/f) AD: source-parameter sensitivities rx, ry, rz, x0, Vx, tL ----
            // One recorded BuildSource call (discrete choices frozen) + the differentiable kernel
            // src_kernel.hpp, differentiated by Enzyme forward mode. No finite differences.
#ifdef USE_ENZYME_AD
            if (!g_src_fd_legacy && NW > LI_RX) {
                SrcTape tape; Vector fchk(ndof);
                {   // BuildSource may touch last_dir / seam counters; this extra recorded call must not perturb them
                    array<double,3> sv_ld = source.last_dir; long sv_a = source.seam_skip_count, sv_b = source.seam_total_count;
                    source.BuildSource(laser, t + dt_try, dt_try, fchk, &tape);
                    source.last_dir = sv_ld; source.seam_skip_count = sv_a; source.seam_total_count = sv_b;
                }
                const int nsub = (int)tape.subs.size();
                const SrcCell *cells = tape.cells.data();
                const bool o2 = (derivative_order >= 2);
                std::vector<Vector> D1(6), D2(6);
                double *d1p[6], *d2p[6];
                for (int j = 0; j < 6; j++) { D1[j].SetSize(ndof); D1[j] = 0.0; D2[j].SetSize(ndof); D2[j] = 0.0;
                                              d1p[j] = D1[j].HostReadWrite(); d2p[j] = D2[j].HostReadWrite(); }
                const bool mix = o2 && g_sens_mixed;
                std::vector<Vector> DM(mix ? 15 : 0);   // mixed a<b among the 6 source parameters
                double *dmp[15];
                for (int k = 0; k < (int)DM.size(); k++) { DM[k].SetSize(ndof); DM[k] = 0.0; dmp[k] = DM[k].HostReadWrite(); }
                static int n_check = 0; const bool do_check = (n_check++ < 3);
                Vector fk(ndof); fk = 0.0; double *fkp = fk.HostReadWrite();
                #pragma omp parallel
                {
                    std::vector<double> buf(SRCK_MAXOUT);
                    #pragma omp for schedule(dynamic, 16)
                    for (int si = 0; si < nsub; si++) {
                        const SrcSub &r = tape.subs[si];
                        const int nout = r.ncell * 8;
                        const SrcCell *rc = cells + r.first_cell;
                        auto scatter = [&](double *dst, const double *val) {
                            for (int k = 0; k < nout; k++) { const double v = val[k]; if (v == 0.0) continue;
                                #pragma omp atomic
                                dst[rc[k/8].dofs[k%8]] += v; } };
                        const int j0 = (r.mode == 2) ? 0 : 3;       // point source: rx,ry,rz do not enter
                        for (int j = j0; j < 6; j++) {
                            ak_src_d1(&r, cells, j, buf.data()); scatter(d1p[j], buf.data());
                            if (o2) { ak_src_d2(&r, cells, j, buf.data()); scatter(d2p[j], buf.data()); }
                        }
                        if (do_check) { ak_src_eval(&r, cells, buf.data()); scatter(fkp, buf.data()); }
                        if (mix) {
                            std::vector<double> H21((size_t)21 * nout);
                            ak_src_hess(&r, cells, H21.data());
                            int k21 = 0, k15 = 0;
                            for (int a = 0; a < 6; a++) for (int b = a; b < 6; b++, k21++) {
                                if (a == b) continue;
                                if (r.mode == 2 || a >= 3) scatter(dmp[k15], H21.data() + (size_t)k21*nout);
                                k15++;
                            }
                        }
                    }
                }
                const double scl = (g_pscale * g_etascale) / (tape.t1 - tape.t0);
                if (do_check) {   // the kernel must reproduce BuildSource's nominal source exactly
                    double md = 0.0, mf = 0.0; const double *fc = fchk.HostRead();
                    for (int i = 0; i < ndof; i++) { md = max(md, std::abs(fkp[i]/(tape.t1 - tape.t0) - fc[i])); mf = max(mf, std::abs(fc[i])); }
                    cout << "  [SRC_AD check] step " << step << ": " << nsub << " sub-samples, " << tape.cells.size()
                         << " cells; max|kernel f - BuildSource f| = " << md << " (max|f| = " << mf << ")\n" << flush;
                }
                for (int j = 0; j < 6; j++) {
                    double *a = D1[j].HostReadWrite(), *b2 = D2[j].HostReadWrite();
                    for (int i = 0; i < ndof; i++) {
                        if (!node_active[i]) { a[i] = 0.0; b2[i] = 0.0; continue; }
                        a[i] *= scl; b2[i] *= scl; }
                    const int li = LI_RX + j;
                    if (NW > li) { double *rr = RHS[li].HostReadWrite(); for (int i = 0; i < ndof; i++) rr[i] += a[i]; }
                }
                if (o2) { src2_fd_rx = D2[0]; src2_fd_ry = D2[1]; src2_fd_rz = D2[2];
                          src2_x0 = D2[3]; src2_vx = D2[4]; src2_tl = D2[5]; }
                if (mix) {
                    for (int k = 0; k < 15; k++) { double *a = DM[k].HostReadWrite();
                        for (int i = 0; i < ndof; i++) a[i] = node_active[i] ? a[i] * scl : 0.0; }
                    srcD1keep = D1; srcHkeep = DM;
                }
            }
#endif
            if (g_src_fd_legacy && NW > LI_RX) {
                const double H_RXYZ = 1.0e-3, inv2h = 0.5 / H_RXYZ;
                // A3 (2026-09-12): the 2nd-order stencil (fp-2f0+fm)/h^2 needs
                // a LARGER h than the 1st-order central difference above --
                // dividing by h^2 instead of 2h amplifies any noise (Goldak
                // quadrature-cell/FindElement discreteness near an element
                // boundary, same category of effect as this session's
                // position-sensitivity element-crossing findings) by
                // 1/h^2=1e6 at h=H_RXYZ, badly aliasing the true curvature.
                // Reusing fp/fm at H_RXYZ gave a real, systematic ~2x error
                // on rx (confirmed via FD validation, job 810987) -- not a
                // formula bug (matches the reference driver's own stencil
                // exactly), a precision issue specific to this 2nd
                // difference. H2_RXYZ is 10x larger, computed via its OWN
                // separate BuildSource pair, leaving the validated 1st-order
                // fp/fm/H_RXYZ path above completely untouched. FIXED and
                // FD-validated at H2_RXYZ=1e-2 (job 810992): rx now matches
                // to rd=0.0001-0.0005; ry/rz's own remaining O(0.01-0.2)
                // relative error is a tiny-absolute-magnitude noise-floor
                // effect (same accepted pattern as first-order's own ry
                // finding), not a further bug -- confirmed by ry matching
                // to rd=0.0021 at the one probe time its magnitude is
                // actually large enough (~1e-4) to compare meaningfully.
                const double H2_RXYZ = 1.0e-2, invh2_2 = 1.0 / (H2_RXYZ*H2_RXYZ);
                double s_af = source.goldak_af, s_ar = source.goldak_ar,
                       s_b  = source.goldak_b,  s_c  = source.goldak_c;
                // BUG FIX (2026-10-07): BuildSource() mutates source.last_dir (and the seam debug
                // counters) as a side effect even though it is logically "const" -- a fallback travel
                // direction carried across calls for degenerate (zero-length) toolpath segments. The
                // perturbed calls below (rx/ry/rz central FD on the Goldak shape) were leaving last_dir
                // in whatever state the LAST perturbed call set it to, which then silently corrupted the
                // very next (real, nominal) BuildSource() call for the following time step -- making the
                // nominal forward solution itself depend on whether -dorder>=1 was requested. Save/restore
                // here, mirroring the existing goldak_af/ar/b/c save/restore immediately below.
                array<double,3> saved_last_dir = source.last_dir;
                long saved_seam_skip = source.seam_skip_count, saved_seam_total = source.seam_total_count;
                Vector fp(ndof), fm(ndof);
                const double *f0 = f_src.HostRead();  // nominal, already scaled+masked (A3 2nd-order stencil)
                for (int s = 0; s < 3; s++) {
                    int li = (s == 0) ? LI_RX : (s == 1) ? LI_RY : LI_RZ;
                    if (NW <= li) break;
                    if (s == 0) { source.goldak_af = s_af*(1.0+H_RXYZ); source.goldak_ar = s_ar*(1.0+H_RXYZ); }
                    else if (s == 1) { source.goldak_b = s_b*(1.0+H_RXYZ); }
                    else { source.goldak_c = s_c*(1.0+H_RXYZ); }
                    source.BuildSource(laser, t + dt_try, dt_try, fp);
                    fp *= (g_pscale * g_etascale);
                    for (int i = 0; i < ndof; i++) if (!node_active[i]) fp[i] = 0.0;
                    source.goldak_af = s_af; source.goldak_ar = s_ar; source.goldak_b = s_b; source.goldak_c = s_c;

                    if (s == 0) { source.goldak_af = s_af*(1.0-H_RXYZ); source.goldak_ar = s_ar*(1.0-H_RXYZ); }
                    else if (s == 1) { source.goldak_b = s_b*(1.0-H_RXYZ); }
                    else { source.goldak_c = s_c*(1.0-H_RXYZ); }
                    source.BuildSource(laser, t + dt_try, dt_try, fm);
                    fm *= (g_pscale * g_etascale);
                    for (int i = 0; i < ndof; i++) if (!node_active[i]) fm[i] = 0.0;
                    source.goldak_af = s_af; source.goldak_ar = s_ar; source.goldak_b = s_b; source.goldak_c = s_c;

                    double *r1 = RHS[li].HostReadWrite();
                    const double *fpp = fp.HostRead(), *fmp = fm.HostRead();
                    for (int i = 0; i < ndof; i++) r1[i] += (fpp[i] - fmp[i]) * inv2h;

                    if (derivative_order >= 2) {
                        Vector fp2(ndof), fm2(ndof);
                        if (s == 0) { source.goldak_af = s_af*(1.0+H2_RXYZ); source.goldak_ar = s_ar*(1.0+H2_RXYZ); }
                        else if (s == 1) { source.goldak_b = s_b*(1.0+H2_RXYZ); }
                        else { source.goldak_c = s_c*(1.0+H2_RXYZ); }
                        source.BuildSource(laser, t + dt_try, dt_try, fp2);
                        fp2 *= (g_pscale * g_etascale);
                        for (int i = 0; i < ndof; i++) if (!node_active[i]) fp2[i] = 0.0;
                        source.goldak_af = s_af; source.goldak_ar = s_ar; source.goldak_b = s_b; source.goldak_c = s_c;

                        if (s == 0) { source.goldak_af = s_af*(1.0-H2_RXYZ); source.goldak_ar = s_ar*(1.0-H2_RXYZ); }
                        else if (s == 1) { source.goldak_b = s_b*(1.0-H2_RXYZ); }
                        else { source.goldak_c = s_c*(1.0-H2_RXYZ); }
                        source.BuildSource(laser, t + dt_try, dt_try, fm2);
                        fm2 *= (g_pscale * g_etascale);
                        for (int i = 0; i < ndof; i++) if (!node_active[i]) fm2[i] = 0.0;
                        source.goldak_af = s_af; source.goldak_ar = s_ar; source.goldak_b = s_b; source.goldak_c = s_c;

                        double *r2 = (s == 0) ? src2_fd_rx.HostWrite()
                                   : (s == 1) ? src2_fd_ry.HostWrite()
                                              : src2_fd_rz.HostWrite();
                        const double *fpp2 = fp2.HostRead(), *fmp2 = fm2.HostRead();
                        for (int i = 0; i < ndof; i++) r2[i] = (fpp2[i] - 2.0*f0[i] + fmp2[i]) * invh2_2;
                    }
                }
                // restore (see save comment above): undo last_dir/seam-counter drift from the
                // perturbed BuildSource() calls so the NEXT step's real nominal call is unaffected.
                source.last_dir = saved_last_dir;
                source.seam_skip_count = saved_seam_skip; source.seam_total_count = saved_seam_total;
            }

            tk_mark(5);
            // ---- (f) toolpath position/timing sensitivity: x0, Vx, tL ----
            // P5 (NW=16). REPLACED (2026-09-12) the frozen-element analytic
            // BuildSourcePosGrad approach with plain central FD directly on
            // BuildSource, exactly mirroring block (e)'s already-validated
            // pattern for rx/ry/rz. Root cause of the original approach's
            // ~9% relative L2 error (root-caused via jobs 810932-953,
            // concentrated at the substrate/row1 tie interface): the
            // frozen-element gradient assumes FindElement's element
            // assignment is stable under an infinitesimal position
            // perturbation, but FindElement's own bounds test uses
            // inclusive >=/<= against per-element AABBs, so a query point
            // sitting on the coincident z=0 face shared by two
            // independently-meshed regions can satisfy BOTH a substrate
            // element's AABB and a row1 element's AABB -- resolution then
            // falls to hash-bucket insertion order, not physics, which is
            // inherently unstable to tiny perturbations right at that seam.
            // Two z-coordinate-based guards were tried and reverted (one
            // never fired, the other zeroed the signal since row1 is only
            // one element thick and every row1 element touches z=0).
            // Plain central FD sidesteps the frozen-element assumption
            // entirely: BuildSource is called with g_xshift/g_yshift/
            // g_zshift set (a true rigid translation of the whole toolpath,
            // applied inside TableLaserParams::position), so element
            // reassignment near any boundary -- including the tie seam --
            // is captured exactly as it truly happens, the same way it
            // already is for rx/ry/rz above.
            if (g_src_fd_legacy && NW > LI_X0) {
                const double H_POS = 1.0e-3, inv2hp = 0.5 / H_POS;
                Vector gx(ndof), gy(ndof), gz(ndof);
                Vector fp(ndof), fm(ndof);
                // Same last_dir/seam-counter corruption risk as the rx/ry/rz block above (see its
                // comment): this block's own comment explicitly says it's probing near-seam/boundary
                // behavior under a true positional shift, which is exactly where BuildSource()'s
                // degenerate-segment fallback (last_dir) is most likely to be touched by a perturbed call.
                array<double,3> saved_last_dir_x0 = source.last_dir;
                long saved_seam_skip_x0 = source.seam_skip_count, saved_seam_total_x0 = source.seam_total_count;
                // BUG FIX (2026-10-07): this block was resetting g_xshift/g_yshift/g_zshift to a
                // hardcoded 0.0 after each perturbed call, instead of restoring them to whatever the
                // user actually set via -xshift/-yshift/-zshift (e.g. -zshift 0.18, required in every
                // production run to align the laser with the row-1 top surface). From the first time
                // this block ran (derivative_order>=1, NW>LI_X0), the real z-shift was silently and
                // PERMANENTLY zeroed for the rest of the run, putting the laser at the wrong z-height
                // and zeroing f_src wherever the correct (shifted) and wrong (unshifted) positions landed
                // in different elements. Save/restore the real values instead (mirrors the goldak_af/
                // ar/b/c and last_dir/seam-counter save/restore above).
                const double saved_xshift = g_xshift, saved_yshift = g_yshift, saved_zshift = g_zshift;

                g_xshift = saved_xshift + H_POS;  source.BuildSource(laser, t + dt_try, dt_try, fp); g_xshift = saved_xshift;
                g_xshift = saved_xshift - H_POS;  source.BuildSource(laser, t + dt_try, dt_try, fm); g_xshift = saved_xshift;
                { double *g1=gx.HostWrite(); const double *p1=fp.HostRead(), *m1=fm.HostRead();
                  for (int i=0;i<ndof;i++) g1[i]=(p1[i]-m1[i])*inv2hp; }

                g_yshift = saved_yshift + H_POS;  source.BuildSource(laser, t + dt_try, dt_try, fp); g_yshift = saved_yshift;
                g_yshift = saved_yshift - H_POS;  source.BuildSource(laser, t + dt_try, dt_try, fm); g_yshift = saved_yshift;
                { double *g1=gy.HostWrite(); const double *p1=fp.HostRead(), *m1=fm.HostRead();
                  for (int i=0;i<ndof;i++) g1[i]=(p1[i]-m1[i])*inv2hp; }

                g_zshift = saved_zshift + H_POS;  source.BuildSource(laser, t + dt_try, dt_try, fp); g_zshift = saved_zshift;
                g_zshift = saved_zshift - H_POS;  source.BuildSource(laser, t + dt_try, dt_try, fm); g_zshift = saved_zshift;
                { double *g1=gz.HostWrite(); const double *p1=fp.HostRead(), *m1=fm.HostRead();
                  for (int i=0;i<ndof;i++) g1[i]=(p1[i]-m1[i])*inv2hp; }

                gx *= (g_pscale * g_etascale); gy *= (g_pscale * g_etascale); gz *= (g_pscale * g_etascale);
                for (int i = 0; i < ndof; i++) if (!node_active[i]) { gx[i]=0.0; gy[i]=0.0; gz[i]=0.0; }

                double tmc = t + 0.5*dt_try, eps = 1e-8;
                double xp_,yp_,zp_,xm_,ym_,zm_;
                laser.position(tmc+eps, xp_,yp_,zp_);
                laser.position(tmc-eps, xm_,ym_,zm_);
                double vx=(xp_-xm_)/(2*eps), vy=(yp_-ym_)/(2*eps), vz=(zp_-zm_)/(2*eps);
                double wVx = tmc - t_start;

                const double *gxp=gx.HostRead(), *gyp=gy.HostRead(), *gzp=gz.HostRead();
                double *rx0 = RHS[LI_X0].HostReadWrite();
                for (int i = 0; i < ndof; i++) rx0[i] += gxp[i];
                if (NW > LI_VX) {
                    double *rv = RHS[LI_VX].HostReadWrite();
                    for (int i = 0; i < ndof; i++) rv[i] += wVx*(vx*gxp[i]+vy*gyp[i]+vz*gzp[i]);
                }
                if (NW > LI_TL) {
                    double *rt = RHS[LI_TL].HostReadWrite();
                    for (int i = 0; i < ndof; i++) rt[i] += -(vx*gxp[i]+vy*gyp[i]+vz*gzp[i]);
                }
                source.last_dir = saved_last_dir_x0;
                source.seam_skip_count = saved_seam_skip_x0; source.seam_total_count = saved_seam_total_x0;
            }

            // Bottom substrate / not-yet-active rows are parameter independent.
            for (int p = 0; p < NW; p++) {
                double *rp = RHS[p].HostReadWrite();
                for (int q = 0; q < ess_tdof_list.Size(); q++) rp[ess_tdof_list[q]] = 0.0;
            }
            // v8 (2026-10-02): Tw -- build-platform heater, i.e. the substrate-bottom Dirichlet value
            // T_sub_bc = Tw_nominal*(1+s_Tw). Essential rows force dTds=RHS there directly (same
            // identity-row mechanism the general zeroing above relies on), so injecting T_sub_bc (the
            // NOMINAL value, same "value*(1+s)" convention as T0's own IC seed) at bc_dofs ONLY (not
            // the other essential dofs, i.e. not-yet-active rows, which have zero Tw-dependence) gives
            // the exact derivative with no other RHS term needed -- unlike every other wired parameter,
            // Tw has no domain/boundary/nodal residual dependence of its own, only this BC value.
            if (NW > LI_TW) {
                double *rtw = RHS[LI_TW].HostReadWrite();
                for (int i = 0; i < bc_dofs.Size(); i++) rtw[bc_dofs[i]] = T_sub_bc;
            }

            // ---- DEBUG (2026-09-11): FD-check RHS[] directly against the
            // residual formula at the FIXED converged u_iter/u_n, isolating
            // whether the bug is in RHS-building (blocks a/c) vs. downstream
            // in the Jacobian/solve. Throwaway FachinottiResidualOp copies
            // with their OWN Vectors (not sharing M_cp_lump/etc with the
            // real `residual`) so this cannot corrupt any state needed later.
            if (rhs_check) {
                double h = 1e-3;
                // Test MULTIPLE dofs, not just the probe -- RHS_CHECK/JAC_CHECK
                // so far only ever verified probe_dof[0] itself. If there's an
                // assembly bug affecting OTHER dofs (wrong sign, bad scatter,
                // an attribute-boundary issue), it wouldn't show up there, but
                // WOULD corrupt the coupled linear solve's answer at the probe
                // through J^-1. Check: row1 probe, substrate diagnostic probe,
                // and (if it exists) row2's own probe.
                std::vector<std::pair<std::string,int>> test_dofs = {
                    {"row1_probe", probe_dof[0]},
                    {"substrate_probe", sub_probe_dof},
                };
                if (n_rows > 1 && probe_dof[1] >= 0) test_dofs.push_back({"row2_probe", probe_dof[1]});
                auto eval_r_at = [&](double ksc, double cpsc, int dof) -> double {
                    g_kscale = ksc; g_cpscale = cpsc;
                    Vector Mc_(ndof), Ml_(ndof), Ln_(ndof), Meff_(ndof), Rb_(ndof);
                    for (auto v : {&Mc_,&Ml_,&Ln_,&Meff_,&Rb_}) v->UseDevice(true);
                    FachinottiResidualOp r_op(fespace, attr_active, bdry_qps, u_n, L_n, V_node,
                                              Mc_, Ml_, Ln_, Meff_, Rb_, f_src, dt_try,
                                              IN625::T_solidus, IN625::T_liquidus, DT_mushy,
                                              h_conv, emissivity, sigma_sb, T_ambient,
                                              substrate_loss_scale, bridge_loss_scale, interface_loss_scale,
                                              ess_tdof_list, ndof);
                    Vector r(ndof); r_op.Mult(u_iter, r);
                    g_kscale = 1.0; g_cpscale = 1.0;
                    return r[dof];
                };
                // P2 T0 isolation (2026-09-12): dT/dT0 mismatched FD by
                // ~11x at step 1, decaying toward FD over later steps --
                // classic symptom of a bad initial-condition seed. Perturb
                // T_ambient DIRECTLY (a constructor parameter, not a
                // global toggle like g_kscale/g_cpscale) to isolate
                // whether RHS[LI_T0]-building itself is already wrong
                // (this test), separate from anything downstream.
                auto eval_r_at_t0 = [&](double ta, int dof) -> double {
                    Vector Mc_(ndof), Ml_(ndof), Ln_(ndof), Meff_(ndof), Rb_(ndof);
                    for (auto v : {&Mc_,&Ml_,&Ln_,&Meff_,&Rb_}) v->UseDevice(true);
                    FachinottiResidualOp r_op(fespace, attr_active, bdry_qps, u_n, L_n, V_node,
                                              Mc_, Ml_, Ln_, Meff_, Rb_, f_src, dt_try,
                                              IN625::T_solidus, IN625::T_liquidus, DT_mushy,
                                              h_conv, emissivity, sigma_sb, ta,
                                              substrate_loss_scale, bridge_loss_scale, interface_loss_scale,
                                              ess_tdof_list, ndof);
                    Vector r(ndof); r_op.Mult(u_iter, r);
                    return r[dof];
                };
                for (auto &td : test_dofs) {
                    const string &name = td.first; int dof = td.second;
                    if (dof < 0) continue;
                    double r_kp = eval_r_at(1.0+h, 1.0, dof), r_km = eval_r_at(1.0-h, 1.0, dof);
                    double r_cpp = eval_r_at(1.0, 1.0+h, dof), r_cpm = eval_r_at(1.0, 1.0-h, dof);
                    double rhs_fd_k  = -(r_kp - r_km) / (2*h);
                    double rhs_fd_cp = -(r_cpp - r_cpm) / (2*h);
                    double r_t0p = eval_r_at_t0(T_ambient*(1.0+h), dof), r_t0m = eval_r_at_t0(T_ambient*(1.0-h), dof);
                    double rhs_fd_t0 = -(r_t0p - r_t0m) / (2*h);
                    cout << "  [RHS_CHECK] " << name << " (dof=" << dof << ") k:  analytic RHS=" << RHS[LI_K][dof]
                         << "  direct-FD RHS=" << rhs_fd_k << "\n"
                         << "  [RHS_CHECK] " << name << " (dof=" << dof << ") cp: analytic RHS=" << RHS[LI_CP][dof]
                         << "  direct-FD RHS=" << rhs_fd_cp << "\n"
                         << "  [RHS_CHECK] " << name << " (dof=" << dof << ") T0: analytic RHS=" << RHS[LI_T0][dof]
                         << "  direct-FD RHS=" << rhs_fd_t0 << "\n" << flush;
                }
                // Also check the AGGREGATE RHS vector norm-of-difference against
                // a per-dof direct-FD vector built the SAME way but for EVERY
                // dof at once (cheap: reuses the same 4 residual evaluations).
                {
                    Vector r_kp_full(ndof), r_km_full(ndof), r_cpp_full(ndof), r_cpm_full(ndof);
                    auto eval_r_full = [&](double ksc, double cpsc, Vector &out) {
                        g_kscale = ksc; g_cpscale = cpsc;
                        Vector Mc_(ndof), Ml_(ndof), Ln_(ndof), Meff_(ndof), Rb_(ndof);
                        for (auto v : {&Mc_,&Ml_,&Ln_,&Meff_,&Rb_}) v->UseDevice(true);
                        FachinottiResidualOp r_op(fespace, attr_active, bdry_qps, u_n, L_n, V_node,
                                                  Mc_, Ml_, Ln_, Meff_, Rb_, f_src, dt_try,
                                                  IN625::T_solidus, IN625::T_liquidus, DT_mushy,
                                                  h_conv, emissivity, sigma_sb, T_ambient,
                                                  substrate_loss_scale, bridge_loss_scale, interface_loss_scale,
                                                  ess_tdof_list, ndof);
                        r_op.Mult(u_iter, out);
                        g_kscale = 1.0; g_cpscale = 1.0;
                    };
                    eval_r_full(1.0+h, 1.0, r_kp_full); eval_r_full(1.0-h, 1.0, r_km_full);
                    Vector rhs_fd_k_full(ndof);
                    { const double *p1=r_kp_full.HostRead(), *m1=r_km_full.HostRead(); double *fp=rhs_fd_k_full.HostWrite();
                      for (int i=0;i<ndof;i++) fp[i] = -(p1[i]-m1[i])/(2*h); }
                    Vector diff_k(rhs_fd_k_full); diff_k -= RHS[LI_K];
                    if (getenv("RHS_LOCATE")) {
                        std::vector<std::pair<double,int>> tp;
                        for (int i = 0; i < ndof; i++) tp.push_back({fabs(diff_k(i)), i});
                        std::partial_sort(tp.begin(), tp.begin() + 8, tp.end(), [](auto &x, auto &y){ return x.first > y.first; });
                        for (int q = 0; q < 8; q++) {
                            int i = tp[q].second; const double *v = mesh.GetVertex(i);
                            cout << "  [RHS_LOCATE] dof " << i << " diff=" << diff_k(i) << " analytic=" << RHS[LI_K](i)
                                 << " fd=" << rhs_fd_k_full(i) << " hist=" << M_cp_lump(i) / dt_try * prev_der[LI_K](i)
                                 << " x=(" << v[0] << "," << v[1] << "," << v[2] << ") T=" << u_iter[i] << "\n";
                        }
                    }
                    cout << "  [RHS_CHECK] FULL-VECTOR k: |RHS_analytic|=" << RHS[LI_K].Norml2()
                         << "  |RHS_direct_fd|=" << rhs_fd_k_full.Norml2()
                         << "  |diff|=" << diff_k.Norml2()
                         << "  max|diff|=" << diff_k.Normlinf() << "\n" << flush;
                }
            }

            // ---- solve: the EXACT sensitivity Jacobian (base J + the dk/dT
            // correction), via GMRES preconditioned by the existing
            // approximate solver (CG/PARDISO on the base, uncorrected J --
            // same -solver choice as the forward Newton solve, matching
            // 2026-09-11's finding that CG alone isn't robust enough right
            // after an activation event). The forward Newton solve itself
            // is untouched -- this correction is used ONLY for the
            // sensitivity solve, per the internal FD check that found
            // dT/dk off by ~9x using the base (chord-approximate) J alone.
            Operator &J_full_sens = residual.GetGradient(u_iter);
            JacobianOp &J_base_sens = static_cast<JacobianOp&>(J_full_sens);
            // J_base_sens (kpa_dynamic-based) is used ONLY as the
            // approximate preconditioner below -- fine, since preconditioners
            // don't need to be exact. The ACTUAL GMRES operator is
            // CorrectedJacobianOp, which builds its OWN manually-assembled
            // domain matrix (see its own comment: kpa_dynamic's assembled
            // diagonal was proven off by ~1.0 at certain rows, regardless of
            // how it's rebuilt -- element-local AssembleElementMatrix,
            // scattered manually, is the only method proven correct).
            CorrectedJacobianOp J_corr_sens(fespace, mesh, attr_active, u_iter,
                                            residual.GetJbdryMat(), Meff_over_dt,
                                            ess_tdof_list, ndof);
            // IMMEDIATE, ISOLATED test of the FULL Mult() chain (domain+
            // boundary+nodal+correction) at the known problem dof, done
            // RIGHT HERE before any diagnostic can touch residual's
            // internal state (which would invalidate Jbdry_mat/etc).
            {
                Vector e41928(ndof); e41928 = 0.0; e41928[41928] = 1.0;
                Vector Je41928(ndof);
                J_corr_sens.Mult(e41928, Je41928);
                cout << "  [FULL_MULT_MARKER] J_corr_sens.Mult(e_41928)[41928]=" << Je41928[41928]
                     << "  (expect ~7.17-7.18 if fix works end-to-end, ~8.17 if not)\n" << std::flush;
                J_corr_sens.DebugBreakdown(41928, e41928);
            }

            // ---- DEBUG (2026-09-11): FD-check J*delta directly against the
            // residual formula at u_iter, isolating whether the (corrected)
            // Jacobian operator itself is right -- independent of BOTH
            // RHS-building (already proven exact via -rhscheck) and the
            // linear solve. delta = uniform unit bump at every non-essential
            // dof; FD_Jv = [r(u_iter+h*delta) - r(u_iter)] / h.
            if (jac_check) {
                int pd0 = probe_dof[0];
                double h_T = 0.01;
                Vector delta(ndof); delta = 1.0;
                for (int q = 0; q < ess_tdof_list.Size(); q++) delta[ess_tdof_list[q]] = 0.0;
                // SECOND test vector: spatially-VARYING (gradient-rich)
                // delta. The uniform bump above has near-zero gradient
                // everywhere, so it barely exercises the domain-diffusion
                // (gradient-coupling) part of J -- exactly the part the
                // dk/dT correction touches. delta_i = x-coordinate of dof i
                // (H1 order-1 => dof index == vertex index in this file's
                // own convention, used elsewhere e.g. bc_dofs/probe_dof).
                Vector delta2(ndof);
                { double *d2 = delta2.HostWrite();
                  for (int i = 0; i < ndof; i++) d2[i] = mesh.GetVertex(i)[0]; }
                for (int q = 0; q < ess_tdof_list.Size(); q++) delta2[ess_tdof_list[q]] = 0.0;
                double h_T2 = 0.01;

                Vector r0(ndof), r1(ndof), r2(ndof);
                residual.Mult(u_iter, r0);
                // Compute BOTH analytic J*delta results HERE, right after r0
                // refreshes the shared Meff_over_dt to u_iter's own state --
                // computing them AFTER a residual.Mult(u_pert*,...) call
                // below would read that shared vector stale at the wrong
                // state instead (bug found 2026-09-11 via JAC_CHECK3 showing
                // a spurious "+delta"-sized error that vanished once this
                // ordering was fixed there).
                Vector analytic_Jv(ndof), analytic_Jv_base(ndof);
                J_corr_sens.Mult(delta, analytic_Jv);
                J_base_sens.Mult(delta, analytic_Jv_base);
                Vector analytic_Jv2(ndof), analytic_Jv2_base(ndof);
                J_corr_sens.Mult(delta2, analytic_Jv2);
                J_base_sens.Mult(delta2, analytic_Jv2_base);

                Vector u_pert(ndof);
                { const double *up = u_iter.HostRead(); double *upp = u_pert.HostWrite();
                  const double *dp = delta.HostRead();
                  for (int i = 0; i < ndof; i++) upp[i] = up[i] + h_T*dp[i]; }
                residual.Mult(u_pert, r1);
                Vector fd_Jv(ndof);
                { const double *r0p=r0.HostRead(), *r1p=r1.HostRead(); double *fp=fd_Jv.HostWrite();
                  for (int i=0;i<ndof;i++) fp[i] = (r1p[i]-r0p[i])/h_T; }
                cout << "  [JAC_CHECK] probe: FD_Jv=" << fd_Jv[pd0]
                     << "  analytic_Jv(corrected)=" << analytic_Jv[pd0]
                     << "  analytic_Jv(base_only)=" << analytic_Jv_base[pd0] << "\n" << flush;

                Vector u_pert2(ndof);
                { const double *up = u_iter.HostRead(); double *upp = u_pert2.HostWrite();
                  const double *dp = delta2.HostRead();
                  for (int i = 0; i < ndof; i++) upp[i] = up[i] + h_T2*dp[i]; }
                residual.Mult(u_pert2, r2);
                Vector fd_Jv2(ndof);
                { const double *r0p=r0.HostRead(), *r2p=r2.HostRead(); double *fp=fd_Jv2.HostWrite();
                  for (int i=0;i<ndof;i++) fp[i] = (r2p[i]-r0p[i])/h_T2; }
                cout << "  [JAC_CHECK2] probe (gradient-rich delta): FD_Jv=" << fd_Jv2[pd0]
                     << "  analytic_Jv(corrected)=" << analytic_Jv2[pd0]
                     << "  analytic_Jv(base_only)=" << analytic_Jv2_base[pd0] << "\n" << flush;

                // residual.Mult(u_pert*,...) above mutated kpa_dynamic's live
                // state -- and critically, EnsureUpdated's own rebuild path
                // (kpa_dynamic already non-null) goes through the SAME
                // "SpMat()=0; Assemble(0)" reuse cycle just proven buggy, so
                // a plain GetGradient(u_iter) restore is NOT enough here;
                // must re-apply the fresh, non-pattern-reused rebuild.
                residual.GetGradient(u_iter);
                residual.RebuildConductivityFreshForSensitivity(u_iter);
            }
            tk_mark(6);
            // Defensive re-fix regardless of whether jac_check ran above:
            // guarantee the fresh (non-reused) kpa_dynamic is what the real
            // solve below actually uses.
            residual.RebuildConductivityFreshForSensitivity(u_iter);
            tk_mark(7);

            if (!tie_active) {
                FachinottiLinearSolver lin_solver_sens(Kd_pa, Kb_diag, Meff_over_dt, ess_tdof_list, ndof);
                lin_solver_sens.SetOperator(J_full_sens);
                GMRESSolver gmres_sens;
                gmres_sens.SetOperator(J_corr_sens);
                gmres_sens.SetPreconditioner(lin_solver_sens);
                gmres_sens.SetRelTol(1e-8); gmres_sens.SetAbsTol(0.0);
                gmres_sens.SetMaxIter(200); gmres_sens.SetKDim(50); gmres_sens.SetPrintLevel(-1);
                for (int p = 0; p < NW; p++) { dTds[p] = 0.0; gmres_sens.Mult(RHS[p], dTds[p]); }
            } else if (sens_direct_on) {
                if (!ao_ex) ao_ex.reset(new AoExactReducedSolver(tie));
                ao_ex->Factor(J_corr_sens);
                std::vector<const Vector*> Bp; std::vector<Vector*> Xp;
                for (int p = 0; p < NW; p++) {
                    dTds[p] = 0.0;
                    if (RHS[p].Normlinf() == 0.0) continue;
                    Bp.push_back(&RHS[p]); Xp.push_back(&dTds[p]);
                }
                if (!Bp.empty()) ao_ex->SolveFull(Bp, Xp);
                std::cout << "  [sens] direct exact solve: " << Bp.size() << " nonzero first-order RHS\n";
            } else {
                ReducedOpWrapper reduced_J_sens(J_full_sens, tie);      // base (approximate), for the preconditioner
                ReducedOpWrapper reduced_J_corr_sens(J_corr_sens, tie); // exact, for the GMRES operator
                ReducedLinearSolver cg_sens(Kd_pa, Kb_diag, Meff_over_dt, tie);
                std::unique_ptr<PardisoReducedSolver> pardiso_sens_local;
                if (pard_persist_on) { if (!pardiso_sens_persist) pardiso_sens_persist.reset(new PardisoReducedSolver(tie)); }
                else pardiso_sens_local.reset(new PardisoReducedSolver(tie));
                PardisoReducedSolver &pardiso_sens = pard_persist_on ? *pardiso_sens_persist : *pardiso_sens_local;
                Solver *precond_solver = (string(lin_solver_kind) == "pardiso")
                                     ? static_cast<Solver*>(&pardiso_sens)
                                     : static_cast<Solver*>(&cg_sens);
                int gmres_maxit = 200, gmres_kdim = 50;
                // v5: sensitivity solve stays on PARDISO (AMG preconditioning measured slower: ~15 vs ~2 GMRES its/param)
                // v6: SENS_REUSE (default on; SENS_REUSE=0 disables) keeps the PARDISO factorisation of the base sensitivity
                // Jacobian across steps. GMRES always runs on the exact corrected Jacobian of THIS step, so a stale factor only
                // costs GMRES iterations; answers agree with a fresh factor to the GMRES tolerance (1e-8), not bit-for-bit.
                // Refactor at layer births, after SENS_MAX_AGE reuses, when any parameter needed > SENS_REFAC_ITS GMRES
                // iterations in the previous step, and (immediately) if GMRES fails to converge.
                static const bool sens_reuse = pard_persist_on && !(std::getenv("SENS_REUSE") && std::string(std::getenv("SENS_REUSE")) == "0");
                static const int sens_max_age = std::getenv("SENS_MAX_AGE") ? atoi(std::getenv("SENS_MAX_AGE")) : 10;
                static const int sens_refac_its = std::getenv("SENS_REFAC_ITS") ? atoi(std::getenv("SENS_REFAC_ITS")) : 6;
                static int sens_age = 0; static bool sens_force = true;
                const bool sens_refactor = !sens_reuse || sens_force || activation_changed || sens_age >= sens_max_age
                                           || string(lin_solver_kind) != "pardiso";
                if (sens_refactor) { precond_solver->SetOperator(reduced_J_sens); sens_age = 0; sens_force = false; }
                else ++sens_age;
                int sens_max_its = 0; bool sens_refac_now = false;
                std::cout << "  [sens] factorisation " << (sens_refactor ? "refreshed" : "reused") << " (age " << sens_age << ")" << std::endl;
                GMRESSolver gmres_sens;
                gmres_sens.SetOperator(reduced_J_corr_sens);
                gmres_sens.SetPreconditioner(*precond_solver);
                gmres_sens.SetRelTol(1e-8); gmres_sens.SetAbsTol(0.0);
                gmres_sens.SetMaxIter(gmres_maxit); gmres_sens.SetKDim(gmres_kdim); gmres_sens.SetPrintLevel(-1);
                if (const char *dd = getenv("RHSDUMP_DIR")) {
                    static int rdstep = 0;
                    std::ofstream of(std::string(dd) + "/d1rhs_" + std::to_string(rdstep++) + ".bin", std::ios::binary);
                    of.write(reinterpret_cast<const char*>(RHS[LI_K].HostRead()), sizeof(double) * ndof);
                }
                Vector RHS_free(tie.ndof_free), dTds_free(tie.ndof_free);
                for (int p = 0; p < NW; p++) {
                    tie.Contract(RHS[p], RHS_free);
                    dTds_free = 0.0;
                    gmres_sens.Mult(RHS_free, dTds_free);
                    if (!gmres_sens.GetConverged() && !sens_refactor && !sens_refac_now) {   // stale factor failed: refresh + redo
                        std::cout << "  [sens] GMRES not converged with reused factorisation -> refactor" << std::endl;
                        precond_solver->SetOperator(reduced_J_sens); sens_refac_now = true; sens_age = 0;
                        dTds_free = 0.0; gmres_sens.Mult(RHS_free, dTds_free);
                    }
                    sens_max_its = std::max(sens_max_its, gmres_sens.GetNumIterations());
                    // TRUE (unpreconditioned) residual check -- an extra operator application per parameter, only for
                    // diagnostics in v6 (SENS_CHECK=1); prints -1 otherwise.
                    static const bool sens_check = (std::getenv("SENS_CHECK") != nullptr);
                    Vector true_resid(tie.ndof_free); true_resid = 0.0;
                    if (sens_check) { reduced_J_corr_sens.Mult(dTds_free, true_resid); true_resid -= RHS_free; }
                    cout << "  [GMRES_SENS DEBUG] p=" << p
                         << " |RHS_free|=" << RHS_free.Norml2()
                         << " |dTds_free|=" << dTds_free.Norml2()
                         << " converged=" << gmres_sens.GetConverged()
                         << " iters=" << gmres_sens.GetNumIterations()
                         << " final_norm=" << gmres_sens.GetFinalNorm()
                         << " init_norm=" << gmres_sens.GetInitialNorm()
                         << " TRUE_resid_norm=" << (sens_check ? true_resid.Norml2() : -1.0) << "\n" << flush;
                    tie.Expand(dTds_free, dTds[p]);
                }
                if (sens_max_its > sens_refac_its) sens_force = true;   // stale factor getting weak: refresh next step
            }

            tk_mark(8);
            // =================================================================
            // A3: second-order (diagonal Hessian d^2T/ds_p^2 only, no cross
            // terms d^2T/ds_p ds_q -- matches the reference driver's own
            // documented scope) sensitivity pass. Differentiates the SAME
            // first-order sensitivity equation J*dT/ds_p = RHS_p ONCE MORE
            // w.r.t. s_p (same IFT logic used to derive RHS_p from the
            // residual) to get J*d2T/ds_p^2 = RHS2_p, reusing the SAME J.
            //
            // UNLIKE the single-layer reference driver (Picard-lagged
            // k(Tn), boundary evaluated at Tn -- LINEAR in T_next), THIS
            // driver's k(T)/boundary terms are FULLY IMPLICIT, evaluated at
            // T_next itself (see blocks (a)/(b)/(c) above, all of which
            // hold T_next UNSEEDED at first order since T_next's own total
            // derivative is recovered entirely by the J-solve). At second
            // order this is no longer sufficient: differentiating a
            // residual piece that is NONLINEAR in T_next (k(T_next) in the
            // domain term; the T_next^4 radiation term; rho(T_next)*cp
            // (T_next) in the nodal term) TWICE produces a genuine
            // d^2(.)/dT_next^2 * (dT_next/ds_p)^2 contribution that must be
            // computed EXPLICITLY using THIS STEP's just-solved dTds (now
            // KNOWN), while the OTHER half of the second total derivative
            // (through T_next's own SECOND-order sensitivity, the unknown
            // d2T/ds_p^2) must stay EXCLUDED from RHS2 -- it is exactly the
            // J*d2T/ds_p^2 piece the linear solve below recovers. Seeding
            // T_next's e(WB(p)) with dTds_p (known) while leaving its
            // e_pp(WB(p)) at 0 (unknown) makes OTI's own chain rule split
            // these two pieces correctly and automatically for any FUSED
            // OTI expression (blocks (b)/(c), where the OTI-typed factor
            // and its "argument" are the same evaluation) -- no manual
            // correction needed there. Block (a) is the one exception: k_q
            // (OTI) and grad(T_next) (a plain-double geometric gradient,
            // assembled OUTSIDE OTI, exactly as at first order) are
            // SEPARATE factors multiplied by hand, so the product-rule
            // cross term 2*dk_dsp*grad(dTds_p) must be added explicitly
            // (grad(dTds_p) being the gradient of this step's ALREADY-KNOWN
            // first-order sensitivity -- itself not an OTI quantity).
            // =================================================================
            if (derivative_order >= 2) {
                for (int p = 0; p < NW; p++) RHS2[p] = 0.0;
                const double t2_0 = now_ms(); double t2_a = 0, t2_b = 0, t2_c = 0, t2_e = 0, t2_s = 0;
                double *rmp[NPAIR > 0 ? NPAIR : 1]; const double *pd2mv[NPAIR > 0 ? NPAIR : 1];
                if (mixed_on_run) for (int k = 0; k < NPAIR; k++) { RHSm[k] = 0.0; rmp[k] = RHSm[k].HostReadWrite(); pd2mv[k] = prev_der2m[k].HostRead(); }

                // ---- (a2) domain diffusion, second order ----
                {
                    const double *up = u_iter.HostRead();
                    const double *dsv[NW];
                    for (int p = 0; p < NW; p++) dsv[p] = dTds[p].HostRead();
                    double *r2ps[NW];
                    for (int p = 0; p < NW; p++) r2ps[p] = RHS2[p].HostReadWrite();
                    const int NE_a2 = dom_cache.ne;
                    const double *Mv2 = g_ever_melted ? g_ever_melted->data() : nullptr;
                    #pragma omp parallel for schedule(dynamic, 64)
                    for (int e = 0; e < NE_a2; e++) {
                        if (!attr_active[mesh.GetAttribute(e)]) continue;
                        const Array<int> &vdofs = dom_cache.vdofs[e];
                        int dof = vdofs.Size();
                        Vector Tnext_loc(dof), dTds_loc[NW];
                        for (int p = 0; p < NW; p++) dTds_loc[p].SetSize(dof);
                        for (int a = 0; a < dof; a++) {
                            int g = vdofs[a];
                            Tnext_loc(a) = up[g];
                            for (int p = 0; p < NW; p++) dTds_loc[p](a) = dsv[p][g];
                        }
                        int nq = (int)dom_cache.weight[e].size();
                        std::vector<double> accm(mixed_on_run ? (size_t)NPAIR * dof : 0, 0.0);   // mixed, element-local
                        for (int q = 0; q < nq; q++) {
                            const Vector &shape = dom_cache.shape[e][q];
                            const DenseMatrix &gshape_e = dom_cache.gshape[e][q];
                            double w = dom_cache.weight[e][q];
                            int dm = gshape_e.Width();
                            double Tnext_q = 0.0, ds_q[NW] = {0};
                            for (int a = 0; a < dof; a++) {
                                Tnext_q += shape(a) * Tnext_loc(a);
                                for (int p = 0; p < NW; p++) ds_q[p] += shape(a) * dTds_loc[p](a);
                            }
                            double gT[3] = {0,0,0}, gDs[NW][3];
                            for (int p = 0; p < NW; p++) { gDs[p][0]=0.0; gDs[p][1]=0.0; gDs[p][2]=0.0; }
                            for (int a = 0; a < dof; a++)
                                for (int d = 0; d < dm; d++) {
                                    gT[d] += gshape_e(a,d) * Tnext_loc(a);
                                    for (int p = 0; p < NW; p++)
                                        gDs[p][d] += gshape_e(a,d) * dTds_loc[p](a);
                                }
#ifdef USE_ENZYME_AD
                            double dkv[NW], d2kv[NW];
                            if (use_hess) {
                                // full Hessian of the powder-blended k(T) w.r.t. [T, S_K, S_KP] at this quadrature point
                                double mfq = 1.0;
                                if (Mv2 && !g_noblend2) { mfq = 0.0; for (int a = 0; a < dof; a++) mfq += shape(a) * Mv2[vdofs[a]]; }
                                const double xk[3] = {Tnext_q, 1.0, 1.0}, fk2[2] = {mfq, IN625::KP_FRACTION};
                                double gk[3], Hk[9];
                                fb_local_hess(0, xk, fk2, gk, Hk);
                                double Xp[NACT][3], HXp[NACT][3]; int act[NACT]; int nact = 0;
                                for (int p = 0; p < NACT; p++) {
                                    Xp[p][0] = ds_q[p]; Xp[p][1] = (p == LI_K); Xp[p][2] = (p == LI_KP);
                                    for (int u = 0; u < 3; u++) HXp[p][u] = Hk[u*3]*Xp[p][0] + Hk[u*3+1]*Xp[p][1] + Hk[u*3+2]*Xp[p][2];
                                    dkv[p] = gk[0]*Xp[p][0] + gk[1]*Xp[p][1] + gk[2]*Xp[p][2];
                                    d2kv[p] = Xp[p][0]*HXp[p][0] + Xp[p][1]*HXp[p][1] + Xp[p][2]*HXp[p][2];
                                    bool nz = (Xp[p][0] != 0.0 || Xp[p][1] != 0.0 || Xp[p][2] != 0.0);
                                    if (!nz) for (int a = 0; a < dof && !nz; a++) for (int d = 0; d < dm; d++) if (gDs[p][d] != 0.0) { nz = true; break; }
                                    if (nz) act[nact++] = p;
                                }
                                for (int p = NACT; p < NW; p++) { dkv[p] = 0.0; d2kv[p] = 0.0; }
                                if (mixed_on_run) {
                                double gd[8], gDa[NACT][8];
                                for (int a = 0; a < dof; a++) {
                                    double s0 = 0.0; for (int d = 0; d < dm; d++) s0 += gshape_e(a,d)*gT[d]; gd[a] = s0;
                                    for (int ii = 0; ii < nact; ii++) { const int p = act[ii]; double s1 = 0.0; for (int d = 0; d < dm; d++) s1 += gshape_e(a,d)*gDs[p][d]; gDa[p][a] = s1; }
                                }
                                for (int ii = 0; ii < nact; ii++) for (int jj = ii + 1; jj < nact; jj++) {
                                    const int i = act[ii], j = act[jj];
                                    const double d2 = Xp[i][0]*HXp[j][0] + Xp[i][1]*HXp[j][1] + Xp[i][2]*HXp[j][2];
                                    if (d2 == 0.0 && dkv[i] == 0.0 && dkv[j] == 0.0) continue;
                                    double *acc = &accm[(size_t)g_pair_index[i][j] * dof];
                                    for (int a = 0; a < dof; a++) acc[a] -= (d2*gd[a] + dkv[i]*gDa[j][a] + dkv[j]*gDa[i][a]) * w;
                                }
                                }   // mixed_on_run
                            } else
                            fb2_dk(Tnext_q, ds_q, NW, dkv, d2kv);
#else
                            OTI Tnext_oti(Tnext_q);
                            for (int p = 0; p < NW; p++) Tnext_oti.set_e(WB(p), ds_q[p]);
                            // powder/bulk blend, same as the first-order block (a) and the forward model (OTI_BLEND2_OFF=1: old reference)
                            double mf2 = 1.0;
                            if (Mv2 && !g_oti_blend2_off) { mf2 = 0.0; for (int a = 0; a < dof; a++) mf2 += shape(a) * Mv2[vdofs[a]]; }
                            OTI k_q = OTI(mf2) * IN625::k_oti(Tnext_oti) + OTI(1.0 - mf2) * IN625::kp_oti(Tnext_oti);
#endif
                            for (int p = 0; p < NW; p++) {
#ifdef USE_ENZYME_AD
                                double dk_dsp = dkv[p], d2k_dsp2 = d2kv[p];
#else
                                double dk_dsp   = k_q.get_e(WB(p));
                                double d2k_dsp2 = 2.0 * k_q.get_epp(WB(p));
#endif
                                if (dk_dsp == 0.0 && d2k_dsp2 == 0.0) continue;
                                for (int a = 0; a < dof; a++) {
                                    double gdot = 0.0, gdotDs = 0.0;
                                    for (int d = 0; d < dm; d++) {
                                        gdot   += gshape_e(a,d)*gT[d];
                                        gdotDs += gshape_e(a,d)*gDs[p][d];
                                    }
                                    #pragma omp atomic
                                    r2ps[p][vdofs[a]] -= (d2k_dsp2*gdot + 2.0*dk_dsp*gdotDs) * w;
                                }
                            }
                        }
                        if (mixed_on_run)
                            for (int pr = 0; pr < NPAIR; pr++) for (int a = 0; a < dof; a++) {
                                const double v = accm[(size_t)pr * dof + a];
                                if (v != 0.0) {
                                    #pragma omp atomic
                                    rmp[pr][vdofs[a]] += v;
                                }
                            }
                    }
                }

                t2_a = now_ms() - t2_0;
                // ---- (b2) boundary, second order ----
                if (emissivity > 0.0) {
                    const double *up = u_iter.HostRead();
                    const double *dsv[NW];
                    for (int p = 0; p < NW; p++) dsv[p] = dTds[p].HostRead();
                    double *b2ps[NW];
                    for (int p = 0; p < NW; p++) b2ps[p] = RHS2[p].HostReadWrite();
                    const double *Mv2b = g_ever_melted ? g_ever_melted->data() : nullptr;
                    const int nqp = (int)bdry_qps.size();
                    #pragma omp parallel for schedule(dynamic, 512)
                    for (int qi = 0; qi < nqp; qi++) {
                        const BdryQP &qp = bdry_qps[qi];
                        double qscale = (qp.kind == 1) ? substrate_loss_scale
                                      : (qp.kind == 2) ? bridge_loss_scale
                                                        : interface_loss_scale;
                        int nd = (int)qp.dof.size();
                        double Tq = 0.0, ds_q[NW] = {0};
                        for (int a = 0; a < nd; a++) {
                            Tq += qp.shp[a] * up[qp.dof[a]];
                            for (int p = 0; p < NW; p++) ds_q[p] += qp.shp[a] * dsv[p][qp.dof[a]];
                        }
#ifdef USE_ENZYME_AD
                        double d2qv[NW];
                        if (use_hess) {
                            double mfb = 1.0;
                            if (Mv2b && !g_noblend2) { mfb = 0.0; for (int a = 0; a < nd; a++) mfb += qp.shp[a] * Mv2b[qp.dof[a]]; }
                            const double xq[4] = {Tq, 1.0, 1.0, 1.0};
                            const double fq[6] = {mfb, qscale * h_conv, qscale * h_conv * IN625::HP_OVER_HC,
                                                  qscale * emissivity * sigma_sb, T_ambient, 0.0};
                            double gq[4], Hq[16];
                            fb_local_hess(1, xq, fq, gq, Hq);
                            double Xp[NACT][4]; int act[NACT]; int nact = 0;
                            for (int p = 0; p < NACT; p++) {
                                Xp[p][0] = ds_q[p]; Xp[p][1] = (p == LI_HC); Xp[p][2] = (p == LI_T0); Xp[p][3] = (p == LI_HP);
                                double d2 = 0.0;
                                for (int u = 0; u < 4; u++) for (int v = 0; v < 4; v++) d2 += Xp[p][u] * Hq[u*4+v] * Xp[p][v];
                                d2qv[p] = d2;
                                if (Xp[p][0] != 0.0 || Xp[p][1] != 0.0 || Xp[p][2] != 0.0 || Xp[p][3] != 0.0) act[nact++] = p;
                            }
                            for (int p = NACT; p < NW; p++) d2qv[p] = 0.0;
                            if (mixed_on_run)
                            for (int ii = 0; ii < nact; ii++) for (int jj = ii + 1; jj < nact; jj++) {
                                const int i = act[ii], j = act[jj];
                                double d2 = 0.0;
                                for (int u = 0; u < 4; u++) for (int v = 0; v < 4; v++) d2 += Xp[i][u] * Hq[u*4+v] * Xp[j][v];
                                if (d2 == 0.0) continue;
                                const int pr = g_pair_index[i][j];
                                for (int a = 0; a < nd; a++) {
                                    #pragma omp atomic
                                    rmp[pr][qp.dof[a]] -= d2 * qp.shp[a] * qp.w;
                                }
                            }
                        } else
                        fb2_q(Tq, ds_q, NW, qscale * h_conv, qscale * emissivity * sigma_sb, T_ambient, d2qv);
#else
                        OTI Tq_oti(Tq);
                        for (int p = 0; p < NW; p++) Tq_oti.set_e(WB(p), ds_q[p]);
                        double mfb2 = 1.0;
                        if (Mv2b && !g_oti_blend2_off) { mfb2 = 0.0; for (int a = 0; a < nd; a++) mfb2 += qp.shp[a] * Mv2b[qp.dof[a]]; }
                        OTI hconv_oti = OTI(mfb2) * (OTI(qscale * h_conv) * IN625::S_HC)
                                      + OTI(1.0 - mfb2) * (OTI(qscale * h_conv * IN625::HP_OVER_HC) * IN625::S_HP);
                        OTI Tinf_oti  = OTI(T_ambient) * IN625::S_T0;
                        OTI qval_oti = hconv_oti * (Tq_oti - Tinf_oti)
                                     + OTI(qscale * emissivity * sigma_sb)
                                       * (Tq_oti*Tq_oti*Tq_oti*Tq_oti - Tinf_oti*Tinf_oti*Tinf_oti*Tinf_oti);
#endif
                        for (int p = 0; p < NW; p++) {
#ifdef USE_ENZYME_AD
                            double d2q_dsp2 = d2qv[p];
#else
                            double d2q_dsp2 = 2.0 * qval_oti.get_epp(WB(p));
#endif
                            if (d2q_dsp2 == 0.0) continue;
                            for (int a = 0; a < nd; a++) {
                                #pragma omp atomic
                                b2ps[p][qp.dof[a]] -= d2q_dsp2 * qp.shp[a] * qp.w;
                            }
                        }
                    }
                }

                t2_b = now_ms() - t2_0 - t2_a;
                // ---- (c2) nodal capacitive+latent term, second order ----
                {
                    double *rp[NW];
                    for (int p = 0; p < NW; p++) rp[p] = RHS2[p].HostReadWrite();
                    const double *Tnext_p = u_iter.HostRead();
                    const double *Tn_p = u_n.HostRead();
                    const double *Vp = V_node.HostRead();
                    const double *dsv2[NW], *pderv[NW], *pder2v[NW];
                    for (int p = 0; p < NW; p++) {
                        dsv2[p] = dTds[p].HostRead();
                        pderv[p] = prev_der[p].HostRead();
                        pder2v[p] = prev_der2[p].HostRead();
                    }
                    double idt_sens2 = 1.0 / dt_try;
#ifdef USE_ENZYME_AD
                    const double *Mv_c2 = ever_melted.data();
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < ndof; i++) {
                        double ds_i[NW], pd_i[NW], pd2_i[NW], r2_i[NW];
                        for (int p = 0; p < NW; p++) { ds_i[p] = dsv2[p][i]; pd_i[p] = pderv[p][i]; pd2_i[p] = pder2v[p][i]; }
                        if (use_hess) {
                            const double Mvn = g_noblend2 ? 1.0 : Mv_c2[i];
                            const double xn[8] = {Tnext_p[i], Tn_p[i], 1, 1, 1, 1, 1, 1};
                            const double fn[6] = {Mvn, IN625::RHOP_FRACTION, Vp[i], idt_sens2, IN625::T_solidus, IN625::T_liquidus};
                            double gn[8], Hn[64];
                            fb_local_hess(2, xn, fn, gn, Hn);
                            double X[NACT][8]; int act[NACT]; int nact = 0;
                            for (int p = 0; p < NACT; p++) {
                                X[p][0] = ds_i[p]; X[p][1] = pd_i[p];
                                X[p][2] = (p == LI_CP); X[p][3] = (p == LI_RHO); X[p][4] = (p == LI_LFUS);
                                X[p][5] = (p == LI_TS); X[p][6] = (p == LI_TLIQ); X[p][7] = (p == LI_RHOP);
                                bool nz = false; for (int u = 0; u < 8; u++) if (X[p][u] != 0.0) { nz = true; break; }
                                if (nz) act[nact++] = p;
                            }
                            double HX[NACT][8];
                            for (int ii = 0; ii < nact; ii++) { const int p = act[ii];
                                for (int u = 0; u < 8; u++) { double a_ = 0.0; for (int v = 0; v < 8; v++) a_ += Hn[u*8+v] * X[p][v]; HX[p][u] = a_; } }
                            for (int p = 0; p < NW; p++) r2_i[p] = 0.0;
                            for (int ii = 0; ii < nact; ii++) { const int p = act[ii];
                                double d2 = gn[1] * pd2_i[p];
                                for (int u = 0; u < 8; u++) d2 += X[p][u] * HX[p][u];
                                r2_i[p] = -d2; }
                            if (mixed_on_run)
                            for (int ii = 0; ii < nact; ii++) for (int jj = ii + 1; jj < nact; jj++) {
                                const int a_ = act[ii], b_ = act[jj];
                                const int pr = g_pair_index[a_][b_];
                                double d2 = gn[1] * pd2mv[pr][i];
                                for (int u = 0; u < 8; u++) d2 += X[a_][u] * HX[b_][u];
                                rmp[pr][i] += -d2;
                            }
                        } else
                        fb2_nodal(Tnext_p[i], Tn_p[i], pd_i, pd2_i, ds_i, NW, Vp[i], idt_sens2,
                                  IN625::T_solidus, IN625::T_liquidus, r2_i);
                        for (int p = 0; p < NW; p++) rp[p][i] += r2_i[p];
                    }
#else
                    Kokkos::parallel_for("block_c2_nodal_sens", ndof, KOKKOS_LAMBDA(int i) {
                        OTI Tnext_i(Tnext_p[i]);
                        for (int p = 0; p < NW; p++) Tnext_i.set_e(WB(p), dsv2[p][i]);
                        OTI Tn_i(Tn_p[i]);
                        for (int p = 0; p < NW; p++) {
                            Tn_i.set_e(WB(p), pderv[p][i]);
                            Tn_i.set_epp(WB(p), 0.5*pder2v[p][i]);
                        }
                        OTI Vi(Vp[i]);
                        const double Mvo = g_oti_blend2_off ? 1.0 : ever_melted[i];
                        OTI rho_i2 = OTI(Mvo) * IN625::rho_oti(Tnext_i) + OTI(1.0 - Mvo) * IN625::rhop_oti(Tnext_i);   // same blend as block (c)
                        OTI Mc_i = Vi * rho_i2 * IN625::cp_oti(Tnext_i);
                        OTI L_i  = Vi * rho_i2 * IN625::Lfusion_oti() * IN625::f_L_oti(Tnext_i);
                        OTI Ln_i = Vi * rho_i2 * IN625::Lfusion_oti() * IN625::f_L_oti(Tn_i);
                        OTI nodal = Mc_i * OTI(idt_sens2) * (Tnext_i - Tn_i) + (L_i - Ln_i) * OTI(idt_sens2);
                        for (int p = 0; p < NW; p++) rp[p][i] += -2.0 * nodal.get_epp(WB(p));
                    });
#endif
                }

                // ---- (e2) explicit source 2nd-derivative for rx,ry,rz ----
                // x0/Vx/tL left at exactly 0 (documented approximation,
                // matching the reference driver's own choice): x0 is EXACT
                // since f_src restricted to a fixed axis-aligned trilinear
                // H1 hex is linear in x, so d^2/dx0^2=0 identically; Vx/tL's
                // small along-path mixed second derivatives are neglected,
                // a deferred refinement (see block (e)'s own comment).
                if (NW > LI_RX) {
                    double *r_rx = RHS2[LI_RX].HostReadWrite();
                    const double *s2rx = src2_fd_rx.HostRead();
                    for (int i = 0; i < ndof; i++) r_rx[i] += s2rx[i];
                    if (NW > LI_RY) {
                        double *r_ry = RHS2[LI_RY].HostReadWrite();
                        const double *s2ry = src2_fd_ry.HostRead();
                        for (int i = 0; i < ndof; i++) r_ry[i] += s2ry[i];
                    }
                    if (NW > LI_RZ) {
                        double *r_rz = RHS2[LI_RZ].HostReadWrite();
                        const double *s2rz = src2_fd_rz.HostRead();
                        for (int i = 0; i < ndof; i++) r_rz[i] += s2rz[i];
                    }
                }

                t2_c = now_ms() - t2_0 - t2_a - t2_b;
                if (!g_src_fd_legacy && NW > LI_RX) {
                    // AD version: explicit d2f/dp2 for x0, Vx, tL too (the FD version defined these as 0)
                    const Vector *s2x[3] = {&src2_x0, &src2_vx, &src2_tl};
                    const int lix[3] = {LI_X0, LI_VX, LI_TL};
                    for (int q = 0; q < 3; q++) if (NW > lix[q] && s2x[q]->Size() == ndof) {
                        double *rr = RHS2[lix[q]].HostReadWrite(); const double *sv = s2x[q]->HostRead();
                        for (int i = 0; i < ndof; i++) rr[i] += sv[i];
                    }
                }
                if (mixed_on_run) {   // explicit mixed source terms: f = f0 (1+s_P)(1+s_eta) g(q)
                    MFEM_VERIFY((int)srcHkeep.size() == 15 && (int)srcD1keep.size() == 6, "SENS_MIXED: AD source tape missing");
                    auto addv = [&](int i, int j, const Vector &v) { double *rr = rmp[g_pair_index[i][j]]; const double *vv = v.HostRead();
                                                                     for (int n = 0; n < ndof; n++) rr[n] += vv[n]; };
                    addv(LI_P, LI_ETA, f_src);
                    for (int q = 0; q < 6; q++) { addv(LI_P, LI_RX + q, srcD1keep[q]); addv(LI_ETA, LI_RX + q, srcD1keep[q]); }
                    int k15 = 0;
                    for (int a = 0; a < 6; a++) for (int b = a + 1; b < 6; b++, k15++) addv(LI_RX + a, LI_RX + b, srcHkeep[k15]);
                    for (int k = 0; k < NPAIR; k++) for (int q = 0; q < ess_tdof_list.Size(); q++) rmp[k][ess_tdof_list[q]] = 0.0;
                }

                // Bottom substrate / not-yet-active rows are parameter independent.
                for (int p = 0; p < NW; p++) {
                    double *rp = RHS2[p].HostReadWrite();
                    for (int q = 0; q < ess_tdof_list.Size(); q++) rp[ess_tdof_list[q]] = 0.0;
                }

                t2_e = now_ms() - t2_0 - t2_a - t2_b - t2_c;
                const double t2_s0 = now_ms();
                // Reuse the SAME Jop/solver setup as the first-order solve
                // above -- J is unchanged (same converged Newton Jacobian).
                // Rebuilding fresh solver wrapper objects here (rather than
                // keeping the first-order ones in scope) is cheap: the
                // underlying PARDISO factorization is cached beneath these
                // wrappers keyed on the operator itself (already proven by
                // the first-order loop above rebuilding fresh every time
                // step without a measured perf regression).
                if (!tie_active) {
                    FachinottiLinearSolver lin_solver_sens2(Kd_pa, Kb_diag, Meff_over_dt, ess_tdof_list, ndof);
                    lin_solver_sens2.SetOperator(J_full_sens);
                    GMRESSolver gmres_sens2;
                    gmres_sens2.SetOperator(J_corr_sens);
                    gmres_sens2.SetPreconditioner(lin_solver_sens2);
                    gmres_sens2.SetRelTol(1e-8); gmres_sens2.SetAbsTol(0.0);
                    gmres_sens2.SetMaxIter(200); gmres_sens2.SetKDim(50); gmres_sens2.SetPrintLevel(-1);
                    for (int p = 0; p < NW; p++) { dT2ds[p] = 0.0; gmres_sens2.Mult(RHS2[p], dT2ds[p]); }
                    if (mixed_on_run) for (int k = 0; k < NPAIR; k++) { dT2m[k] = 0.0; if (RHSm[k].Normlinf() != 0.0) gmres_sens2.Mult(RHSm[k], dT2m[k]); }
                } else if (sens_direct_on) {
                    // same exact factorisation as the first-order solve of this step
                    std::vector<const Vector*> Bp; std::vector<Vector*> Xp;
                    for (int p = 0; p < NW; p++) {
                        dT2ds[p] = 0.0;
                        if (RHS2[p].Normlinf() == 0.0) continue;
                        Bp.push_back(&RHS2[p]); Xp.push_back(&dT2ds[p]);
                    }
                    if (mixed_on_run) for (int k = 0; k < NPAIR; k++) {
                        dT2m[k] = 0.0;
                        if (RHSm[k].Normlinf() == 0.0) continue;
                        Bp.push_back(&RHSm[k]); Xp.push_back(&dT2m[k]);
                    }
                    if (!Bp.empty()) ao_ex->SolveFull(Bp, Xp);
                } else {
                    ReducedOpWrapper reduced_J_sens2(J_full_sens, tie);
                    ReducedOpWrapper reduced_J_corr_sens2(J_corr_sens, tie);
                    ReducedLinearSolver cg_sens2(Kd_pa, Kb_diag, Meff_over_dt, tie);
                    // v6: the second-order solve uses the SAME base Jacobian as the first-order one; v5 built a brand-new
                    // PARDISO object (full analysis + factorisation) for it every step. Reuse the first-order solver's
                    // factorisation unless SENS2_OWN=1.
                    static const bool sens2_own = (std::getenv("SENS2_OWN") != nullptr);
                    const bool share2 = !sens2_own && pard_persist_on && pardiso_sens_persist && string(lin_solver_kind) == "pardiso";
                    std::unique_ptr<PardisoReducedSolver> pardiso_sens2_own;
                    if (!share2) pardiso_sens2_own.reset(new PardisoReducedSolver(tie));
                    Solver *precond_solver2 = (string(lin_solver_kind) != "pardiso") ? static_cast<Solver*>(&cg_sens2)
                                            : share2 ? static_cast<Solver*>(pardiso_sens_persist.get())
                                                     : static_cast<Solver*>(pardiso_sens2_own.get());
                    if (!share2) precond_solver2->SetOperator(reduced_J_sens2);
                    GMRESSolver gmres_sens2;
                    gmres_sens2.SetOperator(reduced_J_corr_sens2);
                    gmres_sens2.SetPreconditioner(*precond_solver2);
                    gmres_sens2.SetRelTol(1e-8); gmres_sens2.SetAbsTol(0.0);
                    gmres_sens2.SetMaxIter(200); gmres_sens2.SetKDim(50); gmres_sens2.SetPrintLevel(-1);
                    Vector RHS2_free(tie.ndof_free), dT2ds_free(tie.ndof_free);
                    for (int p = 0; p < NW; p++) {
                        tie.Contract(RHS2[p], RHS2_free);
                        dT2ds_free = 0.0;
                        gmres_sens2.Mult(RHS2_free, dT2ds_free);
                        tie.Expand(dT2ds_free, dT2ds[p]);
                    }
                    if (mixed_on_run) for (int k = 0; k < NPAIR; k++) {
                        if (RHSm[k].Normlinf() == 0.0) { dT2m[k] = 0.0; continue; }
                        tie.Contract(RHSm[k], RHS2_free);
                        dT2ds_free = 0.0;
                        gmres_sens2.Mult(RHS2_free, dT2ds_free);
                        tie.Expand(dT2ds_free, dT2m[k]);
                    }
                }
                t2_s = now_ms() - t2_s0;
                cout << "  [SENS2 timers ms] a2=" << (long)t2_a << " b2=" << (long)t2_b << " c2=" << (long)t2_c
                     << " src=" << (long)t2_e << " solve=" << (long)t2_s << (mixed_on_run ? "  (mixed)" : "") << "\n";
            }
            // ---- DEBUG (2026-09-11): the DECISIVE Jacobian check. Previous
            // JAC_CHECK/JAC_CHECK2 only tested J on STRUCTURED vectors
            // (uniform bump, linear-in-x ramp); TRUE_resid_norm above only
            // checks J*dTds against RHS as MY OWN CODE computes both (a
            // self-consistency check, not an independent verification). This
            // is the missing piece: an INDEPENDENT finite-difference of the
            // residual in the EXACT direction of the just-computed dTds[k]
            // (a spatially-complex, localized vector, unlike the simple
            // structured test vectors), compared against J's analytic action
            // on that SAME vector.
            if (jac_check) {
                int pd0 = probe_dof[0];
                double dmax = dTds[LI_K].Normlinf();
                double scale = (dmax > 1e-12) ? (0.01 / dmax) : 1.0;
                cout << "  [JAC_CHECK3 SCALE] dmax=" << dmax << " scale=" << scale << "\n" << flush;
                Vector u_pert3(ndof);
                { const double *up = u_iter.HostRead(); const double *dp = dTds[LI_K].HostRead();
                  double *upp = u_pert3.HostWrite();
                  for (int i = 0; i < ndof; i++) upp[i] = up[i] + scale*dp[i]; }
                Vector r0b(ndof), r3(ndof);
                residual.Mult(u_iter, r0b);
                // residual.Mult() above just triggered EnsureUpdated's
                // rebuild via the buggy "SpMat()=0;Assemble(0)" reuse cycle
                // again (kpa_dynamic already non-null) -- re-apply the fresh
                // rebuild so THIS diagnostic actually tests the fix, not the
                // old bug. Meff_over_dt itself is correctly refreshed to
                // u_iter by the residual.Mult(u_iter,...) call just above
                // (that part isn't touched by RebuildConductivityFresh*, so
                // no ordering issue with it here).
                residual.RebuildConductivityFreshForSensitivity(u_iter);
                Vector analytic_J_dTds(ndof), analytic_J_dTds_base(ndof);
                J_corr_sens.Mult(dTds[LI_K], analytic_J_dTds);
                J_base_sens.Mult(dTds[LI_K], analytic_J_dTds_base);
                residual.Mult(u_pert3, r3);
                Vector fd_J_dTds(ndof);
                { const double *r0p=r0b.HostRead(), *r3p=r3.HostRead(); double *fp=fd_J_dTds.HostWrite();
                  for (int i=0;i<ndof;i++) fp[i] = (r3p[i]-r0p[i])/scale; }
                cout << "  [JAC_CHECK3] probe (delta=dTds[k], the ACTUAL solution vector): "
                     << "FD_Jv=" << fd_J_dTds[pd0]
                     << "  analytic_Jv(corrected)=" << analytic_J_dTds[pd0]
                     << "  analytic_Jv(BASE, no dk/dT correction)=" << analytic_J_dTds_base[pd0]
                     << "  |FD_Jv full|=" << fd_J_dTds.Norml2()
                     << "  |analytic_Jv full|=" << analytic_J_dTds.Norml2()
                     << "  |analytic_Jv_BASE full|=" << analytic_J_dTds_base.Norml2() << "\n" << flush;

                // Find WHERE the two vectors diverge most, to localize the bug.
                int worst_i = -1;  // hoisted out of the block below -- also used by the JAC_ROW matrix-entry check further down
                {
                    const double *fp = fd_J_dTds.HostRead();
                    const double *ap = analytic_J_dTds.HostRead();
                    double worst_d = -1.0;
                    double sum_d2 = 0.0;
                    for (int i = 0; i < ndof; i++) {
                        double d = fabs(fp[i] - ap[i]);
                        sum_d2 += d*d;
                        if (d > worst_d) { worst_d = d; worst_i = i; }
                    }
                    double *c = mesh.GetVertex(worst_i);
                    static const double k_bp[11] = {294.0,311.0,366.0,477.0,589.0,700.0,811.0,922.0,1033.0,1144.0,1255.0};
                    double T_here = u_iter[worst_i];
                    double nearest_bp = k_bp[0]; double bp_dist = fabs(T_here - k_bp[0]);
                    for (int kk = 1; kk < 11; kk++) { double d2 = fabs(T_here - k_bp[kk]); if (d2 < bp_dist) { bp_dist = d2; nearest_bp = k_bp[kk]; } }
                    cout << "  [JAC_CHECK3 LOCATE] max|diff|=" << worst_d
                         << " at dof=" << worst_i
                         << " coord=(" << c[0] << "," << c[1] << "," << c[2] << ")"
                         << " FD=" << fp[worst_i] << " analytic(corrected)=" << ap[worst_i]
                         << " analytic(BASE, no correction)=" << analytic_J_dTds_base[worst_i]
                         << " r0b_here=" << r0b[worst_i] << " r3_here=" << r3[worst_i]
                         << " raw_delta_T_here=" << scale*dTds[LI_K][worst_i]
                         << " T_here=" << T_here << " dTds_k_here=" << dTds[LI_K][worst_i]
                         << " nearest_k_breakpoint=" << nearest_bp << " dist=" << bp_dist
                         << " node_active=" << (node_active[worst_i] ? 1 : 0)
                         << " is_bc=" << (is_bc[worst_i] ? 1 : 0)
                         << " is_slave=" << (tie.is_slave[worst_i] ? 1 : 0)
                         << " is_ess=" << (std::find(ess_tdof_list.begin(), ess_tdof_list.end(), worst_i) != ess_tdof_list.end() ? 1 : 0)
                         << "  |diff|_L2=" << sqrt(sum_d2) << "\n" << flush;
                    // Also show the top-5 elements touching this dof and their attributes.
                    Array<int> vdofs;
                    for (int e = 0; e < mesh.GetNE(); e++) {
                        fespace.GetElementDofs(e, vdofs);
                        bool touches = false;
                        for (int a = 0; a < vdofs.Size(); a++) if (vdofs[a] == worst_i) { touches = true; break; }
                        if (touches) cout << "    touched by element " << e << " attribute=" << mesh.GetAttribute(e) << "\n";
                    }
                }

                // ---- DECISIVE: assemble the actual BASE Jacobian matrix
                // and test EVERY individual entry in worst_i's own row
                // against a per-column finite difference of the residual.
                // This pinpoints EXACTLY which coupling term (which
                // neighbor dof) is wrong, instead of only seeing the
                // aggregate matrix-vector-product error.
                {
                    SparseMatrix *Jmat = J_base_sens.AssembleFull();
                    const int *Iptr = Jmat->GetI();
                    const int *Jcol = Jmat->GetJ();
                    const double *Dval = Jmat->GetData();
                    int rstart = Iptr[worst_i], rend = Iptr[worst_i+1];
                    cout << "  [JAC_ROW] row=" << worst_i << " nnz=" << (rend-rstart) << "\n";
                    double h_col = 0.01;
                    Vector u_col_pert(u_iter);
                    for (int idx = rstart; idx < rend; idx++) {
                        int col = Jcol[idx];
                        double val = Dval[idx];
                        u_col_pert = u_iter;
                        u_col_pert[col] += h_col;
                        Vector r_col(ndof);
                        residual.Mult(u_col_pert, r_col);
                        double fd_entry = (r_col[worst_i] - r0b[worst_i]) / h_col;
                        double diff = val - fd_entry;
                        double *cc = mesh.GetVertex(col);
                        int attr_col = -1;
                        Array<int> vd3;
                        for (int e = 0; e < mesh.GetNE() && attr_col < 0; e++) {
                            fespace.GetElementDofs(e, vd3);
                            for (int a = 0; a < vd3.Size(); a++) if (vd3[a] == col) { attr_col = mesh.GetAttribute(e); break; }
                        }
                        cout << "    col=" << col << " analytic=" << val << " FD=" << fd_entry
                             << " diff=" << diff
                             << " coord=(" << cc[0] << "," << cc[1] << "," << cc[2] << ")"
                             << " attr=" << attr_col
                             << (col == worst_i ? "  <-- DIAGONAL" : "") << "\n";
                    }
                    delete Jmat;

                    // Is the diagonal +1 error specific to AssembleFull()
                    // (the PARDISO-only matrix-assembly path), or does the
                    // SAME error show up via Mult() (the matrix-free path
                    // GMRES/CG actually use throughout every -solver cg
                    // test so far)? Test via a unit-vector Mult().
                    Vector e_diag(ndof); e_diag = 0.0; e_diag[worst_i] = 1.0;
                    Vector Je_diag(ndof);
                    J_base_sens.Mult(e_diag, Je_diag);
                    cout << "  [JAC_DIAG_MULT_CHECK] diagonal via Mult()=" << Je_diag[worst_i]
                         << "  (vs AssembleFull()'s value, vs true FD)\n" << flush;
                    // Split the diagonal into its 3 additive pieces: domain
                    // (kd), boundary (jb), nodal capacitance (Meff_over_dt).
                    // Meff_over_dt is directly accessible; kd+jb = the rest.
                    double md_here = Meff_over_dt[worst_i];
                    cout << "  [JAC_DIAG_SPLIT] Meff_over_dt(nodal)=" << md_here
                         << "  domain+boundary(kd+jb)=" << (Je_diag[worst_i] - md_here) << "\n" << flush;
                    // Compare against a few NEIGHBORING dofs' own Meff_over_dt
                    // (structurally similar, same T-range) to see if 41928's
                    // own value is itself the outlier.
                    for (int nb : {41927, 41929, 41926, 41960, 41958}) {
                        cout << "    Meff_over_dt[" << nb << "]=" << Meff_over_dt[nb]
                             << "  T=" << u_iter[nb] << "\n";
                    }
                    cout << flush;
                }

                // ---- ELEMENT-LOCAL: compute each of the 4 elements' own
                // local stiffness contribution to worst_i's diagonal
                // directly via DiffusionIntegrator::AssembleElementMatrix,
                // bypassing global assembly entirely. If one element's
                // local value is itself anomalous, that pinpoints the
                // element/quadrature; if all 4 look normal but sum wrong,
                // the bug is in global assembly instead.
                {
                    GridFunction u_iter_gf(&fespace); u_iter_gf = u_iter;
                    ActiveTDepKCoefficient k_coeff_test(u_iter_gf, attr_active);
                    DiffusionIntegrator diff_test(k_coeff_test);
                    int check_elems[4] = {38265, 38266, 39015, 39016};
                    double sum_local = 0.0;
                    for (int ce : check_elems) {
                        const FiniteElement &fe = *fespace.GetFE(ce);
                        ElementTransformation *Trans = fespace.GetElementTransformation(ce);
                        DenseMatrix elmat;
                        diff_test.AssembleElementMatrix(fe, *Trans, elmat);
                        Array<int> vdofs_e;
                        fespace.GetElementDofs(ce, vdofs_e);
                        int local_idx = -1;
                        for (int a = 0; a < vdofs_e.Size(); a++) if (vdofs_e[a] == worst_i) { local_idx = a; break; }
                        double contrib = (local_idx >= 0) ? elmat(local_idx, local_idx) : 0.0;
                        sum_local += contrib;
                        cout << "  [ELEM_LOCAL] elem=" << ce << " local_idx=" << local_idx
                             << " elmat(local,local)=" << contrib << "\n";
                    }
                    cout << "  [ELEM_LOCAL] SUM of 4 elements' own local diagonal contributions=" << sum_local
                         << "  (compare to global kd assembled value)\n" << flush;
                }

                residual.GetGradient(u_iter);  // restore state

                // Is dTds[k]'s own global max PHYSICALLY plausible (near the
                // laser/row1) or a spurious spike far from the heat source?
                // Print the top-10 |dTds_k| locations to check.
                {
                    std::vector<std::pair<double,int>> ranked(ndof);
                    const double *dp = dTds[LI_K].HostRead();
                    for (int i = 0; i < ndof; i++) ranked[i] = {fabs(dp[i]), i};
                    std::partial_sort(ranked.begin(), ranked.begin()+std::min(10,ndof), ranked.end(),
                                       [](auto &a, auto &b){ return a.first > b.first; });
                    cout << "  [JAC_CHECK3 TOP10 |dTds_k|]:\n";
                    for (int r = 0; r < std::min(10,ndof); r++) {
                        int idx = ranked[r].second;
                        double *cc = mesh.GetVertex(idx);
                        int attr_here = -1;
                        Array<int> vd2;
                        for (int e = 0; e < mesh.GetNE() && attr_here < 0; e++) {
                            fespace.GetElementDofs(e, vd2);
                            for (int a = 0; a < vd2.Size(); a++) if (vd2[a] == idx) { attr_here = mesh.GetAttribute(e); break; }
                        }
                        cout << "    #" << r << " dof=" << idx << " |dTds_k|=" << ranked[r].first
                             << " coord=(" << cc[0] << "," << cc[1] << "," << cc[2] << ")"
                             << " attr=" << attr_here << "\n";
                    }
                    cout << flush;
                }
            }

            for (int p = 0; p < NW; p++) prev_der[p] = dTds[p];
            if (const char *dd = getenv("DUMP_DIR")) {
                static int dstep = 0;
                std::string fn = std::string(dd) + "/d1_" + std::to_string(dstep++) + ".bin";
                std::ofstream of(fn, std::ios::binary);
                of.write(reinterpret_cast<const char*>(dTds[LI_K].HostRead()), sizeof(double) * ndof);
                std::ofstream ot(std::string(dd) + "/T_" + std::to_string(dstep - 1) + ".bin", std::ios::binary);
                ot.write(reinterpret_cast<const char*>(u_iter.HostRead()), sizeof(double) * ndof);
            }
            if (derivative_order >= 2) for (int p = 0; p < NW; p++) prev_der2[p] = dT2ds[p];
            if (mixed_on_run) for (int k = 0; k < NPAIR; k++) prev_der2m[k] = dT2m[k];

            fsens << scientific << setprecision(15) << (t + dt_try);   // v6: full double precision (was 6 -> 7 significant digits, which hid differences below ~1e-7)
            for (int r = 0; r < n_rows; r++)
                fsens << "," << ((probe_dof[r] >= 0) ? dTds[LI_K][probe_dof[r]] : 0.0);
            fsens << "," << ((probe_dof[0] >= 0) ? dTds[LI_CP][probe_dof[0]] : 0.0)
                  << "," << ((probe_dof[0] >= 0) ? dTds[LI_RHO][probe_dof[0]] : 0.0)
                  << "," << ((probe_dof[0] >= 0) ? dTds[LI_LFUS][probe_dof[0]] : 0.0);
            if (NW > LI_T0)
                fsens << "," << ((probe_dof[0] >= 0) ? dTds[LI_TS][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_TLIQ][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_HC][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_T0][probe_dof[0]] : 0.0);
            if (NW > LI_ETA)
                fsens << "," << ((probe_dof[0] >= 0) ? dTds[LI_P][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_ETA][probe_dof[0]] : 0.0);
            if (NW > LI_RX)
                fsens << "," << ((probe_dof[0] >= 0) ? dTds[LI_RX][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_RY][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_RZ][probe_dof[0]] : 0.0);
            if (NW > LI_X0)
                fsens << "," << ((probe_dof[0] >= 0) ? dTds[LI_X0][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_VX][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_TL][probe_dof[0]] : 0.0);
            // 2026-10-02: y0/Vy/lL/T0p -- not wired (see the header comment above), exact 0.0 every step.
            if (NW > LI_X0) fsens << ",0,0,0,0";
            // v8: Tw (real, wired above) + the 9 not-yet-implemented params -- read dTds directly (see
            // the header comment: this reports real 0.0 now via the solve, not literal text).
            if (NW > LI_W)
                fsens << "," << ((probe_dof[0] >= 0) ? dTds[LI_TW][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_HP][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_KP][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_RHOP][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_BP][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_LP][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_WP][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_B][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_L][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dTds[LI_W][probe_dof[0]] : 0.0);
            // A3 (2026-09-12): second-order (diagonal Hessian) columns,
            // same layout/order as the first-order block above.
            if (derivative_order >= 2) {
                for (int r = 0; r < n_rows; r++)
                    fsens << "," << ((probe_dof[r] >= 0) ? dT2ds[LI_K][probe_dof[r]] : 0.0);
                fsens << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_CP][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_RHO][probe_dof[0]] : 0.0)
                      << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_LFUS][probe_dof[0]] : 0.0);
                if (NW > LI_T0)
                    fsens << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_TS][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_TLIQ][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_HC][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_T0][probe_dof[0]] : 0.0);
                if (NW > LI_ETA)
                    fsens << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_P][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_ETA][probe_dof[0]] : 0.0);
                if (NW > LI_RX)
                    fsens << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_RX][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_RY][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_RZ][probe_dof[0]] : 0.0);
                if (NW > LI_X0)
                    fsens << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_X0][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_VX][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_TL][probe_dof[0]] : 0.0);
                // 2026-10-02: y0/Vy/lL/T0p second derivatives -- not wired, exact 0.0 every step.
                if (NW > LI_X0) fsens << ",0,0,0,0";
                if (NW > LI_W)
                    fsens << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_TW][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_HP][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_KP][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_RHOP][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_BP][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_LP][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_WP][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_B][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_L][probe_dof[0]] : 0.0)
                          << "," << ((probe_dof[0] >= 0) ? dT2ds[LI_W][probe_dof[0]] : 0.0);
            }
            fsens << "\n";
            fsens.flush();
            if (mixed_on_run) {
                fhess << t + dt_try;
                const int pdof = probe_dof[0];
                for (int i = 0; i < NACT; i++) for (int j = i; j < NACT; j++) {
                    double v = 0.0;
                    if (pdof >= 0) v = (i == j) ? dT2ds[i][pdof] : dT2m[g_pair_index[i][j]][pdof];
                    fhess << "," << v;
                }
                fhess << "\n"; fhess.flush();
            }
        }

        t += dt_try;
        u_gf = u_iter;
        L_n = L_now;

        if (pvdc && (step % std::max(1, pv_every) == 0)) {
            update_active_gf();
            pvdc->SetCycle(step);
            pvdc->SetTime(t);
            pvdc->Save();
        }

        // Full-bridge port (2026-09-12): the ancestor's Tp[4] fixed-size
        // array is a stack buffer OVERREAD once n_rows > 4 (the CSV-write
        // loop below iterates r < n_rows) -- 63 rows on this mesh. Sized to
        // n_rows; the console print further below still only shows the
        // first 4 (n_rows is always >= 4 on every mesh this driver targets).
        std::vector<double> Tp(n_rows, 0.0);
        for (int r = 0; r < n_rows; r++)
            Tp[r] = (probe_dof[r]>=0) ? u_gf[probe_dof[r]] : 0.0;
        double Tmax = u_gf.Max();
        double Tsub = (sub_probe_dof>=0) ? u_gf[sub_probe_dof] : 0.0;
        fprobe << scientific << setprecision(12) << t;
        for (int r = 0; r < n_rows; r++) fprobe << "," << Tp[r];
        fprobe << "," << Tmax << "," << Tsub << "\n";
        fprobe.flush();

        if (step % print_freq == 0) {
            double xL=0,yL=0,zL=0; laser.position(t, xL, yL, zL);
            bool act = laser.is_active(t);
            cout<<"  Step "<<setw(5)<<step<<"  t="<<fixed<<setprecision(2)<<t
                <<"  T_max="<<setprecision(1)<<Tmax<<" K"
                <<"  P1-4="<<setprecision(0)<<Tp[0]<<"/"<<Tp[1]<<"/"<<Tp[2]<<"/"<<Tp[3]<<" K"
                <<"  Tsub="<<setprecision(1)<<Tsub<<" K"
                <<"  newt="<<newt_iters_done
                <<"  x="<<setprecision(2)<<xL<<" z="<<zL
                <<(act?" [ON]":" [off]")
                <<"  wall="<<std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now()-_step_t0).count()<<"ms"
                <<"\n"<<flush;
        }
        {
            g_tm.t_tie = g_tie_ms;
            double sw = std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now()-_step_t0).count();
            g_tm.report(sw);
#ifdef USE_HYPRE_AMG
            if (g_amg.calls) { printf("  [amg] calls=%ld setups=%ld cg_its=%ld fallbacks=%ld | build=%.0fms setup=%.0fms solve=%.0fms\n",
                                      g_amg.calls, g_amg.setups, g_amg.iters, g_amg.fallbacks, g_amg.t_build, g_amg.t_setup, g_amg.t_solve); fflush(stdout); g_amg.reset(); }
#endif
            printf("  [timers2 ms] kbuild=%.0f krefill=%.0f kfresh=%.0f | sens: a=%.0f b=%.0f c=%.0f d=%.0f e=%.0f f=%.0f prep=%.0f fresh=%.0f solve=%.0f\n",
                   g_kbuild_ms, g_krefill_ms, g_kfresh_ms, g_tk[1], g_tk[2], g_tk[3], g_tk[4], g_tk[5], g_tk[6], 0.0, g_tk[7], g_tk[8]);
            fflush(stdout);
            g_kbuild_ms = g_krefill_ms = g_kfresh_ms = 0; for (int i = 0; i < 10; i++) g_tk[i] = 0;
        }
        if (!isfinite(Tmax) || Tmax > 1e6) {
            cerr << "ERROR: non-finite or runaway T_max=" << Tmax << " at step " << step << ", t=" << t << ". Aborting.\n";
            write_ckpt();
            return 1;
        }
        if (ckpt_every > 0 && step % ckpt_every == 0) write_ckpt();
    }

    write_ckpt();
    delete k_pa_diff;
    delete pvdc;
    fprobe.close();
    if (fsens.is_open()) fsens.close();

    auto tw1=chrono::high_resolution_clock::now();
    double ws=chrono::duration<double>(tw1-t_wall_start).count();
    cout<<"\n=== BRIDGE multilayer complete ===\n"
        <<"  Accepted steps : "<<step<<"\n"
        <<"  Newton iters   : "<<n_newt_tot<<" (avg "<<fixed<<setprecision(2)
        <<(double)n_newt_tot/max(step,1)<<" / step)\n"
        <<"  Wall time      : "<<ws<<" s ("<<ws/60.0<<" min)\n"
        <<"  Output: "<<probe_file<<"\n";
    return 0;
}
