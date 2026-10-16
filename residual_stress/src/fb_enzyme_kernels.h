// ============================================================================
//  Enzyme forward-mode kernels for the full-bridge single-layer driver
//  (full_bridge_activation_tier1_oti.cpp, built with -DUSE_ENZYME_AD).
//  They replace the OTI arithmetic of blocks (a),(b),(c) [first order] and
//  (a2),(b2),(c2) [second order].  Plain-double kernels, differentiated with
//  Enzyme (clang plugin); the driver itself stays on g++.
//
//  Scale slots of the first-order kernels (11): 0 k, 1 cp, 2 rho, 3 Lfus, 4 Ts,
//  5 Tl, 6 hc, 7 T0, 8 kp, 9 rho_p, 10 hp  (all unit multiplicative scales).
//  The driver maps its LI_* direction index p to a slot (or -1: no explicit
//  scale dependence, the direction enters only through propagated sensitivities).
//  Second-order kernels follow the reference OTI blocks exactly: no powder blend,
//  scale slots 0..7 only (direction p < 8), T_next carries its first-order dT/ds_p.
// ============================================================================
#pragma once
#include "src_kernel.hpp"
#ifdef __cplusplus
extern "C" {
#endif
#define FB_NSLOT 11

// (a) blended conductivity k = mf*k(T)*S_K + (1-mf)*kp(T)*S_KP, derivatives at fixed T
void fb_dk1(double T, double mf, double kp_fraction, double *dk_dK, double *dk_dKP);
// (b) q = [mf*a*S_HC + (1-mf)*hpa*S_HP]*(T - Tamb*S_T0) + b*(T^4 - (Tamb*S_T0)^4), derivatives at fixed T
void fb_dq1(double Tq, double mf, double a, double hpa, double b, double Tamb,
            double *dq_dhc, double *dq_dT0, double *dq_dhp);
// (c) nodal capacitive+latent term; rhs[p] = -d nodal/ds_p at fixed Tnext, Tn seeded with pd[p].
//     rho = Mv*RHO*S_RHO + (1-Mv)*RHO*rhop_fraction*S_RHOP
void fb_nodal1(double Tnext, double Tn, const double *pd, int nw, const int *slot, double Mv,
               double rhop_fraction, double V, double idt, double Ts, double Tl, double *rhs);

// second order (diagonal), fwd-over-fwd along the curve s -> theta_p = 1 + s:
//   T_next(s) = Tnext + s*ds[p],   T_n(s) = Tn + s*pd[p] + s^2/2*pd2[p]
void fb2_dk(double Tnext, const double *ds, int nw, double *dk, double *d2k);
void fb2_q(double Tq, const double *ds, int nw, double a, double b, double Tamb, double *d2q);
void fb2_nodal(double Tnext, double Tn, const double *pd, const double *pd2, const double *ds,
               int nw, double V, double idt, double Ts, double Tl, double *rhs2);   // rhs2 = -g''(0)

// ---- local gradient + full symmetric Hessian of each (blended) kernel w.r.t. its own inputs (Enzyme nested fwd) ----
// kind 0: k      x = [T, S_K, S_KP]                                       fixed = [mf, kp_fraction]                         (m = 3)
// kind 1: q      x = [T, S_HC, S_T0, S_HP]                                fixed = [mf, a, hpa, b, Tamb]                      (m = 4)
// kind 2: nodal  x = [Tnext, Tn, S_CP, S_RHO, S_L, S_TS, S_TL, S_RHOP]    fixed = [Mv, rho_p_fraction, V, idt, Ts, Tl]       (m = 8)
// g[m] = gradient, H[m*m] = Hessian at x0 (scale inputs are normally 1)
void fb_local_hess(int kind, const double *x0, const double *fixed, double *g, double *H);

// ---- laser source (src_kernel.hpp): value and AD derivatives along parameter p (0..5 = rx,ry,rz,x0,Vx,tL) ----
void ak_src_eval(const SrcSub *r, const SrcCell *cells, double *out);
void ak_src_d1(const SrcSub *r, const SrcCell *cells, int p, double *dout);
void ak_src_d2(const SrcSub *r, const SrcCell *cells, int p, double *d2out);
// Hessian w.r.t. the 6 source parameters, upper triangle (a<=b) in row-major pair order, each nout values
void ak_src_hess(const SrcSub *r, const SrcCell *cells, double *H21);
#ifdef __cplusplus
}
#endif
