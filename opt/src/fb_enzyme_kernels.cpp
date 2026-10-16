// Enzyme kernels for the full-bridge single-layer driver -- see fb_enzyme_kernels.h.
// Build: clang++ -O2 -fplugin=ClangEnzyme-21.so -DMFEM_USE_ENZYME -I<MFEM_DIR> -c
#include "fb_enzyme_kernels.h"
#include "general/enzyme.hpp"   // MFEM's Enzyme hooks (__enzyme_fwddiff, enzyme_dup, ...)

int enzyme_dup; int enzyme_dupnoneed; int enzyme_out; int enzyme_const; int enzyme_interleave;

struct FbCurve {            // external linkage (template argument of __enzyme_fwddiff)
    double Tx0, dTx;        // T_next(s)
    double Tn0, pd, pd2;    // T_n(s)
    double V, idt, Ts, Tl;  // nodal
    double a, b, Tamb;      // boundary
    int p;
};

namespace {
enum { S_K = 0, S_CP, S_RHO, S_L, S_TS, S_TL, S_HC, S_T0, S_KP, S_RHOP, S_HP };
constexpr double RHO = 8.44e-9, L_FUSION = 2.72e11;

inline double pwlin(const double *Ta, const double *Va, int n, double T)
{
    if (T <= Ta[0])   return Va[0];
    if (T >= Ta[n-1]) return Va[n-1];
    int i = 0; while (i < n - 2 && Ta[i+1] < T) ++i;
    return Va[i] + (T - Ta[i]) / (Ta[i+1] - Ta[i]) * (Va[i+1] - Va[i]);
}
// tables identical to the driver's IN625::k_oti / cp_oti
const double k_Ta[11]  = {294.,311.,366.,477.,589.,700.,811.,922.,1033.,1144.,1255.};
const double k_Va[11]  = {9.8,10.1,10.8,12.5,14.1,15.7,17.5,19.0,20.8,22.8,25.2};
const double cp_Ta[11] = {294.,366.,477.,589.,700.,811.,922.,1033.,1144.,1255.,1366.};
const double cp_Va[11] = {4.1e8,4.27e8,4.56e8,4.81e8,5.11e8,5.36e8,5.65e8,5.90e8,6.20e8,6.45e8,6.70e8};

double k_fn(double T, const double *S)  { return pwlin(k_Ta, k_Va, 11, T) * S[S_K]; }
double cp_fn(double T, const double *S) { return pwlin(cp_Ta, cp_Va, 11, T) * S[S_CP]; }
// clamp on the value with nominal Ts/Tl (as f_L_oti); d/d(Ts,Tl) via scaled bounds
double fL_fn(double T, const double *S, double Ts, double Tl)
{
    if (T <= Ts) return 0.0;
    if (T >= Tl) return 1.0;
    const double ts = Ts * S[S_TS], tl = Tl * S[S_TL];
    const double xi = (T - ts) / (tl - ts);
    return xi*xi*xi*(10.0 - 15.0*xi + 6.0*xi*xi);
}
// ---- first-order functions with the powder blends (v8 driver) ----
double kblend_fn(double T, const double *S, double mf, double kpf)
{
    return mf * (pwlin(k_Ta, k_Va, 11, T) * S[S_K]) + (1.0 - mf) * (pwlin(k_Ta, k_Va, 11, T) * kpf * S[S_KP]);
}
double qblend_fn(double T, const double *S, double mf, double a, double hpa, double b, double Tamb)
{
    const double Tinf = Tamb * S[S_T0];
    const double hconv = mf * (a * S[S_HC]) + (1.0 - mf) * (hpa * S[S_HP]);
    return hconv * (T - Tinf) + b * (T*T*T*T - Tinf*Tinf*Tinf*Tinf);
}
double nodal_blend_fn(double Tnext, double Tn, const double *S, double Mv, double rpf,
                      double V, double idt, double Ts, double Tl)
{
    const double rho = Mv * (RHO * S[S_RHO]) + (1.0 - Mv) * (RHO * rpf * S[S_RHOP]);
    const double Lf = L_FUSION * S[S_L];
    const double Mc = V * rho * cp_fn(Tnext, S);
    const double L  = V * rho * Lf * fL_fn(Tnext, S, Ts, Tl);
    const double Ln = V * rho * Lf * fL_fn(Tn, S, Ts, Tl);
    return Mc * idt * (Tnext - Tn) + (L - Ln) * idt;
}
// ---- second-order functions: reference OTI blocks (no powder blend, slots 0..7) ----
double nodal_fn(double Tnext, double Tn, const double *S, double V, double idt, double Ts, double Tl)
{
    const double rho = RHO * S[S_RHO], Lf = L_FUSION * S[S_L];
    const double Mc = V * rho * cp_fn(Tnext, S);
    const double L  = V * rho * Lf * fL_fn(Tnext, S, Ts, Tl);
    const double Ln = V * rho * Lf * fL_fn(Tn, S, Ts, Tl);
    return Mc * idt * (Tnext - Tn) + (L - Ln) * idt;
}
double q_fn(double T, const double *S, double a, double b, double Tamb)
{
    const double Tinf = Tamb * S[S_T0];
    return a * S[S_HC] * (T - Tinf) + b * (T*T*T*T - Tinf*Tinf*Tinf*Tinf);
}
inline void unit_seed(int p, double *dS)
{ for (int i = 0; i < FB_NSLOT; i++) dS[i] = (i == p) ? 1.0 : 0.0; }

inline void curve_S(double s, int p, double *S)
{ for (int i = 0; i < FB_NSLOT; i++) S[i] = 1.0; if (p < 8) S[p] = 1.0 + s; }
double g_k(double s, const FbCurve *c)
{ double S[FB_NSLOT]; curve_S(s, c->p, S); return k_fn(c->Tx0 + s*c->dTx, S); }
double g_q(double s, const FbCurve *c)
{ double S[FB_NSLOT]; curve_S(s, c->p, S); return q_fn(c->Tx0 + s*c->dTx, S, c->a, c->b, c->Tamb); }
double g_nodal(double s, const FbCurve *c)
{
    double S[FB_NSLOT]; curve_S(s, c->p, S);
    return nodal_fn(c->Tx0 + s*c->dTx, c->Tn0 + s*c->pd + 0.5*s*s*c->pd2, S, c->V, c->idt, c->Ts, c->Tl);
}
double g_k_d1(double s, const FbCurve *c)
{ return __enzyme_fwddiff<double>((void*)g_k, enzyme_dup, s, 1.0, enzyme_const, c); }
double g_q_d1(double s, const FbCurve *c)
{ return __enzyme_fwddiff<double>((void*)g_q, enzyme_dup, s, 1.0, enzyme_const, c); }
double g_nodal_d1(double s, const FbCurve *c)
{ return __enzyme_fwddiff<double>((void*)g_nodal, enzyme_dup, s, 1.0, enzyme_const, c); }
inline double d1_at0(double (*g)(double, const FbCurve*), const FbCurve *c)
{ return __enzyme_fwddiff<double>((void*)g, enzyme_dup, 0.0, 1.0, enzyme_const, c); }
} // namespace

extern "C" {

void fb_dk1(double T, double mf, double kpf, double *dk_dK, double *dk_dKP)
{
    double S[FB_NSLOT] = {1,1,1,1,1,1,1,1,1,1,1}, dS[FB_NSLOT];
    unit_seed(S_K, dS);
    *dk_dK = __enzyme_fwddiff<double>((void*)kblend_fn, enzyme_const, T,
                 enzyme_dup, (const double*)S, (const double*)dS, enzyme_const, mf, enzyme_const, kpf);
    unit_seed(S_KP, dS);
    *dk_dKP = __enzyme_fwddiff<double>((void*)kblend_fn, enzyme_const, T,
                 enzyme_dup, (const double*)S, (const double*)dS, enzyme_const, mf, enzyme_const, kpf);
}

void fb_dq1(double Tq, double mf, double a, double hpa, double b, double Tamb,
            double *dq_dhc, double *dq_dT0, double *dq_dhp)
{
    double S[FB_NSLOT] = {1,1,1,1,1,1,1,1,1,1,1}, dS[FB_NSLOT];
    double *out[3] = {dq_dhc, dq_dT0, dq_dhp};
    const int sl[3] = {S_HC, S_T0, S_HP};
    for (int j = 0; j < 3; j++) {
        unit_seed(sl[j], dS);
        *out[j] = __enzyme_fwddiff<double>((void*)qblend_fn, enzyme_const, Tq,
                     enzyme_dup, (const double*)S, (const double*)dS,
                     enzyme_const, mf, enzyme_const, a, enzyme_const, hpa, enzyme_const, b, enzyme_const, Tamb);
    }
}

void fb_nodal1(double Tnext, double Tn, const double *pd, int nw, const int *slot, double Mv,
               double rpf, double V, double idt, double Ts, double Tl, double *rhs)
{
    double S[FB_NSLOT] = {1,1,1,1,1,1,1,1,1,1,1}, dS[FB_NSLOT];
    for (int p = 0; p < nw; p++) {
        if (pd[p] == 0.0 && slot[p] < 0) { rhs[p] = 0.0; continue; }
        unit_seed(slot[p], dS);   // slot -1 -> all zero
        rhs[p] = -__enzyme_fwddiff<double>((void*)nodal_blend_fn,
                    enzyme_const, Tnext, enzyme_dup, Tn, pd[p],
                    enzyme_dup, (const double*)S, (const double*)dS,
                    enzyme_const, Mv, enzyme_const, rpf,
                    enzyme_const, V, enzyme_const, idt, enzyme_const, Ts, enzyme_const, Tl);
    }
}

void fb2_dk(double Tnext, const double *ds, int nw, double *dk, double *d2k)
{
    FbCurve c{}; c.Tx0 = Tnext;
    for (int p = 0; p < nw; p++) {
        c.p = p; c.dTx = ds[p];
        dk[p] = d1_at0(g_k, &c); d2k[p] = d1_at0(g_k_d1, &c);
    }
}

void fb2_q(double Tq, const double *ds, int nw, double a, double b, double Tamb, double *d2q)
{
    FbCurve c{}; c.Tx0 = Tq; c.a = a; c.b = b; c.Tamb = Tamb;
    for (int p = 0; p < nw; p++) { c.p = p; c.dTx = ds[p]; d2q[p] = d1_at0(g_q_d1, &c); }
}

void fb2_nodal(double Tnext, double Tn, const double *pd, const double *pd2, const double *ds,
               int nw, double V, double idt, double Ts, double Tl, double *rhs2)
{
    FbCurve c{}; c.Tx0 = Tnext; c.Tn0 = Tn; c.V = V; c.idt = idt; c.Ts = Ts; c.Tl = Tl;
    for (int p = 0; p < nw; p++) {
        c.p = p; c.dTx = ds[p]; c.pd = pd[p]; c.pd2 = pd2[p];
        rhs2[p] = -d1_at0(g_nodal_d1, &c);
    }
}


// ---- local Hessians (full-Hessian / mixed second derivatives) ----
} // extern "C"
struct FbH { int kind; double x0[8]; double fx[6]; };   // external linkage (template argument of __enzyme_fwddiff)
namespace {
double fbh_eval(const double *x, const FbH *h)
{
    double S[FB_NSLOT] = {1,1,1,1,1,1,1,1,1,1,1};
    if (h->kind == 0) { S[S_K] = x[1]; S[S_KP] = x[2]; return kblend_fn(x[0], S, h->fx[0], h->fx[1]); }
    if (h->kind == 1) { S[S_HC] = x[1]; S[S_T0] = x[2]; S[S_HP] = x[3];
                        return qblend_fn(x[0], S, h->fx[0], h->fx[1], h->fx[2], h->fx[3], h->fx[4]); }
    S[S_CP] = x[2]; S[S_RHO] = x[3]; S[S_L] = x[4]; S[S_TS] = x[5]; S[S_TL] = x[6]; S[S_RHOP] = x[7];
    return nodal_blend_fn(x[0], x[1], S, h->fx[0], h->fx[1], h->fx[2], h->fx[3], h->fx[4], h->fx[5]);
}
double fbh_curve(double s, double t, const FbH *h, int a, int b)
{
    double x[8]; for (int i = 0; i < 8; i++) x[i] = h->x0[i];
    x[a] += s; x[b] += t;
    return fbh_eval(x, h);
}
double fbh_ds(double t, const FbH *h, int a, int b)
{ return __enzyme_fwddiff<double>((void*)fbh_curve, enzyme_dup, 0.0, 1.0, enzyme_const, t, enzyme_const, h, enzyme_const, a, enzyme_const, b); }

// 1-D curve through source-parameter space (th = s e_p), differentiated once or twice in s
void srck_curve(double s, const SrcSub *r, const SrcCell *cells, int p, double *out)
{ double th[6] = {0, 0, 0, 0, 0, 0}; th[p] = s; srck_eval<double>(*r, cells, th, out); }
void srck_curve_d1(double s, const SrcSub *r, const SrcCell *cells, int p, double *dout)
{
    double tmp[SRCK_MAXOUT];
    __enzyme_fwddiff<void>((void*)srck_curve, enzyme_dup, s, 1.0,
                           enzyme_const, r, enzyme_const, cells, enzyme_const, p,
                           enzyme_dupnoneed, tmp, dout);
}
void srck_curve2(double s, double t, const SrcSub *r, const SrcCell *cells, int a, int b, double *out)
{ double th[6] = {0, 0, 0, 0, 0, 0}; th[a] += s; th[b] += t; srck_eval<double>(*r, cells, th, out); }
void srck_curve2_ds(double t, const SrcSub *r, const SrcCell *cells, int a, int b, double *dout)
{
    double tmp[SRCK_MAXOUT];
    __enzyme_fwddiff<void>((void*)srck_curve2, enzyme_dup, 0.0, 1.0, enzyme_const, t,
                           enzyme_const, r, enzyme_const, cells, enzyme_const, a, enzyme_const, b,
                           enzyme_dupnoneed, tmp, dout);
}
} // namespace

extern "C" {
void fb_local_hess(int kind, const double *x0, const double *fixed, double *g, double *H)
{
    FbH h{}; h.kind = kind;
    const int m = (kind == 0) ? 3 : (kind == 1) ? 4 : 8;
    for (int i = 0; i < m; i++) h.x0[i] = x0[i];
    for (int i = 0; i < 6; i++) h.fx[i] = fixed ? fixed[i] : 0.0;
    for (int a = 0; a < m; a++) {
        g[a] = fbh_ds(0.0, &h, a, a);
        for (int b = a; b < m; b++) {
            const double v = __enzyme_fwddiff<double>((void*)fbh_ds, enzyme_dup, 0.0, 1.0,
                                                      enzyme_const, &h, enzyme_const, a, enzyme_const, b);
            H[a*m + b] = v; H[b*m + a] = v;
        }
    }
}

void ak_src_eval(const SrcSub *r, const SrcCell *cells, double *out) { srck_curve(0.0, r, cells, 0, out); }
void ak_src_d1(const SrcSub *r, const SrcCell *cells, int p, double *dout) { srck_curve_d1(0.0, r, cells, p, dout); }
void ak_src_d2(const SrcSub *r, const SrcCell *cells, int p, double *d2out)
{
    double dtmp[SRCK_MAXOUT];
    __enzyme_fwddiff<void>((void*)srck_curve_d1, enzyme_dup, 0.0, 1.0,
                           enzyme_const, r, enzyme_const, cells, enzyme_const, p,
                           enzyme_dupnoneed, dtmp, d2out);
}
void ak_src_hess(const SrcSub *r, const SrcCell *cells, double *H21)
{
    const int nout = r->ncell * 8;
    int k = 0;
    for (int a = 0; a < 6; a++)
        for (int b = a; b < 6; b++, k++) {
            double dtmp[SRCK_MAXOUT];
            __enzyme_fwddiff<void>((void*)srck_curve2_ds, enzyme_dup, 0.0, 1.0,
                                   enzyme_const, r, enzyme_const, cells, enzyme_const, a, enzyme_const, b,
                                   enzyme_dupnoneed, dtmp, H21 + (size_t)k * nout);
        }
}
} // extern "C"
