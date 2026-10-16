// mock_kernels.cpp -- stand-in for fb_enzyme_kernels.o on a machine without Enzyme (`make selftest`).
// Implements the same C interface with the hyper-dual reference model (ref_model.hpp). It only checks that the test
// HARNESS builds and runs end to end (seeding, layouts, identities); the real verification is `make test`, which
// links the Enzyme-differentiated kernels instead.
#include "fb_enzyme_kernels.h"
#include "ref_model.hpp"
#include <cstring>
#include <vector>

using ref::HD;

namespace {
void curveS(int p, HD *S) { ref::unit_S(S); if (p < 8) S[p] = ref::seed2(1.0, 1.0); }
HD loc(int kind, const HD *x, const double *fx)
{
    HD S[ref::NSLOT]; ref::unit_S(S);
    if (kind == 0) { S[ref::S_K] = x[1]; S[ref::S_KP] = x[2]; return ref::kblend(x[0], S, fx[0], fx[1]); }
    if (kind == 1) { S[ref::S_HC] = x[1]; S[ref::S_T0] = x[2]; S[ref::S_HP] = x[3]; return ref::qblend(x[0], S, fx[0], fx[1], fx[2], fx[3], fx[4]); }
    S[ref::S_CP] = x[2]; S[ref::S_RHO] = x[3]; S[ref::S_L] = x[4]; S[ref::S_TS] = x[5]; S[ref::S_TL] = x[6]; S[ref::S_RHOP] = x[7];
    return ref::nodal_blend(x[0], x[1], S, fx[0], fx[1], fx[2], fx[3], fx[4], fx[5]);
}
void src(const SrcSub *r, const SrcCell *c, int a, int b, std::vector<HD> &out)
{
    HD th[6]; if (a >= 0) th[a].a = 1.0; if (b >= 0) th[b].b = 1.0;
    out.assign((size_t)r->ncell * 8, HD(0.0));
    srck_eval<HD>(*r, c, th, out.data());
}
}

extern "C" {
void fb_dk1(double T, double mf, double kpf, double *dK, double *dKP)
{
    HD S[ref::NSLOT]; ref::unit_S(S); S[ref::S_K].a = 1.0; S[ref::S_KP].b = 1.0;
    const HD k = ref::kblend(HD(T), S, mf, kpf); *dK = k.a; *dKP = k.b;
}
void fb_dq1(double Tq, double mf, double a, double hpa, double b, double Tamb, double *dhc, double *dT0, double *dhp)
{
    const int sl[3] = {ref::S_HC, ref::S_T0, ref::S_HP}; double *out[3] = {dhc, dT0, dhp};
    for (int j = 0; j < 3; j++) {
        HD S[ref::NSLOT]; ref::unit_S(S); S[sl[j]].a = 1.0;
        *out[j] = ref::qblend(HD(Tq), S, mf, a, hpa, b, Tamb).a;
    }
}
void fb_nodal1(double Tnext, double Tn, const double *pd, int nw, const int *slot, double Mv, double rpf, double V,
               double idt, double Ts, double Tl, double *rhs)
{
    for (int p = 0; p < nw; p++) {
        if (pd[p] == 0.0 && slot[p] < 0) { rhs[p] = 0.0; continue; }
        HD S[ref::NSLOT]; ref::unit_S(S); if (slot[p] >= 0) S[slot[p]].a = 1.0;
        rhs[p] = -ref::nodal_blend(HD(Tnext), ref::seed1(Tn, pd[p]), S, Mv, rpf, V, idt, Ts, Tl).a;
    }
}
void fb2_dk(double Tnext, const double *ds, int nw, double *dk, double *d2k)
{
    for (int p = 0; p < nw; p++) { HD S[ref::NSLOT]; curveS(p, S); const HD g = ref::k_plain(ref::seed2(Tnext, ds[p]), S); dk[p] = g.a; d2k[p] = g.ab; }
}
void fb2_q(double Tq, const double *ds, int nw, double a, double b, double Tamb, double *d2q)
{
    for (int p = 0; p < nw; p++) { HD S[ref::NSLOT]; curveS(p, S); d2q[p] = ref::q_plain(ref::seed2(Tq, ds[p]), S, a, b, Tamb).ab; }
}
void fb2_nodal(double Tnext, double Tn, const double *pd, const double *pd2, const double *ds, int nw, double V,
               double idt, double Ts, double Tl, double *rhs2)
{
    for (int p = 0; p < nw; p++) {
        HD S[ref::NSLOT]; curveS(p, S);
        rhs2[p] = -ref::nodal_plain(ref::seed2(Tnext, ds[p]), ref::seed2(Tn, pd[p], pd2[p]), S, V, idt, Ts, Tl).ab;
    }
}
void fb_local_hess(int kind, const double *x0, const double *fixed, double *g, double *H)
{
    const int m = (kind == 0) ? 3 : (kind == 1) ? 4 : 8;
    for (int a = 0; a < m; a++)
        for (int b = a; b < m; b++) {
            HD x[8]; for (int i = 0; i < m; i++) x[i] = HD(x0[i]);
            x[a].a = 1.0; x[b].b = 1.0;
            const HD f = loc(kind, x, fixed);
            if (a == b) g[a] = f.a;
            H[a * m + b] = f.ab; H[b * m + a] = f.ab;
        }
}
void ak_src_eval(const SrcSub *r, const SrcCell *c, double *out)
{ const double th[6] = {0, 0, 0, 0, 0, 0}; srck_eval<double>(*r, c, th, out); }
void ak_src_d1(const SrcSub *r, const SrcCell *c, int p, double *d)
{ std::vector<HD> o; src(r, c, p, -1, o); for (size_t i = 0; i < o.size(); i++) d[i] = o[i].a; }
void ak_src_d2(const SrcSub *r, const SrcCell *c, int p, double *d2)
{ std::vector<HD> o; src(r, c, p, p, o); for (size_t i = 0; i < o.size(); i++) d2[i] = o[i].ab; }
void ak_src_hess(const SrcSub *r, const SrcCell *c, double *H21)
{
    const size_t n = (size_t)r->ncell * 8; int k = 0;
    for (int a = 0; a < 6; a++)
        for (int b = a; b < 6; b++, k++) { std::vector<HD> o; src(r, c, a, b, o); for (size_t i = 0; i < n; i++) H21[k * n + i] = o[i].ab; }
}
} // extern "C"
