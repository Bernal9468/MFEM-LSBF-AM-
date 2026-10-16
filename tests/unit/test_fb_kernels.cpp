// test_fb_kernels.cpp -- unit test of the thermal Enzyme kernels (opt/src/fb_enzyme_kernels.{h,cpp}).
//
// Every exported entry point is compared with EXACT derivatives of an independent reference model
// (ref_model.hpp: physics transcribed from the driver's forward IN625 functions, differentiated with hyper-dual
// numbers), plus exact identities. Both sides are exact to round-off, so the tolerances are ~1e-11: a wrong seed,
// a wrong slot, a missing chain-rule term or a drifted material table fails by many orders of magnitude.
//
//   fb_dk1, fb_dq1                 closed-form derivatives
//   fb_nodal1                      blended nodal term, Tn seeded with the propagated history, all 26 directions
//   fb2_dk, fb2_q, fb2_nodal       second-order curve s -> (T + s dT, Tn + s pd + s^2/2 pd2, S_p = 1 + s)
//   fb_local_hess (kinds 0, 1, 2)  gradient + full Hessian; symmetry; consistency with fb_nodal1
//   ak_src_eval / _d1 / _d2 / _hess  laser-source kernel (point and Goldak mode); energy conservation
//
// States avoid the kinks of the piecewise-linear k(T)/cp(T) tables and the solidus/liquidus clamps, where the
// derivative is one-sided by construction. Exit status 0 = all checks passed.
#include "fb_enzyme_kernels.h"
#include "ref_model.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using ref::HD;

namespace {
int g_fail = 0, g_checks = 0;

// one named group of comparisons; err = |got - want| / max(|want|, floor)
struct Check {
    std::string name; double tol, floor, worst = 0.0, got_w = 0, want_w = 0; long n = 0;
    Check(std::string nm, double t, double f) : name(std::move(nm)), tol(t), floor(f) {}
    void add(double got, double want, double fl = -1.0)
    {
        const double e = std::fabs(got - want) / std::max(std::fabs(want), fl > 0.0 ? fl : floor);
        if (!(e <= worst)) { worst = e; got_w = got; want_w = want; }   // !(<=) also catches NaN
        n++;
    }
    ~Check()
    {
        const bool ok = (worst <= tol) && n > 0;
        g_checks++; if (!ok) g_fail++;
        std::printf("%-5s %-62s n=%-5ld max rel err %.2e (tol %.0e)", ok ? "PASS" : "FAIL", name.c_str(), n, worst, tol);
        if (!ok) std::printf("\n      worst: got %.15e want %.15e", got_w, want_w);
        std::printf("\n");
    }
};

constexpr double TOL = 1e-11;               // AD vs AD, both exact to round-off
constexpr int NW = 26;                      // driver direction count (LI_K .. LI_W)
// driver block (c) slot map: LI_* direction -> kernel scale slot (-1: enters only through the propagated history)
void driver_slot_map(int *slot)
{
    for (int p = 0; p < NW; p++) slot[p] = -1;
    for (int p = 0; p < 8; p++) slot[p] = p;     // LI_K .. LI_T0
    slot[18] = 8; slot[19] = 9; slot[17] = 10;   // LI_KP, LI_RHOP, LI_HP
}

const double Ts = 1563.0, Tl = 1623.0;           // driver defaults
const double V = 1.0e-6, IDT = 1.0e6;            // nodal volume [mm^3], 1/dt (dt = 1e-6): nodal term O(10)
const double HC = 0.018, HPA = 0.0018, BRAD = 0.45 * 5.670373e-11, TAMB = 313.0;   // driver defaults (C2K = 273)

// (Tnext, Tn): solid; both in the melt band; Tnext liquid / Tn in band; Tnext in band / Tn solid; liquid
const double STATES[5][2] = {{800.3, 795.1}, {1585.2, 1570.4}, {1700.0, 1600.0}, {1590.0, 1500.0}, {2500.0, 2400.0}};

void history(double *pd, double scale)
{
    for (int p = 0; p < NW; p++) pd[p] = scale * (double)((p * 7) % 11 - 5) / 3.0;   // includes exact zeros
    for (int p = 20; p < NW; p++) pd[p] = 0.0;                                        // bp..w: structurally zero
}
double maxabs(const double *v, int n) { double m = 0; for (int i = 0; i < n; i++) m = std::max(m, std::fabs(v[i])); return m; }

// ---------------------------------------------------------------------------------------------------------
void test_dk1()
{
    Check c("fb_dk1: dk/dS_K, dk/dS_KP vs closed form", TOL, 1e-12);
    for (double T : {250.0, 300.5, 701.0, 1100.0, 1250.0, 1400.0})
        for (double mf : {0.0, 0.37, 1.0}) {
            const double kpf = 0.2;
            double dK, dKP; fb_dk1(T, mf, kpf, &dK, &dKP);
            c.add(dK, mf * ref::k_tab(T)); c.add(dKP, (1.0 - mf) * kpf * ref::k_tab(T));
        }
}

void test_dq1()
{
    Check c("fb_dq1: dq/dS_HC, dq/dS_T0, dq/dS_HP vs closed form", TOL, 1e-12);
    for (double T : {313.0, 500.0, 1600.0, 3000.0})
        for (double mf : {0.0, 0.5, 1.0}) {
            double dhc, dT0, dhp; fb_dq1(T, mf, HC, HPA, BRAD, TAMB, &dhc, &dT0, &dhp);
            const double h = mf * HC + (1.0 - mf) * HPA;
            c.add(dhc, mf * HC * (T - TAMB));
            c.add(dT0, -h * TAMB - 4.0 * BRAD * TAMB * TAMB * TAMB * TAMB);
            c.add(dhp, (1.0 - mf) * HPA * (T - TAMB));
        }
}

void test_nodal1()
{
    Check c("fb_nodal1: rhs[p] = -d nodal/ds_p, all 26 directions", TOL, 1e-300);
    Check z("fb_nodal1: exact zero for unseeded, history-free directions", 0.0, 1e-300);
    int slot[NW]; driver_slot_map(slot);
    for (auto &st : STATES)
        for (double Mv : {0.0, 0.35, 1.0}) {
            const double Tnext = st[0], Tn = st[1], rpf = 0.6;
            double pd[NW]; history(pd, 1.7);
            double rhs[NW]; fb_nodal1(Tnext, Tn, pd, NW, slot, Mv, rpf, V, IDT, Ts, Tl, rhs);
            const double fl = 1e-12 * std::max(maxabs(rhs, NW), 1.0);
            for (int p = 0; p < NW; p++) {
                if (slot[p] < 0 && pd[p] == 0.0) { z.add(rhs[p], 0.0); continue; }
                HD S[ref::NSLOT]; ref::unit_S(S); if (slot[p] >= 0) S[slot[p]] = ref::seed1(1.0, 1.0);
                const HD g = ref::nodal_blend(HD(Tnext), ref::seed1(Tn, pd[p]), S, Mv, rpf, V, IDT, Ts, Tl);
                c.add(rhs[p], -g.a, fl);
            }
        }
}

// ---------------------------------------------------------------------------------------------------------
void test_fb2()
{
    Check ck("fb2_dk: dk, d2k along the second-order curve", TOL, 1e-300);
    Check cq("fb2_q: d2q along the second-order curve", TOL, 1e-300);
    Check cn("fb2_nodal: rhs2 = -g''(0) (Tn(s) to 2nd order)", TOL, 1e-300);
    for (auto &st : STATES) {
        const double Tnext = st[0], Tn = st[1];
        double ds[NW], pd[NW], pd2[NW];
        history(ds, 2.3); history(pd, 1.1); history(pd2, -0.6);
        double dk[NW], d2k[NW], d2q[NW], rhs2[NW];
        fb2_dk(Tnext, ds, NW, dk, d2k);
        fb2_q(Tnext, ds, NW, HC, BRAD, TAMB, d2q);
        fb2_nodal(Tnext, Tn, pd, pd2, ds, NW, V, IDT, Ts, Tl, rhs2);
        const double fk = 1e-12 * std::max({maxabs(dk, NW), maxabs(d2k, NW), 1e-3});
        const double fq = 1e-12 * std::max(maxabs(d2q, NW), 1e-3), fn = 1e-12 * std::max(maxabs(rhs2, NW), 1.0);
        for (int p = 0; p < NW; p++) {
            HD S[ref::NSLOT]; ref::unit_S(S); if (p < 8) S[p] = ref::seed2(1.0, 1.0);   // the kernels' curve_S
            const HD Tx = ref::seed2(Tnext, ds[p]);
            const HD gk = ref::k_plain(Tx, S), gq = ref::q_plain(Tx, S, HC, BRAD, TAMB);
            const HD gn = ref::nodal_plain(Tx, ref::seed2(Tn, pd[p], pd2[p]), S, V, IDT, Ts, Tl);
            ck.add(dk[p], gk.a, fk); ck.add(d2k[p], gk.ab, fk);
            cq.add(d2q[p], gq.ab, fq);
            cn.add(rhs2[p], -gn.ab, fn);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// fb_local_hess: reference function of the kernel's own inputs x (see fb_enzyme_kernels.h)
HD local_fn(int kind, const HD *x, const double *fx)
{
    HD S[ref::NSLOT]; ref::unit_S(S);
    if (kind == 0) { S[ref::S_K] = x[1]; S[ref::S_KP] = x[2]; return ref::kblend(x[0], S, fx[0], fx[1]); }
    if (kind == 1) { S[ref::S_HC] = x[1]; S[ref::S_T0] = x[2]; S[ref::S_HP] = x[3];
                     return ref::qblend(x[0], S, fx[0], fx[1], fx[2], fx[3], fx[4]); }
    S[ref::S_CP] = x[2]; S[ref::S_RHO] = x[3]; S[ref::S_L] = x[4]; S[ref::S_TS] = x[5]; S[ref::S_TL] = x[6]; S[ref::S_RHOP] = x[7];
    return ref::nodal_blend(x[0], x[1], S, fx[0], fx[1], fx[2], fx[3], fx[4], fx[5]);
}

void test_local_hess()
{
    Check cg("fb_local_hess: gradient (kinds 0,1,2)", TOL, 1e-300);
    Check ch("fb_local_hess: Hessian (kinds 0,1,2)", TOL, 1e-300);
    Check cs("fb_local_hess: Hessian symmetry", 0.0, 1e-300);
    Check cc("fb_local_hess(kind 2) gradient vs -fb_nodal1 (same directions)", TOL, 1e-300);
    int slot[NW]; driver_slot_map(slot);
    for (auto &st : STATES)
        for (double mix : {0.0, 0.4, 1.0})
            for (int kind = 0; kind < 3; kind++) {
                const int m = (kind == 0) ? 3 : (kind == 1) ? 4 : 8;
                double x0[8] = {st[0], st[1], 1, 1, 1, 1, 1, 1};
                if (kind != 2) x0[1] = 1.0;
                double fx[6] = {0, 0, 0, 0, 0, 0};
                if (kind == 0) { fx[0] = mix; fx[1] = 0.2; }
                else if (kind == 1) { fx[0] = mix; fx[1] = HC; fx[2] = HPA; fx[3] = BRAD; fx[4] = TAMB; }
                else { fx[0] = mix; fx[1] = 0.6; fx[2] = V; fx[3] = IDT; fx[4] = Ts; fx[5] = Tl; }
                double g[8], H[64];
                fb_local_hess(kind, x0, fx, g, H);
                const double gfl = 1e-12 * std::max(maxabs(g, m), 1e-6), hfl = 1e-12 * std::max(maxabs(H, m * m), 1e-6);
                for (int a = 0; a < m; a++)
                    for (int b = a; b < m; b++) {
                        HD x[8]; for (int i = 0; i < m; i++) x[i] = HD(x0[i]);
                        x[a].a = 1.0; x[b].b = 1.0;           // e1 along x_a, e2 along x_b (same variable if a == b)
                        const HD f = local_fn(kind, x, fx);
                        if (a == b) cg.add(g[a], f.a, gfl);
                        ch.add(H[a * m + b], f.ab, hfl);
                        cs.add(H[a * m + b], H[b * m + a]);
                    }
                if (kind == 2) {   // -rhs_p of fb_nodal1 = g[slot -> x index] + g[Tn] * pd_p
                    double pd[NW]; history(pd, 1.7);
                    double rhs[NW]; fb_nodal1(st[0], st[1], pd, NW, slot, mix, 0.6, V, IDT, Ts, Tl, rhs);
                    const int xi_of_slot[ref::NSLOT] = {-1, 2, 3, 4, 5, 6, -1, -1, -1, 7, -1};
                    for (int p = 0; p < NW; p++) {
                        const int sl = slot[p];
                        // k, hc, T0, kp, hp are not inputs of the nodal term: they enter only through the Tn history
                        const double want = ((sl >= 0 && xi_of_slot[sl] >= 0) ? g[xi_of_slot[sl]] : 0.0) + g[1] * pd[p];
                        cc.add(-rhs[p], want, gfl);
                    }
                }
            }
}

// ---------------------------------------------------------------------------------------------------------
// laser source kernel: one distorted hex element large enough to hold the whole Goldak box
struct SrcCase { SrcSub r; std::vector<SrcCell> cells; int nout; };

void element_vertices(double *X)
{
    static const int V8[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    for (int a = 0; a < 8; a++) {
        const double u = V8[a][0], v = V8[a][1], w = V8[a][2];
        X[a * 3 + 0] = -1.0 + 3.0 * u + 0.15 * u * v - 0.05 * w;   // trilinear distortion: nonlinear inverse map
        X[a * 3 + 1] = -1.0 + 2.0 * v + 0.10 * u * w;
        X[a * 3 + 2] = -1.0 + 1.5 * w + 0.08 * u * v;
    }
}

void nominal_xi(const double *X, const double p[3], double xi[3])
{
    double g[3] = {0.5, 0.5, 0.5};
    for (int it = 0; it < 30; it++) { srck_invmap<double>(X, g, p, xi); std::memcpy(g, xi, sizeof g); }
}

SrcCase make_case(int mode)
{
    SrcCase c{}; SrcSub &r = c.r;
    r.mode = mode; r.tm = 4.0e-4; r.t_s = 1.0e-4; r.seg_linear = 1; r.ts0 = 0.0; r.ts1 = 1.0e-3;
    const double xa[3] = {0.0, 0.0, 0.0}, xb[3] = {0.8, 0.1, 0.0};
    for (int d = 0; d < 3; d++) { r.xa[d] = xa[d]; r.xb[d] = xb[d]; }
    r.Esub = 0.5;
    const double e1[3] = {1, 0, 0}, e2[3] = {0, 1, 0}, e3[3] = {0, 0, -1};
    for (int d = 0; d < 3; d++) { r.e1[d] = e1[d]; r.e2[d] = e2[d]; r.e3[d] = e3[d]; }
    r.af = 0.1; r.ar = 0.12; r.b = 0.09; r.c = 0.11; r.ff = 0.8; r.fr = 1.2; r.box = 3.0; r.nq = 3;
    r.first_cell = 0;
    double X[24]; element_vertices(X);
    const double f = (r.tm - r.ts0) / (r.ts1 - r.ts0);
    double p0[3]; for (int d = 0; d < 3; d++) p0[d] = xa[d] + f * (xb[d] - xa[d]);
    auto cell_at = [&](int ix, int iy, int iz, const double p[3]) {
        SrcCell cl{}; cl.ix = ix; cl.iy = iy; cl.iz = iz; std::memcpy(cl.X, X, sizeof X);
        for (int a = 0; a < 8; a++) cl.dofs[a] = a;
        cl.const_shape = 0; nominal_xi(X, p, cl.xi0); return cl;
    };
    if (mode != 2) { r.ncell = 1; c.cells.push_back(cell_at(0, 0, 0, p0)); }
    else {
        const int nq = r.nq; const double box = r.box;
        const double dx = box * (r.af + r.ar) / nq, dy = 2.0 * box * r.b / nq, dz = box * r.c / nq;
        for (int ix = 0; ix < nq; ix++) for (int iy = 0; iy < nq; iy++) for (int iz = 0; iz < nq; iz++) {
            const double lx = -box * r.ar + (ix + 0.5) * dx, ly = -box * r.b + (iy + 0.5) * dy, lz = (iz + 0.5) * dz;
            double p[3]; for (int d = 0; d < 3; d++) p[d] = p0[d] + lx * e1[d] + ly * e2[d] + lz * e3[d];
            c.cells.push_back(cell_at(ix, iy, iz, p));
        }
        r.ncell = (int)c.cells.size();
    }
    c.nout = r.ncell * 8;
    return c;
}

// reference: srck_eval<HD> with th_a on e1 and th_b on e2 -> value, d/dth_a, d/dth_b, d2/dth_a dth_b
void src_ref(const SrcCase &c, int a, int b, std::vector<HD> &out)
{
    HD th[6]; th[a].a = 1.0; th[b].b = 1.0;
    out.assign(c.nout, HD(0.0));
    srck_eval<HD>(c.r, c.cells.data(), th, out.data());
}

void test_source()
{
    for (int mode : {0, 2}) {
        SrcCase c = make_case(mode);
        const std::string tag = mode == 2 ? " (Goldak)" : " (point)";
        const int n = c.nout;
        std::vector<double> v(n), d(n), d2v(n);
        std::vector<HD> R;
        ak_src_eval(&c.r, c.cells.data(), v.data());
        const double vmax = maxabs(v.data(), n);
        {
            Check ce("ak_src_eval == srck_eval<double>" + tag, 1e-15, 1e-300);
            Check cE("source energy conservation: sum(out) = Esub" + tag, 1e-13, 1e-300);
            std::vector<double> t0(n); const double th0[6] = {0, 0, 0, 0, 0, 0};
            srck_eval<double>(c.r, c.cells.data(), th0, t0.data());
            double sum = 0; for (int i = 0; i < n; i++) { ce.add(v[i], t0[i], 1e-15 * vmax); sum += v[i]; }
            cE.add(sum, c.r.Esub);
        }
        Check c1("ak_src_d1, 6 params (rx ry rz x0 Vx tL)" + tag, TOL, 1e-300);
        Check c2("ak_src_d2, 6 params" + tag, TOL, 1e-300);
        Check cz("source: d(sum out)/dp = 0 (normalised deposition)" + tag, 1e-12, 1.0);
        Check ch("ak_src_hess, 21 pairs" + tag, TOL, 1e-300);
        Check cd("ak_src_hess diagonal == ak_src_d2" + tag, 1e-14, 1e-300);
        std::vector<double> H21((size_t)21 * n);
        ak_src_hess(&c.r, c.cells.data(), H21.data());
        const double hmax = maxabs(H21.data(), 21 * n);
        int k = 0;
        for (int a = 0; a < 6; a++) {
            ak_src_d1(&c.r, c.cells.data(), a, d.data());
            ak_src_d2(&c.r, c.cells.data(), a, d2v.data());
            const double dfl = 1e-12 * std::max(maxabs(d.data(), n), vmax);
            src_ref(c, a, a, R);
            double dsum = 0;
            for (int i = 0; i < n; i++) { c1.add(d[i], R[i].a, dfl); c2.add(d2v[i], R[i].ab, 1e-12 * hmax); dsum += d[i]; }
            cz.add(dsum / std::max(vmax, 1e-300), 0.0);
            for (int b = a; b < 6; b++, k++) {
                if (b != a) src_ref(c, a, b, R);
                for (int i = 0; i < n; i++) {
                    ch.add(H21[(size_t)k * n + i], R[i].ab, 1e-12 * hmax);
                    if (a == b) cd.add(H21[(size_t)k * n + i], d2v[i], 1e-14 * hmax);
                }
            }
        }
    }
}
} // namespace

int main()
{
    std::printf("thermal Enzyme kernel unit test (fb_enzyme_kernels)\n");
    test_dk1();
    test_dq1();
    test_nodal1();
    test_fb2();
    test_local_hess();
    test_source();
    std::printf("%s: %d of %d checks failed\n", g_fail ? "UNIT TEST FAILED" : "UNIT TEST PASSED", g_fail, g_checks);
    return g_fail ? 1 : 0;
}
