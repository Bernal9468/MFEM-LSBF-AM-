// standalone test of the residual-stress element kernel on a unit-size hex
#include "rs_kernels.h"
#include "rs_kernel.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
static const double NODE[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
static void geometry(double h, double *geom)
{
    const double gp = 0.5773502691896257, sgn[2] = {-1.0, 1.0};
    int g = 0;
    for (int k = 0; k < 2; k++) for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++, g++) {
        const double x[3] = {0.5 + 0.5 * gp * sgn[i], 0.5 + 0.5 * gp * sgn[j], 0.5 + 0.5 * gp * sgn[k]};
        for (int a = 0; a < 8; a++) {
            for (int d = 0; d < 3; d++) {
                double v = 1.0;
                for (int c = 0; c < 3; c++) {
                    const double s = NODE[a][c] > 0.5 ? x[c] : 1.0 - x[c];
                    const double ds = NODE[a][c] > 0.5 ? 1.0 : -1.0;
                    v *= (c == d) ? ds : s;
                }
                geom[g * 25 + 3 * a + d] = v / h;
            }
        }
        geom[g * 25 + 24] = 0.125 * h * h * h;
    }
}
int main()
{
    const double h = 1.0; double geom[200]; geometry(h, geom);
    double par[5] = {1, 1, 1, 1, 1}, Tq[8], T0q[8], st[56] = {0}, u[24];
    for (int g = 0; g < 8; g++) { Tq[g] = 800.0; T0q[g] = 313.15; }
    double fint[24], stn[56], sig[48];
    // A: free uniform thermal expansion -> zero stress / force
    const double eth = rs::interp(rs::Ta, rs::alphat, 800.0) * (800.0 - rs::T_ZERO) - rs::interp(rs::Ta, rs::alphat, 313.15) * (313.15 - rs::T_ZERO);
    for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) u[3 * a + c] = eth * NODE[a][c] * h;
    rs_elem(u, Tq, T0q, st, par, geom, fint, stn, sig);
    double fmax = 0, smax = 0; for (int k = 0; k < 24; k++) fmax = std::max(fmax, std::fabs(fint[k])); for (int k = 0; k < 48; k++) smax = std::max(smax, std::fabs(sig[k]));
    printf("A free expansion: |f|max=%.3e |sigma|max=%.3e (expect ~0)\n", fmax, smax);
    // B: tangent AD vs FD, plastic (large random strain) with nonzero committed plastic state
    srand(3);
    for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) u[3 * a + c] = 0.02 * ((double)rand() / RAND_MAX - 0.3) + eth * NODE[a][c] * h;
    for (int g = 0; g < 8; g++) { st[g * 7 + 0] = 1e-3; st[g * 7 + 1] = -5e-4; st[g * 7 + 2] = -5e-4; st[g * 7 + 6] = 2e-3; }
    rs_elem(u, Tq, T0q, st, par, geom, fint, stn, sig);
    double peeq_max = 0; for (int g = 0; g < 8; g++) peeq_max = std::max(peeq_max, stn[g * 7 + 6]);
    printf("B plastic state: max peeq_new=%.3e  sigma[0]=%.4f\n", peeq_max, sig[0]);
    double worst = 0, scale = 0;
    for (int k = 0; k < 24; k++) {
        double du[24] = {0}, dfi[24], dstn[56], dsig[48], dst0[56] = {0}, dp0[5] = {0}, f2[24], s2[56], sg2[48];
        du[k] = 1.0;
        rs_elem_d1(u, du, Tq, T0q, st, dst0, par, dp0, geom, f2, dfi, s2, dstn, sg2, dsig);
        double up[24], um[24], fp[24], fm[24], sp_[56], sm_[56], sgp[48], sgm[48]; memcpy(up, u, sizeof u); memcpy(um, u, sizeof u);
        const double d = 1e-6; up[k] += d; um[k] -= d;
        rs_elem(up, Tq, T0q, st, par, geom, fp, sp_, sgp); rs_elem(um, Tq, T0q, st, par, geom, fm, sm_, sgm);
        for (int i = 0; i < 24; i++) { const double fd = (fp[i] - fm[i]) / (2 * d); worst = std::max(worst, std::fabs(fd - dfi[i])); scale = std::max(scale, std::fabs(fd)); }
    }
    printf("B tangent: max|AD-FD| = %.3e   (max|K| = %.3e)  rel %.2e\n", worst, scale, worst / scale);
    // C: parameter derivatives
    for (int p = 0; p < 5; p++) {
        double dp[5] = {0}, du0[24] = {0}, dst0[56] = {0}, f2[24], dfi[24], s2[56], dstn[56], sg2[48], dsig[48];
        dp[p] = 1.0;
        rs_elem_d1(u, du0, Tq, T0q, st, dst0, par, dp, geom, f2, dfi, s2, dstn, sg2, dsig);
        double pp[5], pm[5], fp[24], fm[24], sp_[56], sm_[56], sgp[48], sgm[48]; memcpy(pp, par, sizeof pp); memcpy(pm, par, sizeof pm);
        const double d = 1e-6; pp[p] += d; pm[p] -= d;
        rs_elem(u, Tq, T0q, st, pp, geom, fp, sp_, sgp); rs_elem(u, Tq, T0q, st, pm, geom, fm, sm_, sgm);
        double w = 0, sc = 0, ws = 0, scs = 0;
        for (int i = 0; i < 24; i++) { const double fd = (fp[i] - fm[i]) / (2 * d); w = std::max(w, std::fabs(fd - dfi[i])); sc = std::max(sc, std::fabs(fd)); }
        for (int i = 0; i < 48; i++) { const double fd = (sgp[i] - sgm[i]) / (2 * d); ws = std::max(ws, std::fabs(fd - dsig[i])); scs = std::max(scs, std::fabs(fd)); }
        printf("C param %d: df: max|AD-FD|=%.2e (scale %.2e)   dsigma: %.2e (scale %.2e)\n", p, w, sc, ws, scs);
    }
    // D: elastic state (small strain)
    for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) u[3 * a + c] = 1e-4 * NODE[a][c] * (c + 1) + eth * NODE[a][c] * h;
    memset(st, 0, sizeof st);
    rs_elem(u, Tq, T0q, st, par, geom, fint, stn, sig);
    double pe = 0; for (int g = 0; g < 8; g++) pe = std::max(pe, stn[g * 7 + 6]);
    printf("D small strain: peeq=%.3e sigma_xx=%.4f (elastic expected E*~1e-4 = %.4f)\n", pe, sig[0], rs::interp(rs::Tt, rs::Et, 800.0) * 1e-4 * 1.0 * 1.0);
    // restore the plastic configuration of test B for the second-derivative test
    srand(11);
    for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) u[3 * a + c] = 0.02 * ((double)rand() / RAND_MAX - 0.3) + eth * NODE[a][c] * h;
    for (int g = 0; g < 8; g++) { for (int k = 0; k < 7; k++) st[g * 7 + k] = 0.0; st[g * 7 + 0] = 1e-3; st[g * 7 + 1] = -5e-4; st[g * 7 + 2] = -5e-4; st[g * 7 + 6] = 2e-3; }
    { double f_[24], sn_[56], sg_[48]; rs_elem(u, Tq, T0q, st, par, geom, f_, sn_, sg_); double pm_ = 0; for (int g = 0; g < 8; g++) pm_ = std::max(pm_, sn_[g * 7 + 6]); printf("E plastic config: max peeq_new = %.3e (peeq_old 2e-3)\n", pm_); }
    // E: second derivative D_b D_a vs central FD of the first derivative (directions mix u, old state and parameters)
    {
        double a_u[24], a_st[56] = {0}, a_p[5] = {0}, b_u[24], b_st[56] = {0}, b_p[5] = {0};
        for (int k = 0; k < 24; k++) { a_u[k] = 1e-3 * ((double)rand() / RAND_MAX - 0.5); b_u[k] = 1e-3 * ((double)rand() / RAND_MAX - 0.5); }
        for (int g = 0; g < 8; g++) for (int k = 0; k < 7; k++) { a_st[g * 7 + k] = 1e-5 * ((double)rand() / RAND_MAX - 0.5); b_st[g * 7 + k] = 1e-5 * ((double)rand() / RAND_MAX - 0.5); }
        a_p[0] = 0.3; a_p[3] = -0.2; b_p[1] = 0.5; b_p[4] = 0.4; b_p[2] = 0.1;
        double f[24], df[24], sn[56], dsn[56], sg[48], dsg[48], bf[24], ddf[24], bsn[56], ddsn[56], bsg[48], ddsg[48];
        rs_elem_d2(u, a_u, Tq, T0q, st, a_st, par, a_p, geom, b_u, b_st, b_p, f, df, sn, dsn, sg, dsg, bf, ddf, bsn, ddsn, bsg, ddsg);
        const double d = 1e-5; double wf = 0, sf = 0, ws = 0, ss = 0, wst = 0, sst = 0;
        double up[24], um[24], stp[56], stm[56], pp[5], pm[5];
        for (int k = 0; k < 24; k++) { up[k] = u[k] + d * b_u[k]; um[k] = u[k] - d * b_u[k]; }
        for (int k = 0; k < 56; k++) { stp[k] = st[k] + d * b_st[k]; stm[k] = st[k] - d * b_st[k]; }
        for (int k = 0; k < 5; k++) { pp[k] = par[k] + d * b_p[k]; pm[k] = par[k] - d * b_p[k]; }
        double fp[24], dfp[24], snp[56], dsnp[56], sgp[48], dsgp[48], fm[24], dfm[24], snm[56], dsnm[56], sgm[48], dsgm[48];
        rs_elem_d1(up, a_u, Tq, T0q, stp, a_st, pp, a_p, geom, fp, dfp, snp, dsnp, sgp, dsgp);
        rs_elem_d1(um, a_u, Tq, T0q, stm, a_st, pm, a_p, geom, fm, dfm, snm, dsnm, sgm, dsgm);
        for (int i = 0; i < 24; i++) { const double fd = (dfp[i] - dfm[i]) / (2 * d); wf = std::max(wf, std::fabs(fd - ddf[i])); sf = std::max(sf, std::fabs(fd)); }
        for (int i = 0; i < 48; i++) { const double fd = (dsgp[i] - dsgm[i]) / (2 * d); ws = std::max(ws, std::fabs(fd - ddsg[i])); ss = std::max(ss, std::fabs(fd)); }
        for (int i = 0; i < 56; i++) { const double fd = (dsnp[i] - dsnm[i]) / (2 * d); wst = std::max(wst, std::fabs(fd - ddsn[i])); sst = std::max(sst, std::fabs(fd)); }
        printf("E second derivative D_bD_a: f: %.2e (scale %.2e)  sigma: %.2e (scale %.2e)  state: %.2e (scale %.2e)\n", wf, sf, ws, ss, wst, sst);
    }
    return 0;
}
