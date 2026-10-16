// randomised test of the second derivative D_bD_a of the element kernel against central FD of the first derivative
#include "rs_kernels.h"
#include "rs_kernel.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
static const double NODE[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
static void geometry(double *geom)
{
    const double gp = 0.5773502691896257, sgn[2] = {-1.0, 1.0}; int g = 0;
    for (int k = 0; k < 2; k++) for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++, g++) {
        const double x[3] = {0.5 + 0.5 * gp * sgn[i], 0.5 + 0.5 * gp * sgn[j], 0.5 + 0.5 * gp * sgn[k]};
        for (int a = 0; a < 8; a++) for (int d = 0; d < 3; d++) { double v = 1.0;
            for (int c = 0; c < 3; c++) { const double s = NODE[a][c] > 0.5 ? x[c] : 1.0 - x[c]; const double ds = NODE[a][c] > 0.5 ? 1.0 : -1.0; v *= (c == d) ? ds : s; }
            geom[g * 25 + 3 * a + d] = v; }
        geom[g * 25 + 24] = 0.125;
    }
}
static double U() { return (double)rand() / RAND_MAX - 0.5; }
int main(int argc, char **argv)
{
    const int N = argc > 1 ? atoi(argv[1]) : 3000; srand(5); double geom[200]; geometry(geom);
    struct Res { double err, T, amp, peeq; int plastic; };
    std::vector<Res> out;
    for (int n = 0; n < N; n++) {
        double par[5] = {1, 1, 1, 1, 1}, Tq[8], T0q[8], st[56] = {0}, u[24];
        const double T = 300.0 + 1900.0 * (double)rand() / RAND_MAX; const double amp = pow(10.0, -3.5 + 2.0 * (double)rand() / RAND_MAX);
        for (int g = 0; g < 8; g++) { Tq[g] = T; T0q[g] = 313.0; }
        const double eth = rs::interp(rs::Ta, rs::alphat, T) * (T - rs::T_ZERO) - rs::interp(rs::Ta, rs::alphat, 313.0) * (313.0 - rs::T_ZERO);
        for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) u[3 * a + c] = amp * U() * 2 + eth * NODE[a][c];
        if (rand() % 2) for (int g = 0; g < 8; g++) { st[g * 7 + 0] = 2e-3 * U(); st[g * 7 + 1] = 2e-3 * U(); st[g * 7 + 2] = -(st[g * 7 + 0] + st[g * 7 + 1]); st[g * 7 + 6] = 5e-3 * (double)rand() / RAND_MAX; }
        double a_u[24], b_u[24], a_st[56] = {0}, b_st[56] = {0}, a_p[5] = {0}, b_p[5] = {0};
        for (int k = 0; k < 24; k++) { a_u[k] = 1e-3 * U(); b_u[k] = 1e-3 * U(); }
        for (int g = 0; g < 8; g++) for (int k = 0; k < 7; k++) { a_st[g * 7 + k] = 1e-5 * U(); b_st[g * 7 + k] = 1e-5 * U(); }
        for (int k = 0; k < 5; k++) { a_p[k] = U(); b_p[k] = U(); }
        double f[24], df[24], sn[56], dsn[56], sg[48], dsg[48], bf[24], ddf[24], bsn[56], ddsn[56], bsg[48], ddsg[48];
        rs_elem_d2(u, a_u, Tq, T0q, st, a_st, par, a_p, geom, b_u, b_st, b_p, f, df, sn, dsn, sg, dsg, bf, ddf, bsn, ddsn, bsg, ddsg);
        const double d = 1e-6; double up[24], um[24], stp[56], stm[56], pp[5], pm[5];
        for (int k = 0; k < 24; k++) { up[k] = u[k] + d * b_u[k]; um[k] = u[k] - d * b_u[k]; }
        for (int k = 0; k < 56; k++) { stp[k] = st[k] + d * b_st[k]; stm[k] = st[k] - d * b_st[k]; }
        for (int k = 0; k < 5; k++) { pp[k] = par[k] + d * b_p[k]; pm[k] = par[k] - d * b_p[k]; }
        double fp[24], dfp[24], snp[56], dsnp[56], sgp[48], dsgp[48], fm[24], dfm[24], snm[56], dsnm[56], sgm[48], dsgm[48];
        rs_elem_d1(up, a_u, Tq, T0q, stp, a_st, pp, a_p, geom, fp, dfp, snp, dsnp, sgp, dsgp);
        rs_elem_d1(um, a_u, Tq, T0q, stm, a_st, pm, a_p, geom, fm, dfm, snm, dsnm, sgm, dsgm);
        double w = 0, s = 0; for (int i = 0; i < 48; i++) { const double fd = (dsgp[i] - dsgm[i]) / (2 * d); w = std::max(w, std::fabs(fd - ddsg[i])); s = std::max(s, std::fabs(fd)); }
        double pe = 0; for (int g = 0; g < 8; g++) pe = std::max(pe, sn[g * 7 + 6] - st[g * 7 + 6]);
        out.push_back({w / std::max(s, 1e-8), T, amp, pe, pe > 0});
    }
    std::sort(out.begin(), out.end(), [](const Res &a, const Res &b) { return a.err > b.err; });
    int bad = 0, bad_pl = 0, npl = 0; for (auto &r : out) { if (r.plastic) npl++; if (r.err > 1e-4) { bad++; if (r.plastic) bad_pl++; } }
    printf("cases %d (plastic %d): relative error of d2 sigma > 1e-4 in %d cases (%d plastic)\n", N, npl, bad, bad_pl);
    printf("worst 8: err, T, strain amp, d(peeq)\n"); for (int k = 0; k < 8; k++) printf("  %.2e  T=%.0f  amp=%.2e  dpeeq=%.2e\n", out[k].err, out[k].T, out[k].amp, out[k].peeq);
    return 0;
}
