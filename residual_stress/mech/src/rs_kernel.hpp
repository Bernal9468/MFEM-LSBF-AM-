// rs_kernel.hpp -- plain-array Hill/J2 thermoplastic element kernel (B-bar hex, 8 Gauss points) for the residual-stress work.
// Same constitutive model as bridge_4layer_kokkos_OTI/mechanical/src/hill_thermoplastic_material.hpp (IN625 tables, Hill
// ratios, linear isotropic hardening capped at sigma_y1), written without any container so Enzyme forward mode can differentiate it.
// Units: mm, MPa, N, K.   Mandel 6-vectors: (xx, yy, zz, sqrt2 xy, sqrt2 xz, sqrt2 yz).
// par[5] = scale factors on the nominal curves: (E, nu, CTE, yield stress, hardening modulus) -- all 1.0 at the nominal state.
#pragma once
#include <cmath>
namespace rs {
constexpr int NQ = 8, NN = 8, NST = 7;                 // quadrature points, nodes, state doubles per qp (ep[6], peeq)
constexpr int GEOM_PER_QP = 25;                        // dN/dx (8 nodes x 3) + weight (detJ * ref weight)
constexpr double T_ANNEAL = 1563.0;                    // plastic state is reset above this temperature (solidus)
constexpr double T_ZERO = 298.15;                      // expansion zero temperature
constexpr double SY0 = 725.0, SY1 = 990.0, EPS_P1 = 0.35;
constexpr double R11 = 1.0, R22 = 1.0, R33 = 0.8483, R12 = 1.0, R13 = 1.0, R23 = 1.0;
constexpr int NTE = 9;
static const double Tt[NTE]  = {294.15, 366.15, 477.15, 589.15, 700.15, 811.15, 922.15, 1033.15, 1144.15};
static const double Et[NTE]  = {207500., 204100., 197900., 191700., 185500., 178600., 170300., 160600., 147500.};
static const double nut[NTE] = {0.278, 0.28, 0.286, 0.29, 0.295, 0.305, 0.321, 0.34, 0.336};
static const double Ta[NTE]  = {366.15, 477.15, 589.15, 700.15, 811.15, 922.15, 1033.15, 1144.15, 1200.15};
static const double alphat[NTE] = {1.28e-05, 1.31e-05, 1.33e-05, 1.37e-05, 1.4e-05, 1.48e-05, 1.53e-05, 1.58e-05, 1.62e-05};

inline double interp(const double *xt, const double *yt, double T)
{
    if (T <= xt[0]) return yt[0];
    if (T >= xt[NTE - 1]) return yt[NTE - 1];
    int i = 0;
    while (i < NTE - 2 && T > xt[i + 1]) ++i;
    const double f = (T - xt[i]) / (xt[i + 1] - xt[i]);
    return yt[i] + f * (yt[i + 1] - yt[i]);
}

inline void solve6(double A[6][6], const double b[6], double x[6])
{
    double M[6][7];
    for (int i = 0; i < 6; i++) { for (int j = 0; j < 6; j++) M[i][j] = A[i][j]; M[i][6] = b[i]; }
    for (int col = 0; col < 6; col++) {
        int piv = col; double best = std::fabs(M[col][col]);
        for (int r = col + 1; r < 6; r++) { const double v = std::fabs(M[r][col]); if (v > best) { best = v; piv = r; } }
        if (piv != col) for (int j = 0; j < 7; j++) { const double t = M[col][j]; M[col][j] = M[piv][j]; M[piv][j] = t; }
        const double diag = M[col][col];
        for (int r = 0; r < 6; r++) {
            if (r == col) continue;
            const double f = M[r][col] / diag;
            for (int j = col; j < 7; j++) M[r][j] -= f * M[col][j];
        }
    }
    for (int i = 0; i < 6; i++) x[i] = M[i][6] / M[i][i];
}

inline void hill_P(double P[6][6])
{
    const double F = 0.5 * (1.0 / (R22 * R22) + 1.0 / (R33 * R33) - 1.0 / (R11 * R11));
    const double G = 0.5 * (1.0 / (R33 * R33) + 1.0 / (R11 * R11) - 1.0 / (R22 * R22));
    const double Hh = 0.5 * (1.0 / (R11 * R11) + 1.0 / (R22 * R22) - 1.0 / (R33 * R33));
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) P[i][j] = 0.0;
    P[0][0] = Hh + G; P[1][1] = F + Hh; P[2][2] = F + G;
    P[0][1] = P[1][0] = -Hh; P[0][2] = P[2][0] = -G; P[1][2] = P[2][1] = -F;
    P[3][3] = 1.5 / (R12 * R12); P[4][4] = 1.5 / (R13 * R13); P[5][5] = 1.5 / (R23 * R23);
}

inline double sigma_eff(const double s[6], const double P[6][6])
{
    double q = 0.0;
    for (int i = 0; i < 6; i++) { double ps = 0.0; for (int j = 0; j < 6; j++) ps += P[i][j] * s[j]; q += s[i] * ps; }
    return std::sqrt(q > 0.0 ? q : 0.0);
}

// Total-strain Hill return map. eps: total strain (Mandel); T: temperature; T0: initial temperature (stress-free reference).
inline void return_map(const double eps[6], double T, double T0, const double ep_old_in[6], double peeq_in, const double par[5],
                       double sig[6], double ep_new[6], double &peeq_new)
{
    double ep_old[6]; double peeq_old = peeq_in;
    for (int i = 0; i < 6; i++) ep_old[i] = ep_old_in[i];
    if (T > T_ANNEAL) { for (int i = 0; i < 6; i++) ep_old[i] = 0.0; peeq_old = 0.0; }
    const double E = par[0] * interp(Tt, Et, T), nu = par[1] * interp(Tt, nut, T);
    const double mu = E / (2.0 * (1.0 + nu)), lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double eth = par[2] * (interp(Ta, alphat, T) * (T - T_ZERO) - interp(Ta, alphat, T0) * (T0 - T_ZERO));
    double ee[6];
    for (int i = 0; i < 6; i++) ee[i] = eps[i] - ep_old[i];
    ee[0] -= eth; ee[1] -= eth; ee[2] -= eth;
    double C[6][6], P[6][6], M[6][6];
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) C[i][j] = 0.0;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) C[i][j] = lam;
    for (int i = 0; i < 3; i++) C[i][i] += 2.0 * mu;
    C[3][3] = C[4][4] = C[5][5] = 2.0 * mu;
    hill_P(P);
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) { double s = 0.0; for (int k = 0; k < 6; k++) s += C[i][k] * P[k][j]; M[i][j] = s; }
    double str[6];
    for (int i = 0; i < 6; i++) { double s = 0.0; for (int j = 0; j < 6; j++) s += C[i][j] * ee[j]; str[i] = s; }
    const double sy0 = par[3] * SY0, Hs = par[4] * (SY1 - SY0) / EPS_P1;
    const double sy_old = sy0 + Hs * (peeq_old < EPS_P1 ? peeq_old : EPS_P1);
    const double se_trial = sigma_eff(str, P);
    double beta = 0.0; double sv[6];
    for (int i = 0; i < 6; i++) sv[i] = str[i];
    if (se_trial > sy_old) {
        for (int it = 0; it < 40; it++) {
            double A[6][6];
            for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) A[i][j] = (i == j ? 1.0 : 0.0) + beta * M[i][j];
            solve6(A, str, sv);
            double Ms[6], t[6], Ps[6];
            for (int i = 0; i < 6; i++) { double s = 0.0; for (int j = 0; j < 6; j++) s += M[i][j] * sv[j]; Ms[i] = s; }
            solve6(A, Ms, t);                                            // dsigma/dbeta = -t
            for (int i = 0; i < 6; i++) { double s = 0.0; for (int j = 0; j < 6; j++) s += P[i][j] * sv[j]; Ps[i] = s; }
            const double se = sigma_eff(sv, P);
            double dse = 0.0; for (int i = 0; i < 6; i++) dse += Ps[i] * (-t[i]); dse /= se;
            const double x = peeq_old + beta * se;
            const double g = se - (sy0 + Hs * (x < EPS_P1 ? x : EPS_P1));
            const double dg = dse - (x < EPS_P1 ? Hs * (se + beta * dse) : 0.0);
            if (std::fabs(g) < 1e-12 * sy_old) break;
            double bn = beta - g / dg; if (bn < 0.0) bn = 0.0;
            beta = bn;
        }
        double A[6][6];
        for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) A[i][j] = (i == j ? 1.0 : 0.0) + beta * M[i][j];
        solve6(A, str, sv);
    }
    const double se_f = sigma_eff(sv, P);
    const double dg_f = beta * se_f;
    peeq_new = peeq_old + dg_f;
    for (int i = 0; i < 6; i++) sig[i] = sv[i];
    for (int i = 0; i < 6; i++) ep_new[i] = ep_old[i];
    if (dg_f > 0.0 && se_f > 1e-12) {
        const double f = dg_f / se_f;
        for (int i = 0; i < 6; i++) { double s = 0.0; for (int j = 0; j < 6; j++) s += P[i][j] * sv[j]; ep_new[i] += f * s; }
    }
}

// Element response. u[24] (node-major: u[3*a+c]), Tq/T0q: temperature / initial temperature at the 8 qps, st_old[56],
// geom[8*25] (dN/dx per node then weight), par[5].  Outputs: fint[24], st_new[56], sig_out[48] (Mandel stress per qp).
inline void elem_response(const double *u, const double *Tq, const double *T0q, const double *st_old, const double *par,
                          const double *geom, double *fint, double *st_new, double *sig_out)
{
    const double s2 = 1.4142135623730951;
    double e[NQ][3][3], trg[NQ], wq[NQ], wsum = 0.0, trbar = 0.0;
    for (int g = 0; g < NQ; g++) {
        const double *gm = geom + g * GEOM_PER_QP;
        double gu[3][3];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { double s = 0.0; for (int a = 0; a < NN; a++) s += u[3 * a + i] * gm[3 * a + j]; gu[i][j] = s; }
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) e[g][i][j] = 0.5 * (gu[i][j] + gu[j][i]);
        trg[g] = e[g][0][0] + e[g][1][1] + e[g][2][2];
        wq[g] = gm[24]; wsum += wq[g]; trbar += wq[g] * trg[g];
    }
    trbar /= wsum;
    double S[NQ][6];
    for (int g = 0; g < NQ; g++) {
        const double dv = (trbar - trg[g]) / 3.0;
        double eps[6] = {e[g][0][0] + dv, e[g][1][1] + dv, e[g][2][2] + dv, s2 * e[g][0][1], s2 * e[g][0][2], s2 * e[g][1][2]};
        double ep_new[6], peeq_new;
        return_map(eps, Tq[g], T0q[g], st_old + g * NST, st_old[g * NST + 6], par, S[g], ep_new, peeq_new);
        for (int i = 0; i < 6; i++) { st_new[g * NST + i] = ep_new[i]; sig_out[g * 6 + i] = S[g][i]; }
        st_new[g * NST + 6] = peeq_new;
    }
    double gbar[NN][3];
    for (int a = 0; a < NN; a++) for (int i = 0; i < 3; i++) { double s = 0.0; for (int g = 0; g < NQ; g++) s += wq[g] * geom[g * GEOM_PER_QP + 3 * a + i]; gbar[a][i] = s / wsum; }
    for (int k = 0; k < 24; k++) fint[k] = 0.0;
    for (int g = 0; g < NQ; g++) {
        const double *gm = geom + g * GEOM_PER_QP;
        const double is2 = 1.0 / s2;
        const double Sg[3][3] = {{S[g][0], is2 * S[g][3], is2 * S[g][4]}, {is2 * S[g][3], S[g][1], is2 * S[g][5]}, {is2 * S[g][4], is2 * S[g][5], S[g][2]}};
        const double tr3 = (S[g][0] + S[g][1] + S[g][2]) / 3.0;
        for (int a = 0; a < NN; a++) for (int i = 0; i < 3; i++) {
            double s = 0.0;
            for (int j = 0; j < 3; j++) s += Sg[i][j] * gm[3 * a + j];
            s += tr3 * (gbar[a][i] - gm[3 * a + i]);
            fint[3 * a + i] += wq[g] * s;
        }
    }
}
} // namespace rs
