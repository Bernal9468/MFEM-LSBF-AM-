// anyorder_eval.hpp -- MFEM-free residual evaluator for the any-order
// sensitivity driver.
//
#include <cstdio>
#include <cstdlib>
// Given nodal Taylor-coefficient fields for T_next and T_n (coefficient-major
// layout [c*ndof + i], c = flat multi-index rank, normalized coefficients),
// AnyOrderEval::residual() evaluates the FULL discrete residual
//
//   R_i = (K(T)T)_i + Rb_i + Mc_i/dt (T_i - Tn_i) + (L(T_i) - L(Tn_i))/dt - f_src_i
//
// as a Taylor number in the query variables t_1..t_M and returns EVERY
// coefficient.  The driver reads coefficient alpha of order n (with the order-n
// unknown fields of T_next held at zero) as the forcing of the order-n
// sensitivity solve J T^(alpha) = -R_alpha.
//
// All raw arrays, no MFEM types, so each (M,N) shape is a small translation
// unit (eval_shape_<M>_<N>.cpp) behind the virtual interface below.
#pragma once

#include "oti_query_type.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace aoq {

// ---------------------------------------------------------------- parameters
static constexpr int NP = 10;
enum ParamId { P_K = 0, P_CP, P_RHO, P_LFUS, P_TS, P_TLIQ, P_HC, P_T0, P_P, P_ETA };
static const char *PARAM_NAMES[NP] = {"k", "cp", "rho", "Lfus", "Ts", "Tliq", "hc", "T0", "P", "Eta"};

// S_p = base[p] * (1 + sum_j W[p*M + j] * t_j)
struct ParamSpec {
    double base[NP];
    std::vector<double> W;   // NP x M, row-major
};

// ------------------------------------------------------------------ geometry
struct EvalGeom {
    int ndof = 0, ne = 0;
    std::vector<int> attr;                 // per element, mesh attribute
    std::vector<int> e_dof_off, vdofs;     // CSR of element dofs
    std::vector<int> e_q_off;              // qp offset per element (ne+1)
    std::vector<int> dim;                  // per element spatial dim
    std::vector<long> shape_off, gshape_off;  // offsets into shape / gshape
    std::vector<double> shape;             // [q][a] per element
    std::vector<double> gshape;            // [q][a][d] per element
    std::vector<double> weight;            // per global qp
};

struct EvalBoundary {
    std::vector<int> off, dof;             // CSR
    std::vector<double> shp;               // aligned with dof
    std::vector<double> w;                 // per qp
    std::vector<int> kind;                 // 1 substrate, 2 bridge, else interface
};

struct StepInputs {
    double dt = 1.0, h_conv = 0.0, emis = 0.0, sigma = 0.0, T_ambient = 0.0;
    double T_solidus = 0.0, T_liquidus = 0.0;
    double qscale[3] = {1.0, 1.0, 1.0};    // substrate, bridge, interface
    const char *attr_active = nullptr;     // indexed by mesh attribute
    const double *V_node = nullptr;
    const double *f_src = nullptr;         // already scaled by g_p*g_eta and masked
    const char *ess = nullptr;             // per dof, nonzero => residual forced to 0
};

// ----------------------------------------------------------------- interface
class AnyOrderEval {
public:
    virtual ~AnyOrderEval() {}
    virtual int M() const = 0;
    virtual int N() const = 0;
    virtual int NC() const = 0;
    virtual int order_of(int c) const = 0;
    virtual int order_offset(int n) const = 0;
    virtual double fact(int c) const = 0;
    virtual std::vector<int> alpha_of(int c) const = 0;
    virtual int index_of(const std::vector<int> &a) const = 0;
    virtual bool self_test() const = 0;
    virtual void set_geometry(const EvalGeom *g) = 0;
    virtual void set_params(const ParamSpec &ps) = 0;
    // Tnext, Tn: [c*ndof + i].  R (same layout) is overwritten.
    virtual void residual(const StepInputs &si, const EvalBoundary &bd,
                          const double *Tnext, const double *Tn, double *R) const = 0;
};

// Provided by eval_registry.cpp (catalog of compiled shapes).
std::unique_ptr<AnyOrderEval> make_eval(int M, int N);
std::vector<std::pair<int, int>> registered_shapes();

// ----------------------------------------------------------------- material
namespace mat {
static const double kTa[11] = {294.0, 311.0, 366.0, 477.0, 589.0, 700.0,
                               811.0, 922.0, 1033.0, 1144.0, 1255.0};
static const double kVa[11] = {9.8, 10.1, 10.8, 12.5, 14.1, 15.7,
                               17.5, 19.0, 20.8, 22.8, 25.2};
static const double cTa[11] = {294., 366., 477., 589., 700., 811., 922., 1033., 1144., 1255., 1366.};
static const double cVa[11] = {4.1e8, 4.27e8, 4.56e8, 4.81e8, 5.11e8, 5.36e8, 5.65e8, 5.90e8, 6.20e8, 6.45e8, 6.70e8};
static const double L_fusion = 2.72e11;
static const double rho0 = 8.44e-9;

template <class J>
inline J pwlin(const double *Ta, const double *Va, int n, const J &T) {
    double Tr = T.real();
    if (Tr <= Ta[0]) return J(Va[0]);
    if (Tr >= Ta[n - 1]) return J(Va[n - 1]);
    int i = 0;
    while (i < n - 2 && Ta[i + 1] < Tr) ++i;
    return (T - Ta[i]) * ((Va[i + 1] - Va[i]) / (Ta[i + 1] - Ta[i])) + Va[i];   // scalar ops only
}
}  // namespace mat

// ------------------------------------------------------------ implementation
template <int MV, int NV>
class AnyOrderEvalImpl : public AnyOrderEval {
public:
    using J = Jet<MV, NV>;
    static constexpr int NCO = J::NC;
    static constexpr int MAXDOF = 27;

    AlphaTable<MV, NV> tab;
    const EvalGeom *geom = nullptr;
    J S_K, S_CP, S_RHO, S_L, S_TS, S_TLIQ, S_HC, S_T0, S_P, S_ETA;

    int M() const override { return MV; }
    int N() const override { return NV; }
    int NC() const override { return J::NC; }
    int order_of(int c) const override { return tab.order[c]; }
    int order_offset(int n) const override { return tab.off[n]; }
    double fact(int c) const override { return tab.fact[c]; }
    std::vector<int> alpha_of(int c) const override {
        return std::vector<int>(tab.alpha[c].begin(), tab.alpha[c].end());
    }
    int index_of(const std::vector<int> &a) const override { return tab.index_of(a); }
    bool self_test() const override { return tab.self_test(); }
    void set_geometry(const EvalGeom *g) override { geom = g; }

    void set_params(const ParamSpec &ps) override {
        J *S[NP] = {&S_K, &S_CP, &S_RHO, &S_L, &S_TS, &S_TLIQ, &S_HC, &S_T0, &S_P, &S_ETA};
        for (int p = 0; p < NP; p++) {
            *S[p] = J(ps.base[p]);
            for (int j = 0; j < MV; j++) {
                std::vector<int> a(MV, 0); a[j] = 1;
                int c = tab.index_of(a);
                (*S[p])[c] = ps.base[p] * ps.W[(size_t)p * MV + j];
            }
        }
    }

    void residual(const StepInputs &si, const EvalBoundary &bd,
                  const double *Tnext, const double *Tn, double *R) const override {
        const EvalGeom &g = *geom;
        const int nd = g.ndof;
        std::fill(R, R + (size_t)NCO * nd, 0.0);

        // Temperature-dependent phase-change bracket (built after S_TS/S_TLIQ).
        const J Ts_j = J(si.T_solidus) * S_TS;
        const J Tl_j = J(si.T_liquidus) * S_TLIQ;
        const double Ts_r = si.T_solidus, Tl_r = si.T_liquidus;

        auto k_jet = [&](const J &T) { return mat::pwlin<J>(mat::kTa, mat::kVa, 11, T) * S_K; };
        auto cp_jet = [&](const J &T) { return mat::pwlin<J>(mat::cTa, mat::cVa, 11, T) * S_CP; };
        auto rho_jet = [&]() { return J(mat::rho0) * S_RHO; };
        auto fL_jet = [&](const J &T) -> J {
            double Tr = T.real();
            if (Tr <= Ts_r) return J(0.0);
            if (Tr >= Tl_r) return J(1.0);
            J xi = (T - Ts_j) / (Tl_j - Ts_j);
            return xi * xi * xi * (10.0 - 15.0 * xi + 6.0 * xi * xi);
        };

        for (int e = 0; e < g.ne; e++)
            if (g.e_dof_off[e + 1] - g.e_dof_off[e] > MAXDOF) throw std::runtime_error("element dof > MAXDOF");

        // ---- diffusion: sum_qp w * k(Tq) gradT . grad phi_a
#pragma omp parallel for schedule(dynamic, 64)
        for (int e = 0; e < g.ne; e++) {
            if (!si.attr_active[g.attr[e]]) continue;
            const int dof = g.e_dof_off[e + 1] - g.e_dof_off[e];
            const int *vd = &g.vdofs[g.e_dof_off[e]];
            const int dm = g.dim[e];
            const int nq = g.e_q_off[e + 1] - g.e_q_off[e];
            double Tloc[NCO][MAXDOF];
            double Rl[NCO][MAXDOF];
            for (int c = 0; c < NCO; c++)
                for (int a = 0; a < dof; a++) { Tloc[c][a] = Tnext[(size_t)c * nd + vd[a]]; Rl[c][a] = 0.0; }
            for (int q = 0; q < nq; q++) {
                const double *sh = &g.shape[g.shape_off[e] + (long)q * dof];
                const double *gs = &g.gshape[g.gshape_off[e] + (long)q * dof * dm];
                const double w = g.weight[g.e_q_off[e] + q];
                J Tq, gT[3];
                for (int c = 0; c < NCO; c++) {
                    double s = 0.0;
                    for (int a = 0; a < dof; a++) s += sh[a] * Tloc[c][a];
                    Tq[c] = s;
                    for (int d = 0; d < dm; d++) {
                        double gsum = 0.0;
                        for (int a = 0; a < dof; a++) gsum += gs[a * dm + d] * Tloc[c][a];
                        gT[d][c] = gsum;
                    }
                }
                const J kq = k_jet(Tq);
                J flux[3];
                for (int d = 0; d < dm; d++) flux[d] = kq * gT[d];
                for (int a = 0; a < dof; a++)
                    for (int c = 0; c < NCO; c++) {
                        double acc = 0.0;
                        for (int d = 0; d < dm; d++) acc += gs[a * dm + d] * flux[d][c];
                        Rl[c][a] += acc * w;
                    }
            }
            // v7 perf: one atomic per (dof, coefficient) per element instead of per quadrature point
            for (int a = 0; a < dof; a++)
                for (int c = 0; c < NCO; c++)
                    if (Rl[c][a] != 0.0) {
#pragma omp atomic
                        R[(size_t)c * nd + vd[a]] += Rl[c][a];
                    }
        }

        // ---- boundary film + radiation
        if (si.emis > 0.0 || si.h_conv > 0.0) {
            const int nqp = (int)bd.w.size();
#pragma omp parallel for schedule(dynamic, 512)
            for (int qi = 0; qi < nqp; qi++) {
                const int b0 = bd.off[qi], b1 = bd.off[qi + 1];
                const double qs = (bd.kind[qi] == 1) ? si.qscale[0]
                                : (bd.kind[qi] == 2) ? si.qscale[1] : si.qscale[2];
                J Tq;
                for (int c = 0; c < NCO; c++) {
                    double s = 0.0;
                    for (int a = b0; a < b1; a++) s += bd.shp[a] * Tnext[(size_t)c * nd + bd.dof[a]];
                    Tq[c] = s;
                }
                const J hconv = S_HC * (qs * si.h_conv);
                const J Tinf = S_T0 * si.T_ambient;
                const J T2 = Tq * Tq, Ti2 = Tinf * Tinf;
                const J qval = hconv * (Tq - Tinf) + (T2 * T2 - Ti2 * Ti2) * (qs * si.emis * si.sigma);
                for (int a = b0; a < b1; a++)
                    for (int c = 0; c < NCO; c++) {
                        double v = qval[c] * bd.shp[a] * bd.w[qi];
                        if (v != 0.0) {
#pragma omp atomic
                            R[(size_t)c * nd + bd.dof[a]] += v;
                        }
                    }
            }
        }

        // ---- nodal capacitive + latent + source
        const double idt = 1.0 / si.dt;
        const J rho = rho_jet();
        const J Lf = J(mat::L_fusion) * S_L;
        const J SPE = S_P * S_ETA;
        const J rhoLf = rho * Lf;
        // v7 perf: same terms as Vi*rho*cp*(Tx-Tp)/dt + Vi*rho*Lf*(fL(Tx)-fL(Tp))/dt - f_src*P*eta, regrouped so scalars
        // stay scalars (2-3 Jet products per node instead of ~9); the latent term is skipped when fL(Tx) and fL(Tp)
        // are the same constant branch (both solid or both liquid), where it is exactly zero.
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nd; i++) {
            J Tx, Tp;
            for (int c = 0; c < NCO; c++) { Tx[c] = Tnext[(size_t)c * nd + i]; Tp[c] = Tn[(size_t)c * nd + i]; }
            const double Vdt = si.V_node[i] * idt;
            J nodal = (rho * cp_jet(Tx)) * (Tx - Tp) * Vdt;
            if (getenv("AOBLOCKC_TRACE") && i == atoi(getenv("AOBLOCKC_TRACE"))) {
                printf("  [AOBLOCKC] i=%d Tx0=%.6f Tp0=%.6f Tx1=%.6e Tp1=%.6e nodal1=%.6e idt=%.6e rho=%.6e cp=%.6e\n",
                       i, Tx.real(), Tp.real(), Tx[1], Tp[1], nodal[1], idt, rho.real(), cp_jet(Tx).real());
            }
            const double tx = Tx.real(), tp = Tp.real();
            const bool same_branch = (tx <= Ts_r && tp <= Ts_r) || (tx >= Tl_r && tp >= Tl_r);
            if (!same_branch) nodal += rhoLf * (fL_jet(Tx) - fL_jet(Tp)) * Vdt;
            if (si.f_src[i] != 0.0) nodal -= SPE * si.f_src[i];
            for (int c = 0; c < NCO; c++) R[(size_t)c * nd + i] += nodal[c];
        }

        // ---- essential dofs
        for (int i = 0; i < nd; i++)
            if (si.ess[i])
                for (int c = 0; c < NCO; c++) R[(size_t)c * nd + i] = 0.0;
    }

};

}  // namespace aoq
