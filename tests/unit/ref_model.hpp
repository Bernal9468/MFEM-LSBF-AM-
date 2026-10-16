// ref_model.hpp -- reference model for the thermal Enzyme kernels (opt/src/fb_enzyme_kernels.*).
//
// * Physics transcribed from the DRIVER's forward model (opt/src/full_bridge_activation_tier1_oti.cpp, namespace
//   IN625: k(), cp_dev(), rho_dev(), f_L_dev(), and the powder blends of PersistentK::RefillInto / UpdateBoundary /
//   UpdateCapacitance), NOT from fb_enzyme_kernels.cpp: a table or formula that drifts in the kernels shows up as a
//   test failure instead of being copied into the reference.
// * Templated on the scalar type. Instantiated with ref::HD (hyper-dual numbers, x + a e1 + b e2 + ab e1e2,
//   e1^2 = e2^2 = 0) it gives EXACT first, second and mixed directional derivatives by operator overloading -- an AD
//   method independent of Enzyme, with no finite-difference step to tune.
//
// Scale slots (fb_enzyme_kernels.h): 0 k, 1 cp, 2 rho, 3 Lfus, 4 Ts, 5 Tl, 6 hc, 7 T0, 8 kp, 9 rho_p, 10 hp.
#pragma once
#include <cmath>

namespace ref {
enum { S_K = 0, S_CP, S_RHO, S_L, S_TS, S_TL, S_HC, S_T0, S_KP, S_RHOP, S_HP, NSLOT };

// ---------------------------------------------------------------------------------------------------------------
// hyper-dual number: v + a e1 + b e2 + ab e1e2.  f(x + e1 da + e2 db) = f + f' da e1 + f' db e2 + f'' da db e1e2
struct HD {
    double v, a, b, ab;
    HD(double x = 0.0) : v(x), a(0.0), b(0.0), ab(0.0) {}
    HD(double v_, double a_, double b_, double ab_) : v(v_), a(a_), b(b_), ab(ab_) {}
};
inline HD operator+(const HD &x, const HD &y) { return {x.v + y.v, x.a + y.a, x.b + y.b, x.ab + y.ab}; }
inline HD operator-(const HD &x, const HD &y) { return {x.v - y.v, x.a - y.a, x.b - y.b, x.ab - y.ab}; }
inline HD operator-(const HD &x) { return {-x.v, -x.a, -x.b, -x.ab}; }
inline HD operator*(const HD &x, const HD &y)
{ return {x.v * y.v, x.v * y.a + x.a * y.v, x.v * y.b + x.b * y.v, x.v * y.ab + x.a * y.b + x.b * y.a + x.ab * y.v}; }
// scalar function lift: f, f', f'' at x.v
inline HD lift(const HD &x, double f, double f1, double f2) { return {f, f1 * x.a, f1 * x.b, f1 * x.ab + f2 * x.a * x.b}; }
inline HD inv(const HD &y) { const double i = 1.0 / y.v; return lift(y, i, -i * i, 2.0 * i * i * i); }
inline HD operator/(const HD &x, const HD &y) { return x * inv(y); }
inline HD exp(const HD &x) { const double e = std::exp(x.v); return lift(x, e, e, e); }
inline HD srck_exp(const HD &x) { return exp(x); }   // found by ADL inside src_kernel.hpp's srck_eval<HD>
inline bool operator<(const HD &x, const HD &y) { return x.v < y.v; }
inline bool operator<=(const HD &x, const HD &y) { return x.v <= y.v; }
inline bool operator>(const HD &x, const HD &y) { return x.v > y.v; }
inline bool operator>=(const HD &x, const HD &y) { return x.v >= y.v; }
inline double val(double x) { return x; }
inline double val(const HD &x) { return x.v; }
inline HD seed1(double v, double da) { return {v, da, 0.0, 0.0}; }                  // first derivative along da
inline HD seed2(double v, double d, double d2 = 0.0) { return {v, d, d, d2}; }      // curve x(s) = v + s d + s^2/2 d2

// ---------------------------------------------------------------------------------------------------------------
constexpr double RHO = 8.44e-9, L_FUSION = 2.72e11;

// driver IN625::k(): conductivity table of the bridge deck, constant extrapolation
template <class T> T k_tab(const T &t)
{
    static const double Ta[11] = {294.0, 311.0, 366.0, 477.0, 589.0, 700.0, 811.0, 922.0, 1033.0, 1144.0, 1255.0};
    static const double Va[11] = {9.8, 10.1, 10.8, 12.5, 14.1, 15.7, 17.5, 19.0, 20.8, 22.8, 25.2};
    if (t <= Ta[0]) return T(Va[0]);
    if (t >= Ta[10]) return T(Va[10]);
    int i = 0;
    while (i < 9 && Ta[i + 1] < t) i++;
    return Va[i] + (t - Ta[i]) / (Ta[i + 1] - Ta[i]) * (Va[i + 1] - Va[i]);
}

// driver IN625::cp_dev(): same branch structure as the driver (half-open intervals)
template <class T> T cp_tab(const T &t)
{
    static const double Ta[11] = {294.0, 366.0, 477.0, 589.0, 700.0, 811.0, 922.0, 1033.0, 1144.0, 1255.0, 1366.0};
    static const double Va[11] = {4.1e8, 4.27e8, 4.56e8, 4.81e8, 5.11e8, 5.36e8, 5.65e8, 5.90e8, 6.20e8, 6.45e8, 6.70e8};
    if (t <= Ta[0]) return T(Va[0]);
    for (int i = 0; i < 10; i++)
        if (t < Ta[i + 1]) return Va[i] + (t - Ta[i]) / (Ta[i + 1] - Ta[i]) * (Va[i + 1] - Va[i]);
    return T(Va[10]);
}

// driver IN625::f_L_dev() / f_L_oti(): branch on the NOMINAL Ts/Tl, interior uses the scaled bounds
template <class T> T fL(const T &t, const T *S, double Ts, double Tl)
{
    if (t <= Ts) return T(0.0);
    if (t >= Tl) return T(1.0);
    const T ts = Ts * S[S_TS], tl = Tl * S[S_TL];
    const T xi = (t - ts) / (tl - ts);
    return xi * xi * xi * (10.0 - 15.0 * xi + 6.0 * xi * xi);
}

// ---- first-order (powder-blended) quantities ----
template <class T> T kblend(const T &t, const T *S, double mf, double kpf)
{ return mf * k_tab(t) * S[S_K] + (1.0 - mf) * k_tab(t) * kpf * S[S_KP]; }

template <class T> T qblend(const T &t, const T *S, double mf, double a, double hpa, double b, double Tamb)
{
    const T Tinf = Tamb * S[S_T0];
    const T h = mf * a * S[S_HC] + (1.0 - mf) * hpa * S[S_HP];
    return h * (t - Tinf) + b * (t * t * t * t - Tinf * Tinf * Tinf * Tinf);
}

template <class T> T nodal_blend(const T &Tnext, const T &Tn, const T *S, double Mv, double rpf,
                                 double V, double idt, double Ts, double Tl)
{
    const T rho = Mv * RHO * S[S_RHO] + (1.0 - Mv) * RHO * rpf * S[S_RHOP];
    const T Lf = L_FUSION * S[S_L];
    const T Mc = V * rho * cp_tab(Tnext) * S[S_CP];
    return Mc * idt * (Tnext - Tn) + V * rho * Lf * (fL(Tnext, S, Ts, Tl) - fL(Tn, S, Ts, Tl)) * idt;
}

// ---- second-order reference blocks (no powder blend; kernels fb2_*) ----
template <class T> T k_plain(const T &t, const T *S) { return k_tab(t) * S[S_K]; }
template <class T> T q_plain(const T &t, const T *S, double a, double b, double Tamb)
{
    const T Tinf = Tamb * S[S_T0];
    return a * S[S_HC] * (t - Tinf) + b * (t * t * t * t - Tinf * Tinf * Tinf * Tinf);
}
template <class T> T nodal_plain(const T &Tnext, const T &Tn, const T *S, double V, double idt, double Ts, double Tl)
{
    const T rho = RHO * S[S_RHO], Lf = L_FUSION * S[S_L];
    const T Mc = V * rho * cp_tab(Tnext) * S[S_CP];
    return Mc * idt * (Tnext - Tn) + V * rho * Lf * (fL(Tnext, S, Ts, Tl) - fL(Tn, S, Ts, Tl)) * idt;
}

template <class T> void unit_S(T *S) { for (int i = 0; i < NSLOT; i++) S[i] = T(1.0); }
} // namespace ref
