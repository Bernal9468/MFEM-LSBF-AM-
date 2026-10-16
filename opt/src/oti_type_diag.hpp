// oti_type_diag.hpp -- DIAGONAL-ONLY second-order OTI type (drop-in for OTI in oti_type_generic30.hpp).
// Stores only {real, e_i, e_i^2} (normalised Taylor coefficients): 1 + 2N = 55 numbers for N = 27 bases instead of the full
// multivariate C(N+2,2) = 406. The diagonal coefficient of any product/quotient depends only on the operands' real, e_i and
// e_i^2 parts -- (ab)_ii = a0 b_ii + a_i b_i + a_ii b0 -- so every get_e/get_epp the driver reads is IDENTICAL to the full
// type's; the cross coefficients e_i e_j (never read by the diagonal second-order driver) are simply not carried.
// Select with -DOTI_DIAG_ONLY (the full-Hessian path never uses the OTI type; the Enzyme build handles cross terms).
#pragma once
#include <iostream>
#include <cstdlib>

#ifndef OTI_FUNCTION
#define OTI_FUNCTION inline
#endif
static constexpr int OTI_NUM_BASES = 27;
#ifndef OTI_ORDER_VAL
#define OTI_ORDER_VAL 2
#endif
static constexpr int OTI_ORDER        = OTI_ORDER_VAL;
static constexpr int OTI_NEWTON_BASIS = OTI_NUM_BASES;
static constexpr int OTI_NUM_WIRED    = 26;

class OTI {
    static constexpr int N = OTI_NUM_BASES;
    double r;
    double e[N];     // d f / d x_i
    double q[N];     // (1/2) d^2 f / d x_i^2
public:
    OTI() : r(0.0) { for (int i = 0; i < N; i++) { e[i] = 0.0; q[i] = 0.0; } }
    OTI(double v) : r(v) { for (int i = 0; i < N; i++) { e[i] = 0.0; q[i] = 0.0; } }
    double real() const { return r; }
    void set_real(double v) { r = v; }
    double get_e(int i) const { return e[i - 1]; }
    void set_e(int i, double v) { e[i - 1] = v; }
    double get_epp(int i) const { return q[i - 1]; }
    void set_epp(int i, double v) { q[i - 1] = v; }
    double get_epp_cross(int i, int j) const { if (i == j) return q[i - 1]; std::abort(); }
    void set_epp_cross(int i, int j, double v) { if (i == j) { q[i - 1] = v; return; } std::abort(); }

    OTI operator+(const OTI &b) const { OTI c; c.r = r + b.r; for (int i = 0; i < N; i++) { c.e[i] = e[i] + b.e[i]; c.q[i] = q[i] + b.q[i]; } return c; }
    OTI operator-(const OTI &b) const { OTI c; c.r = r - b.r; for (int i = 0; i < N; i++) { c.e[i] = e[i] - b.e[i]; c.q[i] = q[i] - b.q[i]; } return c; }
    OTI operator-() const { OTI c; c.r = -r; for (int i = 0; i < N; i++) { c.e[i] = -e[i]; c.q[i] = -q[i]; } return c; }
    OTI operator*(const OTI &b) const {
        OTI c; c.r = r * b.r;
        for (int i = 0; i < N; i++) {
            c.e[i] = r * b.e[i] + e[i] * b.r;
            c.q[i] = r * b.q[i] + e[i] * b.e[i] + q[i] * b.r;
        }
        return c;
    }
    OTI operator/(const OTI &b) const {
        // 1/b: i0 = 1/b0, i_i = -b_i/b0^2, i_ii = -b_ii/b0^2 + b_i^2/b0^3
        OTI inv; const double i0 = 1.0 / b.r, i02 = i0 * i0;
        inv.r = i0;
        for (int i = 0; i < N; i++) { inv.e[i] = -b.e[i] * i02; inv.q[i] = -b.q[i] * i02 + b.e[i] * b.e[i] * i02 * i0; }
        return (*this) * inv;
    }
    OTI &operator+=(const OTI &b) { *this = *this + b; return *this; }
    OTI &operator-=(const OTI &b) { *this = *this - b; return *this; }
    OTI &operator*=(const OTI &b) { *this = *this * b; return *this; }
    OTI &operator/=(const OTI &b) { *this = *this / b; return *this; }
    friend OTI operator+(double a, const OTI &b) { return OTI(a) + b; }
    friend OTI operator+(const OTI &a, double b) { return a + OTI(b); }
    friend OTI operator-(double a, const OTI &b) { return OTI(a) - b; }
    friend OTI operator-(const OTI &a, double b) { return a - OTI(b); }
    friend OTI operator*(double a, const OTI &b) { OTI c; c.r = a * b.r; for (int i = 0; i < N; i++) { c.e[i] = a * b.e[i]; c.q[i] = a * b.q[i]; } return c; }
    friend OTI operator*(const OTI &a, double b) { return b * a; }
    friend OTI operator/(double a, const OTI &b) { return OTI(a) / b; }
    friend OTI operator/(const OTI &a, double b) { return a * (1.0 / b); }
    bool operator>(const OTI &b) const { return r > b.r; }
    bool operator<(const OTI &b) const { return r < b.r; }
    bool operator>=(const OTI &b) const { return r >= b.r; }
    bool operator<=(const OTI &b) const { return r <= b.r; }
    bool operator==(const OTI &b) const { return r == b.r; }
    bool operator!=(const OTI &b) const { return r != b.r; }
    friend std::ostream &operator<<(std::ostream &os, const OTI &x) { return os << x.r; }
};
inline OTI oti_seed(double val, int idx) { OTI x(val); x.set_e(idx, 1.0); return x; }
