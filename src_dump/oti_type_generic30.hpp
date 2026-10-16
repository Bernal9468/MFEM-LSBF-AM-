// oti_type_generic30.hpp -- Kokkos-portable drop-in OTI type for the
// 30-PARAMETER bridge sensitivity driver, backed by the GENERIC
// oti::otinum<M,N> library (arbitrary parameter count and truncation order).
//
// Same class name (OTI) and public API as the 4-parameter
// oti_type_second_generic.hpp / oti_type_second.hpp, so the driver only needs
// its #include line changed -- every other call site (set_e/get_e/get_epp/
// operators/oti_sqrt/oti_abs/oti_seed) is unchanged.
//
// Sizing (see ../../PLAN.md):
//   * 30 physical-parameter basis slots (1..30), numbered to match the
//     bare-plate 30-param IDX_* enum (IDX_P=1 .. IDX_W=30). Only 16 are
//     actually wired into the bridge model; the rest stay seeded-but-unused
//     (zero derivatives on this geometry) so the seeding procedure and the
//     library size match the original 30-parameter design.
//   * basis 31 = the phantom "Newton-tangent" direction the nodal blocks use
//     to extract d(nodal_residual)/d(Tnext) via the same chain-rule
//     machinery (was e5 in the 4-param baseline; moved clear of the physical
//     parameters -- e5 now belongs to IDX_RZ).
//   * order N = 2, full multivariate (cross Hessian terms included), exactly
//     as oti::otinum is natively designed. ncoeffs = C(31+2,2) = 528.
//
// Validated: the generic otinum library was checked byte-for-byte against
// this project's own Taylor2<M> reimplementation and the Fortran otilib at
// M=6/N=2 earlier this session (bridge_local_oti_second_kokkos_genoti).

#pragma once

#include "otinum/otinum.hpp"
#include <iostream>

// Basis count. Default = COMPACT: the 16 wireable bridge directions on bases
// 1..16 (contiguous, same scheme as the bare-plate's own compact OTI30) plus
// one Newton-tangent basis at 17. Define OTI_BASES_FULL30 to instead reserve
// the full 30-parameter enum (IDX_P=1..IDX_W=30) + Newton at 31 -- byte-exact
// for every wired direction, ~3x more expensive per OTI op (P0 measured).
#ifdef OTI_BASES_FULL30
static constexpr int OTI_NUM_BASES = 31;
#else
static constexpr int OTI_NUM_BASES = 27;   // v8 (2026-10-02): 26 wired (16 original + Tw..w) + 1 Newton-tangent
#endif
#ifndef OTI_ORDER_VAL
#define OTI_ORDER_VAL 2
#endif
static constexpr int OTI_ORDER        = OTI_ORDER_VAL;   // 1 = first-order only builds (-dorder <= 1)
static constexpr int OTI_NEWTON_BASIS  = OTI_NUM_BASES;   // 1-indexed: the last basis
// How many of the 30 are wireable into this single-body bridge model.
static constexpr int OTI_NUM_WIRED     = 26;   // v8: Tw,hp,kp,rho_p,bp,lp,wp,b,l,w added (basis 17-26)

class OTI {
public:
    using Backing = oti::otinum<OTI_NUM_BASES, OTI_ORDER, double>;
    using Alpha   = Backing::alpha_type;
    Backing data;

    OTI_FUNCTION OTI() : data() {}
    OTI_FUNCTION OTI(double r) : data(r) {}
    OTI_FUNCTION explicit OTI(const Backing& d) : data(d) {}

    OTI_FUNCTION double real() const { return data.real(); }
    OTI_FUNCTION void set_real(double v) { data[0] = v; }

    // 1-indexed accessors matching the Taylor2-backed header's convention.
    // otinum's coeff(alpha) is the NORMALIZED Taylor coefficient
    // (1/alpha!) d^|alpha|f : for a pure order-1 index that is df/dx_i, and
    // for a pure order-2 diagonal index it is (1/2) d^2f/dx_i^2 -- matching
    // get_e / get_epp exactly, no extra factorial bookkeeping here.
    OTI_FUNCTION static Alpha alpha_e(int i) {
        Alpha a{};
        a[i - 1] = 1;
        return a;
    }
    OTI_FUNCTION static Alpha alpha_epp(int i) {
        Alpha a{};
        a[i - 1] = 2;
        return a;
    }
    OTI_FUNCTION static Alpha alpha_cross(int i, int j) {
        Alpha a{};
        a[i - 1] += 1;
        a[j - 1] += 1;
        return a;
    }

    OTI_FUNCTION double get_e(int i) const { return data.coeff(alpha_e(i)); }
    OTI_FUNCTION void set_e(int i, double val) { data.set_coeff(alpha_e(i), val); }
    OTI_FUNCTION double get_epp(int i) const { if constexpr (OTI_ORDER >= 2) { return data.coeff(alpha_epp(i)); } else { (void)i; return 0.0; } }
    OTI_FUNCTION void set_epp(int i, double val) { if constexpr (OTI_ORDER >= 2) { data.set_coeff(alpha_epp(i), val); } else { (void)i; (void)val; } }

    // Mixed second derivative d^2f/dx_i/dx_j (i != j). otinum's coeff for a
    // (1,1) multi-index is exactly d^2f/dx_i/dx_j (alpha! = 1), no 1/2 factor.
    OTI_FUNCTION double get_epp_cross(int i, int j) const {
        if (i == j) return get_epp(i);
        if constexpr (OTI_ORDER >= 2) { return data.coeff(alpha_cross(i, j)); } else { return 0.0; }
    }
    OTI_FUNCTION void set_epp_cross(int i, int j, double val) {
        if (i == j) { set_epp(i, val); return; }
        if constexpr (OTI_ORDER >= 2) { data.set_coeff(alpha_cross(i, j), val); } else { (void)val; }
    }

    OTI_FUNCTION OTI operator+(const OTI& b) const { return OTI(data + b.data); }
    OTI_FUNCTION OTI operator-(const OTI& b) const { return OTI(data - b.data); }
    OTI_FUNCTION OTI operator*(const OTI& b) const { return OTI(data * b.data); }
    OTI_FUNCTION OTI operator/(const OTI& b) const { return OTI(data / b.data); }
    OTI_FUNCTION OTI operator-() const { return OTI(-data); }

    OTI_FUNCTION OTI& operator+=(const OTI& b) { *this = *this + b; return *this; }
    OTI_FUNCTION OTI& operator-=(const OTI& b) { *this = *this - b; return *this; }
    OTI_FUNCTION OTI& operator*=(const OTI& b) { *this = *this * b; return *this; }
    OTI_FUNCTION OTI& operator/=(const OTI& b) { *this = *this / b; return *this; }

    friend OTI_FUNCTION OTI operator+(double a, const OTI& b) { return OTI(a) + b; }
    friend OTI_FUNCTION OTI operator+(const OTI& a, double b) { return a + OTI(b); }
    friend OTI_FUNCTION OTI operator-(double a, const OTI& b) { return OTI(a) - b; }
    friend OTI_FUNCTION OTI operator-(const OTI& a, double b) { return a - OTI(b); }
    friend OTI_FUNCTION OTI operator*(double a, const OTI& b) { return OTI(a) * b; }
    friend OTI_FUNCTION OTI operator*(const OTI& a, double b) { return a * OTI(b); }
    friend OTI_FUNCTION OTI operator/(double a, const OTI& b) { return OTI(a) / b; }
    friend OTI_FUNCTION OTI operator/(const OTI& a, double b) { return a / OTI(b); }

    OTI_FUNCTION bool operator>(const OTI& b) const { return data.real() > b.data.real(); }
    OTI_FUNCTION bool operator<(const OTI& b) const { return data.real() < b.data.real(); }
    OTI_FUNCTION bool operator>=(const OTI& b) const { return data.real() >= b.data.real(); }
    OTI_FUNCTION bool operator<=(const OTI& b) const { return data.real() <= b.data.real(); }
    OTI_FUNCTION bool operator==(const OTI& b) const { return data.real() == b.data.real(); }
    OTI_FUNCTION bool operator!=(const OTI& b) const { return data.real() != b.data.real(); }

    friend std::ostream& operator<<(std::ostream& os, const OTI& x) {
        return os << x.data.real();
    }
};

inline OTI oti_sqrt(const OTI& x) { return OTI(oti::sqrt(x.data)); }
inline OTI oti_abs(const OTI& x) { return OTI(oti::abs(x.data)); }
inline OTI oti_seed(double val, int param_idx) {
    OTI result(val);
    result.set_e(param_idx, 1.0);
    return result;
}
