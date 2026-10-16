// oti_query_type.hpp -- MFEM-free Taylor-number type + multi-index table for
// the any-order sensitivity driver.
//
// Jet<M,N> wraps oti::otinum<M,N,double>: M query variables t_1..t_M, total
// truncation order N, ncoeffs = C(M+N,N) normalized Taylor coefficients in the
// library's flat graded layout (flat 0 = real part, order-k coefficients
// contiguous).  All access in hot code is by FLAT index (operator[]); the
// AlphaTable is built once on the host and never used inside kernels.
#pragma once

#include "otinum/otinum.hpp"
#include <array>
#include <cmath>
#include <vector>

namespace aoq {

template <int M, int N>
struct Jet {
    using Backing = oti::otinum<M, N, double>;
    static constexpr int NC = Backing::ncoeffs;
    Backing d;

    Jet() {}
    Jet(double r) : d(r) {}
    explicit Jet(const Backing &b) : d(b) {}

    double real() const { return d.real(); }
    double &operator[](int c) { return d[c]; }
    double operator[](int c) const { return d[c]; }

    Jet operator+(const Jet &b) const { return Jet(d + b.d); }
    Jet operator-(const Jet &b) const { return Jet(d - b.d); }
    Jet operator*(const Jet &b) const { return Jet(d * b.d); }
    Jet operator/(const Jet &b) const { return Jet(d / b.d); }
    Jet operator-() const { return Jet(-d); }
    Jet &operator+=(const Jet &b) { *this = *this + b; return *this; }
    Jet &operator-=(const Jet &b) { *this = *this - b; return *this; }
    Jet &operator*=(const Jet &b) { *this = *this * b; return *this; }

    // v7 perf (thermal_kokkos_oti copy): scalar operands use otinum's scalar operations (O(NC)) instead of being
    // promoted to a Jet and convolved / inverted (O(nproducts)); same values up to roundoff.
    friend Jet operator+(double a, const Jet &b) { Jet r(b); r.d += a; return r; }
    friend Jet operator+(const Jet &a, double b) { Jet r(a); r.d += b; return r; }
    friend Jet operator-(double a, const Jet &b) { Jet r(-b.d); r.d += a; return r; }
    friend Jet operator-(const Jet &a, double b) { Jet r(a); r.d -= b; return r; }
    friend Jet operator*(double a, const Jet &b) { return Jet(b.d * a); }
    friend Jet operator*(const Jet &a, double b) { return Jet(a.d * b); }
    friend Jet operator/(double a, const Jet &b) { return Jet(a) / b; }
    friend Jet operator/(const Jet &a, double b) { return Jet(a.d * (1.0 / b)); }
};

// Multi-index bookkeeping for one (M,N) shape, built once on the host.
template <int M, int N>
struct AlphaTable {
    using Backing = oti::otinum<M, N, double>;
    using TT = typename Backing::table_type;
    static constexpr int NC = Backing::ncoeffs;

    std::vector<std::array<int, M>> alpha;   // exponents of each flat coefficient
    std::vector<int> order;                  // |alpha|
    std::vector<double> fact;                // alpha!  (partial = fact * coeff)
    std::vector<int> off;                    // off[n] = first flat index of order n, off[N+1] = NC

    AlphaTable() {
        alpha.resize(NC); order.resize(NC); fact.resize(NC); off.resize(N + 2);
        for (int i = 0; i < NC; i++) {
            auto a = TT::alpha_at(i);
            for (int j = 0; j < M; j++) alpha[i][j] = a[static_cast<std::size_t>(j)];
            order[i] = TT::order_of_value(i);
            fact[i] = TT::factorial_alpha_value(i);
        }
        for (int n = 0; n <= N + 1; n++) off[n] = (n <= N) ? TT::order_offset_value(n) : NC;
    }

    // -1 if alpha is not representable (wrong length / order > N).
    int index_of(const std::vector<int> &a) const {
        if ((int)a.size() != M) return -1;
        int s = 0;
        for (int v : a) { if (v < 0) return -1; s += v; }
        if (s > N) return -1;
        for (int i = off[s]; i < off[s + 1]; i++) {
            bool eq = true;
            for (int j = 0; j < M; j++) if (alpha[i][j] != a[j]) { eq = false; break; }
            if (eq) return i;
        }
        return -1;
    }

    // Host self-test: table layout agrees with the library's own rank().
    bool self_test() const {
        for (int i = 0; i < NC; i++) {
            oti::detail::alpha_t<M> a{};
            for (int j = 0; j < M; j++) a[static_cast<std::size_t>(j)] = alpha[i][j];
            if (oti::detail::rank<M, N>(a) != i) return false;
        }
        return true;
    }
};

}  // namespace aoq
