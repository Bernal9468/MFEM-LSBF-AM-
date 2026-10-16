// ============================================================================
//  Differentiable laser-source deposition kernel (bridge_case_enzyme, 2026-09-28)
//
//  Replaces the central finite differences on SpatialHashSource::BuildSource that
//  were used for the source-parameter sensitivities (rx, ry, rz, x0, Vx, tL).
//  BuildSource records, while it builds the nominal source, every DISCRETE choice
//  it makes for one time sub-sample (position segment, Goldak frame, which cells
//  are kept, the element each lands in and its reference coordinates). This
//  kernel re-evaluates that record as a SMOOTH function of the six parameters
//
//      th = [ s_rx, s_ry, s_rz, x0, s_Vx, tL ]   (all 0 at nominal)
//      Goldak semi-axes  af,ar *(1+s_rx),  b *(1+s_ry),  c *(1+s_rz)
//      laser position    p(tau) + x0*e_x,   tau = t_s + (1+s_Vx)(t - t_s) - tL
//
//  so its derivatives can be taken by AD: Enzyme (double instantiation, in the
//  clang TU) or OTI (OTI instantiation, in the driver). Discrete choices are
//  held fixed, i.e. the derivative is the within-element derivative.
//
//  The reference coordinates come from two Newton steps of the trilinear hex map
//  started at the nominal solution: exact value at nominal, exact first and
//  second derivatives.
//  Shape functions: MFEM order-1 hex, vertex order (0,0,0),(1,0,0),(1,1,0),
//  (0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1) on [0,1]^3.
// ============================================================================
#pragma once
#include <cmath>

#define SRCK_MAXCELL 216              // up to 6^3 Goldak cells per sub-sample
#define SRCK_MAXOUT  (SRCK_MAXCELL*8)

struct SrcCell {
    int ix, iy, iz;        // Goldak cell index (0 for point mode)
    double X[24];          // element vertex coordinates, [a*3+d]
    double xi0[3];         // nominal reference coordinates of the deposition point
    int dofs[8];           // element dofs (same order as the vertices)
    int const_shape;       // point mode, point not inside element -> shape at xi0 held constant
};

struct SrcSub {
    int mode;              // 0 = point ("Concentrated"), 2 = Goldak
    double tm, t_s;        // sub-sample time, Vx pivot time
    int seg_linear;        // 1: position linear on the frozen segment; 0: frozen endpoint (clamped)
    double ts0, ts1, xa[3], xb[3];
    double Esub;           // energy of this sub-sample [mJ] (frozen: zero-order-hold power)
    double e1[3], e2[3], e3[3];
    double af, ar, b, c, ff, fr, box;
    int nq;
    int first_cell, ncell; // kept cells in the tape's cell array
};

inline double srck_exp(double v) { return std::exp(v); }
// constant of scalar type T (T(v) for double/OTI/CoDiPack; specialised for aggregate dual numbers)
template <class T> struct SrcKMake { static T c(double v) { return T(v); } };

template <class T> inline void srck_shape(const T xi[3], T N[8]) {
    const T x = xi[0], y = xi[1], z = xi[2];
    const T mx = 1.0 - x, my = 1.0 - y, mz = 1.0 - z;
    N[0] = mx*my*mz; N[1] = x*my*mz; N[2] = x*y*mz; N[3] = mx*y*mz;
    N[4] = mx*my*z;  N[5] = x*my*z;  N[6] = x*y*z;  N[7] = mx*y*z;
}
template <class T> inline void srck_dshape(const T xi[3], T dN[8][3]) {
    const T x = xi[0], y = xi[1], z = xi[2];
    const T mx = 1.0 - x, my = 1.0 - y, mz = 1.0 - z;
    dN[0][0] = -my*mz; dN[0][1] = -mx*mz; dN[0][2] = -mx*my;
    dN[1][0] =  my*mz; dN[1][1] = -x*mz;  dN[1][2] = -x*my;
    dN[2][0] =  y*mz;  dN[2][1] =  x*mz;  dN[2][2] = -x*y;
    dN[3][0] = -y*mz;  dN[3][1] =  mx*mz; dN[3][2] = -mx*y;
    dN[4][0] = -my*z;  dN[4][1] = -mx*z;  dN[4][2] =  mx*my;
    dN[5][0] =  my*z;  dN[5][1] = -x*z;   dN[5][2] =  x*my;
    dN[6][0] =  y*z;   dN[6][1] =  x*z;   dN[6][2] =  x*y;
    dN[7][0] = -y*z;   dN[7][1] =  mx*z;  dN[7][2] =  mx*y;
}

// two Newton steps of x(xi) = p, started at the nominal xi0
template <class T> inline void srck_invmap(const double X[24], const double xi0[3], const T p[3], T xi[3]) {
    for (int d = 0; d < 3; d++) xi[d] = SrcKMake<T>::c(xi0[d]);
    for (int it = 0; it < 2; it++) {
        T N[8], dN[8][3];
        srck_shape(xi, N); srck_dshape(xi, dN);
        T r[3], J[3][3];
        for (int d = 0; d < 3; d++) { r[d] = -p[d]; for (int k = 0; k < 3; k++) J[d][k] = SrcKMake<T>::c(0.0); }
        for (int a = 0; a < 8; a++)
            for (int d = 0; d < 3; d++) {
                r[d] = r[d] + N[a]*X[a*3+d];
                for (int k = 0; k < 3; k++) J[d][k] = J[d][k] + X[a*3+d]*dN[a][k];
            }
        // solve J dxi = r (Cramer), xi -= dxi
        const T det = J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
                    - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
                    + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
        const T d0 = r[0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
                   - J[0][1]*(r[1]*J[2][2]-J[1][2]*r[2])
                   + J[0][2]*(r[1]*J[2][1]-J[1][1]*r[2]);
        const T d1 = J[0][0]*(r[1]*J[2][2]-J[1][2]*r[2])
                   - r[0]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
                   + J[0][2]*(J[1][0]*r[2]-r[1]*J[2][0]);
        const T d2 = J[0][0]*(J[1][1]*r[2]-r[1]*J[2][1])
                   - J[0][1]*(J[1][0]*r[2]-r[1]*J[2][0])
                   + r[0]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
        xi[0] = xi[0] - d0/det; xi[1] = xi[1] - d1/det; xi[2] = xi[2] - d2/det;
    }
}

// out[k*8+a] = energy deposited on dof a of kept cell k  [mJ]
template <class T> void srck_eval(const SrcSub &r, const SrcCell *cells, const T th[6], T *out) {
    // laser position at the transformed time on the frozen segment
    const T tau = r.t_s + (1.0 + th[4])*(r.tm - r.t_s) - th[5];
    T p0[3];
    if (r.seg_linear) {
        const T f = (tau - r.ts0) / (r.ts1 - r.ts0);
        for (int d = 0; d < 3; d++) p0[d] = r.xa[d] + f*(r.xb[d] - r.xa[d]);
    } else {
        for (int d = 0; d < 3; d++) p0[d] = SrcKMake<T>::c(r.xa[d]);
    }
    p0[0] = p0[0] + th[3];

    if (r.mode != 2) {                                   // point source: one "cell"
        const SrcCell &c = cells[r.first_cell];
        T xi[3], N[8];
        if (c.const_shape) { for (int d = 0; d < 3; d++) xi[d] = SrcKMake<T>::c(c.xi0[d]); }
        else srck_invmap(c.X, c.xi0, p0, xi);
        srck_shape(xi, N);
        for (int a = 0; a < 8; a++) out[a] = r.Esub * N[a];
        return;
    }

    // Goldak double ellipsoid on the box of the SCALED semi-axes, subdivided nq^3
    const T af = r.af*(1.0 + th[0]), ar = r.ar*(1.0 + th[0]);
    const T gb = r.b*(1.0 + th[1]),  gc = r.c*(1.0 + th[2]);
    const int nq = r.nq; const double box = r.box;
    const T dx = box*(af + ar)/nq, dy = 2.0*box*gb/nq, dz = box*gc/nq;
    const double dx0 = box*(r.af + r.ar)/nq;
    // normalisation over the complete box (front/rear split frozen at nominal)
    T full = SrcKMake<T>::c(0.0);
    for (int ix = 0; ix < nq; ix++) for (int iy = 0; iy < nq; iy++) for (int iz = 0; iz < nq; iz++) {
        const bool front = (-box*r.ar + (ix + 0.5)*dx0) >= 0.0;
        const T lx = -box*ar + (ix + 0.5)*dx, ly = -box*gb + (iy + 0.5)*dy, lz = (iz + 0.5)*dz;
        const T ax = front ? af : ar; const double frac = front ? r.ff : r.fr;
        full = full + frac/ax * srck_exp(-3.0*(lx*lx/(ax*ax) + ly*ly/(gb*gb) + lz*lz/(gc*gc)));
    }
    for (int k = 0; k < r.ncell; k++) {
        const SrcCell &c = cells[r.first_cell + k];
        const bool front = (-box*r.ar + (c.ix + 0.5)*dx0) >= 0.0;
        const T lx = -box*ar + (c.ix + 0.5)*dx, ly = -box*gb + (c.iy + 0.5)*dy, lz = (c.iz + 0.5)*dz;
        const T ax = front ? af : ar; const double frac = front ? r.ff : r.fr;
        const T raw = frac/ax * srck_exp(-3.0*(lx*lx/(ax*ax) + ly*ly/(gb*gb) + lz*lz/(gc*gc)));
        const T dE = r.Esub * raw / full;
        T p[3];
        for (int d = 0; d < 3; d++) p[d] = p0[d] + lx*r.e1[d] + ly*r.e2[d] + lz*r.e3[d];
        T xi[3], N[8];
        srck_invmap(c.X, c.xi0, p, xi);
        srck_shape(xi, N);
        for (int a = 0; a < 8; a++) out[k*8 + a] = dE * N[a];
    }
}
