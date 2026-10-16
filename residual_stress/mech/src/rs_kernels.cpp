// Enzyme forward-mode wrappers for the residual-stress element kernel. Build: clang++ -O2 -fplugin=ClangEnzyme -DMFEM_USE_ENZYME
#include "rs_kernel.hpp"
#include "rs_kernels.h"
#include "general/enzyme.hpp"
int enzyme_dup; int enzyme_dupnoneed; int enzyme_out; int enzyme_const; int enzyme_interleave;
namespace {
void impl(const double *u, const double *Tq, const double *T0q, const double *st_old, const double *par, const double *geom,
          double *f, double *st_new, double *sig)
{ rs::elem_response(u, Tq, T0q, st_old, par, geom, f, st_new, sig); }

// first derivative along a = (du, dst_old, dpar) -- itself differentiated again by rs_elem_d2
void d1_impl(const double *u, const double *du, const double *Tq, const double *T0q, const double *st_old, const double *dst_old,
             const double *par, const double *dpar, const double *geom,
             double *fint, double *dfint, double *st_new, double *dst_new, double *sig, double *dsig)
{
    __enzyme_fwddiff<void>((void *)impl,
        enzyme_dup, u, du, enzyme_const, Tq, enzyme_const, T0q, enzyme_dup, st_old, dst_old, enzyme_dup, par, dpar,
        enzyme_const, geom, enzyme_dup, fint, dfint, enzyme_dup, st_new, dst_new, enzyme_dup, sig, dsig);
}
}
extern "C" {
void rs_elem(const double *u, const double *Tq, const double *T0q, const double *st_old, const double *par, const double *geom,
             double *fint, double *st_new, double *sig)
{ impl(u, Tq, T0q, st_old, par, geom, fint, st_new, sig); }

void rs_elem_d1(const double *u, const double *du, const double *Tq, const double *T0q, const double *st_old, const double *dst_old,
                const double *par, const double *dpar, const double *geom,
                double *fint, double *dfint, double *st_new, double *dst_new, double *sig, double *dsig)
{ d1_impl(u, du, Tq, T0q, st_old, dst_old, par, dpar, geom, fint, dfint, st_new, dst_new, sig, dsig); }

void rs_elem_d2(const double *u, const double *du, const double *Tq, const double *T0q, const double *st_old, const double *dst_old,
                const double *par, const double *dpar, const double *geom, const double *bu, const double *bst, const double *bpar,
                double *f, double *df, double *st_new, double *dst_new, double *sig, double *dsig,
                double *bf, double *ddf, double *bst_new, double *ddst_new, double *bsig, double *ddsig)
{
    double z24[24] = {0}, z56[56] = {0}, z5[5] = {0};
    __enzyme_fwddiff<void>((void *)d1_impl,
        enzyme_dup, u, bu, enzyme_dup, du, z24, enzyme_const, Tq, enzyme_const, T0q, enzyme_dup, st_old, bst, enzyme_dup, dst_old, z56,
        enzyme_dup, par, bpar, enzyme_dup, dpar, z5, enzyme_const, geom,
        enzyme_dup, f, bf, enzyme_dup, df, ddf, enzyme_dup, st_new, bst_new, enzyme_dup, dst_new, ddst_new, enzyme_dup, sig, bsig, enzyme_dup, dsig, ddsig);
}
}
