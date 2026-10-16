#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// element response (see rs_kernel.hpp): outputs fint[24], st_new[56], sig[48]
void rs_elem(const double *u, const double *Tq, const double *T0q, const double *st_old, const double *par, const double *geom,
             double *fint, double *st_new, double *sig);
// forward-mode directional derivative along (du, dst_old, dpar); outputs value and derivative
void rs_elem_d1(const double *u, const double *du, const double *Tq, const double *T0q, const double *st_old, const double *dst_old,
                const double *par, const double *dpar, const double *geom,
                double *fint, double *dfint, double *st_new, double *dst_new, double *sig, double *dsig);
// second directional derivative D_b D_a (nested forward-over-forward). a = (du, dst_old, dpar) [first level], b = (bu, bst, bpar) [second level].
// Outputs: value+first-derivative as in rs_elem_d1, then the b-shadows: bf = D_b f, ddf = D_b D_a f (= D^2 f[a,b]), etc.
void rs_elem_d2(const double *u, const double *du, const double *Tq, const double *T0q, const double *st_old, const double *dst_old,
                const double *par, const double *dpar, const double *geom, const double *bu, const double *bst, const double *bpar,
                double *f, double *df, double *st_new, double *dst_new, double *sig, double *dsig,
                double *bf, double *ddf, double *bst_new, double *ddst_new, double *bsig, double *ddsig);
#ifdef __cplusplus
}
#endif
