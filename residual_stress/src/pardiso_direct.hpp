// pardiso_direct.hpp -- thin wrapper around Intel MKL PARDISO, the EXACT
// solver family the reference paper's HYPAD-FEM uses ("The solver used for
// this study was the Intel MKL PARDISO"). Same one-factorization-many-RHS
// architecture as umfpack_direct.hpp, but using MKL's nested-dissection
// (METIS-based) fill-reducing ordering instead of UMFPack's AMD ordering --
// AMD is known to produce much worse fill-in on 3D meshes, which is the
// likely reason umfpack_direct.hpp's factorization didn't show the expected
// win. This does not require MKL headers/libs in MFEM's own build -- calls
// the PARDISO C API directly (headers from the Intel oneapi module,
// runtime from the conda MKL single dynamic library).
#pragma once
#include "mfem.hpp"
#include <mkl_pardiso.h>
#include <mkl_types.h>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <algorithm>

namespace sensei {

class PardisoDirectSolver {
    std::vector<double> Ax;
    std::vector<MKL_INT> Ap, Ai;
    MKL_INT n = 0;
    void *pt[64];
    MKL_INT iparm[64];
    MKL_INT maxfct = 1, mnum = 1, mtype = 2; // real symmetric positive definite
                                              // (diffusion + Robin boundary +
                                              // lumped capacitance, all
                                              // positive-definite terms)
    MKL_INT msglvl = 0; // verbose (=1) was fine for the old one-shot
                        // estimate but floods stdout once PARDISO becomes
                        // the live per-direction solver (Solve() called 8x/
                        // step) -- the print I/O itself was contaminating
                        // the timing being measured.
    std::vector<MKL_INT> perm; // size n -- PARDISO reads/writes this as a
                                // real array even with iparm[4]=0 (no user
                                // permutation); a single scalar here caused
                                // a segfault (buffer overrun).
    bool initialized = false;
    bool has_symbolic = false; // phase 11 (analysis/reordering) done at least
                                // once -- see Refactorize()'s comment.
    std::vector<MKL_INT> Ap_ref, Ai_ref; // sparsity pattern from the call
                                // that established the symbolic factorization,
                                // kept only to sanity-check later calls
                                // (Refactorize() verifies the new matrix's
                                // pattern still matches before reusing pt).

    void CallPhase(MKL_INT phase, double *b, double *x, MKL_INT nrhs_override = -1) {
        MKL_INT n_ = n, nrhs = (nrhs_override >= 0) ? nrhs_override : ((b && x) ? 1 : 0), error = 0;
        double ddum;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n_,
                Ax.data(), Ap.data(), Ai.data(), perm.data(), &nrhs,
                iparm, &msglvl, b ? b : &ddum, x ? x : &ddum, &error);
        MFEM_VERIFY(error == 0, "PARDISO phase " << phase
                    << " failed with error code " << error);
    }

    // Extract the upper triangle (i<=j) in 0-based CSR -- shared by
    // Factorize() and Refactorize().
    const bool drop_zeros_ = (std::getenv("PARD_KEEP_ZEROS") == nullptr);
    long n_dropped_ = 0;
    void ExtractUpperCSR(const mfem::SparseMatrix &A) {
        n = A.Height();
        perm.assign(n, 0);
        const int *I = A.GetI(), *J = A.GetJ();
        const double *D = A.GetData();
        Ap.assign(n + 1, 0);
        n_dropped_ = 0;
        std::vector<MKL_INT> Ai_tmp; std::vector<double> Ax_tmp;
        for (int i = 0; i < n; i++) {
            Ap[i] = (MKL_INT)Ai_tmp.size();
            for (int k = I[i]; k < I[i+1]; k++) {
                int j = J[k];
                if (j < i && mtype != 11 && mtype != 1) continue;   // mtype 1/11 (real unsymmetric): full CSR, not the upper triangle
                // PERF (v3): drop exact-zero off-diagonal entries. The full-bridge Newton matrix is laid on a pattern
                // spanning ALL 63 layers (stable across births), so most of its entries are structural zeros from
                // not-yet-deposited elements; PARDISO would order/factor them all (11x the non-zeros of the active
                // system on the 2x mesh). The matrix is mathematically unchanged; only the elimination order differs.
                // PARD_KEEP_ZEROS=1 restores the old behaviour.
                if (drop_zeros_ && j != i && D[k] == 0.0) {
                    // mtype 1 (structurally symmetric) needs a symmetric pattern: keep a zero whose transpose is non-zero
                    // (e.g. identity essential rows with their columns kept). Columns of A must be sorted.
                    bool keep = false;
                    if (mtype == 1) {
                        const int *b = J + I[j], *e = J + I[j+1], *f = std::lower_bound(b, e, i);
                        keep = (f != e && *f == i && D[f - J] != 0.0);
                    }
                    if (!keep) { ++n_dropped_; continue; }
                }
                Ai_tmp.push_back(j); Ax_tmp.push_back(D[k]);
            }
        }
        Ap[n] = (MKL_INT)Ai_tmp.size();
        Ai = std::move(Ai_tmp); Ax = std::move(Ax_tmp);
    }

public:
    PardisoDirectSolver() = default;
    // v7 (2026-09-24, ported from bridge_case_enzyme/activation_4layer's wrapper): mt = 11 selects the real
    // nonsymmetric factorisation (full CSR is passed); default 2 = SPD (upper triangle).
    explicit PardisoDirectSolver(MKL_INT mt) : mtype(mt) {}
    MKL_INT PerturbedPivots() const { return iparm[13]; }
    ~PardisoDirectSolver() {
        if (initialized) {
            double ddum; MKL_INT idum, error = 0, phase = -1, nrhs = 0;
            pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n,
                    Ax.data(), Ap.data(), Ai.data(), &idum, &nrhs,
                    iparm, &msglvl, &ddum, &ddum, &error);
        }
    }

    // Full factorize: analysis (phase 11, symbolic reordering) + numerical
    // factorization (phase 22). SparseMatrix must be finalized and
    // SYMMETRIC (only the upper triangle, including diagonal, is sent to
    // PARDISO -- that is its required storage convention for symmetric
    // matrix types). Always safe to call; establishes/re-establishes the
    // symbolic pattern for later Refactorize() calls to reuse.
    // Release MKL's internal analysis + factor storage (phase -1). REQUIRED before re-analysing on an
    // already-initialised handle: Factorize() resets pt[] to null, so without this every pattern change
    // (each layer activation, for a persistent solver) leaked a full factorization (~1 GB per system).
    void ReleaseInternal() {
        if (!initialized) return;
        double ddum; MKL_INT idum, error = 0, phase = -1, nrhs = 0;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n,
                Ax.data(), Ap.data(), Ai.data(), &idum, &nrhs,
                iparm, &msglvl, &ddum, &ddum, &error);
        initialized = false; has_symbolic = false;
    }
    double Factorize(const mfem::SparseMatrix &A) {
        ReleaseInternal();   // before ExtractUpperCSR overwrites the arrays the old handle was built from
        ExtractUpperCSR(A);
        int n_empty_rows = 0, n_no_diag = 0;
        for (int i = 0; i < n; i++) {
            if (Ap[i] == Ap[i+1]) n_empty_rows++;
            bool has_diag = false;
            for (MKL_INT k = Ap[i]; k < Ap[i+1]; k++) if (Ai[k] == i) has_diag = true;
            if (!has_diag) n_no_diag++;
        }
        std::cerr << "  [PARDISO DEBUG] n=" << n << " nnz_upper=" << Ai.size() << " zeros_dropped=" << n_dropped_
                  << " empty_rows=" << n_empty_rows << " no_diag=" << n_no_diag
                  << "\n" << std::flush;

        // Populate iparm[] with MKL's own valid defaults FIRST -- manually
        // zeroing the array and relying on iparm[0]=0 to mean "use defaults"
        // (the previous approach here) skips this and can leave entries
        // (fill-reducing ordering choice, iterative refinement, etc.) in a
        // state pardiso() doesn't handle gracefully during METIS reordering
        // on a real-sized matrix -- suspected root cause of the phase-11
        // segfault seen on the 19,716-DOF matrix (never reproduced on a
        // trivial 3x3 test, consistent with a size-dependent path).
        for (int i = 0; i < 64; i++) pt[i] = nullptr;
        pardisoinit(pt, &mtype, iparm);
        iparm[34] = 1; // zero-based indexing (override the default 1-based)
        // v7 tuning knobs for the nonsymmetric (mtype 11) sensitivity factorisation; unset = MKL defaults.
        // v7 default: two-level parallel factorisation for the nonsymmetric sensitivity matrix (iparm[23]=10; measured
        // 0.95 -> 0.60 s/step on the 553k mesh, identical residual). PARD11_TWOLEVEL overrides (0 = classic).
        // Not for mtype 1: mtype 1 + iparm[23]=1 segfaulted (job 821099).
        if (mtype == 11) iparm[23] = 10;
        // SPD (Newton / v6 sensitivity) factorisation: two-level only on request (PARD2_TWOLEVEL=1), not yet validated.
        if (mtype == 2) { if (const char *e = std::getenv("PARD2_TWOLEVEL")) iparm[23] = (MKL_INT)std::atoi(e); }
        if (mtype == 11 || mtype == 1) {
            auto knob = [](const char *nm, MKL_INT &v) { if (const char *e = std::getenv(nm)) v = (MKL_INT)std::atoi(e); };
            knob("PARD11_IR", iparm[7]); knob("PARD11_SCALE", iparm[10]); knob("PARD11_MATCH", iparm[12]);
            knob("PARD11_PSOLVE", iparm[24]); knob("PARD11_PIVOT", iparm[9]); knob("PARD11_TWOLEVEL", iparm[23]);
        }
        iparm[17] = -1; // report the number of non-zeros in the factors
        initialized = true;

        auto t0 = std::chrono::high_resolution_clock::now();
        CallPhase(11, nullptr, nullptr); // analysis (reordering)
        CallPhase(22, nullptr, nullptr); // numerical factorization
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cerr << "  [PARDISO DEBUG] factorize_s="
                  << std::chrono::duration<double>(t1-t0).count() << " mtype=" << mtype << " factor_nnz=" << iparm[17]
                  << " peak_mem_kb=" << std::max(iparm[14], iparm[15] + iparm[16])
                  << " twolevel=" << iparm[23] << " ir=" << iparm[7] << " scale=" << iparm[10] << " match=" << iparm[12] << " psolve=" << iparm[24] << "\n" << std::flush;
        Ap_ref = Ap; Ai_ref = Ai;
        has_symbolic = true;
        return std::chrono::duration<double>(t1 - t0).count();
    }

    // Cheap refactorize: reuses the symbolic analysis (phase 11) from a
    // prior Factorize() call, redoing ONLY the numerical factorization
    // (phase 22) with the new matrix VALUES. Valid only when the sparsity
    // pattern (which entries exist) is unchanged from the original
    // Factorize() call -- true for this driver's matrix, which is always
    // built from the same fixed-mesh DiffusionIntegrator/MassIntegrator/
    // diagonal-addition sequence, so only VALUES (not structure) change
    // step to step. Verifies the pattern still matches before reusing pt;
    // falls back to a full Factorize() (redoing phase 11) if it doesn't,
    // so this is always safe to call even if that assumption is ever wrong.
    double Refactorize(const mfem::SparseMatrix &A) {
        if (!has_symbolic) return Factorize(A);
        ExtractUpperCSR(A);
        bool pattern_ok = (Ap.size() == Ap_ref.size()) && (Ai.size() == Ai_ref.size())
                        && std::equal(Ap.begin(), Ap.end(), Ap_ref.begin())
                        && std::equal(Ai.begin(), Ai.end(), Ai_ref.begin());
        if (!pattern_ok) return Factorize(A);

        auto t0 = std::chrono::high_resolution_clock::now();
        CallPhase(22, nullptr, nullptr); // numerical factorization only
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }

    // Cheap: reuses the factorization from Factorize().
    void Solve(const mfem::Vector &b, mfem::Vector &x) const {
        x.SetSize(n);
        const_cast<PardisoDirectSolver*>(this)->CallPhase(
            33, const_cast<double*>(b.GetData()), x.GetData());
    }

    // Batched multi-RHS solve: all `nrhs` right-hand sides in ONE phase-33
    // call instead of `nrhs` separate calls, each paying PARDISO's own
    // fixed per-call overhead (thread team setup/teardown, cache-cold
    // triangular sweep) independently. Ported from the validated pattern in
    // bare_plate_30_param/parallelization/CPU/tier1_pardiso_threading/
    // pardiso_direct.hpp, where this was "the single largest win of the
    // entire" Tier1 optimization effort (-70% on solve cost there).
    //
    // B/X are packed in PARDISO's required column-major n x nrhs layout:
    // RHS k's n components occupy [k*n, (k+1)*n) of the flat buffer.
    void SolveMulti(const std::vector<const mfem::Vector*> &B,
                     std::vector<mfem::Vector*> &X) {
        int nrhs = (int)B.size();
        Bpack_.resize((size_t)n * nrhs);
        Xpack_.resize((size_t)n * nrhs);
        for (int k = 0; k < nrhs; k++) {
            const double *bp = B[k]->GetData();
            std::copy(bp, bp + n, Bpack_.data() + (size_t)k * n);
        }
        CallPhase(33, Bpack_.data(), Xpack_.data(), (MKL_INT)nrhs);
        for (int k = 0; k < nrhs; k++) {
            X[k]->SetSize(n);
            std::copy(Xpack_.data() + (size_t)k * n,
                      Xpack_.data() + (size_t)(k + 1) * n, X[k]->GetData());
        }
    }

    // Transposed batched solve A^T X = B on the same factorisation (iparm[11]=2), e.g. for adjoints.
    void SolveMultiT(const std::vector<const mfem::Vector*> &B, std::vector<mfem::Vector*> &X) {
        iparm[11] = 2; SolveMulti(B, X); iparm[11] = 0;
    }

private:
    std::vector<double> Bpack_, Xpack_; // scratch buffers for SolveMulti,
                                         // reused across calls to avoid
                                         // reallocating every step.
};

} // namespace sensei
