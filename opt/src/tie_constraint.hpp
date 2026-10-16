// ============================================================
//  Exact tie-constraint elimination (P / P^T expand-contract)
//  ------------------------------------------------------------
//  Validated in isolation (test_tie_constraint.cpp) against a plain
//  linear diffusion problem under a real two-sided gradient: exact
//  continuity at the interface, smooth monotonic profile, CG converges.
//  This header wraps the SAME mechanism around the REAL (nonlinear)
//  FachinottiResidualOp/JacobianOp from xtrack_multilayer.cpp, UNCHANGED
//  -- the wrapper only ever calls their existing Mult()/GetGradient() on
//  expanded full-space vectors, then contracts the result. No physics
//  code is touched or duplicated.
//
//  Slave DOFs (bridge BOTTOM_SURF nodes) are NEVER independently solved;
//  their value is always reconstructed from a weighted combination of
//  master (substrate TOP_SURF) DOFs. Free-space DOF count = ndof_full -
//  n_slave.
// ============================================================
#pragma once
#include "mfem.hpp"
#include <fstream>
#include <vector>
#include <array>
#include <chrono>

// per-step tie expand/contract wall accumulator (see StepTimers in the driver)
extern double g_tie_ms;
static inline double _tie_now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class TieMap {
public:
    int ndof_full, ndof_free, n_slave;
    std::vector<int> slave_of;                    // slave idx -> full vertex id
    std::vector<std::array<int,4>> masters;        // slave idx -> 4 master full ids
    std::vector<std::array<double,4>> weights;     // slave idx -> 4 weights (sum=1)
    std::vector<bool> is_slave;                    // full idx -> is it a slave?
    std::vector<int> free_of_full;                 // full idx -> free idx (-1 if slave)
    std::vector<int> full_of_free;                 // free idx -> full idx

    TieMap() = default;

    void Load(const std::string &tie_weights_path, int ndof_full_)
    {
        ndof_full = ndof_full_;
        std::ifstream tf(tie_weights_path);
        tf >> n_slave;
        slave_of.resize(n_slave);
        masters.resize(n_slave);
        weights.resize(n_slave);
        is_slave.assign(ndof_full, false);
        for (int s = 0; s < n_slave; s++)
        {
            int sv; tf >> sv;
            slave_of[s] = sv;
            is_slave[sv] = true;
            for (int k = 0; k < 4; k++) tf >> masters[s][k] >> weights[s][k];
        }
        free_of_full.assign(ndof_full, -1);
        for (int i = 0; i < ndof_full; i++)
            if (!is_slave[i]) { free_of_full[i] = (int)full_of_free.size(); full_of_free.push_back(i); }
        ndof_free = (int)full_of_free.size();
    }

    // free-space vector -> full-space vector (slaves reconstructed)
    void Expand(const mfem::Vector &xr, mfem::Vector &xfull) const
    {
        const double *xp = xr.HostRead();
        double *fp = xfull.HostWrite();
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ndof_free; j++) fp[full_of_free[j]] = xp[j];
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < n_slave; s++)
        {
            double v = 0.0;
            for (int k = 0; k < 4; k++) v += weights[s][k] * xp[free_of_full[masters[s][k]]];
            fp[slave_of[s]] = v;
        }
    }

    // full-space vector -> free-space vector (slave rows scattered onto masters)
    void Contract(const mfem::Vector &yfull, mfem::Vector &yr) const
    {
        const double *yp = yfull.HostRead();
        double *rp = yr.HostWrite();
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ndof_free; j++) rp[j] = yp[full_of_free[j]];
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < n_slave; s++)
            for (int k = 0; k < 4; k++) {
                double add = weights[s][k] * yp[slave_of[s]];
                #pragma omp atomic
                rp[free_of_full[masters[s][k]]] += add;
            }
    }

    // Sparse prolongation P (ndof_full x ndof_free) such that xfull = P*xr
    // reproduces Expand() exactly, and P^T reproduces Contract(). Used by the
    // direct (PARDISO) reduced solver, which needs the ASSEMBLED reduced
    // Jacobian Jr = P^T J_full P rather than the matrix-free expand/contract.
    mfem::SparseMatrix* BuildProlongation() const
    {
        mfem::SparseMatrix *P = new mfem::SparseMatrix(ndof_full, ndof_free);
        for (int j = 0; j < ndof_free; j++)
            P->Add(full_of_free[j], j, 1.0);
        for (int s = 0; s < n_slave; s++)
            for (int k = 0; k < 4; k++)
                P->Add(slave_of[s], free_of_full[masters[s][k]], weights[s][k]);
        P->Finalize();
        return P;
    }

    // Reduced Jacobi-diagonal approximation: diag_free[j] = diag_full[full_of_free[j]]
    // + sum over slaves tied to j of w_k^2 * diag_full[slave] (approximates P^T diag(A) P).
    void ContractDiag(const mfem::Vector &diag_full, mfem::Vector &diag_free) const
    {
        for (int j = 0; j < ndof_free; j++) diag_free[j] = diag_full[full_of_free[j]];
        for (int s = 0; s < n_slave; s++)
            for (int k = 0; k < 4; k++)
                diag_free[free_of_full[masters[s][k]]] +=
                    weights[s][k]*weights[s][k] * diag_full[slave_of[s]];
    }
};

// Wraps any full-space Operator (residual or Jacobian) into reduced
// (tie-eliminated) space via expand/contract. Does not care whether the
// wrapped operator is linear or nonlinear -- it is simply evaluated on
// the expanded full-space vector each call, exactly as it already is
// everywhere else in the codebase.
class ReducedOpWrapper : public mfem::Operator {
public:
    mfem::Operator &full_op;
    const TieMap &tie;
    mutable mfem::Vector x_full, y_full;
    ReducedOpWrapper(mfem::Operator &op, const TieMap &t)
        : mfem::Operator(t.ndof_free), full_op(op), tie(t),
          x_full(t.ndof_full), y_full(t.ndof_full)
    { x_full.UseDevice(true); y_full.UseDevice(true); }
    void Mult(const mfem::Vector &xr, mfem::Vector &yr) const override
    {
        double _t = _tie_now_ms();
        tie.Expand(xr, x_full);
        g_tie_ms += _tie_now_ms() - _t;
        full_op.Mult(x_full, y_full);
        _t = _tie_now_ms();
        tie.Contract(y_full, yr);
        g_tie_ms += _tie_now_ms() - _t;
    }
};

// Wraps a Newton-style residual operator (one that also implements
// GetGradient(), returning a JacobianOp&) into reduced space. Mult()
// wraps the residual directly; GetGradient() expands x, calls the real
// GetGradient() (which internally refreshes the Jacobian's nodal terms
// exactly as it already does), then returns a ReducedOpWrapper around
// the resulting full-space Jacobian operator.
class ReducedResidualOp : public mfem::Operator {
    mfem::Operator &full_residual;
    const TieMap &tie;
    mutable mfem::Vector x_full, r_full;
    mutable ReducedOpWrapper *reduced_jac = nullptr;
public:
    ReducedResidualOp(mfem::Operator &fr, const TieMap &t)
        : mfem::Operator(t.ndof_free), full_residual(fr), tie(t),
          x_full(t.ndof_full), r_full(t.ndof_full)
    { x_full.UseDevice(true); r_full.UseDevice(true); }
    ~ReducedResidualOp() override { delete reduced_jac; }
    void Mult(const mfem::Vector &xr, mfem::Vector &rr) const override
    {
        double _t = _tie_now_ms();
        tie.Expand(xr, x_full);
        g_tie_ms += _tie_now_ms() - _t;
        full_residual.Mult(x_full, r_full);
        _t = _tie_now_ms();
        tie.Contract(r_full, rr);
        g_tie_ms += _tie_now_ms() - _t;
    }
    mfem::Operator &GetGradient(const mfem::Vector &xr) const override
    {
        tie.Expand(xr, x_full);
        mfem::Operator &Jfull = full_residual.GetGradient(x_full);
        delete reduced_jac;
        reduced_jac = new ReducedOpWrapper(Jfull, tie);
        return *reduced_jac;
    }
};

// Reduced-space Jacobi-preconditioned CG, mirroring FachinottiLinearSolver
// but building its diagonal via TieMap::ContractDiag instead of reading it
// directly -- same CG/Jacobi structure, same convergence tolerances,
// operating on the reduced (tie-eliminated) system. (A dk/dT Jacobian term
// briefly required GMRES here -- see bridge_multilayer.cpp's JacobianOp
// comment -- but was removed after controlled testing showed it had no
// measurable effect on convergence; reverted to CG accordingly.)
class ReducedLinearSolver : public mfem::Solver {
    mfem::CGSolver cg;
    const mfem::Vector &Kd_pa, &Kb_diag, &Meff_over_dt; // FULL-space diagonal sources
    const TieMap &tie;
    mutable mfem::Vector Jdiag_full, Jdiag_free;
    mutable mfem::OperatorJacobiSmoother *jac = nullptr;
    const mfem::Operator *A = nullptr;
public:
    ReducedLinearSolver(const mfem::Vector &kd, const mfem::Vector &kb,
                         const mfem::Vector &meff, const TieMap &t)
        : mfem::Solver(t.ndof_free), Kd_pa(kd), Kb_diag(kb), Meff_over_dt(meff),
          tie(t), Jdiag_full(t.ndof_full), Jdiag_free(t.ndof_free)
    {
        Jdiag_full.UseDevice(true); Jdiag_free.UseDevice(true);
        cg.iterative_mode = false;
        cg.SetRelTol(1e-8);
        cg.SetAbsTol(0.0);
        cg.SetMaxIter(2000);
        cg.SetPrintLevel(-1);
    }
    ~ReducedLinearSolver() override { delete jac; }
    void SetOperator(const mfem::Operator &op) override { A = &op; cg.SetOperator(op); }
    void Mult(const mfem::Vector &b, mfem::Vector &x) const override
    {
        MFEM_VERIFY(A != nullptr, "ReducedLinearSolver operator is not set.");
        for (int i = 0; i < tie.ndof_full; i++)
            Jdiag_full[i] = Kd_pa[i] + Kb_diag[i] + Meff_over_dt[i];
        tie.ContractDiag(Jdiag_full, Jdiag_free);
        delete jac;
        jac = new mfem::OperatorJacobiSmoother(Jdiag_free, mfem::Array<int>(), 1.0);
        const_cast<mfem::CGSolver&>(cg).SetPreconditioner(*jac);
        const_cast<mfem::CGSolver&>(cg).Mult(b, x);
    }
};
