// rs_mech.cpp -- incremental thermo-mechanical solve on the single-layer full-bridge mesh: residual stress from the nominal
// thermal history (temperature snapshots). Hill/J2 thermoplastic element kernel (rs_kernel.hpp), Enzyme tangent, tie constraint
// (bridge bottom -> substrate top), substrate bottom clamped, PARDISO. Nominal solve (sensitivities are a later stage).
#include "mfem.hpp"
#include "tie_constraint.hpp"
#include "pardiso_direct.hpp"
#include "rs_kernels.h"
#include "rs_kernel.hpp"
#include <omp.h>
#include <fstream>
#include <sstream>
#include <map>
#include <chrono>
#include <sys/stat.h>
using namespace mfem; using namespace std;
double g_tie_ms = 0.0;
static double now_s() { return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char *argv[])
{
    const char *mesh_file = "full_bridge.mesh", *tie_file = "tie_weights_full.txt";
    string snap_dir, out_dir = "rs_out"; int nmax_inc = 1 << 30, newt_max = 30, ckpt_every = 20; bool do_restart = false; int dorder = 0; double pxyz[3] = {2.5, 0.05, 0.18};
    double tolr = 1e-8, tola = 1e-6; double par[5] = {1, 1, 1, 1, 1};
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh."); args.AddOption(&tie_file, "-tie", "--tie", "Tie weights (scalar).");
    args.AddOption(&snap_dir, "-snap", "--snap", "Directory with T_<step>.bin and snapshots.csv.");
    args.AddOption(&out_dir, "-o", "--out", "Output directory.");
    args.AddOption(&nmax_inc, "-ninc", "--ninc", "Use only the first N snapshots (tests).");
    args.AddOption(&newt_max, "-nmax", "--newton-max", "Max Newton iterations per increment.");
    args.AddOption(&tolr, "-tolr", "--tolr", "Newton relative tolerance."); args.AddOption(&tola, "-tola", "--tola", "Newton absolute tolerance (N).");
    args.AddOption(&ckpt_every, "-ckpt-every", "--ckpt-every", "Checkpoint every N increments.");
    args.AddOption(&do_restart, "-restart", "--restart", "-no-restart", "--no-restart", "Restart from out/ckpt.bin.");
    args.AddOption(&par[0], "-sE", "--sE", "Scale on E."); args.AddOption(&par[1], "-snu", "--snu", "Scale on nu.");
    args.AddOption(&par[2], "-sa", "--sa", "Scale on CTE."); args.AddOption(&par[3], "-ssy", "--ssy", "Scale on yield stress.");
    args.AddOption(&par[4], "-sH", "--sH", "Scale on hardening modulus.");
    args.AddOption(&dorder, "-dorder", "--dorder", "0 nominal, 1 + first-order, 2 + full second-order Hessian (Enzyme forward mode).");
    args.AddOption(&pxyz[0], "-px", "--px", "Probe x [mm]."); args.AddOption(&pxyz[1], "-py", "--py", "Probe y [mm]."); args.AddOption(&pxyz[2], "-pz", "--pz", "Probe z [mm].");
    args.Parse(); if (!args.Good()) { args.PrintUsage(cout); return 1; }
    mkdir(out_dir.c_str(), 0775);

    // ---------------- mesh, geometry cache ----------------
    Mesh mesh(mesh_file, 1, 1); const int ne = mesh.GetNE();
    H1_FECollection fec(1, 3); FiniteElementSpace fes(&mesh, &fec, 1); const int nn = fes.GetNDofs();
    const IntegrationRule &ir = IntRules.Get(Geometry::CUBE, 3); MFEM_VERIFY(ir.GetNPoints() == 8, "expected 2x2x2 Gauss");
    vector<int> edof((size_t)ne * 8); vector<double> geom((size_t)ne * 200);
    double Ntab[8][8];
    {
        DenseMatrix dsh(8, 3), Jinv(3, 3); Vector sh(8); Array<int> d;
        for (int e = 0; e < ne; e++) {
            fes.GetElementDofs(e, d); MFEM_VERIFY(d.Size() == 8, "hex8 expected");
            for (int a = 0; a < 8; a++) edof[(size_t)e * 8 + a] = d[a];
            ElementTransformation &Tr = *mesh.GetElementTransformation(e); const FiniteElement &fe = *fes.GetFE(e);
            for (int g = 0; g < 8; g++) {
                const IntegrationPoint &ip = ir.IntPoint(g); Tr.SetIntPoint(&ip);
                fe.CalcShape(ip, sh); fe.CalcDShape(ip, dsh); CalcInverse(Tr.Jacobian(), Jinv);
                if (e == 0) for (int a = 0; a < 8; a++) Ntab[g][a] = sh(a);
                double *gm = &geom[(size_t)e * 200 + g * 25];
                for (int a = 0; a < 8; a++) for (int j = 0; j < 3; j++) { double s = 0; for (int k = 0; k < 3; k++) s += dsh(a, k) * Jinv(k, j); gm[3 * a + j] = s; }
                gm[24] = ip.weight * Tr.Weight();
            }
        }
    }
    cout << "mesh: " << ne << " elements, " << nn << " nodes\n";

    // ---------------- initial temperature (same rule as the thermal driver) ----------------
    const double T_amb = 40.0 + 273.15, T_sub = 80.0 + 273.15;
    vector<double> T0(nn, T_amb);
    for (int e = 0; e < ne; e++) if (mesh.GetAttribute(e) == 1) for (int a = 0; a < 8; a++) T0[edof[(size_t)e * 8 + a]] = T_sub;

    // ---------------- tie + clamp -> free dofs, prolongation P (full 3nn x free) ----------------
    TieMap tie; tie.Load(tie_file, nn);
    double zmin = 1e300; for (int v = 0; v < mesh.GetNV(); v++) zmin = min(zmin, mesh.GetVertex(v)[2]);
    vector<char> clamped(nn, 0); { Array<int> vd; int nc = 0;
        for (int e = 0; e < ne; e++) for (int a = 0; a < 8; a++) { const int n = edof[(size_t)e * 8 + a]; (void)n; }
        // node coordinates through the element vertices (order-1 dofs == vertices)
        for (int v = 0; v < mesh.GetNV(); v++) if (fabs(mesh.GetVertex(v)[2] - zmin) < 1e-9) { clamped[v] = 1; nc++; }
        cout << "bottom clamp: z=" << zmin << " mm, nodes=" << nc << "\n"; }
    vector<int> free_of(3 * (size_t)nn, -1); int nfree = 0;
    for (int n = 0; n < nn; n++) if (!tie.is_slave[n] && !clamped[n]) for (int c = 0; c < 3; c++) free_of[3 * (size_t)n + c] = nfree++;
    SparseMatrix P(3 * nn, nfree);
    for (int n = 0; n < nn; n++) if (free_of[3 * (size_t)n] >= 0) for (int c = 0; c < 3; c++) P.Set(3 * n + c, free_of[3 * (size_t)n + c], 1.0);
    for (int s = 0; s < tie.n_slave; s++) for (int c = 0; c < 3; c++) for (int k = 0; k < 4; k++) {
        const int fc = free_of[3 * (size_t)tie.masters[s][k] + c]; if (fc >= 0) P.Add(3 * tie.slave_of[s] + c, fc, tie.weights[s][k]); }
    P.Finalize(); SparseMatrix *Pt = Transpose(P);
    cout << "dofs: full " << 3 * nn << "  free " << nfree << "  (slave nodes " << tie.n_slave << ")\n";

    // ---------------- snapshots ----------------
    vector<pair<int, double>> snaps; { ifstream f(snap_dir + "/snapshots.csv"); string line; map<int, double> m;
        while (getline(f, line)) { int st; double t, dt; char c; stringstream ss(line); if (ss >> st >> c >> t >> c >> dt) m[st] = t; }
        for (auto &kv : m) snaps.push_back({kv.first, kv.second}); }
    MFEM_VERIFY(!snaps.empty(), "no snapshots found in " << snap_dir);
    const int ninc = min((int)snaps.size(), nmax_inc); cout << "snapshots: " << snaps.size() << " (using " << ninc << ")\n";

    // ---------------- state ----------------
    vector<double> st_old((size_t)ne * 56, 0.0), st_new((size_t)ne * 56, 0.0), sig((size_t)ne * 48, 0.0);
    vector<double> u(3 * (size_t)nn, 0.0), Tn(nn), fe_all((size_t)ne * 24), Ke_all((size_t)ne * 576);
    int inc0 = 0;
    const string ckfile = out_dir + "/ckpt.bin";
    if (do_restart) { ifstream f(ckfile, ios::binary); if (f) { f.read((char *)&inc0, sizeof(int)); f.read((char *)u.data(), u.size() * 8); f.read((char *)st_old.data(), st_old.size() * 8);
        cout << "restart from increment " << inc0 << "\n"; } }

    // ---------------- sensitivity storage (dorder >= 1) ----------------
    const int NP = 5; vector<array<int, 2>> pairs; for (int i = 0; i < NP; i++) for (int j = i; j < NP; j++) pairs.push_back({i, j});
    const int NPR = (int)pairs.size();
    vector<vector<double>> dso, dsn, du1, d2so, d2sn, du2;      // first/second-order: old state, new state, displacement
    vector<double> tmp_ddst, tmp_ddsig;
    vector<vector<double>> dsig1;                               // first-order stress qp (kept for the last increment)
    vector<vector<double>> d2sig;
    if (dorder >= 1) { dso.assign(NP, vector<double>((size_t)ne * 56, 0.0)); dsn = dso; du1.assign(NP, vector<double>(3 * (size_t)nn, 0.0)); dsig1.assign(NP, vector<double>((size_t)ne * 48, 0.0)); }
    if (dorder >= 2) { d2so.assign(NPR, vector<double>((size_t)ne * 56, 0.0)); d2sn = d2so; du2.assign(NPR, vector<double>(3 * (size_t)nn, 0.0)); d2sig.assign(NPR, vector<double>((size_t)ne * 48, 0.0));
                       tmp_ddst.assign((size_t)ne * 56, 0.0); tmp_ddsig.assign((size_t)ne * 48, 0.0); }
    // probe element: closest bridge-element centroid to (px,py,pz)
    int probe_e = -1; { double best = 1e300; Array<int> vs; for (int e = 0; e < ne; e++) { if (mesh.GetAttribute(e) != 2) continue; mesh.GetElementVertices(e, vs); double c[3] = {0, 0, 0};
        for (int a = 0; a < vs.Size(); a++) for (int k = 0; k < 3; k++) c[k] += mesh.GetVertex(vs[a])[k] / vs.Size();
        const double d = (c[0] - pxyz[0]) * (c[0] - pxyz[0]) + (c[1] - pxyz[1]) * (c[1] - pxyz[1]) + (c[2] - pxyz[2]) * (c[2] - pxyz[2]); if (d < best) { best = d; probe_e = e; } }
        cout << "probe element " << probe_e << " (distance " << sqrt(best) << " mm from the requested point)\n"; }
    ofstream fprobe(out_dir + "/probe_hist.csv", do_restart ? ios::app : ios::trunc);
    auto elem_avg = [&](const vector<double> &a, int e, double out[6]) { for (int k = 0; k < 6; k++) { double s2 = 0; for (int g = 0; g < 8; g++) s2 += a[(size_t)e * 48 + g * 6 + k]; out[k] = s2 / 8.0; } const double is2 = 1.0 / sqrt(2.0); out[3] *= is2; out[4] *= is2; out[5] *= is2; };
    sensei::PardisoDirectSolver pard(2);
    ofstream flog(out_dir + "/increments.csv", do_restart ? ios::app : ios::trunc);
    if (!do_restart) flog << "inc,step,time,newton,rnorm,max_vm,max_peeq,Tmax,t_assemble,t_solve\n";

    auto assemble = [&](bool wantK, Vector &rfree, SparseMatrix **Kred, double &t_asm) {
        const double t0 = now_s();
        #pragma omp parallel for schedule(dynamic, 64)
        for (int e = 0; e < ne; e++) {
            double ue[24], Tq[8], T0q[8];
            for (int a = 0; a < 8; a++) { const int n = edof[(size_t)e * 8 + a]; for (int c = 0; c < 3; c++) ue[3 * a + c] = u[3 * (size_t)n + c]; }
            for (int g = 0; g < 8; g++) { double s = 0, s0 = 0; for (int a = 0; a < 8; a++) { const int n = edof[(size_t)e * 8 + a]; s += Ntab[g][a] * Tn[n]; s0 += Ntab[g][a] * T0[n]; } Tq[g] = s; T0q[g] = s0; }
            const double *gm = &geom[(size_t)e * 200], *so = &st_old[(size_t)e * 56]; double *sn = &st_new[(size_t)e * 56], *sg = &sig[(size_t)e * 48], *fe = &fe_all[(size_t)e * 24];
            if (!wantK) { rs_elem(ue, Tq, T0q, so, par, gm, fe, sn, sg); continue; }
            double *Ke = &Ke_all[(size_t)e * 576];
            for (int j = 0; j < 24; j++) {
                double du[24] = {0}, dst[56] = {0}, dpar[5] = {0}, df[24], dsn[56], dsg[48], f2[24], s2[56], g2[48];
                du[j] = 1.0;
                rs_elem_d1(ue, du, Tq, T0q, so, dst, par, dpar, gm, f2, df, s2, dsn, g2, dsg);
                if (j == 0) { for (int i = 0; i < 24; i++) fe[i] = f2[i]; for (int i = 0; i < 56; i++) sn[i] = s2[i]; for (int i = 0; i < 48; i++) sg[i] = g2[i]; }
                for (int i = 0; i < 24; i++) Ke[i * 24 + j] = df[i];
            }
        }
        Vector rfull(3 * nn); rfull = 0.0;
        for (int e = 0; e < ne; e++) for (int a = 0; a < 8; a++) { const int n = edof[(size_t)e * 8 + a]; for (int c = 0; c < 3; c++) rfull[3 * n + c] += fe_all[(size_t)e * 24 + 3 * a + c]; }
        rfree.SetSize(nfree); Pt->Mult(rfull, rfree);
        if (wantK) {
            SparseMatrix K(3 * nn, 3 * nn); Array<int> vd(24); DenseMatrix Kd(24, 24);
            for (int e = 0; e < ne; e++) {
                for (int a = 0; a < 8; a++) { const int n = edof[(size_t)e * 8 + a]; for (int c = 0; c < 3; c++) vd[3 * a + c] = 3 * n + c; }
                memcpy(Kd.Data(), &Ke_all[(size_t)e * 576], 576 * 8);           // symmetric to round-off: layout (row-major == column-major)
                K.AddSubMatrix(vd, vd, Kd, 0);
            }
            K.Finalize(); SparseMatrix *KP = Mult(K, P); *Kred = Mult(*Pt, *KP); delete KP; (*Kred)->SortColumnIndices();
        }
        t_asm = now_s() - t0;
    };

    for (int inc = inc0; inc < ninc; inc++) {
        { ifstream f(snap_dir + "/" + ([&]{ char b[64]; snprintf(b, sizeof b, "T_%06d.bin", snaps[inc].first); return string(b); })(), ios::binary);
          MFEM_VERIFY(f, "missing snapshot " << snaps[inc].first); f.read((char *)Tn.data(), nn * 8); }
        Vector r, du_free(nfree), dx(nfree), uf(nfree); double rn0 = 0, rn = 0, t_asm_tot = 0, t_sol_tot = 0; int it = 0; bool conv = false;
        for (it = 0; it < newt_max; it++) {
            SparseMatrix *Kr = nullptr; double ta; assemble(true, r, &Kr, ta); t_asm_tot += ta;
            rn = r.Norml2(); if (it == 0) rn0 = rn;
            if (rn < tola || rn < tolr * rn0) { conv = true; if (dorder >= 1) { const double ts = now_s(); if (it == 0 && inc == inc0) pard.Factorize(*Kr); else pard.Refactorize(*Kr); t_sol_tot += now_s() - ts; } delete Kr; break; }
            const double ts = now_s();
            if (it == 0 && inc == inc0) pard.Factorize(*Kr); else pard.Refactorize(*Kr);
            Vector rhs(r); rhs *= -1.0; pard.Solve(rhs, dx); t_sol_tot += now_s() - ts; delete Kr;
            Vector dfull(3 * nn); P.Mult(dx, dfull);
            for (size_t i = 0; i < u.size(); i++) u[i] += dfull[i];
        }
        if (!conv) cout << "WARNING: increment " << inc << " (step " << snaps[inc].first << ") did not converge, |r|=" << rn << " (rn0 " << rn0 << ")\n";

        // ---------------- sensitivities of this increment (Enzyme forward mode) ----------------
        if (dorder >= 1) {
            const double ts0 = now_s();
            auto elem_T = [&](int e, double *Tq, double *T0q) { for (int g = 0; g < 8; g++) { double a1 = 0, a0 = 0; for (int a = 0; a < 8; a++) { const int n = edof[(size_t)e * 8 + a]; a1 += Ntab[g][a] * Tn[n]; a0 += Ntab[g][a] * T0[n]; } Tq[g] = a1; T0q[g] = a0; } };
            auto gather = [&](const vector<double> &v, int e, double *ve) { for (int a = 0; a < 8; a++) { const int n = edof[(size_t)e * 8 + a]; for (int c = 0; c < 3; c++) ve[3 * a + c] = v[3 * (size_t)n + c]; } };
            auto solve_forcing = [&](vector<double> &du_out) {            // fe_all holds the element forcing; K du = -forcing
                Vector rfull(3 * nn); rfull = 0.0;
                for (int e = 0; e < ne; e++) for (int a = 0; a < 8; a++) { const int n = edof[(size_t)e * 8 + a]; for (int c = 0; c < 3; c++) rfull[3 * n + c] += fe_all[(size_t)e * 24 + 3 * a + c]; }
                Vector rf(nfree), x(nfree), dfull(3 * nn); Pt->Mult(rfull, rf); rf *= -1.0; pard.Solve(rf, x); P.Mult(x, dfull);
                for (size_t k = 0; k < du_out.size(); k++) du_out[k] = dfull[k];
            };
            double unit[5][5]; for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) unit[i][j] = (i == j) ? 1.0 : 0.0;
            const double z5[5] = {0, 0, 0, 0, 0};
            for (int i = 0; i < NP; i++) {
                #pragma omp parallel for schedule(dynamic, 64)
                for (int e = 0; e < ne; e++) {
                    double ue[24], Tq[8], T0q[8], z24[24] = {0}, f2[24], df[24], s2[56], dsn_[56], g2[48], dsg_[48];
                    gather(u, e, ue); elem_T(e, Tq, T0q);
                    rs_elem_d1(ue, z24, Tq, T0q, &st_old[(size_t)e * 56], &dso[i][(size_t)e * 56], par, unit[i], &geom[(size_t)e * 200], f2, df, s2, dsn_, g2, dsg_);
                    for (int k = 0; k < 24; k++) fe_all[(size_t)e * 24 + k] = df[k];
                }
                solve_forcing(du1[i]);
                #pragma omp parallel for schedule(dynamic, 64)
                for (int e = 0; e < ne; e++) {
                    double ue[24], due[24], Tq[8], T0q[8], f2[24], df[24], s2[56], g2[48];
                    gather(u, e, ue); gather(du1[i], e, due); elem_T(e, Tq, T0q);
                    rs_elem_d1(ue, due, Tq, T0q, &st_old[(size_t)e * 56], &dso[i][(size_t)e * 56], par, unit[i], &geom[(size_t)e * 200], f2, df, s2, &dsn[i][(size_t)e * 56], g2, &dsig1[i][(size_t)e * 48]);
                }
            }
            if (dorder >= 2) for (int p = 0; p < NPR; p++) {
                const int i = pairs[p][0], j = pairs[p][1];
                #pragma omp parallel for schedule(dynamic, 32)
                for (int e = 0; e < ne; e++) {
                    double ue[24], au[24], bu[24], z24[24] = {0}, Tq[8], T0q[8];
                    double f[24], df[24], sn[56], dsn_[56], sg[48], dsg_[48], bf[24], ddf[24], bsn[56], ddsn[56], bsg[48], ddsg[48];
                    gather(u, e, ue); gather(du1[i], e, au); gather(du1[j], e, bu); elem_T(e, Tq, T0q);
                    rs_elem_d2(ue, au, Tq, T0q, &st_old[(size_t)e * 56], &dso[i][(size_t)e * 56], par, unit[i], &geom[(size_t)e * 200], bu, &dso[j][(size_t)e * 56], unit[j],
                               f, df, sn, dsn_, sg, dsg_, bf, ddf, bsn, ddsn, bsg, ddsg);
                    double f2[24], dfc[24], s2[56], dsc[56], g2[48], dgc[48];       // c = (0, d2 old state, 0): forcing from the second-order old state
                    rs_elem_d1(ue, z24, Tq, T0q, &st_old[(size_t)e * 56], &d2so[p][(size_t)e * 56], par, z5, &geom[(size_t)e * 200], f2, dfc, s2, dsc, g2, dgc);
                    for (int k = 0; k < 24; k++) fe_all[(size_t)e * 24 + k] = ddf[k] + dfc[k];
                    for (int k = 0; k < 56; k++) tmp_ddst[(size_t)e * 56 + k] = ddsn[k];
                    for (int k = 0; k < 48; k++) tmp_ddsig[(size_t)e * 48 + k] = ddsg[k];
                }
                solve_forcing(du2[p]);
                #pragma omp parallel for schedule(dynamic, 64)
                for (int e = 0; e < ne; e++) {
                    double ue[24], d2u[24], Tq[8], T0q[8], f2[24], df[24], s2[56], dsn_[56], g2[48], dsg_[48];
                    gather(u, e, ue); gather(du2[p], e, d2u); elem_T(e, Tq, T0q);
                    rs_elem_d1(ue, d2u, Tq, T0q, &st_old[(size_t)e * 56], &d2so[p][(size_t)e * 56], par, z5, &geom[(size_t)e * 200], f2, df, s2, dsn_, g2, dsg_);
                    for (int k = 0; k < 56; k++) d2sn[p][(size_t)e * 56 + k] = tmp_ddst[(size_t)e * 56 + k] + dsn_[k];
                    for (int k = 0; k < 48; k++) d2sig[p][(size_t)e * 48 + k] = tmp_ddsig[(size_t)e * 48 + k] + dsg_[k];
                }
            }
            t_sol_tot += now_s() - ts0;
            if (probe_e >= 0) {                                         // probe-element stress and its derivatives
                double v[6]; elem_avg(sig, probe_e, v);
                fprobe << setprecision(12) << inc << "," << snaps[inc].first << "," << snaps[inc].second; for (int k = 0; k < 6; k++) fprobe << "," << v[k];
                for (int i = 0; i < NP; i++) { elem_avg(dsig1[i], probe_e, v); for (int k = 0; k < 6; k++) fprobe << "," << v[k]; }
                if (dorder >= 2) for (int p = 0; p < NPR; p++) { elem_avg(d2sig[p], probe_e, v); for (int k = 0; k < 6; k++) fprobe << "," << v[k]; }
                fprobe << "\n"; fprobe.flush();
            }
            dso = dsn; if (dorder >= 2) d2so = d2sn;                    // commit sensitivities of the state
        }
        else if (probe_e >= 0) { double v[6]; elem_avg(sig, probe_e, v);   // nominal run: probe stress only (finite-difference reference runs)
            fprobe << setprecision(14) << inc << "," << snaps[inc].first << "," << snaps[inc].second; for (int k = 0; k < 6; k++) fprobe << "," << v[k]; fprobe << "\n"; fprobe.flush(); }
        st_old = st_new;                                      // commit
        double mvm = 0, mpe = 0, tmax = 0;
        for (size_t q = 0; q < (size_t)ne * 8; q++) { const double *s = &sig[q * 6]; const double vm = sqrt(max(0.0, 0.5 * ((s[0] - s[1]) * (s[0] - s[1]) + (s[1] - s[2]) * (s[1] - s[2]) + (s[2] - s[0]) * (s[2] - s[0])) + 1.5 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))); mvm = max(mvm, vm); mpe = max(mpe, st_old[q * 7 + 6]); }
        for (int n = 0; n < nn; n++) tmax = max(tmax, Tn[n]);
        flog << inc << "," << snaps[inc].first << "," << snaps[inc].second << "," << it << "," << rn << "," << mvm << "," << mpe << "," << tmax << "," << t_asm_tot << "," << t_sol_tot << "\n"; flog.flush();
        printf("inc %4d step %6d t=%.6g  newton %2d |r|=%.2e  max vM %.1f MPa  max peeq %.4f  Tmax %.1f  (asm %.1fs, solve %.1fs)\n", inc, snaps[inc].first, snaps[inc].second, it, rn, mvm, mpe, tmax, t_asm_tot, t_sol_tot); fflush(stdout);
        if ((inc + 1) % ckpt_every == 0 || inc == ninc - 1) { ofstream f(ckfile, ios::binary); int nx = inc + 1; f.write((char *)&nx, sizeof(int)); f.write((char *)u.data(), u.size() * 8); f.write((char *)st_old.data(), st_old.size() * 8); }
    }
    // ---------------- final fields ----------------
    { ofstream f(out_dir + "/final_u.bin", ios::binary); f.write((char *)u.data(), u.size() * 8); }
    { ofstream f(out_dir + "/final_sigma_qp.bin", ios::binary); f.write((char *)sig.data(), sig.size() * 8); }
    { vector<double> pe((size_t)ne * 8); for (size_t q = 0; q < pe.size(); q++) pe[q] = st_old[q * 7 + 6]; ofstream f(out_dir + "/final_peeq_qp.bin", ios::binary); f.write((char *)pe.data(), pe.size() * 8); }
    if (dorder >= 0) { vector<double> av((size_t)ne * 6); auto dump = [&](const vector<double> &a, const string &nm) { for (int e = 0; e < ne; e++) { double o[6]; elem_avg(a, e, o); for (int k = 0; k < 6; k++) av[(size_t)e * 6 + k] = o[k]; } ofstream f(out_dir + "/" + nm, ios::binary); f.write((char *)av.data(), av.size() * 8); };
        dump(sig, "final_sigma_elem.bin"); for (int i = 0; i < (dorder >= 1 ? NP : 0); i++) dump(dsig1[i], "final_dsigma_elem_" + to_string(i) + ".bin");
        for (int p = 0; p < (dorder >= 2 ? NPR : 0); p++) dump(d2sig[p], "final_d2sigma_elem_" + to_string(pairs[p][0]) + to_string(pairs[p][1]) + ".bin"); }
    cout << "done\n"; return 0;
}
