// query.hpp -- parse the any-order derivative query (MFEM-free, header-only).
//
//   -deriv "k^2*cp ; k^4 ; P*Eta"     each term = product of name[^power];
//                                      d^3T/(dk^2 dcp), d^4T/dk^4, d^2T/(dP dEta)
//   -dir   "v1=k:1,cp:0.5@4 ; v2=hc:-0.2,T0:1@2"
//                                      weight vectors over parameters, all
//                                      orders 1..N along each (default N=1)
//
// Every distinct parameter name in -deriv and every -dir vector is one
// Taylor basis column t_j.  Physical parameter p enters the model as
//   S_p = base_p * (1 + sum_j W[p][j] t_j)      (relative scale s_p)
// so d/dt_j of a named column is the derivative w.r.t. the relative scale of
// that parameter, and a direction column gives derivatives along that vector.
#pragma once

#include "anyorder_eval.hpp"
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace aoq {

struct Query {
    int Mrun = 0, Nrun = 0;
    std::vector<std::string> col_name;              // basis column names
    std::vector<double> W;                          // NP x Mrun, row-major
    std::vector<std::vector<int>> alphas;           // requested multi-indices (length Mrun)
    std::vector<std::string> alpha_name;            // output column base names
    std::string error;
    bool ok() const { return error.empty(); }
    bool empty() const { return alphas.empty(); }
};

inline std::string q_trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}
inline std::vector<std::string> q_split(const std::string &s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep)) out.push_back(q_trim(cur));
    return out;
}
inline int q_param_index(const std::string &n) {
    for (int p = 0; p < NP; p++) if (n == PARAM_NAMES[p]) return p;
    return -1;
}
inline std::string q_param_list() {
    std::string s;
    for (int p = 0; p < NP; p++) s += std::string(p ? " " : "") + PARAM_NAMES[p];
    return s;
}

inline Query parse_query(const std::string &deriv, const std::string &dir) {
    Query q;
    struct Col { std::string name; bool is_dir; std::vector<std::pair<int, double>> w; };
    std::vector<Col> cols;
    struct Term { std::vector<std::pair<int, int>> fac; std::string name; };   // (col, power)
    std::vector<Term> terms;
    struct DirReq { int col; int maxo; std::string name; };
    std::vector<DirReq> dreqs;

    auto col_of_param = [&](int p) {
        for (size_t j = 0; j < cols.size(); j++)
            if (!cols[j].is_dir && cols[j].w.size() == 1 && cols[j].w[0].first == p) return (int)j;
        cols.push_back({PARAM_NAMES[p], false, {{p, 1.0}}});
        return (int)cols.size() - 1;
    };

    if (!q_trim(deriv).empty()) {
        for (const std::string &tstr : q_split(deriv, ';')) {
            if (tstr.empty()) continue;
            Term t;
            std::vector<std::pair<int, int>> pw;   // (param, power) in order of appearance
            for (const std::string &f : q_split(tstr, '*')) {
                std::string nm = f; int power = 1;
                size_t hat = f.find('^');
                if (hat != std::string::npos) {
                    nm = q_trim(f.substr(0, hat));
                    power = std::atoi(f.substr(hat + 1).c_str());
                    if (power < 1) { q.error = "bad power in '" + f + "'"; return q; }
                }
                int p = q_param_index(nm);
                if (p < 0) {
                    q.error = "unknown parameter '" + nm + "' in -deriv (valid: " + q_param_list() +
                              "; Rx/Ry/Rz/X0/Vx/TL are only available in the order-2 driver)";
                    return q;
                }
                bool found = false;
                for (auto &e : pw) if (e.first == p) { e.second += power; found = true; }
                if (!found) pw.push_back({p, power});
            }
            for (auto &e : pw) {
                int c = col_of_param(e.first);
                t.fac.push_back({c, e.second});
                t.name += std::string("_d") + PARAM_NAMES[e.first] + (e.second > 1 ? std::to_string(e.second) : "");
            }
            terms.push_back(t);
        }
    }

    if (!q_trim(dir).empty()) {
        for (const std::string &dstr : q_split(dir, ';')) {
            if (dstr.empty()) continue;
            size_t eq = dstr.find('=');
            if (eq == std::string::npos) { q.error = "-dir item '" + dstr + "' needs name=k:w,..."; return q; }
            std::string vname = q_trim(dstr.substr(0, eq));
            std::string rest = dstr.substr(eq + 1);
            int maxo = 1;
            size_t at = rest.find('@');
            if (at != std::string::npos) { maxo = std::atoi(rest.substr(at + 1).c_str()); rest = rest.substr(0, at); }
            if (maxo < 1) { q.error = "bad @order in '" + dstr + "'"; return q; }
            Col c; c.name = vname; c.is_dir = true;
            for (const std::string &pw : q_split(rest, ',')) {
                if (pw.empty()) continue;
                size_t colon = pw.find(':');
                std::string nm = q_trim(colon == std::string::npos ? pw : pw.substr(0, colon));
                double w = (colon == std::string::npos) ? 1.0 : std::atof(pw.substr(colon + 1).c_str());
                int p = q_param_index(nm);
                if (p < 0) { q.error = "unknown parameter '" + nm + "' in -dir (valid: " + q_param_list() + ")"; return q; }
                c.w.push_back({p, w});
            }
            if (c.w.empty()) { q.error = "-dir '" + vname + "' has no weights"; return q; }
            cols.push_back(c);
            dreqs.push_back({(int)cols.size() - 1, maxo, vname});
        }
    }

    q.Mrun = (int)cols.size();
    if (q.Mrun == 0) return q;   // no query: nominal only
    q.W.assign((size_t)NP * q.Mrun, 0.0);
    for (int j = 0; j < q.Mrun; j++) {
        q.col_name.push_back(cols[j].name);
        for (auto &pw : cols[j].w) q.W[(size_t)pw.first * q.Mrun + j] += pw.second;
    }
    for (auto &t : terms) {
        std::vector<int> a(q.Mrun, 0);
        int ord = 0;
        for (auto &f : t.fac) { a[f.first] = f.second; ord += f.second; }
        q.alphas.push_back(a);
        q.alpha_name.push_back("d" + std::to_string(ord) + "T" + t.name);
        q.Nrun = std::max(q.Nrun, ord);
    }
    for (auto &d : dreqs)
        for (int n = 1; n <= d.maxo; n++) {
            std::vector<int> a(q.Mrun, 0); a[d.col] = n;
            q.alphas.push_back(a);
            q.alpha_name.push_back("d" + std::to_string(n) + "T_d" + d.name);
            q.Nrun = std::max(q.Nrun, n);
        }
    return q;
}

}  // namespace aoq
