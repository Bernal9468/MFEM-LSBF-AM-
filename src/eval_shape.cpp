// One (M,N) shape of the residual evaluator; compile with -DJET_M=<M> -DJET_N=<N>.
#include "anyorder_eval.hpp"

#ifndef JET_M
#error "define JET_M and JET_N"
#endif
#define AOQ_CAT_(a, b, c, d) a##b##c##d
#define AOQ_CAT(a, b, c, d) AOQ_CAT_(a, b, c, d)

namespace aoq {
std::unique_ptr<AnyOrderEval> AOQ_CAT(make_eval_, JET_M, _, JET_N)() {
    return std::unique_ptr<AnyOrderEval>(new AnyOrderEvalImpl<JET_M, JET_N>());
}
}  // namespace aoq
