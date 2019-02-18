#ifndef _C4CC_NEGAMAX_H_
#define _C4CC_NEGAMAX_H_

#include "c4cc/board.h"

namespace c4cc {

struct NegamaxResult {
  int32_t eval;
  int best_move = 0;
};

int32_t StaticEval(const Board& b);

NegamaxResult Negamax(const Board& b);

}  // namespace c4cc

#endif
