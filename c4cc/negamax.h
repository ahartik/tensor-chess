#ifndef _C4CC_NEGAMAX_H_
#define _C4CC_NEGAMAX_H_

#include "c4cc/board.h"

namespace c4cc {


int32_t StaticEval(const Board& b);

// Returns eval and best move for the current player.
struct NegamaxResult {
  int32_t eval;
  int best_move;
};
NegamaxResult Negamax(const Board& b, int depth);

}  // namespace c4cc

#endif
