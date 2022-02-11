#ifndef _C4CC_PERFECT_NEGAMAX_H_
#define _C4CC_PERFECT_NEGAMAX_H_

#include "c4cc/board.h"

namespace c4cc {

class PerfectCache {
 public:
  PerfectCache() {}

 private:
  // XXX
  struct HashElem {};
};

inline constexpr int kNoPerfectResult = -100;

// Returns 0 if `b` is a draw, 1 if the current player wins with their last
// piece, -1 if the current player loses on last turn, and so on.
//
// Returns kNoPerfectResult if no conclusive result is reached in `max_depth`.
int PerfectEval(Board b, int max_depth, PerfectCache* cache);

}  // namespace c4cc

#endif
