
#include "c4cc/perfect_negamax.h"

namespace c4cc {

namespace {

int NRec(Board& b, int depth, int32_t alpha, int32_t beta) {
  if (b.is_over()) {
    const Color r = b.result();
    if (r == Color::kEmpty) {
      return 0;
    }
    if (b.result() == b.turn()) {
      return 42 - b.ply();
    } else {
      return -42 + b.ply();
    }
  }
  if (depth == 0) {
    return kNoPerfectResult;
  }
  int32_t best = -42;
  // std::cout << "num moves: " << b.valid_moves().size() << "\n";
  const MoveList moves = b.valid_moves();
  for (int m : moves) {
    b.MakeMove(m);
    const int r = NRec(b, depth - 1, -beta, -alpha);
    if (r == kNoPerfectResult) {
      // Bail out, not enough depth to get a perfect result.
      return kNoPerfectResult;
    }
    const int32_t eval_for_me = -r;
    //     std::cout << "depth=" << depth << " score for " << m << ": " <<
    //     eval_for_me
    //               << "\n";
    b.UndoMove(m);
    if (eval_for_me > best) {
      best = eval_for_me;
    }
    alpha = std::max(alpha, best);
    if (alpha >= beta) {
      break;
    }
  }
  return best;
}

}  // namespace

int PerfectEval(Board b, int max_depth, PerfectCache* cache) {
  return NRec(b, max_depth, -50, 50);
}

}  // namespace c4cc
