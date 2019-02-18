#include "c4cc/negamax.h"

#include <iostream>

namespace c4cc {

namespace {
const int32_t kWinScore = 1e9;
const int32_t kCloseBonus[4] = {0, 1, 10, 100};

}  // namespace

// How good is this board for the person who made last move.
int32_t StaticEval(const Board& b, Color me) {
  // std::cout << "Eval " << b;
  if (b.is_over()) {
    if (b.result() == Color::kEmpty) {
      return 0;
    }
    if (b.result() == me) {
      return kWinScore;
    } else {
      return -kWinScore;
    }
  }
  int32_t e = 0;
  for (int dir = 0; dir < Board::kNumDirs; ++dir) {
    for (const auto start : Board::start_pos_list(dir)) {
      int x = start.first;
      int y = start.second;
      int count = 0;
      Color current = Color::kEmpty;
      // Go go power rangers!
      while (x < 7 && y >= 0 && y < 6) {
        if (b.color(x, y) == current) {
          ++count;
        } else {
          // Give points.
          if (current != Color::kEmpty) {
            assert(count < 4);
            e += ((current == me) ? 1 : -1) * kCloseBonus[count];
          }
          current = b.color(x, y);
          count = 1;
        }
        x += Board::dx(dir);
        y += Board::dy(dir);
      }
      if (current != Color::kEmpty) {
        e += ((current == me) ? 1 : -1) * kCloseBonus[count];
      }
    }
  }
  return e;
}

namespace {

NegamaxResult NRec(Board& b, int depth) {
  if (b.is_over() || depth == 0) {
    int32_t eval = StaticEval(b, b.turn());
    if (b.is_over()) {
      // Try to make wins closer, and loses later.
      if (eval < 0) {
        eval -= depth;
      } else {
        eval += depth;
      }
    }
    return NegamaxResult{eval, -10000};
  }
  int32_t best = -kWinScore * 2;
  int best_move = -20000;
  // std::cout << "num moves: " << b.valid_moves().size() << "\n";
  const MoveList moves = b.valid_moves();
  for (int m : moves) {
    b.MakeMove(m);
    const auto r = NRec(b, depth - 1);
    const int32_t eval_for_me = -r.eval;
    if (eval_for_me > best) {
      best_move = m;
      best = eval_for_me;
    }
    //     std::cout << "depth=" << depth << " score for " << m << ": " <<
    //     eval_for_me
    //               << "\n";
    b.UndoMove(m);
  }
  return NegamaxResult{best, best_move};
}

}  // namespace

NegamaxResult Negamax(const Board& b, int depth) {
  Board copy = b;
  return NRec(copy, depth);
}

}  // namespace c4cc
