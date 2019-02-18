#include "c4cc/negamax.h"

namespace c4cc {

namespace {
const int32_t kWinScore = 1e9;
const int32_t kCloseBonus[4] = {0, 1, 10, 100};

}  // namespace

// How good is this board for the person who made last move.
int32_t StaticEval(const Board& b, Color me) {
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
          assert(count < 4);
          // Give points.
          if (current != Color::kEmpty) {
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

}  // namespace c4cc
