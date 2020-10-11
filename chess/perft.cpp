#include <cstdlib>

#include "chess/board.h"
#include "chess/movegen.h"

namespace chess {

// Undef this to get easier-to-read profile output.
#define OPTIMIZED

int64_t Perft(const Board& b, int d) {
  if (d <= 0) {
    return 1;
  }

  int64_t nodes = 0;
#ifdef OPTIMIZED
  if (d == 1) {
    IterateLegalMoves(b, [&](const Move& m) { ++nodes; });
  } else {
    IterateLegalMoves(
        b, [&](const Move& m) { nodes += Perft(Board(b, m), d - 1); });
  }
#else
  const auto moves = b.valid_moves();
  if (d == 1) {
    nodes += moves.size();
  } else {
    for (const Move& m : moves) {
      nodes += Perft(b, m, d - 1);
    }
  }
#endif
  return nodes;
}

}  // namespace chess
