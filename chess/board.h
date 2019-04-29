#ifndef _CHESS_BOARD_H_
#define _CHESS_BOARD_H_

#include <cstdint>
#include <vector>

#include "util/int-set.h"

namespace chess {

enum class Color : uint8_t {
  kWhite = 0,
  kBlack = 1,
  kEmpty = 2,
};

enum class Piece : uint8_t {
  kPawn = 0,
  kKnight = 1,
  kBishop = 2,
  kRook = 3,
  kQueen = 4,
  kKing = 5,
};
constexpr int kNumPieces = 6;

struct Move {
  int8_t from;
  int8_t to;
  Piece promotion = Piece::kQueen;
};

using MoveList = std::vector<Move>;

// Thoughts:
// Pseudo moves and draw
//
// Good blog post:
// https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
//
//
class Board {
 public:
  // Initializes the board at the starting position.
  Board();
  bool is_over() const;
  // If the game is over, this returns the winner of the game, or Color::kEmpty
  // in case of a draw.
  Color winner() const;

  Color turn() const {
    return (half_move_count_ % 2) == 0 ? Color::kWhite : Color::kBlack;
  }

  MoveList valid_moves() const;

  Board MakeMove(m

  bool operator==(const Board& b) const;

 private:
  template <typename H>
  friend H AbslHashValue(H h, const Board& b);

  uint64_t bitboards_[2][6];
  // Squares where en-passant capture is possible for the current player.
  uint64_t en_passant_;
  int16_t repetition_count_ = 0;
  int half_move_count_ = 0;
  int no_progress_count_ = 0;
  SmallIntSet history_;
};

template <typename H>
H AbslHashValue(H h, const Board& b) {
  // XXX: Implement
  return h;
}

void InitializeMovegen();

}  // namespace chess

#endif
