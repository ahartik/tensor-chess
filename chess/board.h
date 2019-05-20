#ifndef _CHESS_BOARD_H_
#define _CHESS_BOARD_H_

#include <cstdint>
#include <vector>

#include "util/int-set.h"
#include "chess/game.pb.h"

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
  kNone = 6,
};
constexpr int kNumPieces = 6;

struct Move {
  Move() = default;
  MoveProto ToProto() const {
    MoveProto p;
    p.set_from_square(from);
    p.set_to_square(to);
    if (promotion != Piece::kNone) {
      p.set_promotion(static_cast<int>(promotion));
    }
    return p;
  }

  static Move FromProto(const MoveProto& p) {
    Move m;
    m.from = p.from_square();
    m.to = p.to_square();
    if (p.has_promotion()) {
      m.promotion = static_cast<Piece>(p.promotion());
    }
    return m;
  }

  int8_t from = 0;
  int8_t to = 0;
  Piece promotion = Piece::kNone;

  bool operator==(const Move& o) const {
    return from == o.from && to == o.to && promotion == o.promotion;
  }
};

template <typename H>
H AbslHashValue(H h, const Move& m) {
  return H::combine(std::move(h), m.from, m.to, m.promotion);
}

using MoveList = std::vector<Move>;

// Thoughts:
// Pseudo moves and draw
//
// Good blog post:
// https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
//
// Implementation plan:
// 1. Create million test cases by using python-chess and downloaded games
// 2. Implement basic moves.
// 3. Keep iterating until my code agrees with python-chess in 100% of cases.
//
// Use cases and "correctness":
// 1. Actual game: always have full history.
// 2. Actual game training: always have full history.
// 3. Puzzle cases (testing or training): No full history, or last move.
//
// 
class Board {
 public:
  // Initializes the board at the starting position.
  Board();

  // TODO: Add constructor for puzzle cases.
  // static Board MakeTestCase();

  bool is_over() const;
  // If the game is over, this returns the winner of the game, or Color::kEmpty
  // in case of a draw.
  Color winner() const;

  Color turn() const {
    return (half_move_count_ % 2) == 0 ? Color::kWhite : Color::kBlack;
  }

  MoveList valid_moves() const;

  // Hash value for the board, to be used for detecting repetitions.
  uint64_t board_hash() const;
  // Hash value of the state, includes history.
  uint64_t state_hash() const;

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
  uint64_t history_hash_ = 0;
};

template <typename H>
H AbslHashValue(H h, const Board& b) {
  return H::combine(std::move(h), b.state_hash());
}

void InitializeMovegen();

}  // namespace chess

#endif
