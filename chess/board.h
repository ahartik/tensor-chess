#ifndef _CHESS_BOARD_H_
#define _CHESS_BOARD_H_

#include <cstdint>
#include <string>
#include <ostream>
#include <vector>

#include "absl/strings/string_view.h"
#include "chess/game.pb.h"
#include "chess/hashing.h"
#include "chess/square.h"
#include "chess/types.h"
#include "util/int-set.h"

namespace chess {

// Thoughts:
// Pseudo moves and draw
//
// Good blog post:
// https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
//
// Use cases and "correctness":
// 1. Actual game: always have full history.
// 2. Actual game training: always have full history.
// 3. Puzzle cases (testing or training): No full history, or last move.
//
class Board {
 public:
  // Initializes the board at the starting position.
  Board();

  explicit Board(const BoardProto& p);

  // crashes on failure
  explicit Board(absl::string_view fen);

  // Construct a board from given existing board + a move.
  Board(const Board& o, const Move& m);

  Color turn() const {
    return (half_move_count_ % 2) == 0 ? Color::kWhite : Color::kBlack;
  }

  // Hash value for the board, to be used for detecting repetitions.
  uint64_t board_hash() const { return board_hash_; }

  uint64_t bitboard(Color c, Piece p) const {
    return bitboards_[int(c)][int(p)];
  }

  uint64_t en_passant() const { return en_passant_; }

  uint64_t castling_rights() const { return castling_rights_; }

  bool operator==(const Board& o) const;

  std::string ToPrintString() const;
  std::string ToFEN() const;

  PieceColor square(int sq) const;
  PieceColor square(int r, int f) const { return square(r * 8 + f); }

  Move::Type GetMoveType(const Move& m) const;

  // Convenience function for getting valid moves,
  MoveList valid_moves() const;

  uint64_t ComputeOcc() const {
    uint64_t o = 0;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < kNumPieces; ++j) {
        o |= bitboards_[i][j];
      }
    }
    return o;
  }

  // Initialize movegen and hashing.
  static void Init();

  // "half-move clock", for purposes of 50-move rule. Draw occurs at 100.
  int no_progress_count() const { return no_progress_count_; }

 private:
  template <typename H>
  friend H AbslHashValue(H h, const Board& b);

  uint64_t ComputeBoardHash() const;

  uint64_t bitboards_[2][kNumPieces] = {};
  // Squares where en-passant capture is possible for the current player.
  uint64_t en_passant_ = 0;
  uint64_t castling_rights_ = 0;
  int16_t no_progress_count_ = 0;
  int32_t half_move_count_ = 0;
  uint64_t board_hash_ = 0;

  // TODO: Maybe change bitboards to the following:
  // * side
  // * pawns
  // * knights
  // * bishop (including queens)
  // * rooks (including queens)
  // * kings
};
static_assert(sizeof(Board) == 16 * 8);

template <typename H>
H AbslHashValue(H h, const Board& b) {
  return H::combine(std::move(h), b.board_hash());
}

std::ostream& operator<<(std::ostream& o, const Board& b) {
  return o << b.ToFEN();
}

}  // namespace chess

#endif
