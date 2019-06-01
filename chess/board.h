#ifndef _CHESS_BOARD_H_
#define _CHESS_BOARD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "chess/game.pb.h"
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

  // For the most efficient interface, implement function like:
  enum class State {
    kNotOver = 0,
    // Check and the current player has no legal moves.
    kCheckmate,
    // Not a check, but current player has no moves.
    kStalemate,
    // Threefold repetition (possibly returning earlier). Moves were still
    // returned.
    kRepetitionDraw,
    // 50-move rule or so. Moves were still returned.
    kNoProgressDraw,
  };

  // For F callable with signature void(const Move& m);
  template <typename MoveFunc>
  State LegalMoves(const MoveFunc& f, bool return_draw_moves = true ) const;

  // Hash value for the board, to be used for detecting repetitions.
  uint64_t board_hash() const {
    return board_hash_;
  }
  // Hash value of the state, includes history.
  uint64_t state_hash() const {
    return 0;
  }

  bool operator==(const Board& b) const;

  std::string ToPrintString() const;
  std::string ToFEN() const;

  PieceColor square(int sq) const;
  PieceColor square(int r, int f) const { return square(r * 8 + f); }

  Move::Type GetMoveType(const Move& m) const;

  // Initialize movegen and hashing.
  static void Init();

 private:
  template <typename H>
  friend H AbslHashValue(H h, const Board& b);

  template <typename MoveFunc>
  friend class MoveGenerator;

  uint64_t ComputeBoardHash() const;

  uint64_t ComputeOcc() const {
    uint64_t o = 0;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < kNumPieces; ++j) {
        o |= bitboards_[i][j];
      }
    }
    return o;
  }

  uint64_t bitboards_[2][kNumPieces] = {};
  // Squares where en-passant capture is possible for the current player.
  uint64_t en_passant_ = 0;
  uint64_t castling_rights_ = 0;
  int16_t repetition_count_ = 0;
  int16_t no_progress_count_ = 0;
  int32_t half_move_count_ = 0;
  uint64_t board_hash_ = 0;

  // SmallIntSet history_;
  // uint64_t history_hash_ = 0;
  //
  // TODO: Maybe change bitboards to the following:
  // * side
  // * pawns
  // * knights
  // * bishop
  // * rooks
  // * kings
};

template <typename H>
H AbslHashValue(H h, const Board& b) {
  return H::combine(std::move(h), b.state_hash(), b.board_hash());
}

void InitializeMovegen();

}  // namespace chess

#include "board-inl.cpp"

#endif
