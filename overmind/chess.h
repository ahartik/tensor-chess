#ifndef _OVERMIND_CHESS_H_
#define _OVERMIND_CHESS_H_

#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "overmind/board.pb.h"
#include "overmind/persistent-int-map.h"

namespace overmind {

enum class Piece {
  kPawn = 0,
  kKnight,
  kBishop,
  kRook,
  kQueen,
  kKing,
  kEmpty,  // Special sentinel value, not stored.
};

enum class Color {
  kWhite,
  kBlack,
};

struct Move {
  int from_square = 0;
  int to_square = 0;
  Piece promotion = Piece::kQueen;
};

bool IsValidFen(absl::string_view fen);

class ChessBoard {
 public:

  ChessBoard();
  explicit ChessBoard(absl::string_view fen);

  void GetLegalMoves(std::vector<Move>* out) const;
  ChessBoard ApplyMove(const Move& m) const;

  void ToProto(Board* out) const;

  std::string DebugString() const;

 private:
  // Helper for constructor.
  void SetPiece(int square, Piece p, Color color);

  uint64_t pieces_[12] = {};
  uint64_t castling_rights_ = {};
  uint64_t en_passant_ = {};

  int half_move_count_ = 0;
  int no_progress_count_ = 0;
  int repetition_count_ = 0;

  uint64_t hash() const;

  PersistentHashMap<uint64_t, int> state_counts_;
};

}  // namespace overmind

#endif  // _OVERMIND_CHESS_H_
