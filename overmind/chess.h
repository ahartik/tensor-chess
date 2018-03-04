#ifndef _OVERMIND_CHESS_H_
#define _OVERMIND_CHESS_H_

#include <cstdint>
#include <memory>
#include <string>
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
  string ToUCI();
};

// Returns algebraic notation, e.g. "e5"
absl::string_view SquareToString(int square);

int StringToSquare(absl::string_view uci);

bool IsValidFen(absl::string_view fen);

class ChessBoard {
 public:
  ChessBoard();
  explicit ChessBoard(absl::string_view fen);

  void GetLegalMoves(std::vector<Move>* out) const;
  ChessBoard ApplyMove(const Move& m) const;

  void ToProto(Board* out) const;

  Color turn() const {return turn_};
  std::string DebugString() const;

 private:
  // Helper for constructor.
  void SetPiece(int square, Piece p, Color color);

  uint64_t hash() const;

  uint64_t pieces_[12] = {};
  uint64_t castling_rights_ = 0;
  uint64_t en_passant_ = 0;

  Color turn_;
  int halfmove_count_ = 0;
  int no_progress_count_ = 0;
  int repetition_count_ = 0;

  PersistentIntMap<uint64_t, int> state_counts_;
};

}  // namespace overmind

#endif  // _OVERMIND_CHESS_H_
