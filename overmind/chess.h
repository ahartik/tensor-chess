#ifndef _OVERMIND_CHESS_H_
#define _OVERMIND_CHESS_H_

#include <memory>
#include <vector>

#include "overmind/board.pb.h"

namespace overmind {

enum class Piece {
  kPawn = 0,
  kKnight,
  kBishop,
  kRook,
  kQueen,
  kKing,
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

class ChessBoard : std::enable_shared_from_this<ChessBoard> {
 public:
  ChessBoard();
  explicit ChessBoard(const std::string& fen);

  void legal_moves(std::vector<Move>* out) const;
  std::shared_ptr<ChessBoard> ApplyMove(const Move& m) const;

  void ToProto(Board* out) const;

  std::string DebugString() const;

 private:
  uint64_t pieces_[12] = {};
  uint64_t castling_rights_ = {};
  uint64_t en_passant_ = {};

  int half_move_count_ = 0;
  int no_progress_count_ = 0;
  int repetition_count_ = 0;

  std::shared_ptr<ChessBoard> parent_;
};

}  // namespace overmind

#endif  // _OVERMIND_CHESS_H_
