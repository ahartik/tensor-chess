#include "chess/types.h"

#include "absl/strings/str_cat.h"
#include "chess/square.h"

namespace chess {

extern const Piece kPromoPieces[4] = {Piece::kQueen, Piece::kBishop,
                                      Piece::kKnight, Piece::kRook};
std::string Move::ToString() const {
  char p_str[2] = {0, 0};
  if (promotion != Piece::kNone) {
    p_str[0] = PieceChar(promotion, Color::kBlack);
  }
  return absl::StrCat(Square::ToString(from), Square::ToString(to), p_str);
}

}  // namespace chess
