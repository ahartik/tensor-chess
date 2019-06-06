#include "chess/types.h"

#include "absl/strings/str_cat.h"
#include "chess/square.h"

namespace chess {

std::ostream& operator<<(std::ostream& out, Color c) {
  switch (c) {
    case Color::kWhite:
      return out << "white";
    case Color::kBlack:
      return out << "black";
    case Color::kEmpty:
      return out << "-";
  }
  return out;
}

extern const Piece kPromoPieces[4] = {Piece::kQueen, Piece::kBishop,
                                      Piece::kKnight, Piece::kRook};
std::string Move::ToString() const {
  char p_str[2] = {0, 0};
  if (promotion != Piece::kNone) {
    p_str[0] = PieceChar(promotion, Color::kBlack);
  }
  return absl::StrCat(Square::ToString(from), Square::ToString(to), p_str);
}

std::ostream& operator<<(std::ostream& out, Piece p) {
  return out << PieceChar(p);
}

}  // namespace chess
