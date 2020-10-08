#include "chess/types.h"

#include <cassert>
#include <cctype>

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

// static
absl::optional<Move> Move::FromString(absl::string_view str) {
  const absl::string_view orig = str;
  if (str.size() != 4 && str.size() != 5) {
    std::cerr << "Invalid move string '" << orig << "'\n";
    return absl::nullopt;
  }
  Move m;
  bool found_from = false, found_to = false;
  for (int i = 0; i < 64; ++i) {
    if (str.substr(0, 2) == Square::ToString(i)) {
      m.from = i;
      found_from = true;
      break;
    }
  }
  if (!found_from) {
    std::cerr << "Invalid move string '" << orig << "'\n";
    return absl::nullopt;
  }
  str.remove_prefix(2);
  for (int i = 0; i < 64; ++i) {
    if (str.substr(0, 2) == Square::ToString(i)) {
      m.to = i;
      found_to = true;
      break;
    }
  }
  if (!found_to) {
    std::cerr << "Invalid move string '" << orig << "'\n";
    return absl::nullopt;
  }
  str.remove_prefix(2);
  if (str.empty()) {
    return m;
  }
  assert(str.size() == 1);
  for (const Piece p : kPromoPieces) {
    if (std::toupper(str[0]) == PieceChar(p, Color::kWhite)) {
      m.promotion = p;
      return m;
    }
  }
  std::cerr << "Invalid promotion '" << str << "'\n";
  return absl::nullopt;
}

std::ostream& operator<<(std::ostream& out, Piece p) {
  return out << PieceChar(p);
}

}  // namespace chess
