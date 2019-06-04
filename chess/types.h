// Place to put simple types that don't warrant their own file.
#ifndef _CHESS_TYPES_H_
#define _CHESS_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "chess/game.pb.h"

namespace chess {

enum class Color : uint8_t {
  kWhite = 0,
  kBlack = 1,
  kEmpty = 2,
};

inline Color OtherColor(Color c);

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

std::ostream& operator<<(std::ostream& out, Piece p);

extern const Piece kPromoPieces[4];

struct PieceColor {
  Piece p;
  Color c;
};

inline char PieceChar(Piece p, Color c = Color::kWhite) {
  const char kLetters[3 * 7 + 1] =
      "PNBRQK."
      "pnbrqk."
      ".......";
  return kLetters[int(p) + 7 * int(c)];
}

inline char PieceColorChar(PieceColor pc) { return PieceChar(pc.p, pc.c); }

struct Move {
  enum class Type : uint8_t {
    kUnknown = 0,
    // Non-capture moves of (non-pawn) pieces.
    kReversible = 1,
    // Captures and all pawn moves.
    kRegular = 2,
    kPromotion = 3,
    kCastling = 4,
    kEnPassant = 5,
  };
  Move() = default;
  Move(int f, int t, Piece promo) : from(f), to(t), promotion(promo) {
    if (promo != Piece::kNone) {
      type = Type::kPromotion;
    }
  }

  Move(int f, int t, Type ty) : from(f), to(t), type(ty) {}

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

  std::string ToString() const;

  bool operator==(const Move& o) const {
    return from == o.from && to == o.to && promotion == o.promotion;
  }

  bool operator<(const Move& o) const {
    if (from != o.from) {
      return from < o.from;
    }
    if (to != o.to) {
      return to < o.to;
    }
    return promotion < o.promotion;
  }

  int8_t from = 0;
  int8_t to = 0;
  Piece promotion = Piece::kNone;
  // Flags follow:
  Type type = Type::kUnknown;
};
static_assert(sizeof(Move) == 4);

inline std::ostream& operator<<(std::ostream& out, Move m) {
  return out << m.ToString();
}

template <typename H>
H AbslHashValue(H h, const Move& m) {
  return H::combine(std::move(h), m.from, m.to, m.promotion);
}

using MoveList = std::vector<Move>;

// Output of either a neural network prediction, or MCTS evaluation.
struct PredictionResult {
  std::vector<std::pair<Move, double>> policy;
  double value = 0.0;
};

// Implementations of inline methods.

Color OtherColor(Color c) {
  assert(c < Color::kEmpty);
  return Color(int(c) ^ 1);
}

}  // namespace chess

#endif
