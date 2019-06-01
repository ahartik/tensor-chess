#ifndef _CHESS_BOARD_H_
#define _CHESS_BOARD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "chess/game.pb.h"
#include "util/int-set.h"

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

inline char PieceColorChar(PieceColor pc) {
  return PieceChar(pc.p, pc.c);
}

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

  Move(int f, int t, Type ty ) : from(f), to(t), type(ty) {}

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
    // Threefold repetition (possibly returning earlier).
    kRepetitionDraw,
    // 50-move rule or so.
    kNoProgressDraw,
  };

  // For F callable with signature void(const Move& m);
  template <typename MoveFunc>
  State LegalMoves(const MoveFunc& f) const;

  // Hash value for the board, to be used for detecting repetitions.
  uint64_t board_hash() const;
  // Hash value of the state, includes history.
  uint64_t state_hash() const;

  bool operator==(const Board& b) const;

  std::string ToPrintString() const;
  std::string ToFEN() const;

  PieceColor square(int sq) const;
  PieceColor square(int r, int f) const {
    return square(r * 8 + f);
  }

  Move::Type GetMoveType(const Move& m) const;

 private:
  template <typename H>
  friend H AbslHashValue(H h, const Board& b);

  template <typename MoveFunc>
  friend class MoveGenerator;

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
  return H::combine(std::move(h), b.state_hash());
}

void InitializeMovegen();

// Squares
class Square {
 public:
#define SQ(rank, rank_num)                         \
  static constexpr int rank##1 = 0 * 8 + rank_num; \
  static constexpr int rank##2 = 1 * 8 + rank_num; \
  static constexpr int rank##3 = 2 * 8 + rank_num; \
  static constexpr int rank##4 = 3 * 8 + rank_num; \
  static constexpr int rank##5 = 4 * 8 + rank_num; \
  static constexpr int rank##6 = 5 * 8 + rank_num; \
  static constexpr int rank##7 = 6 * 8 + rank_num; \
  static constexpr int rank##8 = 7 * 8 + rank_num

  SQ(A, 0);
  SQ(B, 1);
  SQ(C, 2);
  SQ(D, 3);
  SQ(E, 4);
  SQ(F, 5);
  SQ(G, 6);
  SQ(H, 7);
#undef SQ
  static int Rank(int sq) {
    return sq / 8;
  }
  static int File(int sq) {
    return sq % 8;
  }

  static std::string ToString(int sq);
};
static_assert(Square::A1 == 0);
static_assert(Square::H8 == 63);

}  // namespace chess

#include "board-inl.cpp"

#endif
