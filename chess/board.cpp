#include "chess/board.h"

#include <cassert>
#include <iostream>

#include "absl/strings/str_cat.h"
#include "chess/bitboard.h"
#include "chess/magic.h"

namespace chess {

namespace {

// https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/

struct MoveInput {
  Color turn = Color::kEmpty;
  // Position of the piece we're generating moves for.
  int square = 0;
  // Position of my king.
  uint64_t king_s = 0;
  // Square occupation.
  uint64_t occ = 0;
  // Masks of my and opponent pieces.
  uint64_t my_pieces = 0;
  uint64_t opp_pieces = 0;
  // Which opponent pawns are in en passant.
  uint64_t en_passant = 0;
  // Mask of positions we can't move our king to. Only used for king moves.
  uint64_t king_danger = 0;

  // If we're in single check, these are
  // TODO: Document these, from
  // https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
  uint64_t check_ok = kAllBits;
};

// TODO: Profile and optimize this later.

// For pinned pieces, we can only move them towards or away from the king.
bool SameDirection(int king, int from, int to) {
  uint64_t pt = PushMask(king, to) | OneHot(to);
  uint64_t pf = PushMask(king, from) | OneHot(from);
  return (pt & pf) != 0;
}

}  // namespace

class MoveGenerator {
 public:
  explicit MoveGenerator(const Board& b)
      : b_(b), turn_(b.turn()), ti_(int(turn_)) {
    for (int i = 0; i < 2; ++i) {
      for (int p = 0; p < kNumPieces; ++p) {
        if (i == ti_) {
          my_pieces_ |= b.bitboards_[i][p];
        } else {
          opp_pieces_ |= b.bitboards_[i][p];
        }
        occ_ |= b.bitboards_[i][p];
      }
    }
    king_s_ = GetFirstBit(b_.bitboards_[ti_][5]);

    king_danger_ = ComputeKingDanger();

    in_check_ = ComputeCheck(&check_ok_, &push_mask_);
    check_ok_ |= push_mask_;

    soft_pinned_ = ComputePinnedPieces();
  }

  void PawnMoves(int sq, MoveList* out) const {
    const int dr = turn_ == Color::kWhite ? 8 : -8;
    const int rank = Square::Rank(sq);
    const int file = Square::File(sq);
    assert(rank != 0);
    assert(rank != 7);
    const uint64_t capture_ok = check_ok_ | b_.en_passant_;

    const auto try_forward = [&](int to) {
      if ((occ_ & OneHot(to)) == 0 && (OneHot(to) & check_ok_)) {
        if (IsPinned(sq, to)) {
          // It's possible for a pawn to be soft-pinned, but not hard-pinned:
          // when we're going straight towards an opponent rook.
          return;
        }
        const int to_rank = Square::Rank(to);
        if (to_rank == 0 || to_rank == 7) {
          for (Piece p : kPromoPieces) {
            out->emplace_back(sq, to, p);
          }
        } else {
          out->emplace_back(sq, to);
        }
      }
    };

    const auto try_capture = [&](int to) {
      if (((opp_pieces_ | b_.en_passant_) & OneHot(to) & capture_ok) != 0) {
        if (IsPinned(sq, to)) {
          return;
        }
        if (b_.en_passant_ & OneHot(to)) {
          // This is en passant capture. We need to check if this leaves us in
          // check (which would be illegal).
          if (SquareRank(king_s_) == SquareRank(sq)) {
            // Captured piece and our king are on the same rank. Check if they
            // have other pieces in-between than 'sq' and the pawn previously
            // next to us.
            const uint64_t other_pawn = (turn_ == Color::kWhite)
                                            ? b_.en_passant_ >> 8
                                            : b_.en_passant_ << 8;
            const uint64_t removed_sqs = OneHot(sq) | other_pawn;
            const uint64_t king_rook_moves =
                RookMoveMask(king_s_, (occ_ ^ removed_sqs)) &
                RankMask(SquareRank(king_s_));
#if 0
            std::cout << "Other pawn:\n"
                      << BitboardToString(other_pawn) << "\n";
            std::cout << "Removed sqs:\n"
                      << BitboardToString(removed_sqs) << "\n";
            std::cout << "King rook moves:\n"
                      << BitboardToString(king_rook_moves) << "\n";
#endif
            // Check if opponent rooks match this.
            const uint64_t opp_rooks =
                b_.bitboards_[1 - ti_][3] | b_.bitboards_[1 - ti_][4];
            if (king_rook_moves & opp_rooks) {
              return;
            }
          }
        }
        const int to_rank = Square::Rank(to);
        if (to_rank == 0 || to_rank == 7) {
          for (Piece p : kPromoPieces) {
            out->emplace_back(sq, to, p);
          }
        } else {
          out->emplace_back(sq, to);
        }
      }
    };
    // Single forward move:
    try_forward(sq + dr);
    if (((rank == 1 && turn_ == Color::kWhite) ||
         (rank == 6 && turn_ == Color::kBlack)) &&
        (OneHot(sq + dr) & occ_) == 0) {
      try_forward(sq + 2 * dr);
    }

    // Capture left.
    if (file != 0) {
      try_capture(sq + dr - 1);
    }
    // Capture right.
    if (file != 7) {
      try_capture(sq + dr + 1);
    }
  }

  void KnightMoves(int sq, MoveList* out) const {
    const uint64_t m = KnightMoveMask(sq);
    if (soft_pinned_ & OneHot(sq)) {
      // Pinned knights can't move.
      return;
    }
    for (int to : BitRange(m & ~(my_pieces_)&check_ok_)) {
      out->emplace_back(sq, to);
    }
  }

  void BishopMoves(int sq, MoveList* out) const {
    const uint64_t m = BishopMoveMask(sq, occ_);
    for (int to : BitRange(m & ~my_pieces_ & check_ok_)) {
      if (!IsPinned(sq, to)) {
        out->emplace_back(sq, to);
      }
    }
  }

  void RookMoves(int sq, MoveList* out) const {
    const uint64_t m = RookMoveMask(sq, occ_);
    for (int to : BitRange(m & ~my_pieces_ & check_ok_)) {
      if (!IsPinned(sq, to)) {
        out->emplace_back(sq, to);
      }
    }
  }

  void KingMoves(int sq, MoveList* out) const {
    // Unlike other pieces, king cannot be pinned :)
    const uint64_t m = KingMoveMask(sq);
    for (int to : BitRange(m & ~my_pieces_ & ~king_danger_)) {
      out->emplace_back(sq, to);
    }
  }

  void GenPieceMoves(Piece p, int sq, MoveList* out) const {
    switch (p) {
      case Piece::kPawn:
        return PawnMoves(sq, out);
      case Piece::kKnight:
        return KnightMoves(sq, out);
      case Piece::kBishop:
        return BishopMoves(sq, out);
      case Piece::kRook:
        return RookMoves(sq, out);
      case Piece::kQueen:
        RookMoves(sq, out);
        BishopMoves(sq, out);
        return;
      case Piece::kKing:
        return KingMoves(sq, out);
      case Piece::kNone:
        return;
    }
  }

  MoveList GenerateMoves() const {
    MoveList list;
    for (int p = 0; p < kNumPieces; ++p) {
      for (int sq : BitRange(b_.bitboards_[ti_][p])) {
        GenPieceMoves(Piece(p), sq, &list);
      }
    }
    // Castling.
    if (!in_check_) {
      const int king_rank = ti_ * 7;
      const int king_square = MakeSquare(king_rank, 4);
      const int long_castle = MakeSquare(king_rank, 0);
      const int short_castle = MakeSquare(king_rank, 7);
      const uint64_t long_mask = OneHot(MakeSquare(king_rank, 1)) |
                                 OneHot(MakeSquare(king_rank, 2)) |
                                 OneHot(MakeSquare(king_rank, 3));
      const uint64_t short_mask =
          OneHot(MakeSquare(king_rank, 5)) | OneHot(MakeSquare(king_rank, 6));
      const uint64_t castle_occ =
          occ_ | (king_danger_ & ~(OneHot(Square::B1) | OneHot(Square::B8)));
      // std::cout << "castle occ:\n" << BitboardToString(input.king_danger) <<
      // "\n";
      if ((b_.castling_rights_ & OneHot(long_castle)) &&
          (castle_occ & long_mask) == 0) {
        list.emplace_back(king_square, MakeSquare(king_rank, 2));
      }
      if ((b_.castling_rights_ & OneHot(short_castle)) &&
          ((castle_occ & short_mask) == 0)) {
        list.emplace_back(king_square, MakeSquare(king_rank, 6));
      }
    }

    return list;
  }

  uint64_t ComputeKingDanger() const {
    const int other_ti = 1 - ti_;
    // To account for sliding pieces, remove our king from the occ mask.
    uint64_t occ = occ_ ^ OneHot(king_s_);

    uint64_t danger = 0;

    // Pawns
    for (int s : BitRange(b_.bitboards_[other_ti][0])) {
      // This is reverse, as we're imitating opponent's pawns.
      int dr = turn_ == Color::kWhite ? -1 : 1;
      const int r = SquareRank(s);
      const int f = SquareFile(s);
      if (SquareOnBoard(r + dr, f - 1)) {
        danger |= OneHot(MakeSquare(r + dr, f - 1));
      }
      if (SquareOnBoard(r + dr, f + 1)) {
        danger |= OneHot(MakeSquare(r + dr, f + 1));
      }
    }
    // Knights.
    for (int s : BitRange(b_.bitboards_[other_ti][1])) {
      danger |= KnightMoveMask(s);
    }
    // Bishops and queen diagonal.
    for (int s :
         BitRange(b_.bitboards_[other_ti][2] | b_.bitboards_[other_ti][4])) {
      danger |= BishopMoveMask(s, occ);
    }
    // Rooks and queens
    for (int s :
         BitRange(b_.bitboards_[other_ti][3] | b_.bitboards_[other_ti][4])) {
      danger |= RookMoveMask(s, occ);
    }
    // King:
    for (int s : BitRange(b_.bitboards_[other_ti][5])) {
      danger |= KingMoveMask(s);
    }
    return danger;
  }

  bool ComputeCheck(uint64_t* capture_mask, uint64_t* push_mask) const {
    const int other_ti = 1 - ti_;

    const int r = SquareRank(king_s_);
    const int f = SquareFile(king_s_);

    uint64_t threats = 0;
    uint64_t slider_threats = 0;
    // Pawn
    {
      const int dr = turn_ == Color::kWhite ? 1 : -1;
      uint64_t pawn_mask = 0;
      if (SquareOnBoard(r + dr, f - 1)) {
        pawn_mask |= OneHot(MakeSquare(r + dr, f - 1));
      }
      if (SquareOnBoard(r + dr, f + 1)) {
        pawn_mask |= OneHot(MakeSquare(r + dr, f + 1));
      }
      threats |= pawn_mask & b_.bitboards_[other_ti][0];
      // if (pawn_mask) {
      //   printf("Pawn threat at %i\n", GetFirstBit(pawn_mask));
      // }
    }

    threats |= KnightMoveMask(king_s_) & b_.bitboards_[other_ti][1];
    slider_threats |= BishopMoveMask(king_s_, occ_) &
                      (b_.bitboards_[other_ti][2] | b_.bitboards_[other_ti][4]);
    slider_threats |= RookMoveMask(king_s_, occ_) &
                      (b_.bitboards_[other_ti][3] | b_.bitboards_[other_ti][4]);
    threats |= slider_threats;
    if (threats == 0) {
      *capture_mask = kAllBits;
      *push_mask = kAllBits;
      return false;
    }
    if (PopCount(threats) > 1) {
      // Only king moves help in this case.
      *capture_mask = 0;
      *push_mask = 0;
    } else {
      *capture_mask = threats;
      if (slider_threats == 0) {
        *push_mask = 0;
      } else {
        *push_mask = PushMask(king_s_, GetFirstBit(threats));
      }
    }
    return true;
  }

  uint64_t ComputePinnedPieces() const {
    const int other_ti = 1 - ti_;

    uint64_t pinned = 0;

    const uint64_t king_bishop_mask = BishopMoveMask(king_s_, occ_);
    const uint64_t possible_bishops =
        (b_.bitboards_[other_ti][2] | b_.bitboards_[other_ti][4]) &
        BishopMoveMask(king_s_, 0);
    for (int s : BitRange(possible_bishops)) {
      const uint64_t move_mask = BishopMoveMask(s, occ_);
      pinned |= (move_mask & king_bishop_mask);
    }
    const uint64_t king_rook_mask = RookMoveMask(king_s_, occ_);
    const uint64_t possible_rooks =
        (b_.bitboards_[other_ti][3] | b_.bitboards_[other_ti][4]) &
        RookMoveMask(king_s_, 0);
    for (int s : BitRange(possible_rooks)) {
      const uint64_t move_mask = RookMoveMask(s, occ_);
      pinned |= (move_mask & king_rook_mask);
    }
    return pinned;
  }

 private:
  bool IsPinned(int from, int to) const {
    if ((OneHot(from) & soft_pinned_) == 0) {
      return false;
    }
    // Piece is pinned.  Iff we're not moving towards or from the king, this
    // move must leave our king in check.
    return !SameDirection(king_s_, from, to);
  }

  const Board& b_;

  const Color turn_;
  const int ti_;
  // Square occupation.
  uint64_t occ_ = 0;
  // Masks of my and opponent pieces.
  uint64_t my_pieces_ = 0;
  uint64_t opp_pieces_ = 0;
  // Position of my king.
  uint64_t king_s_ = 0;
  // Mask of positions we can't move our king to. Only used for king moves.
  uint64_t king_danger_ = 0;

  uint64_t soft_pinned_ = 0;

  bool in_check_ = false;

  // If we're in single check, these are
  // TODO: Document these, from
  // https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
  uint64_t check_ok_ = kAllBits;
  uint64_t push_mask_ = kAllBits;
};

extern const Piece kPromoPieces[4] = {Piece::kQueen, Piece::kBishop,
                                      Piece::kKnight, Piece::kRook};

// static
std::string Square::ToString(int sq) {
  char buf[3] = {
      char('a' + (sq % 8)),
      char('1' + (sq / 8)),
      0,
  };
  return buf;
}

void InitializeMovegen() { InitializeMagic(); }

std::string Move::ToString() const {
  char p_str[2] = {0, 0};
  if (promotion != Piece::kNone) {
    p_str[0] = PieceChar(promotion);
  }
  return absl::StrCat(Square::ToString(from), Square::ToString(to), p_str);
}

MoveList Board::valid_moves() const {
  MoveGenerator gen(*this);
  return gen.GenerateMoves();
}

Board::Board() {}

Board::Board(absl::string_view fen) {
  std::cerr << "TODO: implement fen constructor\n";
  abort();
}

Board::Board(const BoardProto& p) {
  assert(p.bitboards_size() == 12);
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < 6; ++i) {
      bitboards_[c][i] = p.bitboards(c * 6 + i);
    }
  }
  en_passant_ = p.en_passant();
  castling_rights_ = p.castling_rights();
  half_move_count_ = p.half_move_count();
  repetition_count_ = p.repetition_count();
}

PieceColor Board::square(int sq) const {
  const uint64_t mask = OneHot(sq);
  for (int c = 0; c < 2; ++c) {
    for (int p = 0; p < kNumPieces; ++p) {
      if (bitboards_[c][p] & mask) {
        return PieceColor{Piece(p), Color(c)};
      }
    }
  }
  return PieceColor{Piece::kNone, Color::kEmpty};
}

std::string Board::ToPrintString() const {
  std::string str = "  a b c d e f g h\n";
  for (int r = 7; r >= 0; --r) {
    str.push_back('1' + r);
    str.push_back(' ');
    for (int f = 0; f < 8; ++f) {
      int sq = r * 8 + f;
      str.push_back(PieceChar(square(sq)));
      str.push_back(' ');
    }
    str.push_back('\n');
  }
  return str;
}

}  // namespace chess
