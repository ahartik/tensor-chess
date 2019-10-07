#ifndef _CHESS_MOVEGEN_H_
#define _CHESS_MOVEGEN_H_

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/magic.h"

namespace chess {

// https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/

// TODO: Profile and optimize this later.

// For pinned pieces, we can only move them towards or away from the king.
template <typename MoveFunc>
class MoveGenerator {
 public:
  explicit MoveGenerator(const Board& b, const MoveFunc& move_func)
      : b_(b),
        move_func_(move_func),
        turn_(b.turn()),
        opp_(OtherColor(b.turn())) {
    assert(turn_ == b_.turn());
    occ_ = b.ComputeOcc();
    for (int p = 0; p < kNumPieces; ++p) {
      my_pieces_ |= b.bitboard(turn_, Piece(p));
    }
    opp_pieces_ = occ_ ^ my_pieces_;
    king_s_ = GetFirstBit(b_.bitboard(turn_, Piece::kKing));

    king_danger_ = ComputeKingDanger();
    if (ABSL_PREDICT_FALSE(king_danger_ & b_.bitboard(turn_, Piece::kKing))) {
      in_check_ = true;
      bool in_check = ComputeCheck(&check_ok_, &push_mask_);
      if (ABSL_PREDICT_FALSE(!in_check)) {
        std::cerr << "ComputeCheck disagrees with king_danger " << b.ToFEN() << "\n";
        abort();
      }
      check_ok_ |= push_mask_;
    } else {
      in_check_ = false;
      push_mask_ = kAllBits;
      check_ok_ = kAllBits;
    }

    soft_pinned_ = ComputePinnedPieces();
  }

  bool IsInCheck() const { return in_check_; }

  int NumMoves() const { return gen_count_; }

  // Enumerates "simple" pawn moves, so not including promotions and en
  // passants.
  template <typename F>
  void EnumSimplePawnMoves(uint64_t pawns, const F& move_func) {
    // Regular forward moves:
    const int dr = turn_ == Color::kWhite ? 8 : -8;
    const uint64_t impossible_push_squares = occ_ | ~check_ok_;
    const uint64_t blocked = turn_ == Color::kWhite
                                 ? (impossible_push_squares >> 8)
                                 : (impossible_push_squares << 8);
    // non-pinned pawns;
    for (int sq : BitRange(pawns & ~blocked & ~soft_pinned_)) {
      int to = sq + dr;
      move_func(sq, to);
    }
    // It's possible for a pawn to be soft-pinned, but not hard-pinned:
    // when we're going straight towards an opponent rook.
    for (int sq : BitRange(pawns & ~blocked & soft_pinned_)) {
      int to = sq + dr;
      if (SameDirection(king_s_, sq, to)) {
        move_func(sq, to);
      }
    }

    // Middle square must be empty, and the target square must also be OK  for
    // check.
    const uint64_t double_blocked =
        turn_ == Color::kWhite
            ? ((occ_ >> 8) | (impossible_push_squares >> 16))
            : ((occ_ << 8) | (impossible_push_squares << 16));
    // Double pushes
    const uint64_t double_mask = RankMask(turn_ == Color::kWhite ? 1 : 6);
    // Unpinned first.
    for (int sq :
         BitRange(pawns & double_mask & ~double_blocked & ~soft_pinned_)) {
      move_func(sq, sq + dr * 2);
    }
    // Pinned double-push (pretty unlikely).
    const uint64_t pinned_doubles =
        pawns & double_mask & ~double_blocked & soft_pinned_;
    for (int sq : BitRange(pinned_doubles)) {
      int to = sq + dr * 2;
      if (SameDirection(king_s_, sq, to)) {
        move_func(sq, to);
      }
    }

    const uint64_t left_mask = ~FileMask(0);
    const uint64_t right_mask = ~FileMask(7);
    const uint64_t pieces_to_capture = opp_pieces_ & check_ok_;
    // Left captures (except en-passant).
    const uint64_t possible_left_captures = turn_ == Color::kWhite
                                                ? (pieces_to_capture >> 7)
                                                : (pieces_to_capture << 9);
    for (int from : BitRange(pawns & left_mask & possible_left_captures)) {
      int dr = turn_ == Color::kWhite ? 7 : -9;
      if (!IsPinned(from, from + dr)) {
        move_func(from, from + dr);
      }
    }
    // Right captures
    const uint64_t possible_right_captures = turn_ == Color::kWhite
                                                 ? (pieces_to_capture >> 9)
                                                 : (pieces_to_capture << 7);
    for (int from : BitRange(pawns & right_mask & possible_right_captures)) {
      int dr = turn_ == Color::kWhite ? 9 : -7;
      // Only case where capture is not pinned is when we're capturing the
      // pinning bishop/queen.
      if (!IsPinned(from, from + dr)) {
        move_func(from, from + dr);
      }
    }
  }

  void PawnMoves(uint64_t pawns) {
    const uint64_t promotion_mask =
        turn_ == Color::kWhite ? RankMask(6) : RankMask(1);
    EnumSimplePawnMoves(pawns & ~promotion_mask, [&](int from, int to) {
      OutputMove(Move(from, to, Move::Type::kRegular));
    });
    if (ABSL_PREDICT_FALSE(pawns & promotion_mask)) {
      // Promotions:
      EnumSimplePawnMoves(pawns & promotion_mask, [&](int from, int to) {
        for (Piece p : kPromoPieces) {
          OutputMove(Move(from, to, p));
        }
      });
    }

    // En passant moves (obviously never promotions):
    if (b_.en_passant() != 0) {
      const int to = GetFirstBit(b_.en_passant());
      int to_rank = SquareRank(to);
      int to_file = SquareFile(to);
      uint64_t captures = 0;
      int dr = turn_ == Color::kWhite ? -1 : 1;
      if (to_file != 0) {
        captures |= OneHot(MakeSquare(to_rank + dr, to_file - 1));
      }
      if (to_file != 7) {
        captures |= OneHot(MakeSquare(to_rank + dr, to_file + 1));
      }
      const uint64_t possible_capturing_pawns = captures & pawns;
      // if (ABSL_PREDICT_TRUE(possible_capturing_pawns == 0)) {
      //   return;
      // }
      for (int from : BitRange(possible_capturing_pawns)) {
        if (IsPinned(from, to)) {
          continue;
        }
        // This is the mask of the pawn we just captured.
        const uint64_t other_pawn = (turn_ == Color::kWhite)
                                        ? b_.en_passant() >> 8
                                        : b_.en_passant() << 8;
        // This is en passant capture. We need to check if this leaves us in
        // check (which would be illegal).
        if (SquareRank(king_s_) == SquareRank(from)) {
          // Captured piece and our king are on the same rank. Check if they
          // have other pieces in-between than 'sq' and the pawn previously
          // next to us.
          const uint64_t removed_sqs = OneHot(from) | other_pawn;
          const uint64_t king_rook_moves =
              RookMoveMask(king_s_, (occ_ ^ removed_sqs)) &
              RankMask(SquareRank(king_s_));
          // Check if opponent rooks match this.
          const uint64_t opp_rooks = b_.bitboard(opp_, Piece::kRook) |
                                     b_.bitboard(opp_, Piece::kQueen);
          if (king_rook_moves & opp_rooks) {
            return;
          }
        }
        // Also, we must double-check that this capture is legal in case
        // we're in check.
        if ((other_pawn & check_ok_) == 0 &&
            (b_.en_passant() & check_ok_) == 0) {
          return;
        }
        OutputMove(Move(from, to, Move::Type::kEnPassant));
      }
    }
  }

  void KnightMoves(uint64_t knights) {
    // Pinned knights can't move.
    for (int from : BitRange(knights & ~soft_pinned_)) {
      const uint64_t m = KnightMoveMask(from);
      for (int to : BitRange(m & ~(my_pieces_)&check_ok_)) {
        const bool is_capture = opp_pieces_ & OneHot(to);
        OutputMove(
            Move(from, to,
                 is_capture ? Move::Type::kRegular : Move::Type::kReversible));
      }
    }
  }

  template <typename MaskFunc>
  void GenerateSlider(uint64_t from_mask, const MaskFunc& mask_func) {
    // Only non-pinned.
    for (int from : BitRange(from_mask & ~soft_pinned_)) {
      const uint64_t m = mask_func(from, occ_);
      for (int to : BitRange(m & ~my_pieces_ & check_ok_)) {
        const bool is_capture = opp_pieces_ & OneHot(to);
        OutputMove(
            Move(from, to,
                 is_capture ? Move::Type::kRegular : Move::Type::kReversible));
      }
    }
    // Only pinned.
    for (int from : BitRange(from_mask & soft_pinned_)) {
      const uint64_t m = mask_func(from, occ_);
      for (int to : BitRange(m & ~my_pieces_ & check_ok_)) {
        if (SameDirection(king_s_, from, to)) {
          const bool is_capture = opp_pieces_ & OneHot(to);
          OutputMove(Move(
              from, to,
              is_capture ? Move::Type::kRegular : Move::Type::kReversible));
        }
      }
    }
  }

  void BishopMoves(uint64_t bishops) {
    GenerateSlider(bishops, &BishopMoveMask);
  }

  void RookMoves(uint64_t rooks) { GenerateSlider(rooks, &RookMoveMask); }

  void KingMoves(uint64_t kings) {
    assert(PopCount(kings) == 1);
    const int from = GetFirstBit(kings);
    // Unlike other pieces, king cannot be pinned :)
    const uint64_t m = KingMoveMask(from);
    for (int to : BitRange(m & ~my_pieces_ & ~king_danger_)) {
      const bool is_capture = opp_pieces_ & OneHot(to);
      OutputMove(
          Move(from, to,
               is_capture ? Move::Type::kRegular : Move::Type::kReversible));
    }
  }

  void GenerateMoves() {
    PawnMoves(b_.bitboard(turn_, Piece::kPawn));
    KnightMoves(b_.bitboard(turn_, Piece::kKnight));
    BishopMoves(b_.bitboard(turn_, Piece::kBishop) |
                b_.bitboard(turn_, Piece::kQueen));
    RookMoves(b_.bitboard(turn_, Piece::kRook) |
              b_.bitboard(turn_, Piece::kQueen));
    KingMoves(b_.bitboard(turn_, Piece::kKing));
    // Castling.
    if (!in_check_) {
      const int king_rank = turn_ == Color::kWhite ? 0 : 7;
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
      // std::cout << "castle occ:\n" << BitboardToString(input.king_danger)
      // << "\n";
      if ((b_.castling_rights() & OneHot(long_castle)) &&
          (castle_occ & long_mask) == 0) {
        OutputMove(
            Move(king_square, MakeSquare(king_rank, 2), Move::Type::kCastling));
      }
      if ((b_.castling_rights() & OneHot(short_castle)) &&
          ((castle_occ & short_mask) == 0)) {
        OutputMove(
            Move(king_square, MakeSquare(king_rank, 6), Move::Type::kCastling));
      }
    }
  }

  uint64_t ComputeKingDanger() const {
    // To account for sliding pieces, remove our king from the occ mask.
    // Otherwise the king could "hide behind itself".
    uint64_t occ = occ_ ^ OneHot(king_s_);

    uint64_t danger = 0;
    // Pawns
    {
      const uint64_t opp_pawns = b_.bitboard(opp_, Piece::kPawn);
      const uint64_t left_mask = ~FileMask(0);
      const uint64_t right_mask = ~FileMask(7);
      // Left and right captures:
      danger |= turn_ == Color::kWhite ? ((opp_pawns & left_mask) >> 9)
                                       : ((opp_pawns & left_mask) << 7);
      danger |= turn_ == Color::kWhite ? ((opp_pawns & right_mask) >> 7)
                                       : ((opp_pawns & right_mask) << 9);
    }

    // Knights.
    for (int s : BitRange(b_.bitboard(opp_, Piece::kKnight))) {
      danger |= KnightMoveMask(s);
    }
    // Bishops and queen diagonal.
    for (int s : BitRange(b_.bitboard(opp_, Piece::kBishop) |
                          b_.bitboard(opp_, Piece::kQueen))) {
      danger |= BishopMoveMask(s, occ);
    }
    // Rooks and queens
    for (int s : BitRange(b_.bitboard(opp_, Piece::kRook) |
                          b_.bitboard(opp_, Piece::kQueen))) {
      danger |= RookMoveMask(s, occ);
    }
    // King:
    danger |= KingMoveMask(GetFirstBit(b_.bitboard(opp_, Piece::kKing)));
    return danger;
  }

  bool ComputeCheck(uint64_t* capture_mask, uint64_t* push_mask) const {
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
      threats |= pawn_mask & b_.bitboard(opp_, Piece::kPawn);
    }

    threats |= KnightMoveMask(king_s_) & b_.bitboard(opp_, Piece::kKnight);
    slider_threats |=
        BishopMoveMask(king_s_, occ_) &
        (b_.bitboard(opp_, Piece::kBishop) | b_.bitboard(opp_, Piece::kQueen));
    slider_threats |=
        RookMoveMask(king_s_, occ_) &
        (b_.bitboard(opp_, Piece::kRook) | b_.bitboard(opp_, Piece::kQueen));
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
    uint64_t pinned = 0;

    const uint64_t king_bishop_mask = BishopMoveMask(king_s_, occ_);
    const uint64_t possible_bishops =
        (b_.bitboard(opp_, Piece::kBishop) | b_.bitboard(opp_, Piece::kQueen)) &
        BishopMoveMask(king_s_, 0);
    for (int s : BitRange(possible_bishops)) {
      const uint64_t move_mask = BishopMoveMask(s, occ_);
      pinned |= (move_mask & king_bishop_mask);
    }
    const uint64_t king_rook_mask = RookMoveMask(king_s_, occ_);
    const uint64_t possible_rooks =
        (b_.bitboard(opp_, Piece::kRook) | b_.bitboard(opp_, Piece::kQueen)) &
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

  void OutputMove(const Move& m) {
    ++gen_count_;
    move_func_(m);
  }

  const Board& b_;
  const MoveFunc& move_func_;
  const Color turn_;
  const Color opp_;

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

  int gen_count_ = 0;

  // If we're in single check, these are
  // TODO: Document these, from
  // https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
  uint64_t check_ok_ = kAllBits;
  uint64_t push_mask_ = kAllBits;
};

// For the most efficient interface, implement function like:
enum class MovegenResult {
  kNotOver = 0,
  // Check and the current player has no legal moves.
  kCheckmate,
  // Not a check, but current player has no moves.
  kStalemate,
};

// For F callable with signature void(const Move& m);
template <typename MoveFunc>
MovegenResult IterateLegalMoves(const Board& b, const MoveFunc& f) {
  MoveGenerator<MoveFunc> gen(b, f);
  gen.GenerateMoves();
  const int num_moves = gen.NumMoves();
  const bool in_check = gen.IsInCheck();
  if (num_moves == 0) {
    if (in_check) {
      return MovegenResult::kCheckmate;
    } else {
      return MovegenResult::kStalemate;
    }
  }
  return MovegenResult::kNotOver;
}

}  // namespace chess

#endif
