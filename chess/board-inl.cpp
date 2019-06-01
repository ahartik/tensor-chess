#include <cassert>
#include <iostream>

#include "absl/base/optimization.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "chess/bitboard.h"
#include "chess/magic.h"

namespace chess {

// https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/

// TODO: Profile and optimize this later.

// For pinned pieces, we can only move them towards or away from the king.
static inline bool SameDirection(int king, int from, int to) {
  uint64_t pt = PushMask(king, to) | OneHot(to);
  uint64_t pf = PushMask(king, from) | OneHot(from);
  return (pt & pf) != 0;
}

template <typename MoveFunc>
class MoveGenerator {
 public:
  explicit MoveGenerator(const Board& b, const MoveFunc& move_func)
      : b_(b), move_func_(move_func), turn_(b.turn()), ti_(int(turn_)) {
    assert(turn_ == b_.turn());
    occ_ = b.ComputeOcc();
    for (int p = 0; p < kNumPieces; ++p) {
      my_pieces_ |= b.bitboards_[ti_][p];
    }
    opp_pieces_ = occ_ ^ my_pieces_;
    king_s_ = GetFirstBit(b_.bitboards_[ti_][5]);

    king_danger_ = ComputeKingDanger();
    if (ABSL_PREDICT_FALSE(king_danger_ & b_.bitboards_[ti_][5])) {
      in_check_ = true;
      bool in_check = ComputeCheck(&check_ok_, &push_mask_);
      assert(in_check);
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

    const uint64_t double_blocked = turn_ == Color::kWhite
                                        ? ((occ_ >> 8) | (occ_ >> 16))
                                        : ((occ_ << 8) | (occ_ << 16));
    // Double moves
    const uint64_t double_mask = RankMask(turn_ == Color::kWhite ? 1 : 6);
    for (int sq : BitRange(pawns & double_mask & ~double_blocked)) {
      int to = sq + dr * 2;
      // Still check compare against check_ok_:
      if (OneHot(to) & check_ok_) {
        if (IsPinned(sq, to)) {
          // It's possible for a pawn to be soft-pinned, but not hard-pinned:
          // when we're going straight towards an opponent rook.
          continue;
        }
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
    if (b_.en_passant_ != 0) {
      const int to = GetFirstBit(b_.en_passant_);
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
      for (int from : BitRange(captures & pawns)) {
        if (IsPinned(from, to)) {
          continue;
        }
        // This is the mask of the pawn we just captured.
        const uint64_t other_pawn = (turn_ == Color::kWhite)
                                        ? b_.en_passant_ >> 8
                                        : b_.en_passant_ << 8;
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
          const uint64_t opp_rooks =
              b_.bitboards_[ti_ ^ 1][3] | b_.bitboards_[ti_ ^ 1][4];
          if (king_rook_moves & opp_rooks) {
            return;
          }
        }
        // Also, we must double-check that this capture is legal in case
        // we're in check.
        if ((other_pawn & check_ok_) == 0 &&
            (b_.en_passant_ & check_ok_) == 0) {
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
    PawnMoves(b_.bitboards_[ti_][0]);
    KnightMoves(b_.bitboards_[ti_][1]);
    BishopMoves(b_.bitboards_[ti_][2] | b_.bitboards_[ti_][4]);
    RookMoves(b_.bitboards_[ti_][3] | b_.bitboards_[ti_][4]);
    KingMoves(b_.bitboards_[ti_][5]);
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
      // std::cout << "castle occ:\n" << BitboardToString(input.king_danger)
      // <<
      // "\n";
      if ((b_.castling_rights_ & OneHot(long_castle)) &&
          (castle_occ & long_mask) == 0) {
        OutputMove(
            Move(king_square, MakeSquare(king_rank, 2), Move::Type::kCastling));
      }
      if ((b_.castling_rights_ & OneHot(short_castle)) &&
          ((castle_occ & short_mask) == 0)) {
        OutputMove(
            Move(king_square, MakeSquare(king_rank, 6), Move::Type::kCastling));
      }
    }
  }

  uint64_t ComputeKingDanger() const {
    const int other_ti = ti_ ^ 1;
    // To account for sliding pieces, remove our king from the occ mask.
    uint64_t occ = occ_ ^ OneHot(king_s_);

    uint64_t danger = 0;
    // Pawns
    for (int s :
         BitRange(KingPawnDanger(king_s_) & b_.bitboards_[other_ti][0])) {
      // This is reverse, since we're imitating opponent's pawns.
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
    danger |= KingMoveMask(GetFirstBit(b_.bitboards_[other_ti][5]));
    return danger;
  }

  bool ComputeCheck(uint64_t* capture_mask, uint64_t* push_mask) const {
    const int other_ti = ti_ ^ 1;

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
    const int other_ti = ti_ ^ 1;

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

  void OutputMove(const Move& m) {
    ++gen_count_;
    move_func_(m);
  }

  const Board& b_;
  const MoveFunc& move_func_;
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

  int gen_count_ = 0;

  // If we're in single check, these are
  // TODO: Document these, from
  // https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
  uint64_t check_ok_ = kAllBits;
  uint64_t push_mask_ = kAllBits;
};

extern const Piece kPromoPieces[4] = {Piece::kQueen, Piece::kBishop,
                                      Piece::kKnight, Piece::kRook};

void InitializeMovegen() { InitializeMagic(); }

std::string Move::ToString() const {
  char p_str[2] = {0, 0};
  if (promotion != Piece::kNone) {
    p_str[0] = PieceChar(promotion, Color::kBlack);
  }
  return absl::StrCat(Square::ToString(from), Square::ToString(to), p_str);
}

MoveList Board::valid_moves() const {
  MoveList list;
  list.reserve(128);
  LegalMoves([&](const Move& m) { list.push_back(m); });
  return list;
}

template <typename MoveFunc>
Board::State Board::LegalMoves(const MoveFunc& f) const {
  MoveGenerator<MoveFunc> gen(*this, f);
  gen.GenerateMoves();
  const int num_moves = gen.NumMoves();
  const bool in_check = gen.IsInCheck();
  if (num_moves == 0) {
    if (in_check) {
      return State::kCheckmate;
    } else {
      return State::kStalemate;
    }
  }
  if (half_move_count_ >= 100) {
    return State::kNoProgressDraw;
  }
  return State::kNotOver;
}

Board::Board()
    : Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") {}

Board::Board(absl::string_view fen) {
  int r = 7;
  int f = 0;
  std::vector<absl::string_view> parts = absl::StrSplit(fen, ' ');
  if (parts.size() != 6) {
    std::cerr << "Invalid fen: \"" << fen << "\"\n";
    abort();
  }
  for (char c : parts[0]) {
    if (c == ' ') {
      break;
    } else if (c == '/') {
      assert(f == 8);
      f = 0;
      r -= 1;
    } else if (std::isdigit(c)) {
      f += c - '0';
    } else {
      const Color col = std::isupper(c) ? Color::kWhite : Color::kBlack;
      Piece p;
      switch (std::tolower(c)) {
        case 'p':
          p = Piece::kPawn;
          break;
        case 'n':
          p = Piece::kKnight;
          break;
        case 'b':
          p = Piece::kBishop;
          break;
        case 'r':
          p = Piece::kRook;
          break;
        case 'q':
          p = Piece::kQueen;
          break;
        case 'k':
          p = Piece::kKing;
          break;
        default:
          std::cerr << "Invalid piece '" << c << "'\n";
          abort();
      }
      bitboards_[int(col)][int(p)] |= OneHot(MakeSquare(r, f));
      ++f;
    }
  }
  char turn_char = parts[1][0];
  if (turn_char == 'w') {
    half_move_count_ = 0;
  } else {
    half_move_count_ = 1;
  }

  castling_rights_ = 0;
  for (char c : parts[2]) {
    switch (c) {
      case 'K':
        castling_rights_ |= OneHot(Square::H1);
        break;
      case 'Q':
        castling_rights_ |= OneHot(Square::A1);
        break;
      case 'k':
        castling_rights_ |= OneHot(Square::H8);
        break;
      case 'q':
        castling_rights_ |= OneHot(Square::A8);
        break;
      case '-':
        break;
      default:
        std::cerr << "Invalid castling_right '" << c << "'\n";
        abort();
    }
  }
  if (parts[3] != "-") {
    for (int s = 0; s < 64; ++s) {
      if (Square::ToString(s) == parts[3]) {
        en_passant_ = OneHot(s);
      }
    }
    if (en_passant_ == 0) {
      std::cerr << "Invalid en passant square \"" << parts[3] << "\"";
      abort();
    }
  }

  int halfmove_clock = 0;
  if (!absl::SimpleAtoi(parts[4], &halfmove_clock)) {
    std::cerr << "Invalid half move clock \"" << parts[4] << "\"";
    abort();
  }
  no_progress_count_ = halfmove_clock;
  int fullmove_clock = 0;
  if (!absl::SimpleAtoi(parts[5], &fullmove_clock)) {
    std::cerr << "Invalid full move clock \"" << parts[5] << "\"";
    abort();
  }
  half_move_count_ += 2 * (fullmove_clock - 1);
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

Board::Board(const Board& o, const Move& m) : Board(o) {
  const uint64_t from_o = OneHot(m.from);
  const uint64_t to_o = OneHot(m.to);
  const int ti = half_move_count_ & 1;

  const Move::Type type = o.GetMoveType(m);
  en_passant_ = 0;
  switch (type) {
    case Move::Type::kReversible:
      ++no_progress_count_;
      for (int i = 0; i < kNumPieces; ++i) {
        if (bitboards_[ti][i] & from_o) {
          // Swap these two bits.
          bitboards_[ti][i] ^= from_o | to_o;
        }
      }
      // If this was a rook, castle rights are lost.
      castling_rights_ &= ~from_o;
      // Or if we took an opponent rook:
      castling_rights_ &= ~to_o;
      // Or if king was moved:
      if (bitboards_[0][5] & to_o) {
        castling_rights_ &= ~RankMask(0);
      } else if (bitboards_[1][5] & to_o) {
        castling_rights_ &= ~RankMask(7);
      }
      break;
    case Move::Type::kRegular:
      no_progress_count_ = 0;
      // Before changing bitmaps, set en passant if needed:

      // Check if this is a pawn move first.
      if (bitboards_[ti][0] & from_o) {
        if (m.to - m.from == 16) {
          // White two-step pawn move.
          en_passant_ = OneHot(m.to - 8);
        }
        if (m.from - m.to == 16) {
          // Black two-step pawn move.
          en_passant_ = OneHot(m.to + 8);
        }
      }
      // Swap the pieces as above.
      for (int i = 0; i < kNumPieces; ++i) {
        if (bitboards_[ti][i] & from_o) {
          bitboards_[ti][i] ^= from_o | to_o;
        }
      }
      // Perform potential captures.
      for (int i = 0; i < kNumPieces; ++i) {
        bitboards_[ti ^ 1][i] &= ~to_o;
      }
      // If this was a rook, castle rights are lost.
      castling_rights_ &= ~from_o;
      // Or if we took an opponent rook:
      castling_rights_ &= ~to_o;
      // Or if king was moved:
      if (bitboards_[0][5] & to_o) {
        castling_rights_ &= ~RankMask(0);
      } else if (bitboards_[1][5] & to_o) {
        castling_rights_ &= ~RankMask(7);
      }
      break;
    case Move::Type::kCastling:
      ++no_progress_count_;
      // This doesn't have to be super fast, castling is not frequent.
      castling_rights_ &= ~RankMask(ti * 7);
      if (m.to == Square::C1) {
        bitboards_[0][5] = OneHot(Square::C1);
        bitboards_[0][3] ^= OneHot(Square::A1) | OneHot(Square::D1);
      } else if (m.to == Square::G1) {
        bitboards_[0][5] = OneHot(Square::G1);
        bitboards_[0][3] ^= OneHot(Square::H1) | OneHot(Square::F1);
      } else if (m.to == Square::C8) {
        bitboards_[1][5] = OneHot(Square::C8);
        bitboards_[1][3] ^= OneHot(Square::A8) | OneHot(Square::D8);
      } else if (m.to == Square::G8) {
        bitboards_[1][5] = OneHot(Square::G8);
        bitboards_[1][3] ^= OneHot(Square::H8) | OneHot(Square::F8);
      } else {
        std::cerr << "bad castle move: " << m.ToString() << "\n";
        abort();
      }
      break;
    case Move::Type::kPromotion:
      // Pawn move, so this counter resets.
      no_progress_count_ = 0;
      // Remove pawn.
      bitboards_[ti][0] &= ~from_o;
      // Insert new piece:
      bitboards_[ti][int(m.promotion)] |= to_o;
      // Perform potential captures.
      for (int i = 0; i < kNumPieces; ++i) {
        bitboards_[ti ^ 1][i] &= ~to_o;
      }
      // We might have captured a rook with castling rights.
      castling_rights_ &= ~to_o;
      break;
    case Move::Type::kEnPassant:
      // This must be a pawn move.
      bitboards_[ti][0] ^= from_o | to_o;
      // We're capturing pawn on the same rank as 'from', same file as 'to'.
      bitboards_[ti ^ 1][0] &=
          ~OneHot(MakeSquare(SquareRank(m.from), SquareFile(m.to)));
      break;
    default:
      std::cerr << "broken move type: " << int(type) << "\n";
      abort();
  }
  ++half_move_count_;
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
      str.push_back(PieceColorChar(square(r, f)));
      str.push_back(' ');
    }
    str.push_back('\n');
  }
  return str;
}

std::string Board::ToFEN() const {
  std::string fen;
  for (int r = 7; r >= 0; --r) {
    int rep = 0;
    for (int f = 0; f < 8; ++f) {
      PieceColor pc = square(r, f);
      if (pc.c != Color::kEmpty) {
        if (rep != 0) {
          fen.push_back('0' + rep);
          rep = 0;
        }
        fen.push_back(PieceColorChar(pc));
      } else {
        ++rep;
      }
    }
    if (rep != 0) {
      fen.push_back('0' + rep);
    }
    if (r != 0) {
      fen.push_back('/');
    }
  }
  fen.push_back(' ');
  fen.push_back(turn() == Color::kWhite ? 'w' : 'b');
  fen.push_back(' ');
  if (castling_rights_ & OneHot(Square::H1)) {
    fen.push_back('K');
  }
  if (castling_rights_ & OneHot(Square::A1)) {
    fen.push_back('Q');
  }
  if (castling_rights_ & OneHot(Square::H8)) {
    fen.push_back('k');
  }
  if (castling_rights_ & OneHot(Square::A8)) {
    fen.push_back('q');
  }
  if (castling_rights_ == 0) {
    fen.push_back('-');
  }
  fen.push_back(' ');
  if (en_passant_) {
    fen += Square::ToString(GetFirstBit(en_passant_));
  } else {
    fen.push_back('-');
  }
  absl::StrAppend(&fen, " ", no_progress_count_, " ",
                  1 + (half_move_count_ / 2));

  return fen;
}

Move::Type Board::GetMoveType(const Move& m) const {
  if (m.type != Move::Type::kUnknown) {
    // Already pre-computed.
    return m.type;
  }
  // Rest of the code only occurs when input move is not generated by
  // valid_moves(). Thus the code below doesn't need to be fast: just make
  // sure it's correct.
  const uint64_t to_o = OneHot(m.to);
  const uint64_t from_o = OneHot(m.from);
  const uint64_t occ = ComputeOcc();

  const uint64_t pawns = (bitboards_[0][0] | bitboards_[1][0]);
  // Ok, let's see if this is an en-passant capture:
  if (to_o == en_passant_) {
    // Could be, but not necessarily. Check if the piece being moved was a
    // pawn:
    if (pawns & from_o) {
      return Move::Type::kEnPassant;
    }
  }
  if (m.promotion != Piece::kNone) {
    return Move::Type::kPromotion;
  }
  // Could be castling:
  const uint64_t kings = bitboards_[0][5] | bitboards_[1][5];
  if (from_o & kings) {
    // We're moving a king. It's a castling if there is exactly 2 difference
    // in position.
    if (abs(m.to - m.from) == 2) {
      return Move::Type::kCastling;
    }
  }
  // Is this a pawn move or a capture:
  if ((pawns & from_o) || (to_o & occ)) {
    return Move::Type::kRegular;
  }
  return Move::Type::kReversible;
}

}  // namespace chess
