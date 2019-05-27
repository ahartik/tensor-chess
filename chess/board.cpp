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

void PawnMoves(const MoveInput& in, MoveList* out) {
  const int sq = in.square;
  const int dr = in.turn == Color::kWhite ? 8 : -8;
  const uint64_t occ = in.occ;
  const int rank = Square::Rank(sq);
  const int file = Square::File(sq);
  assert(rank != 0);
  assert(rank != 7);

  const auto try_forward = [&](int to) {
    if ((occ & OneHot(to)) == 0 && (OneHot(to) & in.check_ok)) {
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
    if ((in.opp_pieces & OneHot(to) & in.check_ok) != 0) {
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
  if (((rank == 1 && in.turn == Color::kWhite) ||
      (rank == 6 && in.turn == Color::kBlack)) &&
      (OneHot(sq + dr) & occ) == 0) {
    try_forward(sq + 2 * dr);
  }

  // Capture left.
  if (file != 0) {
    try_capture(sq + dr - 1);
  }
  if (file != 7) {
    try_capture(sq + dr + 1);
  }
}

void KnightMoves(const MoveInput& in, MoveList* out) {
  const uint64_t m = KnightMoveMask(in.square);
  for (int to : BitRange(m & ~(in.my_pieces) & in.check_ok)) {
    out->emplace_back(in.square, to);
  }
}

void BishopMoves(const MoveInput& in, MoveList* out) {
  const uint64_t m = BishopMoveMask(in.square, in.my_pieces | in.opp_pieces);
  for (int to : BitRange(m & ~(in.my_pieces) & in.check_ok)) {
    out->emplace_back(in.square, to);
  }
}

void RookMoves(const MoveInput& in, MoveList* out) {
  const uint64_t m = RookMoveMask(in.square, in.my_pieces | in.opp_pieces);

  for (int to : BitRange(m & ~(in.my_pieces) & in.check_ok)) {
    out->emplace_back(in.square, to);
  }
}

void KingMoves(const MoveInput& in, MoveList* out) {
  const uint64_t m = KingMoveMask(in.square);
  for (int to : BitRange(m & ~in.my_pieces & ~in.king_danger)) {
    out->emplace_back(in.square, to);
  }
}

void QueenMoves(const MoveInput& in, MoveList* out) {
  RookMoves(in, out);
  BishopMoves(in, out);
}

void GenPieceMoves(Piece p, const MoveInput& in, MoveList* out) {
  switch (p) {
    case Piece::kPawn:
      return PawnMoves(in, out);
    case Piece::kKnight:
      return KnightMoves(in, out);
    case Piece::kBishop:
      return BishopMoves(in, out);
    case Piece::kRook:
      return RookMoves(in, out);
    case Piece::kQueen:
      return QueenMoves(in, out);
    case Piece::kKing:
      return KingMoves(in, out);
    case Piece::kNone:
      return;
  }
}


}  // namespace

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
  MoveList list;
  const Color t = turn();
  const int ti = int(t);
  const int other_ti = int(t);

  // Collect
  MoveInput input = {};
  input.turn = t;
  input.king_danger = ComputeKingDanger();
  for (int i = 0; i < 2; ++i) {
    for (int p = 0; p < kNumPieces; ++p) {
      if (t == Color(i)) {
        input.my_pieces |= bitboards_[i][p];
      } else {
        input.opp_pieces |= bitboards_[i][p];
      }
      input.occ |= bitboards_[i][p];
    }
  }
  input.en_passant = en_passant_;

  uint64_t capture_mask = 0;
  uint64_t push_mask = 0;
  const bool is_check = ComputeCheck(input.occ, &capture_mask, &push_mask);
  if (is_check && push_mask) {
    printf("push mask: %016llx\n", push_mask);
  }

  input.check_ok = capture_mask | push_mask;
//   if (capture_mask == 0) {
//     // Only king moves are allowed.
//     const int sq = GetFirstBit(bitboards_[t][5]);
//     input.square = sq;
//     KingMoves(input, &list);
//     return list;
//   }

  for (int p = 0; p < kNumPieces; ++p) {
    for (int sq : BitRange(bitboards_[ti][p])) {
      input.square = sq;
      GenPieceMoves(Piece(p), input, &list);
    }
  }
  // Castling.
  if (!is_check) {
    const int king_rank = (t == Color::kWhite) ? 0 : 7;
    const int king_square = MakeSquare(king_rank, 4);
    const int long_castle = MakeSquare(king_rank, 0);
    const int short_castle = MakeSquare(king_rank, 7);
    const uint64_t long_mask = OneHot(MakeSquare(king_rank, 1)) |
                               OneHot(MakeSquare(king_rank, 2)) |
                               OneHot(MakeSquare(king_rank, 3));
    const uint64_t short_mask =
        OneHot(MakeSquare(king_rank, 5)) | OneHot(MakeSquare(king_rank, 6));
    if ((castling_rights_ & OneHot(long_castle)) &&
        ((input.occ & long_mask) == 0)) {
      list.emplace_back(king_square, MakeSquare(king_rank, 1));
    }
    if ((castling_rights_ & OneHot(short_castle)) &&
        ((input.occ & short_mask) == 0)) {
      list.emplace_back(king_square, MakeSquare(king_rank, 6));
    }
  }
  return list;
}

uint64_t Board::ComputeKingDanger() const {
  const Color t = turn();
  const int ti = int(t);
  const int other_ti = 1 - ti;
  uint64_t occ = 0;
  for (int i = 0; i < 2; ++i) {
    for (int p = 0; p < kNumPieces; ++p) {
      if (t == Color(i) && Piece(p) == Piece::kKing) {
        continue;
      }
      occ |= bitboards_[i][p];
    }
  }

  uint64_t danger = 0;

  // Pawns
  for (int s : BitRange(bitboards_[other_ti][0])) {
    int dr = t == Color::kWhite ? 1 : -1;
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
  for (int s : BitRange(bitboards_[other_ti][1])) {
    danger |= KnightMoveMask(s);
  }
  // Bishops and queen diagonal.
  for (int s : BitRange(bitboards_[other_ti][2] | bitboards_[other_ti][4])) {
    danger |= BishopMoveMask(s, occ);
  }
  // Rooks and queens
  for (int s : BitRange(bitboards_[other_ti][3] | bitboards_[other_ti][4])) {
    danger |= RookMoveMask(s, occ);
  }
  // King:
  for (int s : BitRange(bitboards_[other_ti][5])) {
    danger |= KingMoveMask(s);
  }
  return danger;
}

bool Board::ComputeCheck(uint64_t occ, uint64_t* capture_mask,
                         uint64_t* push_mask) const {
  const Color t = turn();
  const int ti = int(t);
  const int other_ti = 1 - ti;

  const int king_s = GetFirstBit(bitboards_[ti][5]);
  const int r = SquareRank(king_s);
  const int f = SquareFile(king_s);

  uint64_t threats = 0;
  uint64_t slider_threats = 0;
  // Pawn
  {
    // This is reverse, as we're imitating opponent's pawns.
    const int dr = t == Color::kWhite ? -1 : 1;
    uint64_t pawn_mask = 0;
    if (SquareOnBoard(r + dr, f - 1)) {
      pawn_mask |= MakeSquare(r + dr, f - 1);
    }
    if (SquareOnBoard(r + dr, f + 1)) {
      pawn_mask |= MakeSquare(r + dr, f + 1);
    }
    threats |= pawn_mask & bitboards_[other_ti][0];
  }

  threats |= KnightMoveMask(king_s) & bitboards_[other_ti][1];
  slider_threats |= BishopMoveMask(king_s, occ) &
                    (bitboards_[other_ti][2] | bitboards_[other_ti][4]);
  slider_threats |= RookMoveMask(king_s, occ) &
                    (bitboards_[other_ti][3] | bitboards_[other_ti][4]);
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
      *push_mask = PushMask(king_s, GetFirstBit(threats));
    }
  }
  return true;
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
