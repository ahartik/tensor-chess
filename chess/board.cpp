#include "chess/board.h"

#include <cassert>
#include <iostream>

#include "absl/base/optimization.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "chess/bitboard.h"
#include "chess/hashing.h"
#include "chess/movegen.h"

namespace chess {

void Board::Init() {
  InitMagic();
  InitHashing();
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
  board_hash_ = ComputeBoardHash();
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
  board_hash_ = ComputeBoardHash();
}

// This constructor is our only "MakeMove()" function.
Board::Board(const Board& o, const Move& m) : Board(o) {
  const uint64_t from_o = OneHot(m.from);
  const uint64_t to_o = OneHot(m.to);
  const int ti = half_move_count_ & 1;

  const Move::Type type = o.GetMoveType(m);
  // Remove en passant and castling from hash. They are added back later after
  // they've been (potentially) modified.
  board_hash_ ^= en_passant_;
  board_hash_ ^= castling_rights_;
  en_passant_ = 0;

  switch (type) {
    case Move::Type::kReversible: {
      ++no_progress_count_;
      for (int i = 0; i < kNumPieces; ++i) {
        if (bitboards_[ti][i] & from_o) {
          // Swap these two bits.
          bitboards_[ti][i] ^= from_o | to_o;
          board_hash_ ^= zobrist[ti][i][m.from] ^ zobrist[ti][i][m.to];
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
    }
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
        board_hash_ ^= en_passant_;
      }
      // Swap the pieces as above.
      for (int i = 0; i < kNumPieces; ++i) {
        if (bitboards_[ti][i] & from_o) {
          bitboards_[ti][i] ^= from_o | to_o;
          board_hash_ ^= zobrist[ti][i][m.from] ^ zobrist[ti][i][m.to];
        }
      }
      // Perform potential captures.
      for (int i = 0; i < kNumPieces; ++i) {
        if (bitboards_[ti ^ 1][i] & to_o) {
          bitboards_[ti ^ 1][i] &= ~to_o;
          board_hash_ ^= zobrist[ti ^ 1][i][m.to];
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
    case Move::Type::kCastling:
      ++no_progress_count_;
      // This doesn't have to be super fast, castling is not frequent.
      castling_rights_ &= ~RankMask(ti * 7);
      if (m.to == Square::C1) {
        bitboards_[0][5] = OneHot(Square::C1);
        bitboards_[0][3] ^= OneHot(Square::A1) | OneHot(Square::D1);
        board_hash_ ^= zobrist[0][5][Square::E1] ^ zobrist[0][5][Square::C1] ^
                       zobrist[0][3][Square::A1] ^ zobrist[0][3][Square::D1];
      } else if (m.to == Square::G1) {
        bitboards_[0][5] = OneHot(Square::G1);
        bitboards_[0][3] ^= OneHot(Square::H1) | OneHot(Square::F1);
        board_hash_ ^= zobrist[0][5][Square::E1] ^ zobrist[0][5][Square::G1] ^
                       zobrist[0][3][Square::H1] ^ zobrist[0][3][Square::F1];
      } else if (m.to == Square::C8) {
        bitboards_[1][5] = OneHot(Square::C8);
        bitboards_[1][3] ^= OneHot(Square::A8) | OneHot(Square::D8);
        board_hash_ ^= zobrist[1][5][Square::E8] ^ zobrist[1][5][Square::C8] ^
                       zobrist[1][3][Square::A8] ^ zobrist[1][3][Square::D8];
      } else if (m.to == Square::G8) {
        bitboards_[1][5] = OneHot(Square::G8);
        bitboards_[1][3] ^= OneHot(Square::H8) | OneHot(Square::F8);
        board_hash_ ^= zobrist[1][5][Square::E8] ^ zobrist[1][5][Square::G8] ^
                       zobrist[1][3][Square::H8] ^ zobrist[1][3][Square::F8];
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

      board_hash_ ^= zobrist[ti][0][m.from];
      board_hash_ ^= zobrist[ti][int(m.promotion)][m.to];
      // Perform potential captures.
      for (int i = 0; i < kNumPieces; ++i) {
        if (bitboards_[ti ^ 1][i] & to_o) {
          bitboards_[ti ^ 1][i] &= ~to_o;
          board_hash_ ^= zobrist[ti ^ 1][i][m.to];
        }
      }
      // We might have captured a rook with castling rights.
      castling_rights_ &= ~to_o;
      break;
    case Move::Type::kEnPassant: {
      // This must be a pawn move.
      bitboards_[ti][0] ^= from_o | to_o;
      // We're capturing pawn on the same rank as 'from', same file as 'to'.
      const int captured_square =
          MakeSquare(SquareRank(m.from), SquareFile(m.to));
      bitboards_[ti ^ 1][0] &= ~OneHot(captured_square);
      board_hash_ ^= zobrist[ti][0][m.from];
      board_hash_ ^= zobrist[ti][0][m.to];
      board_hash_ ^= zobrist[ti ^ 1][0][captured_square];
      break;
    }
    default:
      std::cerr << "broken move type: " << int(type) << "\n";
      abort();
  }
  board_hash_ ^= castling_rights_;

  // Different turn, XOR with a constant.
  constexpr uint64_t kMoveXor = kAllBits / M_SQRT2;
  board_hash_ ^= kMoveXor;
  ++half_move_count_;
  assert(board_hash_ == ComputeBoardHash());
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

bool Board::operator==(const Board& o) const {
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < kNumPieces; ++j) {
      if (bitboards_[i][j] != o.bitboards_[i][j]) {
        return false;
      }
    }
  }
  if (en_passant_ != o.en_passant_) {
    return false;
  }
  if (castling_rights_ != o.castling_rights_) {
    return false;
  }
  // Different turn.
  if (half_move_count_ % 2 != o.half_move_count_) {
    return false;
  }

  return true;
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

uint64_t Board::ComputeBoardHash() const {
  uint64_t h = 0;
  h ^= en_passant_;
  h ^= castling_rights_;

  // This is not the fastest, but this codepath is only used when constructing
  // from proto or FEN.
  for (int s = 0; s < 64; ++s) {
    PieceColor pc = square(s);
    if (pc.c != Color::kEmpty) {
      h ^= ZobristHash(pc.c, pc.p, s);
    }
  }
  return h;
}

MoveList Board::valid_moves() const {
  MoveList list;
  list.reserve(128);
  IterateLegalMoves(*this, [&](const Move& m) { list.push_back(m); });
  return list;
}

std::ostream& operator<<(std::ostream& o, const Board& b) {
  return o << b.ToFEN();
}

}  // namespace chess
