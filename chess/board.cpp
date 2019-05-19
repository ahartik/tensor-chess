#include "chess/board.h"

#include "chess/bitboard.h"
#include "chess/magic.h"

namespace chess {

namespace {

// https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/

struct MoveInput {
  // Position of the piece we're generating moves for.
  int piece_pos = 0;
  // Masks of my and opponent pieces.
  uint64_t my_pieces = 0;
  uint64_t opp_pieces = 0;
  // Which opponent pawns are in en passant.
  uint64_t opp_en_passant = 0;
  // Mask of positions we can't move our king to. Only used for king moves.
  uint64_t king_danger = 0;

  // If we're in single check, these are
  uint64_t capture_mask = -1ull;
  uint64_t push_mask = -1ull;
};

void PawnMoves(const MoveInput& in, MoveList* out) {}

void KingMoves(const MoveInput& in, MoveList* out) {}

void BishopMoves(const MoveInput& in, MoveList* out) {}

void KnightMoves(const MoveInput& in, MoveList* out) {}

void RookMoves(const MoveInput& in, MoveList* out) {}

void QueenMoves(const MoveInput& in, MoveList* out) {}

}  // namespace

void InitializeMovegen() { InitializeMagic(); }

MoveList Board::valid_moves() const {
  MoveList list;

  uint64_t all[2] = {};
  for (int i = 0; i < 2; ++i) {
    for (int p = 0; p < kNumPieces; ++p) {
      all[i] |= bitboards_[i][p];
    }
  }

  const uint64_t occ = all[0] | all[1];
  const int t = static_cast<int>(turn());

  // Start with computing king danger. These are
  return list;
}

}  // namespace chess
