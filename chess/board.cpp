#include "chess/board.h"

#include "chess/bitboard.h"

namespace chess {

uint64_t knight_masks[64];
uint64_t bishop_magic[64][1 << 10];

// Low-level move masks.
uint64_t KnightMoveMask(int pos, uint64_t my, uint64_t opp) {
  return 0;
}

uint64_t BishopMoveMask(int pos, uint64_t my, uint64_t opp) {
  return 0;
}

uint64_t RookMask(int pos, uint64_t my, uint64_t opp) {
  return 0;
}

// 
uint64_t GenerateBishopMask(int pos, uint64_t occ) {
  uint64_t mask = 0;
  int r = PosRank(pos);
  int f = PosFile(pos);
  return mask;
}

void InitializeMagic() {
  // Knights.
  for (int r = 0; r < 8; ++r) {
    for (int f = 0; f < 8; ++f) {
      const int p = MakePos(r, f);
      uint64_t mask = 0;
      for (int dr : {1, 2}) {
        const int df = dr ^ 3;
        // Either can be positive or negative.
        for (int i = 0; i < 4; ++i) {
          int result_r = r + dr * ((i&1) ? 1 : -1);
          int result_f = f + df * ((i&2) ? 1 : -1);
          if (PosOnBoard(result_r, result_f)) {
            mask |= (1ull << MakePos(result_r, result_f));
          }
        }
      }
      knight_masks[p] = mask;
    }
  }
  // Bishops.

}

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

void PawnMoves(const MoveInput& in, MoveList* out) {
}

void KingMoves(const MoveInput& in, MoveList* out) {
}

void BishopMoves(const MoveInput& in, MoveList* out) {
}

void KnightMoves(const MoveInput& in, MoveList* out) {
}

void RookMoves(const MoveInput& in, MoveList* out) {
}

void QueenMoves(const MoveInput& in, MoveList* out) {
}

void InitializeMovegen() {

}

class Board {
 public:
  bool is_over() const;
  // If the game is over, this returns the winner of the game, or Color::kEmpty
  // in case of a draw.
  Color winner() const;

  MoveList valid_moves() const;

  bool operator==(const Board& b) const;

 private:
  template <typename H>
  friend H AbslHashValue(H h, const Board& b);

  uint64_t bitboards_[2][6];
  // Squares where en-passant capture is possible for the current player.
  uint64_t en_passant_;
  int16_t repetition_count_ = 0;
  int half_move_count_ = 0;
  int no_progress_count_ = 0;
  SmallIntSet history_;
};


}  // namespace chess
