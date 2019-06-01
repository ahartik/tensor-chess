#ifndef _CHESS_MAGIC_H_
#define _CHESS_MAGIC_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "chess/bitboard.h"

namespace chess {

// XXX explain better:
// "occ" is a bitboard of all occupied squares. Since we can't distinguish our
// pieces from our opponent's, the returned mask will include self-captures.
// These moves should be filtered out at a higher level.

namespace magic {

#ifdef NDEBUG
// Faster and smaller, but takes longer to generate (about 1s).
constexpr int kBishopLogSize = 10;
constexpr int kRookLogSize = 12;
#else
constexpr int kBishopLogSize = 10;
constexpr int kRookLogSize = 12;
#endif

template <int LogSize>
struct SliderMagic {
  static constexpr int kLogSize = LogSize;
  static constexpr size_t kSize = 1ull << kLogSize;

  size_t log_size() const { return kLogSize; }
  size_t mask_size() const { return kSize; }

  uint64_t GetMask(int sq, uint64_t occ) const {
    occ &= rel_occ[sq];
    uint64_t x = (occ * mul[sq]) >> (64 - log_size());
    assert(x < mask_size());
    return mask[sq][x];
  }

  uint64_t mul[64];
  uint64_t rel_occ[64];
  uint64_t mask[64][kSize];
};

struct Magic {
  uint64_t knight_masks[64];
  uint64_t king_masks[64];

  uint64_t push_masks[64][64];
  uint64_t ray_masks[64][64];

  uint64_t king_pawn_danger[64];

  SliderMagic<kBishopLogSize> bishop_magics;
  SliderMagic<kRookLogSize> rook_magics;
};
extern Magic m;

}  // namespace magic

inline uint64_t KnightMoveMask(int square) {
  return magic::m.knight_masks[square];
}
inline uint64_t KingMoveMask(int square) { return magic::m.king_masks[square]; }
inline uint64_t BishopMoveMask(int square, uint64_t occ) {
  return magic::m.bishop_magics.GetMask(square, occ);
}

inline uint64_t RookMoveMask(int square, uint64_t occ) {
  return magic::m.rook_magics.GetMask(square, occ);
}

inline uint64_t PushMask(int from, int to) {
  return magic::m.push_masks[from][to];
}

inline uint64_t RayMask(int from, int to) {
  return magic::m.ray_masks[from][to];
}

inline uint64_t KingPawnDanger(int sq) { return magic::m.king_pawn_danger[sq]; }

inline bool SameDirection(int king, int from, int to) {
  uint64_t pt = PushMask(king, to) | OneHot(to);
  uint64_t pf = PushMask(king, from) | OneHot(from);
  return (pt & pf) != 0;
}

// Must be called for above functions to work.
void InitMagic();

}  // namespace chess

#endif
