#ifndef _CHESS_MAGIC_H_
#define _CHESS_MAGIC_H_

#include <cstdint>

namespace chess {

// XXX explain better:
// "occ" is a bitboard of all occupied squares. Since we can't distinguish our
// pieces from our opponent's, the returned mask will include self-captures.
// These moves should be filtered out at a higher level.

uint64_t KingMoveMask(int pos);
uint64_t KnightMoveMask(int pos);
uint64_t BishopMoveMask(int pos, uint64_t occ);
uint64_t RookMoveMask(int pos, uint64_t occ);

uint64_t PushMask(int from, int to);

// TODO: Consider if we should have only "occ" instead of both "my" and "opp"
// here. Maybe the removal of self-captures belongs higher up?

void InitializeMagic();

}  // namespace chess

#endif
