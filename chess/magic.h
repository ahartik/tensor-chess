#ifndef _CHESS_MAGIC_H_
#define _CHESS_MAGIC_H_

#include <cstdint>

namespace chess {

uint64_t KnightMoveMask(int pos, uint64_t my, uint64_t opp);
uint64_t BishopMoveMask(int pos, uint64_t my, uint64_t opp);
uint64_t RookMask(int pos, uint64_t my, uint64_t opp);

void InitializeMagic();

}  // namespace chess

#endif
