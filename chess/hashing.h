#ifndef _CHESS_HASHING_H_
#define _CHESS_HASHING_H_

#include <cstdint>
#include <cmath>

#include "chess/types.h"

namespace chess {

extern uint64_t zobrist[2][6][64];

inline uint64_t MixHash(uint64_t key) {
  // From http://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
  key ^= (key >> 30);
  key *= 0xbf58476d1ce4e5b9;
  key ^= (key >> 27);
  key *= 0x94d049bb133111eb;
  key ^= (key >> 31);
  return key;
}

inline uint64_t HashCombine(uint64_t a, uint64_t b) {
  constexpr uint64_t kOnePerPi = ~0ull / M_PI;
  return (MixHash(a) ^ (b >> 33) ^ (b << 15)) + kOnePerPi;
}

inline uint64_t ZobristHash(Color c, Piece p, int sq) {
  return zobrist[int(c)][int(p)][sq];
}

void InitHashing();

}  // namespace chess

#endif
