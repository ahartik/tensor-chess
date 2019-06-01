#include "chess/hashing.h"

#include <random>

namespace chess {

namespace {
std::once_flag generated;

void InitHashingInternal() {
  std::mt19937_64 mt;
  for (int j = 0; j < 2; ++j) {
    for (int p = 0; p < 6; ++p) {
      for (int i = 0; i < 64; ++i) {
        zobrist[j][p][i] = mt();
      }
    }
  }
}

}  // namespace

uint64_t zobrist[2][6][64];

void InitHashing() { std::call_once(generated, &InitHashingInternal); }

}  // namespace chess
