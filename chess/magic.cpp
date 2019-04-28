#include "chess/magic.h"

#include <iostream>
#include <mutex>
#include <random>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "chess/bitboard.h"

namespace chess {

namespace {

std::once_flag generated;

uint64_t knight_masks[64];
constexpr int kBishopLogSize = 10;
constexpr int kRookLogSize = 12;

template <int LogSize>
struct Magics {
  static constexpr int kLogSize = LogSize;
  static constexpr size_t kSize = 1ull << kLogSize;

  size_t log_size() const { return kLogSize; }
  size_t mask_size() const { return kSize; }
  uint64_t mul[64];
  uint64_t rel_occ[64];
  uint64_t mask[64][kSize];
};

Magics<kBishopLogSize> bishop_magics;
Magics<kRookLogSize> rook_magics;

constexpr uint64_t kUnsetSentinel = kAllBits;

std::vector<uint64_t> AllSubsets(uint64_t set) {
  int set_bits[64] = {};
  int j = 0;
  for (int i = 0; i < 64; ++i) {
    if (BitIsSet(set, i)) {
      set_bits[j++] = i;
    }
  }
  int size = PopCount(set);
  std::vector<uint64_t> res;
  res.reserve(1 << size);
  for (uint64_t x = 0; x < (1ull << size); ++x) {
    uint64_t subset = 0;
    for (int i = 0; i < size; ++i) {
      if (BitIsSet(x, i)) {
        subset |= OneHot(set_bits[i]);
      }
    }
    res.push_back(subset);
  }
  return res;
}

uint64_t GeneratePieceMoves(const int dr[4], const int df[4], int square,
                            uint64_t occ) {
  uint64_t mask = 0;
  for (int i = 0; i < 4; ++i) {
    int r = SquareRank(square);
    int f = SquareFile(square);
    r += dr[i];
    f += df[i];
    while (SquareOnBoard(r, f)) {
      int new_square = MakeSquare(r, f);
      mask |= OneHot(new_square);
      if (BitIsSet(occ, new_square)) {
        // This square is occupied. Add it as capture, even though it could be
        // our own piece. We'll remove self-captures later.
        break;
      }
      r += dr[i];
      f += df[i];
    }
  }
  return mask;
}

void GenerateMagic(const int dr[4], const int df[4], int square,
                   uint64_t* out_rel_occ, uint64_t* out_mul, uint64_t* output,
                   int output_logsize) {
  uint64_t rel_occ = 0;
  for (int i = 0; i < 4; ++i) {
    int r = SquareRank(square);
    int f = SquareFile(square);
    r += dr[i];
    f += df[i];
    while (true) {
      int new_square = MakeSquare(r, f);
      r += dr[i];
      f += df[i];
      if (!SquareOnBoard(r, f)) {
        break;
      }
      rel_occ |= OneHot(new_square);
    }
  }
  *out_rel_occ = rel_occ;
  std::mt19937_64 mt;
  const auto subsets = AllSubsets(rel_occ);
  const int output_size = 1ull << output_logsize;
  std::vector<uint64_t> masks;
  masks.reserve(subsets.size());
  assert(subsets.size() <= output_size);
  for (uint64_t occ : subsets) {
    masks.push_back(GeneratePieceMoves(dr, df, square, occ));
  }
  uint64_t mul = 0;
  while (true) {
    // This should result in every 8th bit being set.
    mul = mt() & mt() & mt();
    std::fill(output, output + output_size, kUnsetSentinel);
    bool success = true;
    for (size_t i = 0; i < subsets.size(); ++i) {
      const uint64_t subset = subsets[i];
      // std::cerr << "subset:\n" << BitboardToString(subset);
      const uint64_t mask = masks[i];
      // std::cerr << "mask:\n" << BitboardToString(mask);
      const uint64_t x = (subset * mul) >> (64 - output_logsize);
      // std::cerr << "x: " << BitboardToString(subset);
      assert(x < output_size);
      if (output[x] != kUnsetSentinel && output[x] != mask) {
        success = false;
        break;
      }
      output[x] = mask;
    }
    if (success) {
      break;
    }
  }
  *out_mul = mul;
}

void InitializeMagicInternal() {
  // Knights.
  for (int r = 0; r < 8; ++r) {
    for (int f = 0; f < 8; ++f) {
      const int p = MakeSquare(r, f);
      uint64_t mask = 0;
      for (int dr : {1, 2}) {
        const int df = dr ^ 3;
        // Either can be positive or negative.
        for (int i = 0; i < 4; ++i) {
          int result_r = r + dr * ((i & 1) ? 1 : -1);
          int result_f = f + df * ((i & 2) ? 1 : -1);
          if (SquareOnBoard(result_r, result_f)) {
            mask |= (1ull << MakeSquare(result_r, result_f));
          }
        }
      }
      knight_masks[p] = mask;
    }
  }
  // Bishops.
  {
    std::cerr << "Generating bishops\n";
    const int dr[4] = {1, -1, 1, -1};
    const int df[4] = {1, 1, -1, -1};
    for (int s = 0; s < 64; ++s) {
      GenerateMagic(dr, df, s, &bishop_magics.rel_occ[s], &bishop_magics.mul[s],
                    bishop_magics.mask[s], bishop_magics.log_size());
    }
  }
  {
    std::cerr << "Generating rooks\n";
    const int dr[4] = {1, -1, 0, 0};
    const int df[4] = {0, 0, 1, -1};
    for (int s = 0; s < 64; ++s) {
      GenerateMagic(dr, df, s, &rook_magics.rel_occ[s], &rook_magics.mul[s],
                    rook_magics.mask[s], rook_magics.log_size());
    }
  }
  std::cerr << "Magics ready: " << sizeof(bishop_magics) << ", "
            << sizeof(rook_magics) << "\n";
}

}  // namespace

// Low-level move masks.
uint64_t KnightMoveMask(int square, uint64_t my, uint64_t opp) {
  return knight_masks[square] & (~my);
}

uint64_t BishopMoveMask(int square, uint64_t my, uint64_t opp) {
  const uint64_t occ = (my | opp) & bishop_magics.rel_occ[square];
  uint64_t x = (occ * bishop_magics.mul[square]) >> (64 - kBishopLogSize);
  assert(x < bishop_magics.mask_size());
  return bishop_magics.mask[square][x];
}

uint64_t RookMask(int square, uint64_t my, uint64_t opp) {
  const uint64_t occ = (my | opp) & rook_magics.rel_occ[square];
  uint64_t x = (occ * rook_magics.mul[square]) >> (64 - kRookLogSize);
  assert(x < rook_magics.mask_size());
  return rook_magics.mask[square][x];
}

void InitializeMagic() { std::call_once(generated, &InitializeMagicInternal); }

}  // namespace chess
