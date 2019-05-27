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
uint64_t king_masks[64];
#ifdef NDEBUG
// Faster and smaller, but takes longer to generate (about 1s).
constexpr int kBishopLogSize = 10;
constexpr int kRookLogSize = 12;
#else
constexpr int kBishopLogSize = 12;
constexpr int kRookLogSize = 14;
#endif

uint64_t push_masks[64][64];

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
    // Why do we do AND 2 times? That's for you to figure out ;)
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

int OnlySign(int x) {
  if (x == 0) {
    return 0;
  } else if (x < 0) {
    return -1;
  }
  return 1;
}

uint64_t GenPushMask(int f, int t) {
  if (f == t) {
    return 0;
  }
  if (f > t) {
    std::swap(f, t);
  }
  const int fr = SquareRank(f);
  const int ff = SquareFile(f);
  const int tr = SquareRank(t);
  const int tf = SquareFile(t);
  // std::cout << "p " << f << " " << t << "\n";

  int dr = tr - fr;
  int df = tf - ff;
  uint64_t m = 0;
  if (dr == 0) {
    assert(ff < tf);
    for (int f = ff + 1; f < tf; ++f) {
      m |= OneHot(MakeSquare(fr, f));
    }
  } else if (df == 0) {
    assert(fr < tr);
    for (int r = fr + 1; r < tr; ++r) {
      m |= OneHot(MakeSquare(r, ff));
    }
  } else {
    assert(fr < tr);
    // Not a rook move, check that deltas are equal.
    if (abs(dr) != abs(df)) {
      return 0;
    }
    dr = OnlySign(dr);
    df = OnlySign(df);
    int r = fr + dr;
    int f = ff + df;
    while (r != tr) {
      m |= OneHot(MakeSquare(r, f));
      r += dr;
      f += df;
    }
  }
  return m;
}


void InitializeMagicInternal() {
  // Push masks.
  for (int f = 0; f < 64; ++f) {
    for (int t = 0; t < 64; ++t) {
      push_masks[f][t] = GenPushMask(f, t);
    }
  }

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
  // King
  for (int r = 0; r < 8; ++r) {
    for (int f = 0; f < 8; ++f) {
      const int p = MakeSquare(r, f);
      uint64_t mask = 0;
      for (int dr : {1, 0, -1}) {
        for (int df : {1, 0, -1}) {
          if (df == 0 && dr == 0) {
            continue;
          }
          int result_r = r + dr;
          int result_f = f + df;
          if (SquareOnBoard(result_r, result_f)) {
            mask |= (1ull << MakeSquare(result_r, result_f));
          }
        }
      }
      king_masks[p] = mask;
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

uint64_t KnightMoveMask(int square) {
  return knight_masks[square];
}

uint64_t KingMoveMask(int square) {
  return king_masks[square];
}

uint64_t BishopMoveMask(int square, uint64_t occ) {
  occ &= bishop_magics.rel_occ[square];
  uint64_t x = (occ * bishop_magics.mul[square]) >> (64 - kBishopLogSize);
  assert(x < bishop_magics.mask_size());
  return bishop_magics.mask[square][x];
}

uint64_t RookMoveMask(int square, uint64_t occ) {
  occ &= rook_magics.rel_occ[square];
  uint64_t x = (occ * rook_magics.mul[square]) >> (64 - kRookLogSize);
  assert(x < rook_magics.mask_size());
  return rook_magics.mask[square][x];
}

uint64_t PushMask(int from, int to) {
  return push_masks[from][to];
}

void InitializeMagic() { std::call_once(generated, &InitializeMagicInternal); }

}  // namespace chess
