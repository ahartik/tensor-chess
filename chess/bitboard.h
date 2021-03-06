#ifndef _CHESS_BITBOARD_H_
#define _CHESS_BITBOARD_H_

#include <x86intrin.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>

#include "absl/strings/string_view.h"

namespace chess {

constexpr uint64_t kAllBits = ~(0ULL);

class BitIterator {
 public:
  using value_type = int;
  using pointer = const int*;
  using reference = const int&;
  using iterator_category = std::forward_iterator_tag;

  explicit BitIterator(uint64_t v) : x_(v) {
    if (x_ != 0) {
      bit_ = __builtin_ctzll(x_);
    }
  }
  BitIterator(const BitIterator& o) = default;
  BitIterator& operator=(const BitIterator& o) = default;

  bool operator==(const BitIterator& o) const { return x_ == o.x_; }
  bool operator!=(const BitIterator& o) const { return x_ != o.x_; }

  const int& operator*() const { return bit_; }
  const int* operator->() const { return &bit_; }

  BitIterator& operator++() {
    x_ ^= (1ull << bit_);
    if (x_ != 0) {
      bit_ = __builtin_ctzll(x_);
    }
    return *this;
  }

  // Post-increment.
  BitIterator operator++(int) {
    BitIterator ret = *this;
    x_ ^= (1ull << bit_);
    if (x_ != 0) {
      bit_ = __builtin_ctzll(x_);
    }
    return ret;
  }

 private:
  int bit_ = 0;
  uint64_t x_;
};

class BitRange {
 public:
  explicit BitRange(uint64_t x) : x_(x) {}
  BitIterator begin() const { return BitIterator(x_); }
  BitIterator end() const { return BitIterator(0); }

 private:
  uint64_t x_;
};

inline uint64_t OneHot(int p) { return 1ull << p; }
inline int GetFirstBit(uint64_t x) {
  assert(x != 0);
  return __builtin_ctzll(x);
}

inline bool BitIsSet(uint64_t x, int p) { return (x >> p) & 1; }

inline int PopCount(uint64_t x) { return __builtin_popcountll(x); }
// Returns the index of bit with 'rank'. Requires PopCount(x) > rank.

#if 0
int BitSelect(uint64_t x, int rank) {
  assert(PopCount(x) > rank);
  // https://stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int
  return _pdep_u64(1ULL << rank, x);
}
#endif

inline std::string BitboardToString(uint64_t b) {
  std::string s;
  s.reserve(64 + 10);
  for (int r = 7; r >= 0; --r) {
    for (int f = 0; f < 8; ++f) {
      if (BitIsSet(b, r * 8 + f)) {
        s.push_back('1');
      } else {
        s.push_back('0');
      }
    }
    s.push_back('\n');
  }
  return s;
}

inline uint64_t BitboardFromString(absl::string_view str) {
  uint64_t result = 0;
  int i = 0;
  for (char c : str) {
    if (c == '0' || c == '1') {
      if (c == '1') {
        result += 1ull << i;
      }
      ++i;
    }
  }
  if (i != 64) {
    std::cerr << "Invalid bitboard string: '" << str << "', i=" << i << "\n";
    abort();
  }
  return result;
}

inline constexpr uint64_t RankMask(int r) { return uint64_t{0xff} << (8 * r); }

inline constexpr uint64_t FileMask(int r) {
  constexpr uint64_t first_file = 0x0101010101010101;
  static_assert(first_file & (0xffull << (64 - 8)));
  return (first_file << r) | (first_file >> (64 - r));
}
}  // namespace chess
#endif
