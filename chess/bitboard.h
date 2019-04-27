#ifndef _CHESS_BITBOARD_H_
#define _CHESS_BITBOARD_H_

#include <cstdint>
#include <iterator>

namespace chess {

using Bitboard = uint64_t;

constexpr Bitboard kAllSet = ~(0ULL);

class BitIterator {
 public:
  using value_type = int;
  using pointer = const int*;
  using reference = const int&;
  using iterator_category = std::forward_iterator_tag;

  explicit BitIterator(Bitboard v) : x_(v) {
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
  Bitboard x_;
};

class BitRange {
 public:
  explicit BitRange(Bitboard x) : x_(x) {}
  BitIterator begin() const { return BitIterator(x_); }
  BitIterator end() const { return BitIterator(0); }

 private:
  Bitboard x_;
};

}  // namespace chess

#endif
