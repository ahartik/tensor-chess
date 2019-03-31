#ifndef _C4CC_BOARD_H_
#define _C4CC_BOARD_H_

#include <cassert>
#include <cstring>
#include <cstdint>

#include <iostream>
#include <ostream>
#include <vector>
#include <utility>

namespace c4cc {

class MoveList {
 public:
  MoveList() : size_(0) {}
  MoveList(const MoveList& o) = default;
  MoveList& operator=(const MoveList& o) = default;

  int operator[](int idx) const { return moves_[idx]; }
  int size() const { return size_; }

  const uint8_t* begin() const { return &moves_[0]; }
  const uint8_t* end() const { return begin() + size(); }

  void push_back(int x) {
    assert(x < 7);
    assert(x >= 0);
    assert(size_ < 7);
    moves_[size_] = x;
    ++size_;
  }

 private:
  uint8_t size_;
  uint8_t moves_[7];
};

enum class Color : uint8_t {
  kEmpty = 0,
  kOne = 1,
  kTwo = 2,
};

inline Color OtherColor(Color c) {
  assert(c != Color::kEmpty);
  return c == Color::kOne ? Color::kTwo : Color::kOne;
}

class Board {
 public:
  Board();
  Board(const Board& b) = default;
  Board& operator=(const Board& b) = default;

  bool operator==(const Board& o) const {
    // Turn, result, and valid moves are functions of board data.
    return memcmp(data_, o.data_, sizeof(data_)) == 0;
  }

  Color turn() const { return turn_; }

  MoveList valid_moves() const;

  bool is_over() const { return is_over_; }
  // 1 if kOne wins, -1 if kTwo wins, 0 for draw.
  Color result() const {
    assert(is_over());
    return result_;
  }

  void MakeMove(int move_x);
  void UndoMove(int move_x);

  Color color(int x, int y) const {
    unsigned i = x * 6 + y;
    int byte = i / 4;
    int offset = 2 * (i % 4);
    const int val = (data_[byte] >> offset) & 3;
#ifndef NDEBUG
    if (val == 3 || byte > 10) {
      std::cerr << "Invalid bytes: i=" << i << " val=" << val << "\n";
      abort();
    }
#endif
    return static_cast<Color>(val);
  }

  static constexpr int kNumDirs = 4;
  static int dx(int dir);
  static int dy(int dir);
  static std::pair<int, int> start_pos(int dir, int x, int y);
  static const std::vector<std::pair<int, int>>& start_pos_list(int dir);

 private:
  template <typename H>
  friend H AbslHashValue(H h, const Board& b);

  void RedoMoves();

  void SetColor(int x, int y, Color c) {
    unsigned i = x * 6 + y;
    int byte = i / 4;
    int offset = 2 * (i % 4);
    assert(byte < 11);
    // Clear:
    data_[byte] &= ~(3 << offset);
    // Set:
    data_[byte] |= (static_cast<uint8_t>(c) << offset);
  }

  // 2 * 42 = 84 / 8 = 10.5.
  uint8_t data_[11] = {};
  bool is_over_ = false;
  uint8_t valid_moves_;
  Color turn_ = Color::kOne;
  Color result_ = Color::kEmpty;
};

void PrintBoard(std::ostream& out, const Board& b, const char* one,
                const char* two);

inline std::ostream& operator<<(std::ostream& out, const Board& b) {
  PrintBoard(out, b, " X ", " O ");
  return out;
}

template <typename H>
H AbslHashValue(H h, const Board& b) {
  // Turn, result, and valid moves are functions of board data.
  return H::combine_contiguous(std::move(h), b.data_, 11);
}

}  // namespace c4cc

#endif
