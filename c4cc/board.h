#ifndef _C4CC_BOARD_H_
#define _C4CC_BOARD_H_

#include <cassert>
#include <cstdint>

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
  kOne,
  kTwo,
};

inline Color OtherColor(Color c) {
  assert(c != Color::kEmpty);
  return c == Color::kOne ? Color::kTwo : Color::kOne;
}

class Board {
 public:
  Board();

  Color turn() const { return turn_; }

  const MoveList& valid_moves() const { return valid_moves_; }

  bool is_over() const { return is_over_; }
  // 1 if kOne wins, -1 if kTwo wins, 0 for draw.
  Color result() const {
    assert(is_over());
    return result_;
  }

  void MakeMove(int move_x);
  void UndoMove(int move_x);

  Color color(int x, int y) const { return board_[x][y]; }
  const std::vector<uint8_t>& history() const { return moves_; }

  static constexpr int kNumDirs = 4;
  static int dx(int dir);
  static int dy(int dir);
  static std::pair<int, int> start_pos(int dir, int x, int y);
  static const std::vector<std::pair<int, int>>& start_pos_list(int dir);

 private:
  void RedoMoves();

  Color turn_ = Color::kOne;
  MoveList valid_moves_;
  Color board_[7][6] = {};
  bool is_over_ = false;
  Color result_ = Color::kEmpty;
  std::vector<uint8_t> moves_;
};

void PrintBoard(std::ostream& out, const Board& b, const char* one,
                const char* two);

inline std::ostream& operator<<(std::ostream& out, const Board& b) {
  PrintBoard(out, b, " X ", " O ");
  return out;
}

}  // namespace c4cc

#endif