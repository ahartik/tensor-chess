#include "c4cc/board.h"

#include <iostream>
#include <cstdint>

namespace c4cc {

namespace {
const int kDX[4] = {0, 1, 1, 1};
const int kDY[4] = {1, 0, 1, -1};
uint8_t kStartX[4][7][6];
uint8_t kStartY[4][7][6];

bool InitializeStartXY() {
  // Vertical and horizontal.
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 6; ++j) {
      // Vertical
      kStartX[0][i][j] = i;
      kStartY[0][i][j] = 0;
      // Horizontal
      kStartX[1][i][j] = 0;
      kStartY[1][i][j] = j;

      // Diagonals: Go back until we hit the edge.

      // Upward (direction kDX/kDY[2]):
      int x = i;
      int y = j;
      while (x > 0 && y > 0) {
        --x;
        --y;
      }
      assert(x == 0 || y == 0);
      kStartX[2][i][j] = x;
      kStartY[2][i][j] = y;

      // Downward (direction kDX/kDY[3]):
      x = i;
      y = j;
      while (x > 0 && y < 5) {
        --x;
        ++y;
      }
      assert(x == 0 || y == 5);
      kStartX[3][i][j] = x;
      kStartY[3][i][j] = y;
    }
  }
  return false;
}
bool dummy = InitializeStartXY();

}  // namespace

Board::Board() {
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 6; ++j) {
      board_[i][j] = Color::kEmpty;
    }
  }
  RedoMoves();
}

void Board::MakeMove(int move_x) {
  assert(!is_over());
  moves_.push_back(move_x);
  const Color t = turn();
  int move_y;
  for (move_y = 0; move_y < 6; ++move_y) {
    if (board_[move_x][move_y] == Color::kEmpty) {
      board_[move_x][move_y] = t;
      break;
    }
  }
  assert(move_y < 6);
  // Check for wins here
  // TODO: do bit magic instead if this is slow.
  for (int d = 0; d < 4; ++d) {
    int x = kStartX[d][move_x][move_y];
    int y = kStartY[d][move_x][move_y];
    int count = 0;
    while (x < 7 && y >= 0 && y < 6) {
      if (board_[x][y] == t) {
        ++count;
        if (count == 4) {
          is_over_ = true;
          result_ = t;
          d = 4;
          break;
        }
      } else {
        count = 0;
      }
      x += kDX[d];
      y += kDY[d];
    }
  }
  if (move_y == 5) {
    // Last move for this column, update the list of valid moves.
    RedoMoves();
    if (!is_over_ && valid_moves_.size() == 0) {
      // Draw, no more moves left.
      is_over_ = true;
      result_ = Color::kEmpty;
    }
  }
  turn_ = OtherColor(turn_);
}

void Board::UndoMove(int move_x) {
  is_over_ = false;
  moves_.pop_back();
  turn_ = OtherColor(turn_);
  for (int move_y = 5; move_y >= 0; --move_y) {
    if (board_[move_x][move_y] != Color::kEmpty) {
      assert(board_[move_x][move_y] == turn_);
      board_[move_x][move_y] = Color::kEmpty;
      if (move_y == 5) {
        RedoMoves();
      }
      return;
    }
  }
  assert(false);
}

void Board::RedoMoves() {
  // Last move for this column, update the list of valid moves.
  MoveList new_list;
  static const int kBestOrder[] = {3, 2, 4, 1, 5, 0, 6};
  for (int x : kBestOrder) {
    if (board_[x][5] == Color::kEmpty) {
      new_list.push_back(x);
    }
  }
  valid_moves_ = new_list;
}

// static
int Board::dx(int dir) { return kDX[dir]; }
// static
int Board::dy(int dir) { return kDY[dir]; }
// static
std::pair<int, int> Board::start_pos(int dir, int x, int y) {
  return {kStartX[dir][x][y], kStartY[dir][x][y]};
}

// static
const std::vector<std::pair<int, int>>& Board::start_pos_list(int dir) {
  static const auto make_vec = [](int dir) {
    std::vector<std::pair<int, int>> v;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (kStartX[dir][i][j] == i && kStartY[dir][i][j] == j) {
          v.push_back({i, j});
        }
      }
    }
    return v;
  };
  static const std::vector<std::pair<int, int>> kVecs[4] = {
      make_vec(0),
      make_vec(1),
      make_vec(2),
      make_vec(3),
  };
  return kVecs[dir];
}

void PrintBoard(std::ostream& out, const Board& b, const char* one,
                const char* two) {
  out << "+---+---+---+---+---+---+---+\n";
  for (int y = 5; y >= 0; --y) {
    for (int x = 0; x < 7; ++x) {
      Color c = b.color(x, y);
      switch (c) {
        case Color::kEmpty:
          out << "|   ";
          break;
        case Color::kOne:
          out << "|" << one;
          break;
        case Color::kTwo:
          out << "|" << two;
          break;
      }
    }
    out << "|\n";
    out << "+---+---+---+---+---+---+---+\n";
  }
  out << "  0   1   2   3   4   5   6 \n";
}

}  // namespace c4cc
