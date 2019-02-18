#include "c4cc/board.h"

#include <cstdint>

namespace c4cc {

namespace {
const int kDX[4] = {0, 1, 1, 1};
const int kDY[4] = {1, 0, 1, -1};
uint8_t kStartX[4][6][7];
uint8_t kStartY[4][6][7];

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
      while (x > 1 && y > 1) {
        --x;
        --y;
      }
      assert(x == 0 || y == 0);
      kStartX[2][i][j] = x;
      kStartY[2][i][j] = y;

      // Downward (direction kDX/kDY[3]):
      x = i;
      y = j;
      while (x > 1 && y < 5) {
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
}

void Board::MakeMove(int move_x) {
  assert(!is_over());
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
    MoveList new_list;
    for (int i = 0; i < valid_moves_.size(); ++i) {
      if (valid_moves_[i] != move_x) {
        new_list.push_back(valid_moves_[i]);
      }
    }
    valid_moves_ = new_list;
    if (!is_over_ && valid_moves_.size() == 0) {
      // Draw, no more moves left.
      is_over_ = true;
      result_ = Color::kEmpty;
    }
  }
  turn_ = OtherColor(turn_);
}

}  // namespace c4cc
