#include "c4cc/human_player.h"

namespace c4cc {

int HumanPlayer::GetMove() {
  assert(!current_board_.is_over());
  PrintBoardWithColor(std::cout, current_board_);
  while (true) {
    std::cout << "Enter move:\n";
    int x = -1;
    std::cin >> x;
    for (const int m : current_board_.valid_moves()) {
      if (m == x) {
        return x;
      }
    }
    std::cout << "Invalid move: " << x << "\n";
  }
}

}  // namespace c4cc
