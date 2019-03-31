#include "c4cc/human_player.h"

namespace c4cc {

namespace {
void PrintBoard(const Board& b) {
  const char kOne[] = "\u001b[31;1m X \u001b[0m";
  const char kTwo[] = "\u001b[36;1m O \u001b[0m";
  PrintBoard(std::cout, b, kOne, kTwo);
}

}  // namespace

int HumanPlayer::GetMove() {
  assert(!current_board_.is_over());
  PrintBoard(current_board_);
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
