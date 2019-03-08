#include "c4cc/play_game.h"

#include <iostream>

namespace c4cc {

std::pair<Board, std::vector<int>> PlayGame(
    const std::function<int(const Board&)>& move_picker_1,
    const std::function<int(const Board&)>& move_picker_2) {
  std::vector<int> moves;
  Board b;
  while (!b.is_over()) {
    int m = -1;
    switch (b.turn()) {
      case Color::kOne:
        m = move_picker_1(b);
        break;
      case Color::kTwo:
        m = move_picker_2(b);
        break;
      default:
        std::cerr << "Impossible location";
        abort();
    }
    moves.push_back(m);
    b.MakeMove(m);
  }
  return {b, moves};
}

}  // namespace c4cc
