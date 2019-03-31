#include "c4cc/play_game.h"

#include <iostream>

namespace c4cc {

std::pair<Board, std::vector<int>> PlayGame(Player* player1, Player* player2) {
  std::vector<int> moves;
  Board b;
  player1->SetBoard(b);
  player2->SetBoard(b);
  while (!b.is_over()) {
    int m = -1;
    switch (b.turn()) {
      case Color::kOne:
        m = player1->GetMove();
        break;
      case Color::kTwo:
        m = player2->GetMove();
        break;
      default:
        std::cerr << "Impossible location";
        abort();
    }
    moves.push_back(m);

    b.MakeMove(m);
    player1->PlayMove(m);
    player2->PlayMove(m);
  }
  return {b, moves};
}

}  // namespace c4cc
