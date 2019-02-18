#include "c4cc/negamax.h"
#include "c4cc/play_game.h"
#include "util/init.h"

#include <iostream>

namespace c4cc {
namespace {

const int kDepth = 6;

void PrintBoard(const Board& b) {
  const char kOne[] = "\u001b[31;1m X \u001b[0m";
  const char kTwo[] = "\u001b[36;1m O \u001b[0m";
  PrintBoard(std::cout, b, kOne, kTwo);
}

int HumanPickMove(const Board& b) {
  PrintBoard(b);
  while (true) {
    std::cout << "Enter move:\n";
    int x = -1;
    std::cin >> x;
    for (const int m : b.valid_moves()) {
      if (m == x) {
        return x;
      }
    }
    std::cout << "Invalid move: " << x << "\n";
  }
}

void Play() {
  std::cout << "Start game!\n";
  const Board result = PlayGame(&HumanPickMove, [](const Board& b) -> int {
    auto r = Negamax(b, kDepth);
#if 0
    std::cout << "AI ponder for\n";
    std::cout << b << "\n";
#endif
    std::cout << "eval for AI: " << r.eval << "\n";
    std::cout << "move: " << r.best_move << "\n";
    return r.best_move;
  });
  PrintBoard(result);
  switch (result.result()) {
    case Color::kEmpty:
      std::cout << "Draw!\n";
      break;
    case Color::kOne:
      std::cout << "X won!\n";
      break;
    case Color::kTwo:
      std::cout << "O won!\n";
      break;
  }
}

}  // namespace
}  // namespace c4cc

int main(int argc, const char** argv) {
  NiceInit(argc, argv);
  c4cc::Play();
  return 0;
}
