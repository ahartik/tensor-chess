#include "c4cc/negamax.h"
#include "c4cc/play_game.h"
#include "c4cc/human_player.h"
#include "util/init.h"

#include <iostream>

namespace c4cc {
namespace {

const int kDepth = 6;

void Play() {
  std::cout << "Start game!\n";

  const Board result = PlayGame(&HumanPickMove, &AiPickMove);
  // const Board result = PlayGame(&AiPickMove, &HumanPickMove);
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
