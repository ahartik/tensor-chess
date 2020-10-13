#include <iostream>
#include <cstdlib>
#include <string>

#include "chess/board.h"
#include "absl/strings/string_view.h"

namespace chess {

void Go() {
  while (true) {
    std::string input_fen;
    if (!std::getline(std::cin, input_fen)) {
      break;
    }
    std::string move_str;
    if (!std::getline(std::cin, move_str)) {
      break;
    }
    std::string result_fen;
    if (!std::getline(std::cin, result_fen)) {
      break;
    }

    Board input(input_fen);

    Move parsed_move;
    bool found = false;
    const auto moves = input.valid_moves();
    for (const Move m : moves) {
      if (m.ToString() == move_str) {
        parsed_move = m;
        found = true;
      }
    }
    if (!found) {
      std::cerr << "input:    " << input_fen << "\n";
      std::cerr << "invalid move \"" << move_str << "\"\n";
      for (const Move m : moves) {
        std::cerr << "  move " << m.ToString() << "\n";
      }
      abort();
    }

    Board result(input, parsed_move);
    // This simultaneously tests FEN encoding and decoding. Nice!
    if (result.ToFEN() != result_fen) {
      std::cerr << "input:    " << input_fen << "\n";
      std::cerr << "move:     " << move_str << "\n";
      std::cerr << "expected: " << result_fen << "\n";
      std::cerr << "actual:   " << result.ToFEN() << "\n";
      abort();
    }
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  chess::Go();
  return 0;
}
