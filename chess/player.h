#ifndef _CHESS_PLAYER_H_
#define _CHESS_PLAYER_H_

#include <functional>
#include <iostream>
#include <string>

#include "chess/board.h"
#include "chess/types.h"

namespace chess {

// This class is thread-compatible (but not thread-safe).
class Player {
 public:
  virtual ~Player() {}

  // Modify board state.
  virtual void Reset() = 0;
  virtual void Advance(const Move& m) = 0;

  //
  virtual Move GetMove() = 0;
};

// "Human" player using commandline input.
class CliHumanPlayer : public Player {
 public:
  // Modify board state.
  void Reset() override { b_ = Board(); }

  void Advance(const Move& m) override { b_ = Board(b_, m); }

  // TODO: Move def to .cc
  Move GetMove() override {
    auto valid_moves = b_.valid_moves();
    if (valid_moves.empty()) {
      std::cerr << "Invalid call to GetMove(), no moves available in state "
                << b_.ToFEN();
      abort();
    }
    std::cout << b_.ToPrintString() << "Enter move for " << b_.turn() << ":\n";
    while (true) {
      std::string move_s;
      std::cin >> move_s;
      for (const Move& m : valid_moves) {
        if (m.ToString() == move_s) {
          return m;
        }
      }
      std::cout << "Invalid move '" << move_s << "',  valid moves: ";
      for (const Move& m : valid_moves) {
        std::cout << m.ToString() << " ";
      }
      std::cout << "\n";
      std::cout << "num moves: " << valid_moves.size() << "\n";
    }
  }

 private:
  Board b_;
};

}  // namespace chess

#endif
