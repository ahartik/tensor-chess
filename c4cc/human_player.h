#ifndef _C4CC_HUMAN_PLAYER_H_
#define _C4CC_HUMAN_PLAYER_H_

#include <functional>

#include "c4cc/board.h"
#include "c4cc/play_game.h"

namespace c4cc {

// This class is thread-compatible (but not thread-safe).
class HumanPlayer : public Player {
 public:
  // By default the player is at the start of the game as the first player.
  HumanPlayer() {}
  ~HumanPlayer() override {}

  // Player implementation.
  const Board& board() const override { return current_board_; }
  void SetBoard(const Board& b) override { current_board_ = b; }
  void MakeMove(int move) override { current_board_.MakeMove(move); }
  // See .cc
  int GetMove() override;

 private:
  Board current_board_;
};

}  // namespace c4cc

#endif
