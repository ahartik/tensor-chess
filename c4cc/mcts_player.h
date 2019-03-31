#ifndef _C4CC_MCTS_PLAYER_H_
#define _C4CC_MCTS_PLAYER_H_

#include <random>

#include "c4cc/board.h"
#include "c4cc/mcts.h"
#include "c4cc/model.h"
#include "c4cc/play_game.h"

namespace c4cc {

class MCTSPlayer : public Player {
 public:
  explicit MCTSPlayer(Model* model);
  ~MCTSPlayer() override {}

  const Board& board() const override { return mcts_->current_board(); }

  void SetBoard(const Board& b) override;
  int GetMove() override;
  void MakeMove(int move) override;

 private:
  Model* const model_;
  std::unique_ptr<MCTS> mcts_;
  std::mt19937 rand_;
};

}  // namespace c4cc

#endif
