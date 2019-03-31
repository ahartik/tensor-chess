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
  explicit MCTSPlayer(Model* model, int iters);
  ~MCTSPlayer() override {}

  const Board& board() const override { return mcts_->current_board(); }

  void SetBoard(const Board& b) override;
  int GetMove() override;
  void MakeMove(int move) override;

  Prediction GetPrediction();

 private:
  void RunIterations(int n);

  Model* const model_;
  int ply_ = 0;
  const int iters_per_move_;
  Prediction current_pred_;
  bool pred_ready_;
  std::unique_ptr<MCTS> mcts_;
  std::mt19937 rand_;
};

}  // namespace c4cc

#endif
