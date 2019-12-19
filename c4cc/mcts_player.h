#ifndef _C4CC_MCTS_PLAYER_H_
#define _C4CC_MCTS_PLAYER_H_

#include <random>

#include "c4cc/board.h"
#include "c4cc/mcts.h"
#include "c4cc/play_game.h"
#include "c4cc/prediction_queue.h"

namespace c4cc {

using PredictionCache = absl::node_hash_map<Board, Prediction>;

class MCTSPlayer : public Player {
 public:
  explicit MCTSPlayer(PredictionQueue* queue, int iters,
                      PredictionCache* cache = nullptr, bool hard = false);
  ~MCTSPlayer() override {}

  const Board& board() const override { return mcts_->current_board(); }

  void SetBoard(const Board& b) override;
  int GetMove() override;
  void MakeMove(int move) override;

  Prediction GetPrediction();
  void Reset();

  void LogStats();

 private:
  void RunIterations(int n);

  PredictionQueue* const queue_;
  PredictionCache* const pred_cache_;
  const bool hard_;

  const int iters_per_move_;
  Prediction current_pred_;
  bool pred_ready_;
  std::unique_ptr<MCTS> mcts_;
  std::mt19937 rand_;
};

}  // namespace c4cc

#endif
