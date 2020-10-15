#ifndef _CHESS_MCTS_PLAYER_H_
#define _CHESS_MCTS_PLAYER_H_

#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include "chess/board.h"
#include "chess/player.h"
#include "chess/types.h"
#include "generic/mcts.h"
#include "generic/prediction_queue.h"

namespace chess {

class MCTSPlayer : public Player {
 public:
  explicit MCTSPlayer(generic::PredictionQueue* pq, int iters_per_move);

  //
  void Reset(const Board& b) override;

  //
  void Advance(const Move& m) override;

  Move GetMove() override;

  struct SavedPrediction {
    Board board;
    generic::PredictionResult pred;
  };
  std::vector<SavedPrediction> saved_predictions() const {
    auto saved = saved_predictions_;
    if (!saved.empty() && saved.back().pred.policy.empty()) {
      saved.pop_back();
    }
    return saved_predictions_;
  }

 private:
  void RunIterations(int n);

  Board board_;
  generic::PredictionQueue* const queue_;
  const int iters_per_move_;
  std::vector<SavedPrediction> saved_predictions_;
  std::unique_ptr<generic::MCTS> mcts_;
  std::mt19937 rand_;
};

}  // namespace chess

#endif
