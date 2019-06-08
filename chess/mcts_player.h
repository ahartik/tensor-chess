#ifndef _CHESS_MCTS_PLAYER_H_
#define _CHESS_MCTS_PLAYER_H_

#include <functional>
#include <memory>
#include <iostream>
#include <string>

#include "chess/board.h"
#include "chess/player.h"
#include "chess/prediction_queue.h"
#include "chess/mcts.h"
#include "chess/types.h"

namespace chess {

class MCTSPlayer : public Player {
 public:
  explicit MCTSPlayer(PredictionQueue* pq, int iters_per_move);

  //
  void Reset() override;

  //
  void Advance(const Move& m) override;

  // TODO: Move def to .cc
  Move GetMove() override;

 private:
  void RunIterations(int n);

  PredictionQueue* const queue_;
  const int iters_per_move_;
  std::unique_ptr<MCTS> mcts_;
  std::mt19937 rand_;
};

}  // namespace chess

#endif
