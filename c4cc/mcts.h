#ifndef _C4CC_MCTS_H_
#define _C4CC_MCTS_H_

#include <functional>
#include <memory>

#include "c4cc/board.h"

namespace c4cc {

struct Prediction {
  float move_p[7] = {1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7,
                     1.0 / 7, 1.0 / 7, 1.0 / 7};
  float value = 0.0;
};

struct MCTSState;

class MCTS {
 public:
  explicit MCTS(const Board& start = {});
  ~MCTS();
  const Board& current_board() const;
  // Iteration is split in two steps: StartIteration() and FinishIteration().
  //
  // Finds a new leaf node to explore and returns it. FinishIteration() must be
  // called with predictions for this position to complete the iteration. New
  // iteration must not be started before calling FinishIteration().
  //
  // May return null in case no new leaf node was found (i.e. we hit a terminal
  // node). In that case FinishIteration() must not be called.
  //
  // Caller doesn't take ownership.
  const Board* StartIteration();
  // Given predictions for the position previously returned by StartIteration(),
  // adds the new node to the tree and propagates new weights.
  void FinishIteration(const Prediction& p);

  // Number of times an iteration has been completed.
  int num_iterations() const;

  // This is the prediction for the current board
  Prediction GetMCTSPrediction();

  // Advances the current state with a move.
  void MakeMove(int a);

 private:
  Board current_;
  std::unique_ptr<MCTSState> root_;

  void BubbleCounts(MCTSState* s);

  Board next_prediction_;
  MCTSState* next_parent_;
  int next_a_;
};

}  // namespace c4cc

#endif
