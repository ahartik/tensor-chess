#ifndef _C4CC_MCTS_H_
#define _C4CC_MCTS_H_

#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "c4cc/board.h"

namespace c4cc {

namespace mcts {

struct State;
struct Action;
}  // namespace mcts

struct Prediction {
  double move_p[7] = {1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7,
                      1.0 / 7, 1.0 / 7, 1.0 / 7};
  double value = 0.0;
};

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

  // This is the prediction for the current board. Requires num_iterations > 1.
  Prediction GetMCTSPrediction() const;

  // Advances the current state with a move.
  void MakeMove(int a);

 private:
  Board current_;
  std::shared_ptr<mcts::State> root_;

  int num_iterations_ = 0;

  struct PredictionRequest {
    std::vector<mcts::Action*> picked_path;
    std::shared_ptr<mcts::State> next_parent;
    Board next_board;
    int next_a = -1;
  };
  absl::optional<PredictionRequest> request_;

  // TODO: It might be possible to optimize this memory-wise.
  absl::flat_hash_map<Board, std::weak_ptr<mcts::State>> visited_states_;
};

}  // namespace c4cc

#endif
