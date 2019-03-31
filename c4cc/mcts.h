#ifndef _C4CC_MCTS_H_
#define _C4CC_MCTS_H_

#include <functional>
#include <memory>
#include <random>

#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"
#include "c4cc/board.h"

namespace c4cc {

// MCTS internals, not to be used by callers directly.
namespace mcts {

struct State;
struct Action;
}  // namespace mcts

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
  Prediction GetPrediction() const;

  // Advances the current state with a move.
  void MakeMove(int a);

 private:
  Board current_;
  std::shared_ptr<mcts::State> root_;

  struct ActionRef {
    ActionRef(mcts::State* ss, mcts::Action* aa) : s(ss), a(aa) {}
    mcts::State* s;
    mcts::Action* a;
  };
  struct PredictionRequest {
    // What board we want to get inspected.
    Board board;
    std::vector<ActionRef> picked_path;
    std::shared_ptr<mcts::State> parent;
    // The action from 'parent' leading to this board.
    int parent_a = -1;
  };
  absl::optional<PredictionRequest> request_;

  // TODO: It might be possible to optimize this memory-wise.
  absl::node_hash_map<Board, std::shared_ptr<mcts::State>> visited_states_;
  std::mt19937 rand_;
};

}  // namespace c4cc

#endif
