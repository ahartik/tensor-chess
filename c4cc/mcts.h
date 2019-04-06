#ifndef _C4CC_MCTS_H_
#define _C4CC_MCTS_H_

#include <functional>
#include <memory>
#include <random>

#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"
#include "c4cc/board.h"

namespace c4cc {

// MCTS internals, not to be used by callers directly.
namespace mcts {
struct State;
struct Action;
struct ActionRef {
  ActionRef(mcts::State* ss, mcts::Action* aa) : s(ss), a(aa) {}
  mcts::State* s;
  mcts::Action* a;
};
}  // namespace mcts

class MCTS {
 public:
  explicit MCTS(const Board& start = {});
  ~MCTS();

  // Resets position to 'b'.
  void SetBoard(const Board& b);

  const Board& current_board() const;
  // Iteration is split in two steps: StartIteration() and FinishIteration().
  //
  // Finds a new leaf node to explore and returns a prediction request.
  // FinishIteration() must be called with predictions for this position to
  // complete the iteration. New iteration must not be started before calling
  // FinishIteration().
  //
  // May return null in case no new leaf node was found (i.e. we hit a terminal
  // node). In that case FinishIteration() must not be called.
  class PredictionRequest {
   public:
    PredictionRequest(const PredictionRequest&) = delete;
    PredictionRequest& operator=(const PredictionRequest&) = delete;
    const Board& board() const { return board_; }

   private:
    using PathVec = absl::InlinedVector<mcts::ActionRef, 8>;

    // Only MCTS can construct this.
    PredictionRequest() {}
    // What board we want to get inspected.
    Board board_;
    PathVec picked_path_;
    std::shared_ptr<mcts::State> parent_;
    // The action from 'parent' leading to this board.
    int parent_a_ = -1;

    friend class MCTS;
  };
  std::unique_ptr<PredictionRequest> StartIteration();

  // Given predictions for the position previously returned by StartIteration(),
  // adds the new node to the tree and propagates new weights.
  void FinishIteration(std::unique_ptr<PredictionRequest> req,
                       const Prediction& p);

  // Number of times an iteration has been completed.
  int num_iterations() const;

  // This is the prediction for the current board.
  // Must not be called if there are outstanding
  Prediction GetPrediction() const;
  // Returns the prior prediction for the current (root) position.
  Prediction GetPrior() const;
  Prediction GetChildValues() const;

  // Advances the current state with a move.
  void MakeMove(int a);

 private:
  Board current_;
  std::shared_ptr<mcts::State> root_;

  // TODO: Optimize this memory-wise.
  absl::node_hash_map<Board, std::shared_ptr<mcts::State>> visited_states_;
  std::mt19937 rand_;
};

}  // namespace c4cc

#endif
