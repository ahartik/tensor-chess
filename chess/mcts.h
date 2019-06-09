#ifndef _C4CC_MCTS_H_
#define _C4CC_MCTS_H_

#include <functional>
#include <memory>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"
#include "chess/board.h"

namespace chess {

// MCTS internals, not to be used by callers directly.
namespace mcts {
struct State;
struct Action;
struct ActionRef {
  ActionRef(mcts::State* ss, mcts::Action* aa) : s(ss), a(aa) {}
  mcts::State* s;
  mcts::Action* a;
};

struct StateHasher {
  using is_transparent = void;
  size_t operator()(const State&) const;
  size_t operator()(BoardFP) const;
};

struct StateEquals {
  using is_transparent = void;

  bool operator()(const State&, const State&) const;
  bool operator()(const BoardFP&, const State&) const;
  bool operator()(const State&, const BoardFP&) const;
  bool operator()(const BoardFP&, const BoardFP&) const;
};

}  // namespace mcts

class MCTS {
 public:
  explicit MCTS(const Board& start = {});
  ~MCTS();

  // Resets position to 'b' with empty history.
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
    const MoveList& moves() const { return moves_; }

   private:
    using PathVec = absl::InlinedVector<mcts::ActionRef, 8>;

    // Only MCTS can construct this.
    PredictionRequest() {}
    // What board we want to get inspected.
    Board board_;
    PathVec picked_path_;
    mcts::State* parent_;
    MoveList moves_;
    // The action from 'parent' leading to this board.
    mcts::Action* parent_a_ = nullptr;

    friend class MCTS;
  };
  std::unique_ptr<PredictionRequest> StartIteration();

  // Given predictions for the position previously returned by StartIteration(),
  // adds the new node to the tree and propagates new weights.
  void FinishIteration(std::unique_ptr<PredictionRequest> req,
                       const PredictionResult& p);

  // Number of times an iteration has been completed from the current node.
  int num_iterations() const;

  // This is the prediction for the current board.
  // Must not be called if there are outstanding prediction requests.
  PredictionResult GetPrediction() const;
  // Returns the prior prediction for the current (root) position.
  PredictionResult GetPrior() const;

  // Advances the current state with a move.
  void MakeMove(Move m);

 private:
  Board current_;
  mcts::State* root_;

  absl::flat_hash_map<uint64_t, int> visited_;
  // TODO: Optimize this memory-wise.
  // TODO: Probably no need for shared_ptr here.
  absl::flat_hash_map<BoardFP, std::unique_ptr<mcts::State>> visited_states_;
  std::mt19937 rand_;
};

}  // namespace chess

#endif
