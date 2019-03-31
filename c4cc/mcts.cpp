#include "c4cc/mcts.h"

#include <array>
#include <cmath>

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

namespace mcts {

struct Action {
  // Prior and prior value don't change after construction.
  double prior = 0.0;

  int num_taken = 0;
  double total_value = 0;
  double mean_value = 0;

  void AddResult(double v) {
    ++num_taken;
    total_value += v;
    mean_value = total_value / num_taken;
  }

  std::shared_ptr<State> state;
};
using Actions = std::array<Action, 7>;

struct State {
  State(const Board& b, const Prediction& p) : board(b), prior_value(p.value) {
    for (int i = 0; i < 7; ++i) {
      actions[i].prior = p.move_p[i];
    }

    // Override predictions and everything if the game is over.
    if (b.is_over()) {
      if (b.result() == Color::kEmpty) {
        prior_value = 0.0;
      } else {
        prior_value = b.result() == b.turn() ? 1.0 : -1.0;
      }
    }
  }

  bool is_terminal() const { return board.is_over(); }

  const Board board;
  double prior_value = 0.0;  // "q"
  Action actions[7];
};

// Important points from AlphaGo paper:
//
// To pick multiple nodes at once, use "virtual loss": Continue search as if
// the picked move lost (v=-1). Perform updates later. We can implement this
// with the iteration model: allow starting multiple iterations at once, but
// must finish together.
//

}  // namespace mcts

namespace {

using mcts::Action;
using mcts::State;

const double kPUCT = 1.0;

double UCB(const State& s, int a) {
  const Action& action = s.actions[a];
  // Cache these values.
  int num_sum = 0;
  for (int i = 0; i < 7; ++i) {
    num_sum += s.actions[i].num_taken;
  }
  // We start with a value strongly based on the prior, but as we know more we
  // shift to just using the mean value. THis makes some sense as to begin,
  // results of the simulation may be really really noisy, and the prior is
  // likely to be more accurate.
  return action.mean_value +
         kPUCT * action.prior * std::sqrt(num_sum) / (1.0 + num_sum);
}

}  // namespace

MCTS::MCTS(const Board& start) : current_(start) {}

MCTS::~MCTS() {}

const Board& MCTS::current_board() const { return current_; }

const Board* MCTS::StartIteration() {
  CHECK(!request_.has_value());
  if (root_ == nullptr) {
    return &current_;
  }
  ++num_iterations_;
  // TODO: Use some tricks to avoid keeping 'cur' a shared ptr.
  std::shared_ptr<State> cur = root_;
  // TODO: Use InlinedVector
  std::vector<mcts::Action*> picked_path;
  while (true) {
    if (cur->is_terminal()) {
      // Terminal node, can't iterate.
      for (auto* a : picked_path) {
        a->AddResult(cur->prior_value);
      }
      return nullptr;
    }
    int best_a = 0;
    double best_score = -1e6;
    // (For virtual iterations, we can look up from a map in case an action
    // should be demoted. Maybe "map<Action*, int> virtual_actions_;")
    for (int a : cur->board.valid_moves()) {
      double u = UCB(*cur, a);
      if (u > best_score) {
        best_score = u;
        best_a = a;
      }
    }
    if (cur->actions[best_a].num_taken == 0) {
      // We might have visited this node from another search branch:
      Board next_board = cur->board;
      next_board.MakeMove(best_a);
      auto transpose_it = visited_states_.find(next_board);
      if (transpose_it == visited_states_.end()) {
        // OK, it's a new node: must request a prediction.
        request_.emplace();
        request_->picked_path = std::move(picked_path);
        request_->next_parent = cur;
        request_->next_board = cur->board;
        request_->next_board.MakeMove(best_a);
        return &(request_->next_board);
      }
      // In the game of Connect 4, possible states form an acyclic graph, as
      // each move increments the total number of pieces in the board ("in" as
      // the Connect4 board is vertical :-).
      std::shared_ptr<State> transposed = transpose_it->second.lock();
      CHECK(transposed);
      cur->actions[best_a].state = transposed;
    }
    cur = cur->actions[best_a].state;
    CHECK(cur);
  }
}

void MCTS::FinishIteration(const Prediction& p) {
  CHECK(request_.has_value());
  if (root_ == nullptr) {
    root_ = std::make_shared<State>(current_, p);
  } else {
    const auto& req = request_.value();
    // Initialize next state.
    req.next_parent->actions[req.next_a].state =
        std::make_shared<State>(req.next_board, p);
    for (auto* action : req.picked_path) {
      action->AddResult(p.value);
    }
  }
  request_.reset();
}

int MCTS::num_iterations() const { return num_iterations_;}

Prediction MCTS::GetMCTSPrediction() const {
  Prediction res;
  if (root_ == nullptr) {
    for (int i = 0; i < 7; ++i) {
      res.move_p[i] = 1.0 / 7;
    }
    res.value = 0;
    return res;
  }
  int sum_n = 0;
  for (int i = 0; i < 7; ++i) {
    sum_n += root_->actions[i].num_taken;
  }
  CHECK(sum_n != 0);
  const double inv_sum = 1.0 / sum_n;
  res.value = 0;
  for (int i = 0; i < 7; ++i) {
    const int num = root_->actions[i].num_taken;
    res.move_p[i] = num * inv_sum;
    res.value += num * root_->actions[i].mean_value * inv_sum;
  }
  return res;
}

void MCTS::MakeMove(int a) { }

}  // namespace c4cc
