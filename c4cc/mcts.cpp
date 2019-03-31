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
  State(const State&) = delete;
  State& operator=(const State&) = delete;

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
    request_.emplace();
    request_->board = current_;
    return &(request_->board);
  }
  // TODO: Use some tricks to avoid keeping 'cur' a shared ptr.
  std::shared_ptr<State> cur = root_;
  // TODO: Use InlinedVector
  std::vector<ActionRef> picked_path;
  while (true) {
    if (cur->is_terminal()) {
      // Terminal node, can't iterate.
      for (auto e : picked_path) {
        const double mul = e.s->board.turn() == cur->board.turn() ? 1.0 : -1.0;
        e.a->AddResult(mul * cur->prior_value);
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
    picked_path.emplace_back(cur.get(), &cur->actions[best_a]);
    if (cur->actions[best_a].num_taken == 0) {
      CHECK_EQ(cur->actions[best_a].state, nullptr);
      // We might have visited this node from another search branch:
      Board board = cur->board;
      board.MakeMove(best_a);

      auto transpose_it = visited_states_.find(board);
      std::shared_ptr<State> transposed;
      if (transpose_it != visited_states_.end()) {
        transposed = transpose_it->second.lock();
      }
      if (!transposed) {
        // OK, it's a new node: must request a prediction.
        request_.emplace();
        request_->picked_path = std::move(picked_path);
        request_->parent = cur;
        request_->parent_a = best_a;
        request_->board = board;
        return &(request_->board);
      }
      CHECK(transposed != cur);
      cur->actions[best_a].state = transposed;
    }
    cur = cur->actions[best_a].state;
    CHECK(cur);
  }
}

void MCTS::FinishIteration(const Prediction& p) {
  CHECK(request_.has_value());
  const auto& req = request_.value();
  if (root_ == nullptr) {
    CHECK_EQ(req.board, current_);
    root_ = std::make_shared<State>(current_, p);
  } else {
    // Initialize next state.
    auto state = std::make_shared<State>(req.board, p);
    auto& old_state = visited_states_[req.board];
    // We never request predictions for moves that are already visited and
    // found live in the map.
    CHECK(old_state.expired());
    old_state = state;

    CHECK(req.parent != nullptr);
    CHECK(req.parent->actions[req.parent_a].state == nullptr);
    req.parent->actions[req.parent_a].state = std::move(state);
    CHECK_EQ(req.picked_path.back().s, req.parent.get());
    for (auto e : req.picked_path) {
      const double mul = e.s->board.turn() != req.board.turn() ? 1.0 : -1.0;
      e.a->AddResult(mul * p.value);
    }
  }
  request_.reset();
}

int MCTS::num_iterations() const {
  if (root_ == nullptr) {
    return 0;
  }
  int sum = 0;
  for (int i = 0; i < 7; ++i) {
    sum += root_->actions[i].num_taken;
  }
  return sum;
}

Prediction MCTS::GetPrediction() const {
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

void MCTS::MakeMove(int a) {
  current_.MakeMove(a);
  if (root_ != nullptr) {
    // This may cause multiple refcounts to be zero.
    root_ = root_->actions[a].state;
    if (root_ != nullptr) {
      CHECK_EQ(current_, root_->board);
    }
  }
}

}  // namespace c4cc
