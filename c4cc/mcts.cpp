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
  std::shared_ptr<State> state;

  void AddResult(double v);

};
using Actions = std::array<Action, 7>;

struct State {
  State(const Board& b, const Prediction& p) : board(b) {
    for (int i = 0; i < 7; ++i) {
      actions[i].prior = p.move_p[i];
    }
  }
  State(const State&) = delete;
  State& operator=(const State&) = delete;

  bool is_terminal() const { return board.is_over(); }
  double terminal_value() const {
    CHECK(is_terminal());
    if (board.result() == Color::kEmpty) {
      return 0.0;
    } else {
      return board.result() == board.turn() ? 1.0 : -1.0;
    }
  }

  const Board board;
  Action actions[7];
};

void Action::AddResult(double v) {
  ++num_taken;
  total_value += v;
  mean_value = total_value / num_taken;
  CHECK(state != nullptr);
  if (state != nullptr) {
    if (state->is_terminal()) {
      CHECK_EQ(v, -state->terminal_value());
    }
  }
}

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

const double kPUCT = 0.1;

double UCB(const State& s, int a) {
  const Action& action = s.actions[a];
  // TODO: Cache these values.
  int num_sum = 0;
  for (int i = 0; i < 7; ++i) {
    num_sum += s.actions[i].num_taken;
  }
  if (num_sum == 0) {
    // These numbers are only ever compared against other actions in the same
    // state.
    return action.prior;
  }
  // We start with a value strongly based on the prior, but as we know more we
  // shift to just using the mean value. This makes sense since in the
  // beginning the simulation results may be really really noisy, and the prior
  // is likely to be more accurate.
  return action.mean_value + kPUCT * action.prior * std::sqrt(num_sum) /
                                 (1.0 + s.actions[a].num_taken);
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
  CHECK(!cur->is_terminal());
  // TODO: Use InlinedVector
  std::vector<ActionRef> picked_path;
  while (true) {
    int best_a = 0;
    double best_score = -1e6;
    // (For virtual iterations, we can look up from a map in case an action
    // should be demoted. Maybe "map<Action*, int> virtual_actions_;")
    for (int a : cur->board.valid_moves()) {
      const double u = UCB(*cur, a);
      if (u > best_score) {
        best_score = u;
        best_a = a;
      }
    }
    CHECK_GT(best_score, -100);
    picked_path.emplace_back(cur.get(), &cur->actions[best_a]);
    if (cur->actions[best_a].num_taken == 0) {
      CHECK_EQ(cur->actions[best_a].state, nullptr);
      // We might have visited this node from another search branch:
      Board board = cur->board;
      board.MakeMove(best_a);
      // Terminal node, add this to graph and continue.

      auto transpose_it = visited_states_.find(board);
      std::shared_ptr<State> transposed;
      if (transpose_it != visited_states_.end()) {
        // transposed = transpose_it->second.lock();
        transposed = transpose_it->second;
      }
      if (!transposed && board.is_over()) {
        auto& visit = visited_states_[board];
        visit = std::make_shared<State>(board, Prediction());
        transposed = visit;
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
    if (cur->is_terminal()) {
      // LOG(INFO) << "path len: " << picked_path.size();
      // Terminal node, can't iterate.
      CHECK(!picked_path.empty());
      for (auto e : picked_path) {
        const double mul = e.s->board.turn() == cur->board.turn() ? 1.0 : -1.0;
        e.a->AddResult(mul * cur->terminal_value());
      }
      return nullptr;
    }
  }
}

void MCTS::FinishIteration(const Prediction& p) {
  CHECK(request_.has_value());
  const auto& req = request_.value();
  if (root_ == nullptr) {
    CHECK_EQ(req.board, current_);
    root_ = std::make_shared<State>(current_, p);
    LOG(INFO) << "Root prediction: " << p;
    // Shuffle root priors a little bit.
    for (int i = 0; i < 7; ++i) {
      root_->actions[i].prior = 1.0 / 7;
    }
  } else {
    // Initialize next state.
    auto state = std::make_shared<State>(req.board, p);
    auto& old_state = visited_states_[req.board];
    // We never request predictions for moves that are already visited and
    // found live in the map.
    // CHECK(old_state.expired());
    // CHECK(old_state == nullptr);
    old_state = state;

    CHECK(req.parent != nullptr);
    CHECK(req.parent->actions[req.parent_a].state == nullptr);
    req.parent->actions[req.parent_a].state = std::move(state);
    CHECK_EQ(req.picked_path.back().s, req.parent.get());
    for (auto e : req.picked_path) {
      const double mul = e.s->board.turn() == req.board.turn() ? 1.0 : -1.0;
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
      // Shuffle root priors a little bit.
      for (int i = 0; i < 7; ++i) {
        root_->actions[i].prior = 1.0 / 7;
      }
    }
  }
}

}  // namespace c4cc
