#include "c4cc/mcts.h"

#include <cmath>

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

namespace mcts {

struct Action {
  // Prior and prior value don't change after construction.
  double prior = 0.0;

  int num_virtual = 0;
  int num_taken = 0;
  double total_value = 0;
  std::shared_ptr<State> state;

  void AddResult(double v);
};

struct State {
  State(const Board& b, const Prediction& p) : board(b) {
    // Give a positive prior to all moves, so that we still sometimes explore
    // them in case the prediction network gives a zero weight for the move.
    const double add = 0.05;
    const double new_total = 1.0 + 7 * add;
    for (int i = 0; i < 7; ++i) {
      actions[i].prior = (p.move_p[i] + add) / new_total;
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
  num_taken += 1;
  total_value += v;
  // mean_value = total_value / num_taken;
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
using mcts::ActionRef;
using mcts::State;

const double kPUCT = 1.0;

int PickAction(std::mt19937& rand, const State& s) {
  static const double rand_add = 0.01;
  std::uniform_real_distribution<double> rand_dist(0.0, rand_add);
  int num_sum = 0;
  for (int i = 0; i < 7; ++i) {
    num_sum += s.actions[i].num_taken + s.actions[i].num_virtual;
  }
  double best_score = -10000;
  int best_a = -1;
  for (int m : s.board.valid_moves()) {
    const auto& action = s.actions[m];
    double score;
    if (num_sum == 0) {
      score = action.prior;
    } else {
      const int num = action.num_taken + action.num_virtual;
      // Reminder: virtual moves are counted as losses for both players.
      double mean_value =
          num == 0 ? 0 : (action.total_value - action.num_virtual) / num;
      score =
          mean_value + kPUCT * action.prior * std::sqrt(num_sum) / (1.0 + num);
    }
    score += rand_dist(rand);
    if (score > best_score) {
      best_score = score;
      best_a = m;
    }
  }
  CHECK_GE(best_a, 0);
  return best_a;
}

}  // namespace

MCTS::MCTS(const Board& start) { SetBoard(start); }

MCTS::~MCTS() {}

const Board& MCTS::current_board() const { return current_; }

std::unique_ptr<MCTS::PredictionRequest> MCTS::StartIteration() {
  CHECK(root_ != nullptr);
  // TODO: Use some tricks to avoid keeping 'cur' a shared ptr.
  std::shared_ptr<State> cur = root_;
  CHECK(!cur->is_terminal());
  // TODO: Use InlinedVector
  PredictionRequest::PathVec picked_path;
  while (true) {
    const int best_a = PickAction(rand_, *cur);
    picked_path.emplace_back(cur.get(), &cur->actions[best_a]);
    if (cur->actions[best_a].state == nullptr) {
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

        // Increment virtual counts - this will be undone by Finish.
        for (auto e : picked_path) {
          ++e.a->num_virtual;
        }
        std::unique_ptr<PredictionRequest> request(new PredictionRequest());
        request->picked_path_ = std::move(picked_path);
        request->parent_ = cur;
        request->parent_a_ = best_a;
        request->board_ = board;
        return request;
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

void MCTS::FinishIteration(std::unique_ptr<PredictionRequest> req,
                           const Prediction& p) {
  CHECK(root_ != nullptr);

  CHECK(req->parent_ != nullptr);
  auto& action = req->parent_->actions[req->parent_a_];
  // It's possible that 'state' was already added by a previous call to
  // FinishIteration(). StartIteration() may return the same position twice in
  // case the first iteration has not yet been finished.
  if (action.state == nullptr) {
    // Initialize next state.
    auto state = std::make_shared<State>(req->board(), p);
    auto& old_state = visited_states_[req->board()];
    // We never request predictions for moves that are already visited and
    // found live in the map.
    // CHECK(old_state.expired());
    // CHECK(old_state == nullptr);
    old_state = state;

    action.state = std::move(state);
  }
  CHECK_EQ(req->picked_path_.back().s, req->parent_.get());
  for (auto e : req->picked_path_) {
    const double mul = e.s->board.turn() == req->board().turn() ? 1.0 : -1.0;
    // Bias all values towards the mean, so that actual terminal nodes have
    // more weight than strong prediction outputs. This hopefully makes us seek
    // winning terminal nodes and avoid losing ones harder.
    const double kUncertainty = 1.0;
    e.a->AddResult(mul * p.value * kUncertainty);
    --e.a->num_virtual;
  }
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
  CHECK(root_ != nullptr);
  int sum_n = 0;
  for (int i = 0; i < 7; ++i) {
    sum_n += root_->actions[i].num_taken;
    CHECK(root_->actions[i].num_virtual == 0) << "i=" << i;
  }
  CHECK(sum_n != 0);
  const double inv_sum = 1.0 / sum_n;
  res.value = 0;

  for (int i = 0; i < 7; ++i) {
    const int num = root_->actions[i].num_taken;
    res.move_p[i] = num * inv_sum;
    res.value += root_->actions[i].total_value * inv_sum;
  }

  return res;
}

Prediction MCTS::GetPrior() const {
  Prediction res;
  res.value = 0.0;
  for (int i = 0; i < 7; ++i) {
    res.move_p[i] = root_->actions[i].prior;
  }
  return res;
}

Prediction MCTS::GetChildValues() const {
  Prediction res;
  res.value = 0.0;
  for (int i = 0; i < 7; ++i) {
    const double child_value =
        root_->actions[i].total_value / root_->actions[i].num_taken;
    res.move_p[i] = child_value;
  }
  return res;
}

void MCTS::MakeMove(int a) {
  current_.MakeMove(a);
  root_ = root_->actions[a].state;
  if (root_ != nullptr) {
    CHECK_EQ(current_, root_->board);
    // Shuffle root priors a little bit.
    //     for (int i = 0; i < 7; ++i) {
    //       root_->actions[i].prior = 1.0 / 7;
    //     }
  } else {
    root_ = std::make_shared<State>(current_, Prediction());
  }
  CHECK_EQ(current_, root_->board);
}

void MCTS::SetBoard(const Board& b) {
  current_ = b;
  const auto it = visited_states_.find(b);
  if (it == visited_states_.end()) {
    root_ = std::make_shared<State>(b, Prediction());
  } else {
    root_ = it->second;
  }
}

}  // namespace c4cc
