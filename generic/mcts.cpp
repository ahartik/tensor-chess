#include <cmath>

#include "generic/mcts.h"
#include "tensorflow/core/platform/logging.h"

namespace generic {

namespace mcts {

struct Action {
  Action(int m, float p) : move(m), prior(p) {}
  const int move;
  // Prior don't change after construction.
  const float prior;

  int num_virtual = 0;
  int num_taken = 0;
  double total_value = 0;
  std::shared_ptr<State> state;

  void AddResult(double v);
};

struct State {
  State(std::unique_ptr<Board> b, const PredictionResult& p)
      : board(std::move(b)) {
    // Give a positive prior to all moves, so that we still sometimes explore
    // them in case the prediction network gives a zero weight for the move.
    actions.reserve(p.policy.size());
    for (const auto& [move, prob] : p.policy) {
      actions.emplace_back(move, prob);
    }
  }
  State(const State&) = delete;
  State& operator=(const State&) = delete;

  bool is_terminal() const { return board->is_over(); }
  double terminal_value() const {
    CHECK(is_terminal());
    return board->result();
  }

  const std::unique_ptr<Board> board;
  std::vector<Action> actions;
};

void Action::AddResult(double v) {
  num_taken += 1;
  total_value += v;
  // mean_value = total_value / num_taken;
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

Action* PickAction(std::mt19937& rand, State& s) {
  static const double rand_add = 0.001;
  std::uniform_real_distribution<double> rand_dist(0.0, rand_add);
  int num_sum = 0;
  for (const auto& a : s.actions) {
    num_sum += a.num_taken + a.num_virtual;
  }
  const double num_sum_sqrt = sqrt(num_sum);

  const auto score_move = [num_sum,
                           num_sum_sqrt](const Action& action) -> double {
    if (num_sum == 0) {
      return action.prior;
    } else {
      const int num = action.num_taken + action.num_virtual;
      // Reminder: virtual moves are counted as losses for both players.
      double mean_value =
          num == 0 ? 0 : (action.total_value - action.num_virtual) / num;
      return mean_value + kPUCT * action.prior * num_sum_sqrt / (1.0 + num);
    }
  };

  double best_score = -10000;
  Action* best_a = nullptr;
  for (Action& action : s.actions) {
    const double score = score_move(action) + rand_dist(rand);
    if (score > best_score) {
      best_score = score;
      best_a = &action;
    }
  }
#if 0
  // Add debugging here if below check fails.
  if (best_a < 0) {
    for (int m : s.board.valid_moves()) {
      LOG(INFO) << "m " << m << " s " << score_move(m);
    }
  }
#endif
  CHECK_NE(best_a, nullptr);
  return best_a;
}

}  // namespace

MCTS::MCTS(std::unique_ptr<Board> start) { SetBoard(std::move(start)); }

MCTS::~MCTS() {}

const Board& MCTS::current_board() const { return *root_->board; }

std::unique_ptr<MCTS::PredictionRequest> MCTS::StartIteration() {
  CHECK(root_ != nullptr);
  // TODO: Use some tricks to avoid keeping 'cur' a shared ptr.
  std::shared_ptr<State> cur = root_;
  CHECK(!cur->is_terminal());
  // TODO: Use InlinedVector
  PredictionRequest::PathVec picked_path;
  while (true) {
    Action* const best_a = PickAction(rand_, *cur);
    picked_path.emplace_back(cur.get(), best_a);
    if (best_a->state == nullptr) {
      std::unique_ptr<Board> board = cur->board->Move(best_a->move);
      std::shared_ptr<State> state;
      if (board->is_over()) {
        // Terminal node, can't iterate.
        CHECK(!picked_path.empty());
        const double terminal_value = board->result();
        for (auto e : picked_path) {
          const double mul = e.s->board->turn() == board->turn() ? 1.0 : -1.0;
          e.a->AddResult(mul * terminal_value);
        }
        return nullptr;

      } else {
        // OK, it's a new node: must request a prediction.

        // Increment virtual counts - this will be undone by Finish.
        for (auto e : picked_path) {
          ++e.a->num_virtual;
        }
        std::unique_ptr<PredictionRequest> request(new PredictionRequest());
        request->picked_path_ = std::move(picked_path);
        request->parent_ = cur;
        request->parent_a_ = best_a;
        request->board_ = std::move(board);
        return request;
      }
      best_a->state = std::move(state);
    }
    cur = best_a->state;
  }
}

void MCTS::FinishIteration(std::unique_ptr<PredictionRequest> req,
                           const PredictionResult& p) {
  CHECK(root_ != nullptr);
  CHECK(req != nullptr);
  CHECK(req->board_ != nullptr);

  CHECK(req->parent_ != nullptr);
  Action* const action = req->parent_a_;
  // It's possible that 'state' was already added by a previous call to
  // FinishIteration(). StartIteration() may return the same position twice in
  // case the first iteration has not yet been finished.
  std::shared_ptr<State> state;
  if (action->state == nullptr) {
    // Initialize next state.
    state = std::make_shared<State>(std::move(req->board_), p);
#if 0
    auto& old_state = visited_states_[req->board()];
    // We never request predictions for moves that are already visited and
    // found live in the map.
    // CHECK(old_state.expired());
    // CHECK(old_state == nullptr);
    old_state = state;
#endif
    action->state = state;
  } else {
    state = action->state;
  }
  CHECK_EQ(req->picked_path_.back().a, action);
  CHECK_EQ(req->picked_path_.back().s, req->parent_.get());
  for (auto e : req->picked_path_) {
    CHECK(e.s->board != nullptr);
    CHECK(e.a->state != nullptr);
    CHECK(state != nullptr);
    CHECK(state->board != nullptr);
    const double mul = e.s->board->turn() == state->board->turn() ? 1.0 : -1.0;
    // Bias all values towards the mean, so that actual terminal nodes have
    // more weight than strong prediction outputs. This hopefully makes us seek
    // winning terminal nodes and avoid losing ones harder.
    //
    // XXX: Above comment makes sense if the following value is < 1.0.
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

PredictionResult MCTS::GetPrediction() const {
  PredictionResult res;
  CHECK(root_ != nullptr);
  int sum_n = 0;
  for (const Action& a : root_->actions) {
    sum_n += a.num_taken;
    CHECK(a.num_virtual == 0) << "move=" << a.move;
  }
  CHECK(sum_n != 0);
  const double inv_sum = 1.0 / sum_n;
  res.value = 0;

  for (const Action& a : root_->actions) {
    const int num = a.num_taken;
    res.policy.emplace_back(a.move, num * inv_sum);
    res.value += a.total_value * inv_sum;
  }

  return res;
}

#if 0
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
#endif

void MCTS::MakeMove(int a) {
  std::shared_ptr<State> new_root;
  for (Action& action : root_->actions) {
    if (action.move == a) {
      new_root = action.state;
      break;
    }
  }
  if (new_root != nullptr) {
    root_ = new_root;
  } else {
    SetBoard(root_->board->Move(a));
  }
}

void MCTS::SetBoard(std::unique_ptr<Board> b) {
  PredictionResult equal_p;
  const std::vector<int> moves = b->GetValidMoves();
  for (const int m : moves) {
    equal_p.policy.emplace_back(m, 1.0 / moves.size());
  }
  root_ = std::make_shared<State>(std::move(b), equal_p);
}

}  // namespace generic
