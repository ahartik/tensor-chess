#include "chess/mcts.h"

#include <cmath>

#include "chess/movegen.h"
#include "chess/types.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {

namespace mcts {

struct Action {
  Move move;
  // Prior and prior value don't change after construction.
  double prior = 0.0;

  int num_virtual = 0;
  int num_taken = 0;
  double total_value = 0;
  std::shared_ptr<State> state;

  void AddResult(double v);
};

struct State {
  State(const Board& b, const PredictionResult& p)
      : board(b), is_terminal(false), winner(Color::kEmpty) {
    CHECK_GT(p.policy.size(), 0);
    // Give a positive prior to all moves, so that we still sometimes explore
    // them in case the prediction network gives a zero weight for the move.
    const double add = 0.05 / p.policy.size();
    const double new_total = 1.0 + p.policy.size() * add;
    actions.reserve(p.policy.size());
    for (auto& move : p.policy) {
      actions.emplace_back();
      Action& a = actions.back();
      a.move = move.first;
      a.prior = (move.second + add) / new_total;
    }
  }

  State(const Board& b, Color winner)
      : board(b), is_terminal(true), winner(winner) {}

  State(const State&) = delete;
  State& operator=(const State&) = delete;

  double terminal_value() const {
    CHECK(is_terminal);
    if (winner == Color::kEmpty) {
      return 0.0;
    } else {
      CHECK_NE(winner, board.turn());
      // Terminal nodes are never wins for the player to move.
      return -1;
    }
  }

  const Board board;
  std::vector<Action> actions;
  const bool is_terminal;
  const Color winner;
};

void Action::AddResult(double v) {
  num_taken += 1;
  total_value += v;
  // mean_value = total_value / num_taken;
  CHECK(state != nullptr);
  if (state != nullptr) {
    if (state->is_terminal) {
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

Action* PickAction(std::mt19937& rand, State& s) {
  static const double rand_add = 0.001;
  std::uniform_real_distribution<double> rand_dist(0.0, rand_add);
  int num_sum = 0;
  for (auto& action : s.actions) {
    num_sum += action.num_taken + action.num_virtual;
  }
  double best_score = -10000;
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
  Action* best_action = nullptr;
  for (Action& action : s.actions) {
    const double score = score_move(action) + rand_dist(rand);
    if (score > best_score) {
      best_score = score;
      best_action = &action;
    }
  }
  if (best_action == nullptr) {
    for (const Action& action : s.actions) {
      LOG(INFO) << "m " << action.move << " s " << score_move(action);
    }
  }
  CHECK_NE(best_action, nullptr);
  return best_action;
}

}  // namespace

MCTS::MCTS(const Board& start) { SetBoard(start); }

MCTS::~MCTS() {}

const Board& MCTS::current_board() const { return current_; }

std::unique_ptr<MCTS::PredictionRequest> MCTS::StartIteration() {
  CHECK(root_ != nullptr);
  // TODO: Use some tricks to avoid keeping 'cur' a shared ptr.
  std::shared_ptr<State> cur = root_;
  CHECK(!cur->is_terminal);
  // TODO: Use InlinedVector
  PredictionRequest::PathVec picked_path;
  while (true) {
    Action* best_action = PickAction(rand_, *cur);
    picked_path.emplace_back(cur.get(), best_action);
    if (best_action->state == nullptr) {
      Board board(cur->board, best_action->move);

      std::vector<Move> moves;
      MovegenResult res =
          IterateLegalMoves(board, [&](const Move& m) { moves.push_back(m); });
      bool is_terminal = false;
      Color winner = Color::kEmpty;
      switch (res) {
        case MovegenResult::kCheckmate:
          // player to move lost.
          is_terminal = true;
          winner = OtherColor(board.turn());
          break;
        case MovegenResult::kStalemate:
          is_terminal = true;
          winner = Color::kEmpty;
          break;
        case MovegenResult::kNotOver: {
          // Increment virtual counts - this will be undone by Finish.
          for (auto e : picked_path) {
            ++e.a->num_virtual;
          }
          std::unique_ptr<PredictionRequest> request(new PredictionRequest());
          request->picked_path_ = std::move(picked_path);
          request->parent_ = cur;
          request->parent_a_ = best_action;
          request->board_ = board;
          request->moves_ = std::move(moves);
          return request;
        }
      }
      CHECK(is_terminal);
      // Add terminal node here.
      best_action->state = std::make_shared<State>(board, winner);
    }
    cur = best_action->state;
    CHECK(cur);
    // This could be a new terminal node, or one we've discovered before.
    if (cur->is_terminal) {
      // LOG(INFO) << "path len: " << picked_path.size();
      // Terminal node, can't iterate.
      CHECK(!picked_path.empty());
      for (auto e : picked_path) {
        // Terminal nodes have turn of the loser.
        const double mul = e.s->board.turn() == cur->board.turn() ? 1.0 : -1.0;
        e.a->AddResult(mul * cur->terminal_value());
      }
      return nullptr;
    }
  }
}

void MCTS::FinishIteration(std::unique_ptr<PredictionRequest> req,
                           const PredictionResult& p) {
  CHECK(root_ != nullptr);

  CHECK(req->parent_ != nullptr);
  auto& action = *req->parent_a_;
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
  for (const auto& action : root_->actions) {
    sum += action.num_taken;
  }
  return sum;
}

PredictionResult MCTS::GetPrediction() const {
  PredictionResult res;
  CHECK(root_ != nullptr);
  int sum_n = 0;
  for (const auto& action : root_->actions) {
    sum_n += action.num_taken;
    CHECK(action.num_virtual == 0) << "m=" << action.move;
  }
  CHECK(sum_n != 0);
  const double inv_sum = 1.0 / sum_n;
  res.value = 0;

  for (const auto& action : root_->actions) {
    const int num = action.num_taken;
    res.policy.emplace_back(action.move, num * inv_sum);
    res.value += action.total_value * inv_sum;
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
#endif

void MCTS::MakeMove(Move m) {
  visited_.insert(current_.board_hash());
  bool found = false;
  for (auto& action : root_->actions) {
    if (action.move == m) {
      found = true;
      root_ = action.state;
      break;
    }
  }
  CHECK(found) << current_.ToFEN() << " m " << m << "\n";
  current_ = Board(current_, m);

  if (root_ != nullptr) {
    CHECK_EQ(current_, root_->board);
    // Shuffle root priors a little bit.
    //     for (int i = 0; i < 7; ++i) {
    //       root_->actions[i].prior = 1.0 / 7;
    //     }
  } else {
    SetBoard(current_);
  }
  CHECK_EQ(current_, root_->board);
}

void MCTS::SetBoard(const Board& b) {
  current_ = b;
  visited_.clear();
  const auto it = visited_states_.find(b);
  if (it == visited_states_.end()) {
    // Start with even split over all legal moves.
    // TODO: This is not correct, fix this.
    PredictionResult even;
    MovegenResult res = IterateLegalMoves(
        b, [&](const Move& m) { even.policy.emplace_back(m, 1.0); });
    for (auto& move : even.policy) {
      move.second /= even.policy.size();
    }
    // TODO: Extract this to a function.
    switch (res) {
      case MovegenResult::kCheckmate:
        // player to move lost.
        root_ = std::make_shared<State>(b, OtherColor(b.turn()));
        break;
      case MovegenResult::kStalemate:
        root_ = std::make_shared<State>(b, Color::kEmpty);
        break;
      case MovegenResult::kNotOver:
        root_ = std::make_shared<State>(b, even);
        break;
    }
  } else {
    root_ = it->second;
  }
}

}  // namespace chess
