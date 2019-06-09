#include "chess/mcts.h"

#include <cmath>

#include "absl/container/flat_hash_set.h"
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
  State* state;

  void AddResult(double v);
};

struct State {
  State(const Board& b, const PredictionResult& p)
      : fp(BoardFingerprint(b)),
        turn(b.turn()),
        is_terminal(false),
        winner(Color::kEmpty) {
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
      : fp(BoardFingerprint(b)),
        turn(b.turn()),
        is_terminal(true),
        winner(winner) {}

  State(const State&) = delete;
  State& operator=(const State&) = delete;

  double terminal_value() const {
    CHECK(is_terminal);
    if (winner == Color::kEmpty) {
      return 0.0;
    } else {
      CHECK_NE(winner, turn);
      // Terminal nodes are never wins for the player to move.
      return -1;
    }
  }
  const BoardFP fp;
  const Color turn;
  const bool is_terminal;
  const Color winner;
  std::vector<Action> actions;
};

size_t StateHasher::operator()(const State& state) const {
  return absl::Uint128Low64(state.fp);
}
size_t StateHasher::operator()(BoardFP fp) const {
  return absl::Uint128Low64(fp);
}

bool StateEquals::operator()(const State& a, const State& b) const {
  return a.fp == b.fp;
}
bool StateEquals::operator()(const BoardFP& a, const State& b) const {
  return a == b.fp;
}
bool StateEquals::operator()(const State& a, const BoardFP& b) const {
  return b == a.fp;
}
bool StateEquals::operator()(const BoardFP& a, const BoardFP& b) const {
  return a == b;
}

void Action::AddResult(double v) {
  num_taken += 1;
  total_value += v;
  // mean_value = total_value / num_taken;
  // CHECK(state != nullptr);
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
const double kDirichletA = 0.3;
const double kDirichletMul = 0.2;


Action* PickAction(std::mt19937& rand, State& s, bool dirichlet) {
  std::vector<double> dir;
  if (dirichlet) {
    dir.resize(s.actions.size());
    std::gamma_distribution<double> gamma(kDirichletA, 1.0);
    double gamma_sum = 0;
    for (double& g : dir) {
      g = gamma(rand);
      gamma_sum += g;
    }
    for (double& g : dir) {
      g /= gamma_sum;
    }
  }

  int num_sum = 0;
  for (auto& action : s.actions) {
    num_sum += action.num_taken + action.num_virtual;
  }
  double best_score = -10000;
  const double num_sum_sqrt = sqrt(num_sum);
  const auto score_move = [num_sum, num_sum_sqrt, &dir](const Action& action,
                                                        int i) -> double {
    const double prior = dir.empty() ? action.prior
                                     : action.prior * (1 - kDirichletMul) +
                                           kDirichletMul * dir[i];
    if (num_sum == 0) {
      return prior;
    } else {
      const int num = action.num_taken + action.num_virtual;
      // Reminder: virtual moves are counted as losses for both players.
      double mean_value =
          num == 0 ? 0 : (action.total_value - action.num_virtual) / num;
      return mean_value + kPUCT * prior * num_sum_sqrt / (1.0 + num);
    }
  };
  Action* best_action = nullptr;
  for (int i = 0; i < s.actions.size(); ++i) {
    Action& action = s.actions[i];
    const double score = score_move(action, i);
    if (score > best_score) {
      best_score = score;
      best_action = &action;
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
  absl::flat_hash_set<uint64_t> path_visit;
  path_visit.reserve(16);
  State* cur = root_;
  CHECK(!cur->is_terminal);

  PredictionRequest::PathVec picked_path;
  Board cur_board = current_;
  while (true) {
    CHECK_EQ(cur_board.turn(), cur->turn);
    CHECK_EQ(BoardFingerprint(cur_board), cur->fp) << cur_board;
    Action* best_action = PickAction(rand_, *cur, cur == root_);
    picked_path.emplace_back(cur, best_action);
    cur_board = Board(cur_board, best_action->move);

    const auto cur_fp = BoardFingerprint(cur_board);

    // Check for a repetition draw. This also ensures that we don't get stuck
    // in an infinite loop.
    // TODO: Also compare with visited_, and only check for threefold
    // repetition.
    const bool loop = !path_visit.insert(cur_board.board_hash()).second;
    auto visited_it = visited_.find(cur_board.board_hash());
    const int history_count =
        visited_it == visited_.end() ? 0 : visited_it->second;
    if (loop || history_count > 2) {
      // Don't create a terminal node, but update all actions.
      CHECK(!picked_path.empty());
      for (auto e : picked_path) {
        // Draw.
        e.a->AddResult(0);
      }
      return nullptr;
    }

    if (best_action->state == nullptr) {
      // See if we already visited this state via some other path.
      const auto trans_it = visited_states_.find(cur_fp);
      std::shared_ptr<State> transposed;
      if (trans_it != visited_states_.end()) {
        CHECK(trans_it->second != nullptr);
        best_action->state = trans_it->second.get();
      } else {
        std::vector<Move> moves;
        MovegenResult res = IterateLegalMoves(
            cur_board, [&](const Move& m) { moves.push_back(m); });
        bool is_terminal = false;
        Color winner = Color::kEmpty;
        switch (res) {
          case MovegenResult::kCheckmate:
            // player to move lost.
            is_terminal = true;
            winner = OtherColor(cur_board.turn());
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
            request->board_ = cur_board;
            request->moves_ = std::move(moves);
            return request;
          }
        }
        CHECK(is_terminal);
        // Add terminal node here.
        auto new_term = std::make_unique<State>(cur_board, winner);
        best_action->state = new_term.get();
        CHECK(visited_states_.emplace(cur_fp, std::move(new_term)).second);
      }
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
        const double mul = e.s->turn == cur_board.turn() ? 1.0 : -1.0;
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
    const auto fp = BoardFingerprint(req->board());
    // It's possible that another pending request already populated this state,
    // in which case we must reuse it.
    auto& old_state = visited_states_[fp];
    if (old_state == nullptr) {
      // Initialize next state.
      old_state = std::make_unique<State>(req->board(), p);
    }
    action.state = old_state.get();
  }
  CHECK_EQ(req->picked_path_.back().s, req->parent_);
  for (auto e : req->picked_path_) {
    const double mul = e.s->turn == req->board().turn() ? 1.0 : -1.0;
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
    CHECK_EQ(current_.turn(), root_->turn);
    CHECK_EQ(BoardFingerprint(current_), root_->fp);
  } else {
    SetBoard(current_);
  }
  ++visited_[current_.board_hash()];
}

void MCTS::SetBoard(const Board& b) {
  current_ = b;
  visited_.clear();
  const auto it = visited_states_.find(BoardFingerprint(b));
  if (it == visited_states_.end()) {
    // Start with even split over all legal moves.
    // TODO: This is not correct, fix this.
    PredictionResult even;
    MovegenResult res = IterateLegalMoves(
        b, [&](const Move& m) { even.policy.emplace_back(m, 1.0); });
    for (auto& move : even.policy) {
      move.second /= even.policy.size();
    }
    std::unique_ptr<State> new_root;
    // TODO: Extract this to a function.
    switch (res) {
      case MovegenResult::kCheckmate:
        // player to move lost.
        new_root = std::make_unique<State>(b, OtherColor(b.turn()));
        break;
      case MovegenResult::kStalemate:
        new_root = std::make_unique<State>(b, Color::kEmpty);
        break;
      case MovegenResult::kNotOver:
        new_root = std::make_unique<State>(b, even);
        break;
    }
    root_ = new_root.get();
    visited_states_[root_->fp] = std::move(new_root);
  } else {
    root_ = it->second.get();
  }
}

}  // namespace chess
