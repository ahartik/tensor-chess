#include "c4cc/mcts.h"

#include <cmath>

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

using State = MCTSState;

struct MCTSState {
  MCTSState(const Board& b, const Prediction& p)
      : board(b), pred(p), value(pred.value) {
    if (b.is_over()) {
      if (b.result() == Color::kEmpty) {
        value = 0.0;
      } else {
        value = b.result() == b.turn() ? 1.0 : -1.0;
      }
    }
  }
  int c_num(int a) const {
    return children[a] == nullptr ? 0 : children[a]->num;
  }

  double c_value(int a) const {
    return children[a] == nullptr ? 0.0 : -children[a]->value;
  }

  void CheckAssertions() {
    int total = 0;
    for (int i = 0; i < 7; ++i) {
      total += c_value(i);
    }
    CHECK_EQ(total, num);
  }

  void Fix() {
    is_leaf = false;
    num = 0;
    value = 0.0;
    for (int i = 0; i < 7; ++i) {
      num += c_num(i);
    }
    double inv_num = 1.0 / num;
    for (int i = 0; i < 7; ++i) {
      value += (c_value(i) * c_num(i)) * inv_num;
    }
    sqrt_num = sqrt(num);
    CHECK_GT(num, 0);
  }

  Board board;
  Prediction pred;
  bool is_leaf = true;
  int num = 1;          // "n"
  double sqrt_num = 1;  // "sqrt(n)"
  double value = 0.0;   // "q"
  std::unique_ptr<State> children[7];
  State* parent = nullptr;
};

namespace {

const double kPUCT = 1.0;

double UCB(const MCTSState& s, int a) {
  return s.c_value(a) +
         kPUCT * s.pred.move_p[a] * s.sqrt_num / (1.0 + s.c_num(a));
}

}  // namespace

MCTS::MCTS(const Board& start) : current_(start) {}

MCTS::~MCTS() {}

const Board& MCTS::current_board() const { return current_; }

const Board* MCTS::StartIteration() {
  if (root_ == nullptr) {
    return &current_;
  }
  MCTSState* cur = root_.get();
  while (true) {
    if (cur->board.is_over()) {
      // Hit terminal node. Just perform updates on counts.
      ++cur->num;
      if (cur->parent != nullptr) {
        BubbleCounts(cur->parent);
      }
      return nullptr;
    }
    int best_a = 0;
    double best_score = -1e6;
    for (int a : cur->board.valid_moves()) {
      double u = UCB(*cur, a);
      if (u > best_score) {
        best_score = u;
        best_a = a;
      }
    }
    if (cur->children[best_a] == nullptr) {
      next_parent_ = cur;
      next_a_ = best_a;
      next_prediction_ = cur->board;
      next_prediction_.MakeMove(best_a);
      if (next_prediction_.is_over()) {
        // No need for prediction, just finish the iteration with a default
        // prediction.
        FinishIteration(Prediction());
        return nullptr;
      } else {
        return &next_prediction_;
      }
    }
  }
}

void MCTS::FinishIteration(const Prediction& p) {
  if (root_ == nullptr) {
    root_ = std::make_unique<MCTSState>(current_, p);
  } else {
    next_parent_->children[next_a_] =
        std::make_unique<MCTSState>(next_prediction_, p);
    next_parent_->children[next_a_]->parent = next_parent_;
    BubbleCounts(next_parent_);
    next_parent_ = nullptr;
    next_a_ = -1;
  }
}

int MCTS::num_iterations() const { return root_ == nullptr ? 0 : root_->num; }

void MCTS::BubbleCounts(MCTSState* s) {
  while (s != nullptr) {
    s->Fix();
    s = s->parent;
  }
}

Prediction MCTS::GetMCTSPrediction() {
  Prediction res;
  if (root_ == nullptr) {
    for (int i = 0; i < 7; ++i) {
      res.move_p[i] = 1.0 / 7;
    }
    res.value = 0;
  } else {
    res.value = root_->value;
    for (int i = 0; i < 7; ++i) {
      res.move_p[i] = double(root_->c_num(i)) / root_->num;
    }
  }
  return res;
}

}  // namespace c4cc
