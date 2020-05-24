#include <pthread.h>
#include <time.h>

#include <memory>
#include <vector>

#include "c4cc/generic_board.h"
#include "c4cc/mcts_player.h"
#include "generic/board.h"
#include "tensorflow/core/platform/logging.h"

namespace c4cc {

namespace {}  // namespace

MCTSPlayer::MCTSPlayer(generic::PredictionQueue* queue, int iters, bool hard)
    : queue_(queue),
      hard_(hard),
      iters_per_move_(iters),
      mcts_(std::make_unique<generic::MCTS>(MakeGenericBoard(Board()))) {
  rand_.seed(time(0) ^ reinterpret_cast<intptr_t>(this));
}

void MCTSPlayer::SetBoard(const Board& b) {
  mcts_->SetBoard(MakeGenericBoard(b));
  pred_ready_ = false;
}

void MCTSPlayer::RunIterations(int n) {
  // Smaller number of games per minibatch should result in better accuracy, but
  // will be slower.
  const int minibatch_size = 8;
  std::vector<std::unique_ptr<generic::MCTS::PredictionRequest>> mcts_reqs;
  std::vector<generic::PredictionQueue::Request> pred_reqs;

  const auto flush = [&] {
    CHECK_LE(pred_reqs.size(), minibatch_size);
    queue_->GetPredictions(pred_reqs.data(), pred_reqs.size());
    for (int i = 0; i < pred_reqs.size(); ++i) {
      mcts_->FinishIteration(std::move(mcts_reqs[i]), pred_reqs[i].result);
    }
    pred_reqs.clear();
    mcts_reqs.clear();
  };

  for (int iter = 0; iter < n; ++iter) {
    auto pred_req = mcts_->StartIteration();
    if (pred_req != nullptr) {
      pred_reqs.emplace_back();
      auto& preq = pred_reqs.back();
      preq.board = &pred_req->board();
      mcts_reqs.push_back(std::move(pred_req));
    }
    if (pred_reqs.size() == minibatch_size) {
      flush();
    }
  }
  flush();
}

void MCTSPlayer::LogStats() {
  Prediction pred = GetPrediction();
  PrintBoardWithColor(std::cerr, board());
  LOG(INFO) << "Got " << pred << " with total " << mcts_->num_iterations()
            << " mcts iters";
  // LOG(INFO) << "Pri " << mcts_->GetPrior();
  // LOG(INFO) << "Chi " << mcts_->GetChildValues();
}

int MCTSPlayer::GetMove() {
  CHECK(!board().is_over());

  Prediction pred = GetPrediction();
  // LogStats();

  // TODO: Add options for move selection (i.e. temperature).
  const auto valid_moves = board().valid_moves();
  double total = 0;
  double best_score = -1.0;
  int best_move = -1;
  for (const int move : valid_moves) {
    total += pred.move_p[move];
    if (pred.move_p[move] > best_score) {
      best_score = pred.move_p[move];
      best_move = move;
    }
  }
  CHECK_GT(total, 0.0);
  CHECK_LE(total, 1.001);
  CHECK_GE(best_move, 0);

  // USe more random for early moves.
  double r = std::uniform_real_distribution<double>(0.0, 1.0)(rand_);
  if (hard_ || (board().ply() > 11 && r > 0.05)) {
    return best_move;
  }
  r *= total;
  double sum = 0;
  for (const int move : valid_moves) {
    sum += pred.move_p[move];
    if (sum >= r) {
      return move;
    }
  }
  std::cerr << "weird random move, r=" << r << " sum=" << sum
            << ", returning first legal move\n";
  return *valid_moves.begin();
  // Sum should be ~1 now, if not, return the frs
}

Prediction MCTSPlayer::GetPrediction() {
  CHECK(!board().is_over());
  if (pred_ready_) {
    return current_pred_;
  }
  RunIterations(iters_per_move_);
  generic::PredictionResult gen_pred = mcts_->GetPrediction();
  current_pred_.value = gen_pred.value;
  for (int i = 0; i < 7; ++i) {
    current_pred_.move_p[i] = 0;
  }
  for (const auto& [move, prob] : gen_pred.policy) {
    current_pred_.move_p[move] = prob;
  }
  pred_ready_ = true;
  return current_pred_;
}

void MCTSPlayer::MakeMove(int move) {
  mcts_->MakeMove(move);
  board_.MakeMove(move);
  pred_ready_ = false;
}

}  // namespace c4cc
