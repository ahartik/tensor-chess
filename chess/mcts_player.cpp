#include "chess/mcts_player.h"

#include <time.h>

#include "tensorflow/core/platform/logging.h"

namespace chess {

MCTSPlayer::MCTSPlayer(PredictionQueue* pq, int iters_per_move)
    : queue_(pq), iters_per_move_(iters_per_move) {
  rand_.seed(time(0) ^ reinterpret_cast<intptr_t>(this));
  mcts_ = std::make_unique<MCTS>();
}

void MCTSPlayer::Reset() {
  mcts_->SetBoard(Board());
  saved_predictions_.clear();
}

void MCTSPlayer::Advance(const Move& m) {
  mcts_->MakeMove(m);
  saved_predictions_.emplace_back();
}

void MCTSPlayer::RunIterations(int n) {
  // Smaller number of games per minibatch should result in better accuracy, but
  // will be slower.
  const int minibatch_size = 8;
  std::vector<std::unique_ptr<MCTS::PredictionRequest>> requests;
  PredictionQueue::Request batch[minibatch_size];

  const auto flush = [&] {
    CHECK_LE(requests.size(), minibatch_size);
    for (int i = 0; i < requests.size(); ++i) {
      batch[i].board = &requests[i]->board();
      batch[i].moves = &requests[i]->moves();
    }
    queue_->GetPredictions(batch, requests.size());
    for (int i = 0; i < requests.size(); ++i) {
      mcts_->FinishIteration(std::move(requests[i]), batch[i].result);
    }
    requests.clear();
  };

  while (n >= 0) {
    for (int i = 0; i < minibatch_size && i < n; ++i) {
      auto pred_req = mcts_->StartIteration();
      if (pred_req != nullptr) {
        requests.push_back(std::move(pred_req));
      }
    }
    flush();
    n -= minibatch_size;
  }
}

Move MCTSPlayer::GetMove() {
  RunIterations(iters_per_move_);
  auto pred = mcts_->GetPrediction();

  if (saved_predictions_.empty()) {
    saved_predictions_.emplace_back();
  }
  {
    auto& b = saved_predictions_.back();
    b.board = mcts_->current_board();
    b.pred = pred;
  }

  LOG(INFO) << "v=" << pred.value;
  // After certain ply, do
#if 1
  if (mcts_->current_board().ply() > 12 && (rand_() % 20 != 0)) {
    double best = 0;
    Move best_move;
    for (const auto& m : pred.policy) {
      if (m.second > best) {
        best = m.second;
        best_move = m.first;
      }
    }
    return best_move;
  }
#endif
  double r = std::uniform_real_distribution<double>(0.0, 1.0)(rand_);

  CHECK(!pred.policy.empty());
  for (const auto& m : pred.policy) {
    r -= m.second;
    if (r <= 0) {
      return m.first;
    }
  }
  return pred.policy[0].first;
}

}  // namespace chess
