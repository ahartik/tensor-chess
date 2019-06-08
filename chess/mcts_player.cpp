#include "chess/mcts_player.h"

#include "tensorflow/core/platform/logging.h"

namespace chess {

MCTSPlayer::MCTSPlayer(PredictionQueue* pq, int iters_per_move)
    : queue_(pq), iters_per_move_(iters_per_move) {
  mcts_ = std::make_unique<MCTS>();
}

void MCTSPlayer::Reset() {
  mcts_->SetBoard(Board());
}

void MCTSPlayer::Advance(const Move& m) {
  mcts_->MakeMove(m);
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

  for (int iter = 0; iter < n; ++iter) {
    auto pred_req = mcts_->StartIteration();
    if (pred_req != nullptr) {
      requests.push_back(std::move(pred_req));
    }
    if (requests.size() == minibatch_size) {
      flush();
    }
  }
  flush();
}

Move MCTSPlayer::GetMove() {
  int n = iters_per_move_ - mcts_->num_iterations();
  RunIterations(n);
  auto pred = mcts_->GetPrediction();
  double r = std::uniform_real_distribution<double>(0.0, 1.0)(rand_);
  LOG(INFO) << "v=" << pred.value;

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
