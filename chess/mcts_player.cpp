#include "chess/mcts_player.h"

#include <time.h>

#include "chess/generic_board.h"
#include "chess/tensors.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {

MCTSPlayer::MCTSPlayer(generic::PredictionQueue* pq, int iters_per_move)
    : queue_(pq), iters_per_move_(iters_per_move) {
  rand_.seed(time(0) ^ reinterpret_cast<intptr_t>(this));
  mcts_ = std::make_unique<generic::MCTS>(MakeGenericBoard(board_));
}

void MCTSPlayer::Reset(const Board& b) {
  mcts_->SetBoard(MakeGenericBoard(b));
  board_ = b;
  saved_predictions_.clear();
}

void MCTSPlayer::Advance(const Move& m) {
  mcts_->MakeMove(EncodeMove(board_.turn(), m));
  board_ = Board(board_, m);
  saved_predictions_.emplace_back();
}

void MCTSPlayer::RunIterations(int n) {
  // Smaller number of games per minibatch should result in better accuracy, but
  // will be slower.
  const int minibatch_size = 8;
  std::vector<std::unique_ptr<generic::MCTS::PredictionRequest>> requests;
  generic::PredictionQueue::Request batch[minibatch_size];

  const auto flush = [&] {
    CHECK_LE(requests.size(), minibatch_size);
    for (int i = 0; i < requests.size(); ++i) {
      batch[i].board = &requests[i]->board();
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
  const auto pred = mcts_->GetPrediction();
  CHECK_EQ(mcts_->current_board().fingerprint(),
      BoardFingerprint(board_));

  if (saved_predictions_.empty()) {
    saved_predictions_.emplace_back();
  }
  {
    auto& b = saved_predictions_.back();
    b.board = board_;
    b.pred = pred;
  }
  // queue_->CacheRealPrediction(mcts_->current_board(), pred);

#if 0
  LOG(INFO) << "v=" << pred.value << " for " << board_.turn() << " in\n"
            << board_.ToPrintString();
#endif
  // After certain ply, pick the best move most of the time.
#if 1
  if (board_.ply() > 12 && (rand_() % 20 != 0)) {
    double best = 0;
    Move best_move;
    for (const auto& m : pred.policy) {
      if (m.second > best) {
        best = m.second;
        best_move = DecodeMove(board_, m.first);
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
      return DecodeMove(board_, m.first);
    }
  }

  return DecodeMove(board_, pred.policy[0].first);
}

}  // namespace chess
