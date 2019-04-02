#include "c4cc/mcts_player.h"

#include <time.h>

#include <memory>

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

namespace {}  // namespace

MCTSPlayer::MCTSPlayer(Model* m, int iters, PredictionCache* cache, bool hard)
    : model_(m),
      pred_cache_(cache),
      hard_(hard),
      iters_per_move_(iters),
      mcts_(std::make_unique<MCTS>(Board())) {
  rand_.seed(time(0));
}

void MCTSPlayer::SetBoard(const Board& b) {
  if (b != mcts_->current_board()) {
    mcts_->SetBoard(b);
  }
  pred_ready_ = false;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (mcts_->current_board().color(i, j) != Color::kEmpty) {
        ++ply_;
      }
    }
  }
}

void MCTSPlayer::RunIterations(int n) {
  // TODO: Do multiple iterations in parallel.
  const int K = 8;
  tensorflow::Tensor board_tensor = MakeBoardTensor(K);
  int predicted = 0;
  std::vector<std::unique_ptr<MCTS::PredictionRequest>> requests;
  Prediction predictions[K];

  const auto flush = [&] {
    CHECK_LE(requests.size(), K);
    const bool flip_all = rand_() % 3 == 0;
    for (int i = 0; i < requests.size(); ++i) {
      CHECK(!requests[i]->board().is_over());
      if (flip_all) {
        BoardToTensor(requests[i]->board().GetFlipped(), &board_tensor, i);
      } else {
        BoardToTensor(requests[i]->board(), &board_tensor, i);
      }
    }
    auto prediction_result = model_->Predict(board_tensor);
    ReadPredictions(prediction_result, predictions);
    for (int i = 0; i < requests.size(); ++i) {
      if (flip_all) {
        predictions[i] = predictions[i].GetFlipped();
      }
      if (pred_cache_ != nullptr) {
        pred_cache_->emplace(requests[i]->board(), predictions[i]);
      }
      mcts_->FinishIteration(std::move(requests[i]), predictions[i]);
    }
    requests.clear();
  };

  for (int iter = 0; iter < n; ++iter) {
    auto pred_req = mcts_->StartIteration();
    if (pred_req != nullptr) {
      bool cached = false;
      if (pred_cache_ != nullptr) {
        auto it = pred_cache_->find(pred_req->board());
        if (it != pred_cache_->end()) {
          mcts_->FinishIteration(std::move(pred_req), it->second);
          cached = true;
        }
      }
      if (!cached) {
        requests.push_back(std::move(pred_req));
      }
    }
    if (requests.size() == K) {
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
  LOG(INFO) << "Pri " << mcts_->GetPrior();
  LOG(INFO) << "Chi " << mcts_->GetChildValues();
}

int MCTSPlayer::GetMove() {
  CHECK(!board().is_over());

  Prediction pred = GetPrediction();

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

  // USe random for early moves.
  double r = std::uniform_real_distribution<double>(0.0, 1.0)(rand_);
  if (hard_ || (ply_ > 10 && r > 0.05)) {
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
            << ", returning first legal move";
  return *valid_moves.begin();
  // Sum should be ~1 now, if not, return the frs
}

Prediction MCTSPlayer::GetPrediction() {
  CHECK(!board().is_over());
  if (pred_ready_) {
    return current_pred_;
  }
  const auto valid_moves = board().valid_moves();
  CHECK(valid_moves.size() != 0);

  RunIterations(iters_per_move_);
  current_pred_ = mcts_->GetPrediction();
  pred_ready_ = true;
  return current_pred_;
}

void MCTSPlayer::MakeMove(int move) {
  mcts_->MakeMove(move);
  pred_ready_ = false;
  ++ply_;
}

}  // namespace c4cc
