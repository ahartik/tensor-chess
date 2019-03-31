#include "c4cc/mcts_player.h"

#include <time.h>

#include <memory>

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

namespace {}  // namespace

MCTSPlayer::MCTSPlayer(Model* m, int iters)
    : model_(m),
      iters_per_move_(iters),
      mcts_(std::make_unique<MCTS>(Board())) {
  rand_.seed(time(0));
}

void MCTSPlayer::SetBoard(const Board& b) {
  if (b != mcts_->current_board()) {
    mcts_ = std::make_unique<MCTS>(b);
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
  tensorflow::Tensor board_tensor = MakeBoardTensor(1);
  for (int i = 0; i < n; ++i) {
    const Board* to_predict = mcts_->StartIteration();
    if (to_predict != nullptr) {
      CHECK(!to_predict->is_over());
      Prediction prediction;
      BoardToTensor(*to_predict, &board_tensor, 0);
      auto prediction_result = model_->Predict(board_tensor);
      ReadPredictions(prediction_result, &prediction);
      mcts_->FinishIteration(prediction);
    }
  }
}

int MCTSPlayer::GetMove() {
  CHECK(!board().is_over());

  Prediction pred = GetPrediction();
  PrintBoardWithColor(std::cerr, board());
  LOG(INFO) << "Got " << pred << " with total " << mcts_->num_iterations()
            << " mcts iters";

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
  if (ply_ > 10) {
    return best_move;
  }
  // USe random for early moves.
  const double r = std::uniform_real_distribution<double>(0.0, total)(rand_);
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
