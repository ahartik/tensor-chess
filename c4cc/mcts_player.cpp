#include "c4cc/mcts_player.h"

#include <memory>

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

namespace {

}  // namespace

MCTSPlayer::MCTSPlayer(Model* m, int iters)
    : model_(m),
      iters_per_move_(iters),
      mcts_(std::make_unique<MCTS>(Board())) {}

void MCTSPlayer::SetBoard(const Board& b) {
  if (b != mcts_->current_board()) {
    mcts_ = std::make_unique<MCTS>(b);
  }
}

void MCTSPlayer::RunIterations(int n) {
  // TODO: Do multiple iterations in parallel.
  tensorflow::Tensor board_tensor = MakeBoardTensor(1);
  for (int i = 0; i < n; ++i) {
    const Board* to_predict = mcts_->StartIteration();
    if (to_predict != nullptr) {
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
  const auto valid_moves = board().valid_moves();
  CHECK(valid_moves.size() != 0);

  RunIterations(iters_per_move_);

  Prediction pred = mcts_->GetPrediction();
  LOG(INFO) << "Got " << pred << " with total " << mcts_->num_iterations()
            << " mcts iters";

  // TODO: Add options for the move selection algorith. For example, we could

  double total = 0;
  for (const int move : valid_moves) {
    total += pred.move_p[move];
  }
  CHECK_GT(total, 0.0);
  CHECK_LE(total, 1.001);
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

void MCTSPlayer::MakeMove(int move) {
  mcts_->MakeMove(move);
}

}  // namespace c4cc
