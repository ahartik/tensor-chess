#include "c4cc/mcts_player.h"

#include <memory>

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

namespace {}  // namespace

MCTSPlayer::MCTSPlayer(Model* m)
    : model_(m), mcts_(std::make_unique<MCTS>(Board())) {}

void MCTSPlayer::SetBoard(const Board& b) {
  if (b != mcts_->current_board()) {
    mcts_ = std::make_unique<MCTS>(b);
  }
}

int MCTSPlayer::GetMove() {
  CHECK(!board().is_over());
  const auto valid_moves = board().valid_moves();
  CHECK(valid_moves.size() != 0);

  // TODO: Actually run a few iterations.
  MCTS::Prediction pred = mcts_->GetPrediction();
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
