#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "c4cc/board.h"
#include "c4cc/mcts_player.h"
#include "c4cc/model.h"
#include "c4cc/play_game.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "absl/synchronization/mutex.h"

namespace c4cc {
namespace {

class Trainer {
 public:
  static constexpr int iters = 200;
  Trainer() {}

  void PlayGame() {
    player_.SetBoard(Board());
    std::vector<Board> boards;
    std::vector<Prediction> preds;
    while (!player_.board().is_over()) {
      auto pred = player_.GetPrediction();
      boards.push_back(player_.board());
      preds.push_back(pred);
      player_.MakeMove(player_.GetMove());
    }
    PrintBoardWithColor(std::cout, player_.board());
    const Color winner = player_.board().result();
    for (int i = 0; i < boards.size(); ++i) {
      if (winner == Color::kEmpty) {
        preds[i].value = 0.0;
      } else {
        if (boards[i].turn() == winner) {
          preds[i].value = 1.0;
        } else {
          preds[i].value = -1.0;
        }
      }
      trainer_.Train(boards[i], preds[i]);
    }
    model_->Checkpoint(GetDefaultCheckpoint());
  }

 private:
  std::unique_ptr<Model> model_ = CreateDefaultModel(true);

  absl::Mutex mu_;
  MCTSPlayer player_{model_.get(), iters};
  ShufflingTrainer trainer_{model_.get(), 8, 8};
};

void Go() {
  Trainer t;
  while (true) {
    t.PlayGame();
  }
}

}  // namespace
}  // namespace c4cc

int main(int argc, char* argv[]) {
  // Setup global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  c4cc::Go();
  return 0;
}
