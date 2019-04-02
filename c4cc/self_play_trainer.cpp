#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "c4cc/board.h"
#include "c4cc/mcts_player.h"
#include "c4cc/model.h"
#include "c4cc/play_game.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "util/init.h"

namespace c4cc {
namespace {
  
const bool be_simple = false;

std::atomic<int64_t> num_boards;

class Trainer {
 public:
  Trainer() {}

  void PlayGame(MCTSPlayer* player) {
    player->SetBoard(Board());
    std::vector<Board> boards;
    std::vector<Prediction> preds;
    while (!player->board().is_over()) {
      ++num_boards;
      absl::ReaderMutexLock lock(&mu_);
      auto pred = player->GetPrediction();
      boards.push_back(player->board());
      preds.push_back(pred);
      player->MakeMove(player->GetMove());
    }
    PrintBoardWithColor(std::cout, player->board());
    TrainGame(std::move(boards), std::move(preds), player->board().result());
  }

  std::unique_ptr<Model> model_ = CreateDefaultModel(true);

 private:
  void TrainGame(std::vector<Board> boards, std::vector<Prediction> preds,
                 Color winner) {
    absl::MutexLock lock(&mu_);
    bool trained = false;
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
      trained |= trainer_.Train(boards[i], preds[i]);
      trained |= trainer_.Train(boards[i].GetFlipped(), preds[i].GetFlipped());
      CHECK_EQ(boards[i].GetFlipped().GetFlipped(), boards[i]);
    }
    if (trained) {
      model_->Checkpoint(GetDefaultCheckpoint());
    }
  }

  absl::Mutex mu_;
  ShufflingTrainer trainer_{model_.get(), 64, 1024};
};

void Go() {
  Trainer t;
  auto train_thread = [&t] {
    static constexpr int iters = 400;
    auto player = std::make_unique<MCTSPlayer>(t.model_.get(), iters);
    int i = 0;
    while (true) {
      ++i;
      t.PlayGame(player.get());
      if (i % 100 == 0) {
        player = std::make_unique<MCTSPlayer>(t.model_.get(), iters);
      }
    }
  };
  std::vector<std::thread> threads;
  const int kNumThreads = 2;  // be_simple ? 1 : 2;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(train_thread);
    CHECK(threads.back().joinable());
  }
  while (true) {
    absl::SleepFor(absl::Seconds(1));
    int64_t count = num_boards.exchange(0);
    LOG(INFO) << count << " boards last sec";
  }
  for (int i = 0; i < kNumThreads; ++i) {
    threads[i].join();
  }
  CHECK(false) << "Didn't expect threads to return";
}

}  // namespace
}  // namespace c4cc

int main(int argc, char* argv[]) {
  // Setup global state for TensorFlow.
  NiceInit(argc, argv);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  c4cc::Go();
  return 0;
}
