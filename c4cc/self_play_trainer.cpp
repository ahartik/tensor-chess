#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <atomic>
#include <chrono>
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

// This class is thread-safe.
class Trainer {
 public:
  Trainer() {}

  // As MCTSPlayer is not thread-safe, don't call this with the same player
  // from multiple threads.
  void PlayGame(MCTSPlayer* player) {
    player->SetBoard(Board());
    std::vector<Board> boards;
    std::vector<Prediction> preds;
    while (!player->board().is_over()) {
      ++num_boards;
      auto pred = player->GetPrediction();
      boards.push_back(player->board());
      preds.push_back(pred);
      player->MakeMove(player->GetMove());
    }
    // PrintBoardWithColor(std::cout, player->board());
    TrainGame(std::move(boards), std::move(preds), player->board().result());
  }

  PredictionQueue* queue() const { return &queue_; }

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

  std::unique_ptr<Model> model_ = CreateDefaultModel(true);
  mutable PredictionQueue queue_{model_.get()};

  absl::Mutex mu_;
  ShufflingTrainer trainer_{model_.get(), 384, 2048};
};

void Go() {
  Trainer t;
  auto train_thread = [&t] {
    static constexpr int iters = 400;
    auto player = std::make_unique<MCTSPlayer>(t.queue(), iters);
    int i = 0;
    while (true) {
      ++i;
      t.PlayGame(player.get());
      if (i % 100 == 0) {
        player = std::make_unique<MCTSPlayer>(t.queue(), iters);
      }
    }
  };
  std::vector<std::thread> threads;
  const int kNumThreads = 97;  // be_simple ? 1 : 2;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(train_thread);
    CHECK(threads.back().joinable());
  }
  absl::Time last_log = absl::Now();
  int last_preds = 0;
  int last_boards = 0;
  while (true) {
    absl::SleepFor(absl::Seconds(5));
    const absl::Time now = absl::Now();
    const int64_t preds = t.queue()->num_predictions();
    const int64_t boards = num_boards.load(std::memory_order_relaxed);
    const double secs = absl::ToDoubleSeconds(now - last_log);
    LOG(INFO) << (preds - last_preds) / secs << " preds/s (" << preds
              << " total)";
    LOG(INFO) << (boards - last_boards) / secs << " boards/s";
    LOG(INFO) << "bpb: " << t.queue()->avg_batch_size();
    last_preds = preds;
    last_boards = boards;
    last_log = now;
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
