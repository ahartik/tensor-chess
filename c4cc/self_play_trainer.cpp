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
#include "c4cc/model_collection.h"
#include "c4cc/play_game.h"
#include "c4cc/generic_board.h"
#include "generic/board.h"
#include "generic/model.h"
#include "generic/shuffling_trainer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "util/init.h"

namespace c4cc {
namespace {

const bool be_simple = false;

std::atomic<int64_t> num_boards;

std::unique_ptr<generic::Model> OpenOrCreateModel() {
  auto model = generic::Model::Open(
      kModelPath, GetModelCollection()->CurrentCheckpointDir());
  if (model == nullptr) {
    LOG(INFO) << "Starting from scratch";
    model = generic::Model::New(kModelPath);
  } else {
    LOG(INFO) << "Continuing training";
  }
  return model;
}

// This class is thread-safe.
// TODO: Make this generic so it can be used for other games (i.e. chess) as
// well.
class Trainer {
 public:
  // As MCTSPlayer is not thread-safe, don't call this with the same player
  // from multiple threads.
  void PlayGame(MCTSPlayer* player) {
    player->SetBoard(Board());
    std::vector<Board> boards;
    std::vector<generic::PredictionResult> preds;
    while (!player->board().is_over()) {
      ++num_boards;
      auto pred = player->GetPrediction();
      boards.push_back(player->board());
      // We convert from c4cc::Prediction to generic::PredictionResult. This is
      // a little ugly, ideally there should be a "generic" trainer class doing
      // this self-play training.
      generic::PredictionResult gen_pred;
      gen_pred.value = pred.value;
      for (int m = 0; m < 7; ++m) {
        gen_pred.policy.emplace_back(m, pred.move_p[m]);
      }
      preds.push_back(gen_pred);
      player->MakeMove(player->GetMove());
    }
    // PrintBoardWithColor(std::cout, player->board());
    TrainGame(std::move(boards), std::move(preds), player->board().result());
  }

  generic::PredictionQueue* queue() const { return &queue_; }

 private:
  void TrainGame(std::vector<Board> boards,
                 std::vector<generic::PredictionResult> preds, Color winner) {
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
      trainer_.Train(
          MakeGenericBoard(boards[i]),
          preds[i]
          );
      for (auto& [move, prob] : preds[i].policy) {
        move = 6 - move;
      }
      trainer_.Train(
          MakeGenericBoard(boards[i].GetFlipped()), preds[i]);
      CHECK_EQ(boards[i].GetFlipped().GetFlipped(), boards[i]);
    }

    absl::MutexLock lock(&mu_);
    const int64_t num = trainer_.num_trained();
    if (num > last_num_trained_ + checkpoint_interval_) {
      model_->Checkpoint(GetModelCollection()->CurrentCheckpointDir());
      last_num_trained_ = num;
      LOG(INFO) << "Checkpointed";
    }
  }

  std::unique_ptr<generic::Model> model_{OpenOrCreateModel()};
  mutable generic::PredictionQueue queue_{model_.get()};

  const int checkpoint_interval_ = 10 * 1024;
  generic::ShufflingTrainer trainer_{model_.get(), *MakeGenericBoard(Board()),
                                     256, 2048};
  absl::Mutex mu_;
  int64_t last_num_trained_ = 0;
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
#if 1
      if (i % 10 == 0) {
        player = std::make_unique<MCTSPlayer>(t.queue(), iters);
      }
#endif
    }
  };
  std::vector<std::thread> threads;
  const int kNumThreads = 140;  // be_simple ? 1 : 2;
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
