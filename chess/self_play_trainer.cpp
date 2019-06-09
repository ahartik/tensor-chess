#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "absl/time/time.h"
#include "chess/board.h"
#include "chess/game_state.h"
#include "chess/mcts_player.h"
#include "chess/model.h"
#include "chess/player.h"
#include "chess/prediction_queue.h"
#include "util/init.h"
#include "chess/shuffling_trainer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {

const int kNumIters = 200;

void PlayerThread(int thread_i, PredictionQueue* pred_queue,
                  ShufflingTrainer* trainer) {
  auto player = std::make_unique<MCTSPlayer>(pred_queue, kNumIters);
  const int kGamesPerRefresh = 100;
  int games_to_refresh = kGamesPerRefresh;

  while (true) {
    Game g({player.get()});
    while (!g.is_over()) {
      g.Work();
      if (thread_i == 0) {
        std::cout << "Thread " << thread_i << ":\n"
                  << g.board().ToPrintString() << "\n";
      }
    }
    double res = 0;
    switch (g.winner()) {
      case Color::kEmpty:
        res = 0;
        std::cout << "Draw!\n\n";
        break;
      case Color::kWhite:
        res = 1;
        std::cout << "White wins!\n\n";
        break;
      case Color::kBlack:
        res = -1;
        std::cout << "Black wins!\n\n";
        break;
    }

    // TODO: Log games too
    for (const auto& state : player->saved_predictions()) {
      auto sample = std::make_unique<TrainingSample>();
      sample->board = state.board;
      sample->moves = state.pred.policy;
      sample->value = state.board.turn() == Color::kWhite ? res : -res;
      trainer->Train(std::move(sample));
    }

    --games_to_refresh;
    if (games_to_refresh <= 0) {
      games_to_refresh = kGamesPerRefresh;
      // Player state is reset since we may have learned something and play
      // better now.
      player = std::make_unique<MCTSPlayer>(pred_queue, kNumIters);
    }
  }
}

void PlayGames() {
  Board::Init();
  auto model = CreateDefaultModel(/*allow_init=*/true);
  PredictionQueue pred_queue(model.get(), 384);
  ShufflingTrainer trainer(model.get());
  std::vector<std::thread> threads;

  const int kNumThreads = 128;
  // const int kNumThreads = 4;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(
        [i, &pred_queue, &trainer] { PlayerThread(i, &pred_queue, &trainer); });
  }
  int64_t last_num_preds = 0;
  absl::Time last_log = absl::Now();
  absl::Time last_clear = absl::Now();
  const absl::Duration max_cache_age = absl::Minutes(10);
  while (true) {
    absl::SleepFor(absl::Seconds(5));

    model->Checkpoint(GetDefaultCheckpoint());
    // std::cout << "Saved checkpoint\n";

    absl::Time log_time = absl::Now();
    const int64_t num_preds = pred_queue.num_predictions();
    const double preds_per_sec = (num_preds - last_num_preds) /
                                 absl::ToDoubleSeconds(log_time - last_log);
    printf("Preds per sec: %.2f\n", preds_per_sec);
    printf("Avg batch size: %.2f\n", pred_queue.avg_batch_size());

    last_log = log_time;
    last_num_preds = num_preds;

    if (absl::Now() > last_clear + max_cache_age) {
      pred_queue.EmptyCache();
      last_clear = absl::Now();
    }
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  NiceInit(argc, argv);
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  chess::PlayGames();
  return 0;
}
