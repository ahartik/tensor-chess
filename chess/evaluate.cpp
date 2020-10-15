#include <atomic>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "chess/board.h"
#include "chess/game_state.h"
#include "chess/mcts_player.h"
#include "chess/model_collection.h"
#include "chess/player.h"
#include "chess/policy_player.h"
#include "generic/model.h"
#include "generic/prediction_queue.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {

const int kTourneySize = 400;
const double kPromotionWinrate = 0.55;
const int kNumIters = 400;

bool ShouldPromote(generic::PredictionQueue* new_q,
                   generic::PredictionQueue* old_q) {
  std::atomic<int> score{0};
  std::atomic<int> games_so_far{0};
  std::atomic<int> moves{0};

  auto play_game = [&](int g) {
    MCTSPlayer new_player(new_q, kNumIters);
    MCTSPlayer old_player(old_q, kNumIters);
    // auto old_player = MakePolicyPlayer(old_q);
    std::vector<Player*> players;
    const Color new_color = g % 2 == 0 ? Color::kWhite : Color::kBlack;
    if (new_color == Color::kWhite) {
      players = {&new_player, &old_player};
    } else {
      players = {&old_player, &new_player};
    }
    LOG(INFO) << "new is " << new_color;
    Game game(players);
    while (!game.is_over()) {
      game.Work();
      ++moves;
    }
    if (game.winner() == Color::kEmpty) {
      score += 1;
    } else if (game.winner() == new_color) {
      score += 2;
    }
    LOG(INFO) << "Score: " << score << " after " << games_so_far << " games";
    LOG(INFO) << "new is " << new_color << ":\n"
              << game.board().ToPrintString();
    ++games_so_far;
  };

  constexpr int kNumGamesPerStep = 1;
  for (int g = 0; g < kTourneySize; g += kNumGamesPerStep) {
    std::vector<std::thread> threads;
    for (int i = 0; i < kNumGamesPerStep; ++i) {
      threads.emplace_back([&play_game, i] { play_game(i); });
    }
    while (games_so_far < g + kNumGamesPerStep) {
      LOG(INFO) << moves << " moves so far";
      absl::SleepFor(absl::Seconds(5));
    }
    for (std::thread& t : threads) {
      t.join();
    }
  }
  return score > 2 * kTourneySize * kPromotionWinrate;
}

void PlayGames() {
  const auto* const model_collection = GetModelCollection();
  int next_gen = model_collection->CountNumGens();
  auto old_model = generic::Model::Open(
      kModelPath, model_collection->GenCheckpointDir(next_gen - 1));
  auto old_queue = std::make_unique<generic::PredictionQueue>(old_model.get());
  while (true) {
    auto new_model = generic::Model::Open(
        kModelPath, model_collection->CurrentCheckpointDir());
    auto new_queue =
        std::make_unique<generic::PredictionQueue>(new_model.get());
    if (ShouldPromote(new_queue.get(), old_queue.get())) {
      LOG(INFO) << "Promotion!";
      new_model->Checkpoint(model_collection->GenCheckpointDir(next_gen));
      old_model = std::move(new_model);
      old_queue = std::move(new_queue);
      ++next_gen;
    }
    absl::SleepFor(absl::Minutes(10));
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  chess::PlayGames();
  return 0;
}
