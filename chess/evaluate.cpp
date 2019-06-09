#include <atomic>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "chess/board.h"
#include "chess/game_state.h"
#include "chess/mcts_player.h"
#include "chess/model.h"
#include "chess/player.h"
#include "chess/prediction_queue.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {

const int kTourneySize = 400;
const double kPromotionWinrate = 0.55;
const int kNumIters = 200;

bool ShouldPromote(PredictionQueue* new_q, PredictionQueue* old_q) {
  std::atomic<int> score{0};
  std::atomic<int> games_so_far{0};
  std::atomic<int> moves{0};

  auto play_game = [&](int g) {
    MCTSPlayer new_player(new_q, kNumIters);
    MCTSPlayer old_player(old_q, kNumIters);
    std::vector<Player*> players;
    const Color new_color = g % 2 == 0 ? Color::kWhite : Color::kBlack;
    if (new_color == Color::kWhite) {
      players = {&new_player, &old_player};
    } else {
      players = {&old_player, &new_player};
    }
    Game game(players);
    while (!game.is_over()) {
      game.Work();
      ++moves;
    }
    ++games_so_far;
    if (game.winner() == Color::kEmpty) {
      score += 1;
    } else if (game.winner() == new_color) {
      score += 2;
    }
    LOG(INFO) << "Score: " << score << " after " << games_so_far << " games";
  };

  constexpr int kNumGamesPerStep = 20;
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
  Board::Init();

  int next_gen = GetNumGens();
  auto old_model = CreateDefaultModel(/*allow_init=*/false, next_gen - 1);
  auto old_queue = std::make_unique<PredictionQueue>(old_model.get(), 96);
  while (true) {
    auto new_model = CreateDefaultModel(false, -1);
    auto new_queue = std::make_unique<PredictionQueue>(new_model.get(), 96);
    if (ShouldPromote(new_queue.get(), old_queue.get())) {
      LOG(INFO) << "Promotion!";
      new_model->Checkpoint(GetDefaultCheckpoint(next_gen));
      old_model = std::move(new_model);
      old_queue = std::move(new_queue);
      ++next_gen;
    }
    absl::SleepFor(absl::Seconds(10));
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  chess::PlayGames();
  return 0;
}
