#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <memory>
#include <thread>

#include "absl/synchronization/mutex.h"
#include "c4cc/human_player.h"
#include "c4cc/mcts_player.h"
#include "c4cc/model_collection.h"
#include "c4cc/negamax.h"
#include "c4cc/play_game.h"
#include "generic/prediction_queue.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "util/init.h"

namespace c4cc {
namespace {

enum class PlayerType {
  kHuman,
  kNegamax,
  kMcts,
};

struct Options {
  PlayerType players[2];
  int negamax_depth = 7;
  generic::PredictionQueue* queues[2];
  int mcts_iters = 400;
  int num_games = 1;
  bool hard = true;
};
std::unique_ptr<Player> MakePlayer(int p, Options opts) {
  switch (opts.players[p]) {
    case PlayerType::kHuman:
      return std::make_unique<HumanPlayer>();
    case PlayerType::kNegamax:
      return std::make_unique<NegamaxPlayer>(opts.negamax_depth);
    case PlayerType::kMcts:
      return std::make_unique<MCTSPlayer>(opts.queues[p], opts.mcts_iters,
                                          /*hard=*/opts.hard);
  }
  std::cerr << "Invalid PlayerType " << static_cast<int>(opts.players[p])
            << "\n";
  abort();
  return nullptr;
}

void Play(Options opts) {
  std::cout << "Start game!\n";

  absl::Mutex mu;
  int score[2] = {};
  int num_finished = 0;

  const auto play_game = [&opts, &mu, &score, &num_finished](int i) {
    const int p1_num = i % 2;
    const int p2_num = (i + 1) % 2;
    const auto player1 = MakePlayer(p1_num, opts);
    const auto player2 = MakePlayer(p2_num, opts);
    const Board result = PlayGame(player1.get(), player2.get()).first;
    PrintBoardWithColor(std::cout, result);

    absl::MutexLock lock(&mu);
    switch (result.result()) {
      case Color::kEmpty:
        score[0] += 1;
        score[1] += 1;
        std::cout << "Draw!\n";
        break;
      case Color::kOne:
        score[p1_num] += 2;
        std::cout << "X won!\n";
        break;
      case Color::kTwo:
        score[p2_num] += 2;
        std::cout << "O won!\n";
        break;
    }
    ++num_finished;
    std::cout << score[0] << " - " << score[1] << " after " << num_finished
              << " games\n";
  };

  if (opts.num_games == 1) {
    play_game(0);
  }
  std::vector<std::thread> game_threads;
  for (int i = 0; i < opts.num_games; ++i) {
    game_threads.emplace_back([i, &play_game] { play_game(i); });
  }
  for (auto& t : game_threads) {
    t.join();
  }
}

void Go(int argc, char** argv) {
  // auto model1 = CreateDefaultModel(/*allow_init=*/false, -1, "golden");
  // auto model2 = CreateDefaultModel(/*allow_init=*/false, -1);
  auto model = generic::Model::Open(
      kModelPath, GetModelCollection()->CurrentCheckpointDir());
  CHECK(model != nullptr);

  generic::PredictionQueue q(model.get());
  // PredictionQueue q2(model2.get());

  c4cc::Options opts;
  opts.players[0] = c4cc::PlayerType::kHuman;
  opts.players[1] = c4cc::PlayerType::kMcts;
  opts.mcts_iters = 1000;
  opts.queues[0] = &q;
  opts.queues[1] = &q;
  // opts.queues[1] = &q2;

  c4cc::Play(opts);
}

}  // namespace
}  // namespace c4cc

int main(int argc, char** argv) {
  const char* const* cargv = argv;
  NiceInit(argc, cargv);
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  c4cc::Go(argc, argv);

  return 0;
}
