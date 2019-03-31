#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <memory>

#include "c4cc/human_player.h"
#include "c4cc/mcts_player.h"
#include "c4cc/model.h"
#include "c4cc/negamax.h"
#include "c4cc/play_game.h"
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
  int negamax_depth = 6;
  Model* model;
  int mcts_iters = 1000;
};
std::unique_ptr<Player> MakePlayer(PlayerType type, Options opts) {
  switch (type) {
    case PlayerType::kHuman:
      return std::make_unique<HumanPlayer>();
    case PlayerType::kNegamax:
      return std::make_unique<NegamaxPlayer>(opts.negamax_depth);
    case PlayerType::kMcts:
      return std::make_unique<MCTSPlayer>(opts.model, opts.mcts_iters);
  }
  std::cerr << "Invalid PlayerType " << static_cast<int>(type) << "\n";
  abort();
  return nullptr;
}

void Play(Options opts) {
  std::cout << "Start game!\n";

  const auto player1 = MakePlayer(opts.players[0], opts);
  const auto player2 = MakePlayer(opts.players[1], opts);

  const Board result = PlayGame(player1.get(), player2.get()).first;
  PrintBoardWithColor(std::cout, result);
  switch (result.result()) {
    case Color::kEmpty:
      std::cout << "Draw!\n";
      break;
    case Color::kOne:
      std::cout << "X won!\n";
      break;
    case Color::kTwo:
      std::cout << "O won!\n";
      break;
  }
}

// TODO: Move this to common lib.
bool DirectoryExists(const std::string& dir) {
  struct stat buf;
  return stat(dir.c_str(), &buf) == 0;
}

void Go(int argc, char** argv) {
  const std::string prefix = "/mnt/tensor-data/c4cc";
  const std::string graph_def_filename = prefix + "/graph.pb";
  const std::string checkpoint_dir = prefix + "/checkpoints";
  const std::string checkpoint_prefix = checkpoint_dir + "/checkpoint";
  bool restore = DirectoryExists(checkpoint_dir);

  std::cout << "Loading graph\n";
  Model model(graph_def_filename);
  if (!restore) {
    std::cout << "Model must be trained before playing\n";
    exit(1);
  } else {
    std::cout << "Restoring model weights from checkpoint\n";
    model.Restore(checkpoint_prefix);
  }

  c4cc::Options opts;
  opts.players[0] = c4cc::PlayerType::kMcts;
  opts.players[1] = c4cc::PlayerType::kNegamax;
  opts.mcts_iters = 1000;
  opts.model = &model;

  c4cc::Play(opts);
}

}  // namespace
}  // namespace c4cc

int main(int argc, char** argv) {
  const char* const * cargv = argv;
  NiceInit(argc, cargv);
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  c4cc::Go(argc, argv);

  return 0;
}
