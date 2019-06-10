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

class RandomPlayer : public Player {
 public:
  RandomPlayer() { mt_.seed(100); }

  void Reset() override { b_ = Board(); }
  void Advance(const Move& m) override { b_ = Board(b_, m); }

  // TODO: Move def to .cc
  Move GetMove() override {
    auto moves = b_.valid_moves();
    CHECK_GT(moves.size(), 0);
    std::uniform_int_distribution<int> dist(0, moves.size() - 1);
    int r = dist(mt_);
    return moves[r];
  }

 private:
  std::mt19937_64 mt_;
  Board b_;
};

void PlayGames() {
  Board::Init();
  auto model = CreateDefaultModel(/*allow_init=*/false);
  //auto human_model = CreateDefaultModel(false, -1, "human_new");
  auto human_model = CreateDefaultModel(false, 0);
  PredictionQueue pred_queue(model.get(), 8);
  PredictionQueue human_queue(human_model.get(), 8);
  std::vector<std::thread> threads;

  RandomPlayer random_player;
  CliHumanPlayer human_player;
  PolicyNetworkPlayer policy_player(&pred_queue);
  PolicyNetworkPlayer human_policy(&human_queue);
  MCTSPlayer mcts_player(&pred_queue, 1000);
  MCTSPlayer human_mcts_player(&human_queue, 1000);

  do {
    // Game g({&random_player, &policy_player});
    // Game g({&human_policy, &mcts_player});
    // Game g({&policy_player, &mcts_player});
    Game g({&human_mcts_player, &mcts_player});
    // Game g({&mcts_player, &human_mcts_player});
    // Game g({&human_mcts_player});
    while (!g.is_over()) {
      g.Work();

      std::cout << g.board().ToPrintString() << "\n";
    }
    std::cout << g.board().ToPrintString() << "\n";
    switch (g.winner()) {
      case Color::kEmpty:
        std::cout << "Draw!\n\n";
        break;
      case Color::kWhite:
        std::cout << "White wins!\n\n";
        break;
      case Color::kBlack:
        std::cout << "Black wins!\n\n";
        break;
    }
  } while (false);
}

}  // namespace chess

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  chess::PlayGames();
  return 0;
}
