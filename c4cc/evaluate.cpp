#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <memory>

#include "c4cc/mcts_player.h"
#include "c4cc/model.h"
#include "c4cc/play_game.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "util/init.h"

namespace c4cc {
namespace {

const int games = 400;

int GetScore(Model* m1, Model* m2, PredictionCache* c1, PredictionCache* c2) {
  const int iters = 400;
  int score = 0;
  for (int g = 0; g < games; ++g) {
    MCTSPlayer p1(m1, iters, c1);
    MCTSPlayer p2(m2, iters, c2);

    Color p1_color;
    Board board;
    if (g % 2 == 0) {
      p1_color = Color::kOne;
      board = PlayGame(&p1, &p2).first;
    } else {
      p1_color = Color::kTwo;
      board = PlayGame(&p2, &p1).first;
    }
    const Color winner = board.result();
    PrintBoardWithColor(std::cout, board);
    if (winner == Color::kEmpty) {
      LOG(INFO) << "Draw";
      score += 1;
    } else if (winner == p1_color) {
      LOG(INFO) << "New won";
      score += 2;
    } else {
      LOG(INFO) << "Old won";
    }
    LOG(INFO) << "Score " << score << " after " << (g + 1) << " games";
  }
  return score;
}

void Go(int argc, char** argv) {
  auto current = CreateDefaultModel(false, -1);

  int next_gen = GetNumGens();
  if (next_gen == 0) {
    // No snapshots yet, make first.
    current->Checkpoint(GetDefaultCheckpoint(0));
    next_gen = 1;
  }
  LOG(INFO) << "next gen: " << next_gen;

  auto last = CreateDefaultModel(false, next_gen - 1);
  PredictionCache last_cache;
  PredictionCache current_cache;

  // Need to get 55% of maximum points.
  const int promo_score = 0.55 * 2 * games;
  while (true) {
    const int score =
        GetScore(current.get(), last.get(), &current_cache, &last_cache);
    LOG(INFO) << "Score: " << score;
    if (score > promo_score) {
      LOG(INFO) << "PROMO";
      current->Checkpoint(GetDefaultCheckpoint(next_gen));
      last = std::move(current);
      ++next_gen;

      last_cache.swap(current_cache);
    }
    // Load new model for the current player, maybe it has improved.
    current = CreateDefaultModel(false, -1);
    current_cache.clear();
  }
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
