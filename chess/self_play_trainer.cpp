#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "absl/time/time.h"
#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/game_state.h"
#include "chess/generic_board.h"
#include "chess/magic.h"
#include "chess/mcts_player.h"
#include "chess/model.h"
#include "chess/model_collection.h"
#include "chess/player.h"
#include "generic/prediction_queue.h"
#include "generic/shuffling_trainer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "util/init.h"

namespace chess {

const int kNumIters = 400;

const std::string kTrainingFens[] = {
    // One rook per side
    "2R1K3/8/8/8/8/8/8/3k1r2 w - - 0 1",
    // Rook and pawn
    "2R1K3/8/8/5p2/2P5/8/8/3k1r2 w - - 0 1",
    // Rook and pawn and knight
    "2R1K3/8/8/N4p2/2P5/6n1/8/3k1r2 w - - 0 1",
    // Rook vs two pawns
    "2K5/8/8/PP6/8/8/8/3k1r2 w - - 0 1",
    // Full initial
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    // R+K vs K, should be mate.
    "8/8/1k6/8/8/8/8/3KR3 w - - 0 1",
    // Even position with two rooks and a pawn for each side.
    "2r5/2r5/1k6/p7/7P/6K1/5R2/5R2 b - - 0 1",
};

Board RandomBoardWith(std::mt19937_64& rand, int num_rooks) {
  PieceColor barr[64] = {};
  // 2 rooks per side random position.
  int kings[2];
  bool occ[64] = {};
  kings[0] = rand() % 64;
  while (true) {
    kings[1] = rand() % 64;
    if (kings[0] == kings[1]) {
      continue;
    }
    if ((KingMoveMask(kings[0]) & OneHot(kings[1])) != 0) {
      continue;
    }
    break;
  }
  std::cout << "k " << Square::ToString(kings[0]) << " "
            << Square::ToString(kings[1]) << "\n";
  occ[kings[0]] = occ[kings[1]] = true;
  barr[kings[0]] = {Piece::kKing, Color::kWhite};
  barr[kings[1]] = {Piece::kKing, Color::kBlack};
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < num_rooks; ++i) {
      int r = rand() % 64;
      while (occ[r] || (SquareRank(r) == SquareRank(kings[c ^ 1])) ||
             (SquareFile(r) == SquareFile(kings[c ^ 1]))) {
        r = rand() % 64;
      }
      barr[r] = {Piece::kRook, Color(c)};
      occ[r] = true;
    }
  }
  // Add pawns in random locations (but not in rank 0 or 7 of course).

  return Board(barr);
}

Board CreateStartBoard(std::mt19937_64& rand) {
  return Board();
#if 0
  uint64_t r = rand();
  switch (r % 3) {
    case 0:
      return Board();
    case 1:
      return RandomBoardWith(rand, /*num_rooks=*/2);
    case 2:
      return RandomBoardWith(rand, /*num_rooks=*/1);
  }
  return Board();
#endif
}

void PlayerThread(int thread_i, generic::PredictionQueue* pred_queue,
                  generic::ShufflingTrainer* trainer) {
  auto player = std::make_unique<MCTSPlayer>(pred_queue, kNumIters);
  const int kGamesPerRefresh = 1;
  int games_to_refresh = kGamesPerRefresh;

  std::mt19937_64 rand(thread_i);
  while (true) {
    Game g({player.get()}, CreateStartBoard(rand));
    while (!g.is_over()) {
      g.Work();
    }
    std::cout << "Thread " << thread_i << ":\n"
              << g.board().ToPrintString() << "\n"
              << g.board().ToFEN() << "\n";
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

    // TODO: Write log for games
    for (const auto& state : player->saved_predictions()) {
      auto pred = state.pred;
      CHECK_EQ(pred.policy.size(), state.board.valid_moves().size());
      if (g.winner() == Color::kEmpty) {
        pred.value = 0;
      } else {
        pred.value = g.winner() == state.board.turn() ? 1.0 : -1.0;
      }
      trainer->Train(MakeGenericBoard(state.board), pred);
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
  const auto* const model_collection = GetModelCollection();
  auto model = generic::Model::Open(kModelPath,
                                    model_collection->CurrentCheckpointDir());
  if (model == nullptr) {
    LOG(INFO) << "Starting from scratch";
    model = generic::Model::New(kModelPath);
  } else {
    LOG(INFO) << "Continuing training";
  }

  generic::PredictionQueue pred_queue(model.get(), 256);
  generic::ShufflingTrainer trainer(model.get(), *MakeGenericBoard(Board()));
  std::vector<std::thread> threads;

  const int kNumThreads = 80;
  // const int kNumThreads = 1;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(
        [i, &pred_queue, &trainer] { PlayerThread(i, &pred_queue, &trainer); });
  }
  int64_t last_num_preds = 0;
  absl::Time last_log = absl::Now();
  while (true) {
    absl::SleepFor(absl::Seconds(5));

    model->Checkpoint(model_collection->CurrentCheckpointDir());
    // std::cout << "Saved checkpoint\n";

    absl::Time log_time = absl::Now();
    const int64_t num_preds = pred_queue.num_predictions();
    const double preds_per_sec = (num_preds - last_num_preds) /
                                 absl::ToDoubleSeconds(log_time - last_log);
    printf("Preds per sec: %.2f\n", preds_per_sec);
    printf("Avg batch size: %.2f\n", pred_queue.avg_batch_size());

    last_log = log_time;
    last_num_preds = num_preds;
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  NiceInit(argc, argv);
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  chess::PlayGames();
  return 0;
}
