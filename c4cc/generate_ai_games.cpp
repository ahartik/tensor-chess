#include "c4cc/game.pb.h"
#include "c4cc/negamax.h"
#include "c4cc/play_game.h"
#include "tensorflow/core/platform/logging.h"
#include "util/init.h"
#include "util/recordio.h"

#include <unistd.h>
#include <iostream>
#include <random>

namespace c4cc {
namespace {

int depth = 6;

std::mt19937 mtrand;

int PickMove(const Board& b) {
  // With 1/8 chance pick random move to add variety to games.
  auto r = Negamax(b, depth);
  // If we're winning, always use best move.
  if (r.eval > 1000000) {
    return r.best_move;
  }
  return r.best_move;
}

int PickMoveWithRandom(const Board& b) {
  // With 1/8 chance pick random move to add variety to games.
  auto r = Negamax(b, depth);
  // If we're winning, always use best move.
  if (r.eval > 1000000) {
    return r.best_move;
  }
  if (mtrand() % 8 == 0) {
    auto valid = b.valid_moves();
    return valid[mtrand() % valid.size()];
  }
  return r.best_move;
}

void PlayGames(const char* output, int n) {
  mtrand.seed(getpid() + mtrand());
  util::RecordWriter writer(output);

  std::function<int(const Board&)> ais[] = {
    &PickMove,
    &PickMoveWithRandom
  };
  for (int i = 0; i < n; ++i) {
    const Color good_ai = (i % 2) ? Color::kOne : Color::kTwo;
    Board result;
    GameRecord record;
    if (good_ai == Color::kOne) {
      record.add_players(GameRecord::MINMAX);
      record.add_players(GameRecord::MINMAX_RANDOM);
      result = PlayGame(&PickMove, &PickMoveWithRandom);
    } else {
      record.add_players(GameRecord::MINMAX_RANDOM);
      record.add_players(GameRecord::MINMAX);
      result = PlayGame(&PickMoveWithRandom, &PickMove);
    }
    LOG(INFO) << "Board: \n" << result; 
    switch (result.result()) {
      case Color::kEmpty:
        record.set_game_result(0);
        break;
      case Color::kOne:
        record.set_game_result(1);
        break;
      case Color::kTwo:
        record.set_game_result(-1);
        break;
    }
    for (const int move : result.history()) {
      record.add_moves(move);
    }
    CHECK(writer.Write(record.SerializeAsString()));
    LOG(INFO) << "Result: " << record.game_result();
    writer.Flush();
  }
  CHECK(writer.Finish());
}

}  // namespace
}  // namespace c4cc

int main(int argc, const char** argv) {
  NiceInit(argc, argv);
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " output-file num-games\n";
    return 1;
  }
  const char* output = argv[1];
  const int num = atoi(argv[2]);
  c4cc::PlayGames(output, num);
  return 0;
}
