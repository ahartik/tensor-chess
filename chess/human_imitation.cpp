#include <time.h>

#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "absl/time/time.h"
#include "chess/board.h"
#include "chess/model.h"
#include "chess/shuffling_trainer.h"
#include "chess/tensors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "util/recordio.h"

namespace chess {

int TrainGame(ShufflingTrainer* trainer, const GameRecord& record) {
  Board board;

  int count_moves = 0;
  for (const MoveProto& move_proto : record.moves()) {
    const Move m = Move::FromProto(move_proto);
    auto sample = std::make_unique<TrainingSample>();
    sample->board = board;
    sample->moves.emplace_back(m, 1.0);
    switch (record.result()) {
      case 1:
        sample->winner = Color::kWhite;
        break;
      case -1:
        sample->winner = Color::kBlack;
        break;
      case 0:
        sample->winner = Color::kEmpty;
        break;
    }
    trainer->Train(std::move(sample));

    board = Board(board, m);
    ++count_moves;
  }
  return count_moves;
}
void TrainerThread(int tid, ShufflingTrainer* trainer,
                   const std::vector<std::string>& files) {
  std::mt19937_64 mt(time(0) ^ tid);
  while (true) {
    const int file_ind = mt() % files.size();
    const std::string fname = files[file_ind];
    std::cout << "===============\n";
    std::cout << "Opening " << fname << "\n";
    std::cout << "===============\n";
    GameRecord record;
    util::RecordReader reader(fname);
    std::string buf;
    int count = 0;
    int moves = 0;
    while (reader.Read(buf)) {
      CHECK(record.ParseFromString(buf));
      moves += TrainGame(trainer, record);
      ++count;
    }
    std::cout << fname << " had " << count << " games with " << moves
              << " moves\n";
  }
}

void TrainFiles(const std::vector<std::string>& files) {
  auto model = CreateDefaultModel(/*allow_init=*/true);

  ShufflingTrainer trainer(model.get());
  std::vector<std::thread> threads;

  const int kNumThreads = 4;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(
        [i, &trainer, &files] { TrainerThread(i, &trainer, files); });
  }

  while (true) {
    model->Checkpoint(GetDefaultCheckpoint());
    std::cout << "Saved checkpoint\n";
    absl::SleepFor(absl::Seconds(5));
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    files.emplace_back(argv[i]);
  }

  chess::TrainFiles(files);
  return 0;
}
