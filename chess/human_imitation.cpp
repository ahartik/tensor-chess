#include <iostream>
#include <memory>
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

void TrainGame(ShufflingTrainer* trainer, const GameRecord& record) {
  Board board;

  for (const MoveProto& move_proto : record.moves()) {
    const Move m = Move::FromProto(move_proto);
    auto sample = std::make_unique<TrainingSample>();
    sample->board = board;
    sample->moves.emplace_back(m, 1.0);
    sample->value = record.result();
    if (board.turn() == Color::kBlack) {
      sample->value *= -1;
    }
    trainer->Train(std::move(sample));

    board = Board(board, m);
  }
}

void TrainerThread(ShufflingTrainer* trainer,
                   const std::vector<std::string>& files) {
  while (true) {
    const int file_ind = rand() % files.size();
    const std::string fname = files[file_ind];
    GameRecord record;
    util::RecordReader reader(fname);
    std::string buf;
    while (reader.Read(buf)) {
      CHECK(record.ParseFromString(buf));
      TrainGame(trainer, record);
    }
  }
}

void TrainFiles(const std::vector<std::string>& files) {
  auto model = CreateDefaultModel(/*allow_init=*/true);

  ShufflingTrainer trainer(model.get());
  std::vector<std::thread> threads;

  const int kNumThreads = 2;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&] { TrainerThread(&trainer, files); });
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
