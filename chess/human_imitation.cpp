#include <iostream>

#include "chess/board.h"
#include "chess/model.h"
#include "chess/shuffling_trainer.h"
#include "chess/tensors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {

void TrainFiles(const std::vector<std::string>& files) {
  auto model = CreateDefaultModel(/*allow_init=*/true);

  ShufflingTrainer trainer(model.get());
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
