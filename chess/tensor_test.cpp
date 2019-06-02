#include <iostream>

#include "chess/board.h"
#include "chess/model.h"
#include "chess/tensors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {

void Go() {
  Board b;
  auto tensor = MakeBoardTensor(1);
  BoardToTensor(b, &tensor, 0);

  for (int i = 0; i < tensor.dim_size(1); ++i) {
    printf("\nLayer %i\n", i);
    auto layer = tensor.SubSlice(0).SubSlice(i);
    for (int r = 7; r >= 0; --r) {
      for (int f = 0; f < 8; ++f) {
        float val = layer.vec<float>()(r * 8 + f);
        printf("%.1f ", val);
      }
      printf("\n");
    }
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  chess::Go();
  return 0;
}
