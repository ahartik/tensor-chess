#include "c4cc/board_tensor.h"

namespace c4cc {

tensorflow::Tensor MakeBoardTensor(int batch_size) {
  return tensorflow::Tensor(tensorflow::DT_FLOAT,
                            tensorflow::TensorShape({batch_size, 84}));
}

void BoardToTensor(const Board& b, tensorflow::Tensor* tensor, int i) {
  static const Color kColorOrder[2][2] = {
      {Color::kOne, Color::kTwo},
      {Color::kTwo, Color::kOne},
  };
  int j = 0;
  for (Color c : kColorOrder[b.turn() == Color::kOne]) {
    for (int x = 0; x < 7; ++x) {
      for (int y = 0; y < 6; ++y) {
        const bool set = b.color(x, y) == c;
        tensor->matrix<float>()(i, j) = set ? 1.0 : 0.0;
        ++j;
      }
    }
  }
}

void ReadPredictions(const generic::Model::Prediction& tensor_pred,
                     Prediction* out_arr, int offset, int n) {
  for (int i = 0; i < n; ++i) {
    for (int m = 0; m < 7; ++m) {
      out_arr[i].move_p[m] = tensor_pred.move_p.matrix<float>()(offset + i, m);
    }
    out_arr[i].value = tensor_pred.value.flat<float>()(offset + i);
  }
}

void ReadPredictions(const generic::Model::Prediction& tensor_pred,
                     Prediction* out_arr) {
  const int n = tensor_pred.move_p.dim_size(0);
  ReadPredictions(tensor_pred, out_arr, 0, n);
}

}  // namespace c4cc
