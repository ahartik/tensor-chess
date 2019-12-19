#ifndef _C4CC_MODEL_H_
#define _C4CC_MODEL_H_

#include <atomic>

#include "absl/synchronization/mutex.h"
#include "c4cc/board.h"
#include "generic/model.h"
#include "tensorflow/core/framework/tensor.h"

namespace c4cc {

tensorflow::Tensor MakeBoardTensor(int batch_size);

// Writes 'b' to tensor batch at index i. 'tensor' should have been created
// using MakeBoardTensor().
void BoardToTensor(const Board& b, tensorflow::Tensor* tensor, int i);

void ReadPredictions(const generic::Model::Prediction& tensor_pred,
                     Prediction* out_arr);
void ReadPredictions(const generic::Model::Prediction& tensor_pred,
                     Prediction* out_arr, int offset, int n);

}  // namespace c4cc

#endif
