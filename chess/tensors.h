// Converting chess game objects to tensors and back.
#ifndef _CHESS_TENSORS_H_
#define _CHESS_TENSORS_H_

#include "chess/board.h"
#include "tensorflow/core/framework/tensor.h"

namespace chess {

// Returns an unset tensor of the shape to hold 'batch_size' input boards.
tensorflow::Tensor MakeBoardTensor(int batch_size);
// Writes board state to tensor at given index. Here i < batch_size used to
// create the tensor.
void BoardToTensor(const Board& b, tensorflow::Tensor* tensor);

double MovePriorFromTensor(const tensorflow::Tensor& tensor, const Move& m);

double BoardValueFromTensor(const tensorflow::Tensor& tensor, int i);

}  // namespace chess

#endif
