// Converting chess game objects to tensors and back.
#ifndef _CHESS_TENSORS_H_
#define _CHESS_TENSORS_H_

#include "chess/board.h"
#include "tensorflow/core/framework/tensor.h"

namespace chess {

extern const int kMoveVectorSize;
// Returns an unset tensor of the shape to hold 'batch_size' input boards.
tensorflow::Tensor MakeBoardTensor(int batch_size);
tensorflow::Tensor MakeMoveTensor(int batch_size);
// Writes board state to tensor at given index. Here i < batch_size used to
// create the tensor.
void BoardToTensor(const Board& b, tensorflow::Tensor* tensor);

int EncodeMove(Color turn, Move m);

double MovePriorFromTensor(const tensorflow::Tensor& tensor, Color turn,
                           const Move& m);

}  // namespace chess

#endif
