#ifndef _CHESS_GAME_DATA_H_
#define _CHESS_GAME_DATA_H_

#include <cstdint>
#include <istream>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace chess {

enum class Piece {
  kWP = 0,
  kWR = 1,
  kWB = 2,
  kWN = 3,
  kWQ = 4,
  kWK = 5,
  kBP = 6,
  kBR = 7,
  kBB = 8,
  kBN = 9,
  kBQ = 10,
  kBK = 11,
  kEnPassant = 12,
  kCastleRight = 13,
};

struct RawBoardData {
  // Bitmask indexed by Piece enum.
  uint64_t data[14];
  int32_t move;
  int32_t result;  // Either -1, 0, or 1
};

struct DataBatch {
  tensorflow::Tensor board;
  tensorflow::Tensor move;
  tensorflow::Tensor result;
};

class GameDataIterator {
 public:
  virtual ~GameDataIterator();
  // Returns false on end-of-file, that is, if there are less than n items left
  // in the file.
  virtual bool ReadData(int batch_size, DataBatch* out);
};

std::unique_ptr<GameDataIterator> ReadGameDataFromFile(tensorflow::Env* env,
                                           const std::string& path);

}  // namespace chess

#endif  // _CHESS_GAME_DATA_H_
