#include "chess/game-data.h"

#include <cstring>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace chess {
namespace {

using ::std::string;
using ::tensorflow::DT_FLOAT;
using ::tensorflow::DataTypeVector;
using ::tensorflow::Env;
using ::tensorflow::IteratorBase;
using ::tensorflow::IteratorContext;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::RandomAccessFile;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::error::Code;

}  // namespace

namespace {

class GameDataIteratorImpl : public GameDataIterator {
 public:
  GameDataIteratorImpl(std::unique_ptr<RandomAccessFile> input_file)
      : input_file_(std::move(input_file)) {}

  bool ReadData(int batch_size, DataBatch* out) override {
    out->board = Tensor(DT_FLOAT, TensorShape({batch_size, 8, 8, 14}));
    out->move = Tensor(DT_FLOAT, TensorShape({batch_size, 64 * 64}));
    out->result = Tensor(DT_FLOAT, TensorShape({batch_size, 1}));
    for (int i = 0; i < batch_size; ++i) {
      // 1. Read data
      RawBoardData raw;
      char scratch[sizeof(RawBoardData)];
      tensorflow::StringPiece result;
      const Status read_status =
          input_file_->Read(offset_, sizeof(RawBoardData), &result, scratch);
      if (read_status.code() == Code::OUT_OF_RANGE) {
        return false;
      }
      // Crash on any other error.
      TF_CHECK_OK(read_status);
      memcpy(&raw, result.data(), sizeof(raw));
      offset_ += sizeof(raw);

      // 2. Put it in output tensors
      for (int piece = 0; piece < 14; ++piece) {
        for (int pos = 0; pos < 64; ++pos) {
          const int rank = pos / 8;
          const int file = pos % 8;
          if ((raw.data[piece] >> pos) & 1) {
            out->board.tensor<float, 4>()(i, rank, file, piece) = 1.0;
          }
        }
      }

      out->move.tensor<float, 2>()(i, raw.move) = 1.0;
      out->result.tensor<float, 2>()(i, 0) = 1.0;
    }
    return true;
  }

 private:
  const std::unique_ptr<RandomAccessFile> input_file_;
  uint64_t offset_ = 0;
};

}  // namespace

std::unique_ptr<GameDataIterator> ReadGameDataFromFile(
    tensorflow::Env* env, const std::string& path) {
  std::unique_ptr<RandomAccessFile> input_file;
  TF_CHECK_OK(env->NewRandomAccessFile(path, &input_file));
  return absl::make_unique<GameDataIteratorImpl>(std::move(input_file));
}

}  // namespace chess
