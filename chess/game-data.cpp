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

bool ReadData(std::istream& in, int n, std::vector<RawBoardData>* out) {
  out->reserve(out->size() + n);
  for (int i = 0; i < n; ++i) {
    char buf[sizeof(RawBoardData)];
    if (!in.read(buf, sizeof(RawBoardData))) {
      return i == 0 ? false : true;
    }
    out->emplace_back();
    memcpy(&out->back(), buf, sizeof(RawBoardData));
  }
  return false;
}
}  // namespace

tensorflow::Status ReadGameDataFromFile(std::istream& in, int batch_size,
                                        tensorflow::Tensor* out) {
  std::vector<RawBoardData> data;
  if (!ReadData(in, batch_size, &data)) {
    return tensorflow::Status(tensorflow::error::Code::OUT_OF_RANGE,
                              "end of file reached");
  }
  *out = tensorflow::Tensor(tensorflow::DT_FLOAT,
                            tensorflow::TensorShape({batch_size, 64, 14}));

  return tensorflow::Status::OK();
}

namespace {

class GameDataset : public tensorflow::DatasetBase {
 public:
  explicit GameDataset(tensorflow::Env* env, const string& path)
      : env_(env), path_(path) {}

  std::unique_ptr<IteratorBase> MakeIterator(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

  string DebugString() override { return absl::StrCat("GameDataset:", path_); }

 private:
  tensorflow::Env* const env_;
  const string path_;

  const DataTypeVector output_dtypes_ = {
      tensorflow::DT_FLOAT, tensorflow::DT_FLOAT, tensorflow::DT_FLOAT};
  const std::vector<PartialTensorShape> output_shapes_ = {
      {64, 14}, {64 * 64}, {1}};
};

class GameDatasetIterator : public tensorflow::DatasetIterator<GameDataset> {
 public:
  using Base = tensorflow::DatasetIterator<GameDataset>;
  struct Params {
    const GameDataset* dataset = nullptr;
    string prefix;
    RandomAccessFile* input_file;
  };

  GameDatasetIterator(Params params)
      : Base(Base::Params{params.dataset, params.prefix}),
        input_file_(params.input_file) {}

 protected:
  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    *end_of_sequence = false;

    RawBoardData raw;
    {
      char scratch[sizeof(RawBoardData)];
      tensorflow::StringPiece result;
      const Status read_status =
          input_file_->Read(offset_, sizeof(RawBoardData), &result, scratch);
      if (read_status.code() == Code::OUT_OF_RANGE) {
        *end_of_sequence = true;
        return Status::OK();
      } else if (!read_status.ok()) {
        return read_status;
      }
      memcpy(&raw, result.data(), sizeof(raw));
      offset += sizeof(raw);
    }

    out_tensors->resize(3);
    Tensor& board = (*out_tensors)[0];
    Tensor& move = (*out_tensors)[1];
    Tensor& result = (*out_tensors)[2];

    auto* allocator = ctx->allocator({});

    board = Tensor(allocator, DT_FLOAT, TensorShape({64, 14}));
    move = Tensor(allocator, DT_FLOAT, TensorShape({64 * 64}));
    result = Tensor(allocator, DT_FLOAT, TensorShape({1}));

    for (int p = 0; p < 14; ++p) {
      for (int i = 0; i < 64; ++i) {
        if ((raw.data[p] >> i) & 1) {
          board.matrix<float>()(i, p) = 1.0;
        }
      }
    }
    move.flat<float>()(raw.move) = 1.0;

    result.flat<float>()(0) = raw.result;

    return Status ::OK();
  }

 private:
  RandomAccessFile* input_file_;
  uint64_t offset_ = 0;
};

std::unique_ptr<IteratorBase> GameDataset::MakeIterator(
    const string& prefix) const {
  GameDatasetIterator::Params params;
  params.dataset = this;
  params.prefix = prefix;
  params.input_file = nullptr;

  return absl::make_unique<GameDatasetIterator>(std::move(params));
}

}  // namespace

std::unique_ptr<tensorflow::DatasetBase> DatasetFromFile(const string& path) {
  return absl::make_unique<GameDataset>(path);
}

}  // namespace chess
