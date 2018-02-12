#include "tensorflow/core/framework/reader_op_kernel.h"

#include "chess/board.pb.h"

namespace chess {

using namespace ::tensorflow;

class DecodeBoardOp : public OpKernel {
 public:
  explicit DecodeBoardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  const int kNumChannels = Board::NUM_LAYERS + 3;
  const int kNumMoves = 73 * 64;
  const int kHalfMoveLayer = Board::NUM_LAYERS;
  const int kRepLayer = Board::NUM_LAYERS + 1;
  const int kNoProgressLayer = Board::NUM_LAYERS + 2;

  void Compute(tensorflow::OpKernelContext* ctx) override {
    const Tensor* encoded;
    OpInputList record_defaults;

    OP_REQUIRES_OK(ctx, ctx->input("encoded", &encoded));

    auto encoded_t = encoded->flat<string>();
    OP_REQUIRES(ctx, encoded_t.size() == 1,
                errors::InvalidArgument("Must only have 1 string per op"));

    StringPiece as_str(encoded_t(0));
    Board board_msg;
    const bool can_parse =
        board_msg.ParseFromArray(as_str.data(), as_str.size());
    OP_REQUIRES(ctx, can_parse,
                errors::InvalidArgument("Failed to parse Board proto"));

    OpOutputList output;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));

    Tensor* board = nullptr;
    Tensor* move = nullptr;
    Tensor* result = nullptr;
    OP_REQUIRES_OK(
        ctx,
        output.allocate(0, tensorflow::TensorShape{kNumChannels, 64}, &board));
    OP_REQUIRES_OK(ctx, output.allocate(1, tensorflow::TensorShape{1}, &move));
    OP_REQUIRES_OK(ctx,
                   output.allocate(2, tensorflow::TensorShape{1}, &result));

    OP_REQUIRES(ctx, board_msg.layers_size() == Board::NUM_LAYERS,
                errors::InvalidArgument("Board message is missing layers"));

    for (int i = 0; i < Board::NUM_LAYERS; ++i) {
      for (int j = 0; j < 64; ++j) {
        if ((board_msg.layers(i) >> j) & 1) {
          board->matrix<float>()(i, j) = 1.0f;
        }
      }
    }
    for (int j = 0; j < 64; ++j) {
      board->matrix<float>()(kHalfMoveLayer, j) =
          0.01 * board_msg.half_move_count();
      board->matrix<float>()(kRepLayer, j) =
          0.01 * board_msg.repetition_count();
      board->matrix<float>()(kNoProgressLayer, j) =
          0.01 * board_msg.no_progress_count();
    }
  }

 private:
#if 0
  const std::vector<tensorflow::DataType> out_type_ = {
      DT_FLOAT,
      DT_INT32,
      DT_FLOAT,
  };
#endif
};
}  // namespace chess

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("DecodeBoard").Device(DEVICE_CPU),
                        chess::DecodeBoardOp);
}  // namespace tensorflow
