#include "tensorflow/core/framework/reader_op_kernel.h"

#include <iostream>

#include "chess/board.pb.h"

namespace chess {

using namespace ::tensorflow;
namespace {
constexpr const int kNumChannels = Board::NUM_LAYERS + 3;
constexpr const int kNumMoves = 73 * 64;
constexpr const int kHalfMoveLayer = Board::NUM_LAYERS;
constexpr const int kRepLayer = Board::NUM_LAYERS + 1;
constexpr const int kNoProgressLayer = Board::NUM_LAYERS + 2;
}  // namespace

class DecodeBoardOp : public OpKernel {
 public:
  explicit DecodeBoardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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

    // std::cout << "Parsed: " << board_msg.DebugString();

    // TODO: Optimize to call int-taking functions
    Tensor* board = nullptr;
    Tensor* move = nullptr;
    Tensor* result = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, tensorflow::TensorShape{kNumChannels, 64},
                                  &board));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, tensorflow::TensorShape{}, &move));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(2, tensorflow::TensorShape{}, &result));

    OP_REQUIRES(ctx, board_msg.layers_size() == Board::NUM_LAYERS,
                errors::InvalidArgument("Board message is missing layers"));

    for (int i = 0; i < Board::NUM_LAYERS; ++i) {
      for (int j = 0; j < 64; ++j) {
        if ((board_msg.layers(i) >> j) & 1) {
          board->matrix<float>()(i, j) = 1.0f;
        } else {
          board->matrix<float>()(i, j) = 0.0f;
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
    int encoded_move = 64 * board_msg.move_from() + board_msg.move_to();
    if (board_msg.encoded_move_to() >= 64) {
      encoded_move = board_msg.encoded_move_to() * 64 + board_msg.move_to();
    }
    move->scalar<int>()() = encoded_move;
    result->scalar<float>()() = board_msg.game_result();
  }

 private:
#if 0
  const std::vector<tensorflow::DataType> out_type_ = {
      DT_FLOAT,
      DT_INT32,
      DT_INT32,
  };
#endif
};
}  // namespace chess

namespace tensorflow {

REGISTER_OP("DecodeBoard")
    .Input("encoded: string")
    .Output("board: float32")
    .Output("move: int32")
    .Output("score: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Matrix(chess::kNumChannels, 64));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Parses chessboard from proto to tensor format.
)doc");

REGISTER_KERNEL_BUILDER(Name("DecodeBoard").Device(DEVICE_CPU),
                        chess::DecodeBoardOp);
}  // namespace tensorflow
