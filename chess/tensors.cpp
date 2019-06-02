#include "chess/tensors.h"

#include <functional>

#include "chess/bitboard.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {

const int kMoveVectorSize = 73 * 64;

namespace {

int FlippedSquare(int x) {
  int r = SquareRank(x);
  int f = SquareFile(x);
  r = 7 - r;
  return r * 8 + f;
}

void SetBitBoard(uint64_t bb, bool flip, tensorflow::Tensor* tensor) {
  for (int i = 0; i < 64; ++i) {
    int ind = flip ? FlippedSquare(i) : i;
    bool set = BitIsSet(bb, ind);
    tensor->flat<float>()(i) = set ? 1.0 : 0.0;
  }
}

struct BitboardLayer {
  BitboardLayer(bool my, Piece p) : my_(my), p_(p) {}

  void operator()(const Board& b, tensorflow::Tensor* tensor) const {
    const Color c = my_ ? b.turn() : OtherColor(b.turn());
    SetBitBoard(b.bitboard(c, p_),
                /*flip=*/b.turn() == Color::kBlack, tensor);
  }

  bool my_;
  Piece p_;
};

using LayerFunc = std::function<void(const Board&, tensorflow::Tensor*)>;

const LayerFunc layers[] = {
    BitboardLayer(true, Piece::kPawn),
    BitboardLayer(true, Piece::kKnight),
    BitboardLayer(true, Piece::kBishop),
    BitboardLayer(true, Piece::kRook),
    BitboardLayer(true, Piece::kQueen),
    BitboardLayer(true, Piece::kKing),
    BitboardLayer(false, Piece::kPawn),
    BitboardLayer(false, Piece::kKnight),
    BitboardLayer(false, Piece::kBishop),
    BitboardLayer(false, Piece::kRook),
    BitboardLayer(false, Piece::kQueen),
    BitboardLayer(false, Piece::kKing),
    [](const Board& b, tensorflow::Tensor* tensor) {
      SetBitBoard(b.en_passant(), b.turn() == Color::kBlack, tensor);
    },
    [](const Board& b, tensorflow::Tensor* tensor) {
      SetBitBoard(b.castling_rights(), b.turn() == Color::kBlack, tensor);
    },
};
constexpr int kNumLayers = sizeof(layers) / sizeof(layers[0]);

static_assert(kNumLayers == 14, "Update build_graph.py if this number changes");

}  // namespace

int EncodeMove(Color turn, Move m) {
  if (turn == Color::kBlack) {
    m.to = FlippedSquare(m.to);
    m.from = FlippedSquare(m.from);
  }
  if (m.promotion == Piece::kNone || m.promotion == Piece::kQueen) {
    return 64 * m.from + m.to;
  }
  int encoded_from = 0;
  if (m.to == m.from + 8) {
    encoded_from = 64;
  } else if (m.to == m.from + 7) {
    encoded_from = 67;
  } else if (m.to == m.from + 9) {
    encoded_from = 70;
  }
  switch (m.promotion) {
    case Piece::kRook:
      break;
    case Piece::kBishop:
      encoded_from += 1;
      break;
    case Piece::kKnight:
      encoded_from += 2;
      break;
    default:
      LOG(FATAL) << "Invalid promotion " << m.promotion << "\n";
  }
  return encoded_from * 64 + m.to;
}

tensorflow::Tensor MakeBoardTensor(int batch_size) {
  return tensorflow::Tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({batch_size, kNumLayers, 64}));
}

void BoardToTensor(const Board& b, tensorflow::Tensor* tensor) {
  CHECK_EQ(tensor->dims(), 2);
  CHECK_EQ(tensor->dim_size(0), kNumLayers);
  CHECK_EQ(tensor->dim_size(1), 64);
  for (int layer = 0; layer < kNumLayers; ++layer) {
    auto slice = tensor->SubSlice(layer);
    layers[layer](b, &slice);
  }
}

double MovePriorFromTensor(const tensorflow::Tensor& tensor, Color turn,
                           const Move& m) {
  CHECK_EQ(tensor.dims(), 1);
  CHECK_EQ(tensor.dim_size(0), kMoveVectorSize);
  return tensor.vec<float>()(EncodeMove(turn, m));
}

}  // namespace chess
