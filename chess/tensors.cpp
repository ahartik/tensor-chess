#include "chess/tensors.h"

#include <functional>

#include "chess/bitboard.h"

namespace chess {

namespace {

void SetBitBoard(uint64_t bb, tensorflow::Tensor* tensor) {
  for (int i = 0; i < 64; ++i) {
    bool set = BitIsSet(bb, i);
    tensor->flat<float>()(i) = set ? 1.0 : 0.0;
  }
}

struct BitboardLayer {
  BitboardLayer(Color c, Piece p) : c_(c), p_(p) {}

  void operator()(const Board& b, tensorflow::Tensor* tensor) const {
    SetBitBoard(b.bitboard(c_, p_), tensor);
  }

  Color c_;
  Piece p_;
};

using LayerFunc = std::function<void(const Board&, tensorflow::Tensor*)>;

const LayerFunc layers[] = {
    BitboardLayer(Color::kWhite, Piece::kPawn),
    BitboardLayer(Color::kWhite, Piece::kKnight),
    BitboardLayer(Color::kWhite, Piece::kBishop),
    BitboardLayer(Color::kWhite, Piece::kRook),
    BitboardLayer(Color::kWhite, Piece::kQueen),
    BitboardLayer(Color::kWhite, Piece::kKing),
    BitboardLayer(Color::kBlack, Piece::kPawn),
    BitboardLayer(Color::kBlack, Piece::kKnight),
    BitboardLayer(Color::kBlack, Piece::kBishop),
    BitboardLayer(Color::kBlack, Piece::kRook),
    BitboardLayer(Color::kBlack, Piece::kQueen),
    BitboardLayer(Color::kBlack, Piece::kKing),
    [](const Board& b, tensorflow::Tensor* tensor) {
      SetBitBoard(b.en_passant(), tensor);
    },
    [](const Board& b, tensorflow::Tensor* tensor) {
      SetBitBoard(b.castling_rights(), tensor);
    },
};
constexpr int kNumLayers = sizeof(layers) / sizeof(layers[0]);

}  // namespace

tensorflow::Tensor MakeBoardTensor(int batch_size) {
  return tensorflow::Tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({batch_size, kNumLayers, 64}));
}
void BoardToTensor(const Board& b, tensorflow::Tensor* tensor) {
  for (int layer = 0; layer < kNumLayers; ++layer) {
    auto slice = tensor->SubSlice(layer);
    layers[layer](b, &slice);
  }
}

double MovePriorFromTensor(const tensorflow::Tensor& tensor, int i,
                           const Move& m) {
  return 0.0;
}

double BoardValueFromTensor(const tensorflow::Tensor& tensor, int i) {
  return 0.0;
}

}  // namespace chess
