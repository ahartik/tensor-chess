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

int move_encoding[16][16] = {};
constexpr int kCenter = 8;
int InitMoves() {
  memset(move_encoding, 0xff, sizeof(move_encoding));
  // Queen moves.
  int dr[] = {1, 1, 0, -1, -1, -1, 0, 1};
  int df[] = {0, 1, 1, 1, 0, -1, -1, -1};
  int m = 0;
  for (int d = 0; d < 8; ++d) {
    for (int i = 1; i <= 7; ++i) {
      int r = kCenter + i * dr[d];
      int f = kCenter + i * df[d];
      CHECK_EQ(move_encoding[r][f], -1);
      move_encoding[r][f] = m++;
    }
  }
  CHECK_EQ(m, 56);
  // Knight moves
  for (int dr : {1, 2}) {
    const int df = dr ^ 3;
    // Either can be positive or negative.
    for (int i = 0; i < 4; ++i) {
      int result_r = kCenter + dr * ((i & 1) ? 1 : -1);
      int result_f = kCenter + df * ((i & 2) ? 1 : -1);
      move_encoding[result_r][result_f] = m++;
    }
  }
  CHECK_EQ(m, 64);
  return 0;
}

const int dummy = InitMoves();

int EncodeMoveTo(const Move& m) {
  if (m.promotion != Piece::kNone && m.promotion != Piece::kQueen) {
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
    return encoded_from;
  }
  int dr = SquareRank(m.to) - SquareRank(m.from);
  int df = SquareFile(m.to) - SquareFile(m.from);
  int x = move_encoding[kCenter + dr][kCenter + df];
  CHECK_GE(x, 0) << "dr=" << dr << " df=" << df << " m= " << m;
  return x;
}

}  // namespace

int EncodeMove(Color turn, Move m) {
  if (turn == Color::kBlack) {
    m.to = FlippedSquare(m.to);
    m.from = FlippedSquare(m.from);
  }
  return 64 * EncodeMoveTo(m) + m.from;
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
