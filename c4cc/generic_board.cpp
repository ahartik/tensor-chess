#include "c4cc/generic_board.h"

#include "absl/hash/hash.h"
#include "tensorflow/core/framework/tensor.h"

namespace c4cc {

namespace {

class GenericBoard : public generic::Board {
 public:
  GenericBoard(c4cc::Board b) : b_(b) {}
  GenericBoard(c4cc::Board b, int m) : b_(b) { b_.MakeMove(m); }

  std::unique_ptr<Board> Move(int move) const override {
    return std::make_unique<GenericBoard>(b_, move);
  }

  generic::BoardFP fingerprint() const override {
      return absl::Hash<c4cc::Board>()(b_);
  }

  std::vector<int> GetValidMoves() const override {
    std::vector<int> moves;
    for (const int move : b_.valid_moves()) {
      moves.push_back(move);
    }
    return moves;
  }

  // These are constant per game.
  tensorflow::TensorShape tensor_shape() const override {
    return tensorflow::TensorShape({84});
  }

  // This determines prediction tensor shape.
  int num_possible_moves() const override { return 7; }

  void ToTensor(tensorflow::Tensor* t) const override {
    static const Color kColorOrder[2][2] = {
        {Color::kOne, Color::kTwo},
        {Color::kTwo, Color::kOne},
    };
    int j = 0;
    for (Color c : kColorOrder[b_.turn() == Color::kOne]) {
      for (int x = 0; x < 7; ++x) {
        for (int y = 0; y < 6; ++y) {
          const bool set = b_.color(x, y) == c;
          t->vec<float>()(j) = set ? 1.0 : 0.0;
          ++j;
        }
      }
    }
  }

 private:
  c4cc::Board b_;
};
}  // namespace

std::unique_ptr<generic::Board> MakeGenericBoard(const Board& b) {
  return std::make_unique<GenericBoard>(b);
}

}  // namespace c4cc
