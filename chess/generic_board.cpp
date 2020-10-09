#include "chess/generic_board.h"

#include "chess/movegen.h"
#include "chess/tensors.h"
#include "absl/hash/hash.h"
#include "tensorflow/core/framework/tensor.h"

namespace chess {

namespace {

class GenericBoard : public generic::Board {
 public:
  GenericBoard(chess::Board b) : b_(b) {
    game_state_ = IterateLegalMoves(b, [](const chess::Move& m) {});
  }

  std::unique_ptr<Board> Move(int move) const override {
    return std::make_unique<GenericBoard>(
        Board(b_, 
  }

  generic::BoardFP fingerprint() const override {
    return absl::Hash<chess::Board>()(b_);
  }

  std::vector<int> GetValidMoves() const override {
    std::vector<int> moves;
    IterateLegalMoves(b_, [this, &moves]
        (const chess::Move& m) {
        moves.push_back(EncodeMove(b_.turn(), m));
        });
    return moves;
  }

  bool is_over() const override { return b_.is_over(); }
  int result() const override {
    const auto winner = b_.result();
    if (winner == Color::kEmpty) {
      return 0;
    }
    if (winner == b_.turn()) {
      return 1;
    } else {
      return -1;
    }
  }

  int turn() const override { return b_.turn() == Color::kOne ? 0 : 1; }

  // These are constant per game.
  void GetTensorShape(int n, tensorflow::TensorShape* shape) const override {
    *shape = tensorflow::TensorShape({n, 84});
  }

  // This determines prediction tensor shape.
  int num_possible_moves() const override { return 7; }

  void ToTensor(tensorflow::Tensor* t, int i) const override {
    static const Color kColorOrder[2][2] = {
        {Color::kOne, Color::kTwo},
        {Color::kTwo, Color::kOne},
    };
    int j = 0;
    for (Color c : kColorOrder[b_.turn() == Color::kOne]) {
      for (int x = 0; x < 7; ++x) {
        for (int y = 0; y < 6; ++y) {
          const bool set = b_.color(x, y) == c;
          t->matrix<float>()(i, j) = set ? 1.0 : 0.0;
          ++j;
        }
      }
    }
  }

 private:
  chess::Board b_;
  std::vector<int> encoded_moves_;
  MovegenResult game_state_;
};

}  // namespace

std::unique_ptr<generic::Board> MakeGenericBoard(const Board& b) {
  return std::make_unique<GenericBoard>(b);
}

}  // namespace chess
