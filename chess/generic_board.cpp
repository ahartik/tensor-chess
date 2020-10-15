#include "chess/generic_board.h"

#include "absl/hash/hash.h"
#include "chess/movegen.h"
#include "chess/tensors.h"
#include "tensorflow/core/framework/tensor.h"

namespace chess {

namespace {

class GenericBoard : public generic::Board {
 public:
  GenericBoard(chess::Board b) : b_(b) {
    game_state_ = IterateLegalMoves(b, [](const chess::Move& m) {});
  }

  GenericBoard(chess::Board b, MovegenResult state)
      : b_(b), game_state_(state) {}

  std::unique_ptr<Board> Move(int move) const override {
    return std::make_unique<GenericBoard>(
        chess::Board(b_, DecodeMove(b_, move)));
  }

  std::unique_ptr<Board> Clone() const override {
    return std::make_unique<GenericBoard>(b_, game_state_);
  }

  generic::BoardFP fingerprint() const override { return BoardFingerprint(b_); }

  std::vector<int> GetValidMoves() const override {
    std::vector<int> moves;
    IterateLegalMoves(b_, [this, &moves](const chess::Move& m) {
      moves.push_back(EncodeMove(b_.turn(), m));
    });
    return moves;
  }

  bool is_over() const override {
    return game_state_ != MovegenResult::kNotOver;
  }

  int result() const override {
    switch (game_state_) {
      case MovegenResult::kCheckmate:
        return -1;
      case MovegenResult::kStalemate:
        return 0;
      case MovegenResult::kNotOver:
        std::cerr << "Invalid call to chess::GenericBoard::result(), game not "
                     "over. fen "
                  << b_.ToFEN() << "\n";
        return 0;
      default:
        std::cerr << "Invalid game_state_" << static_cast<int>(game_state_)
                  << "\n";
        abort();
        return 0;
    }
  }

  int turn() const override { return b_.turn() == Color::kWhite ? 0 : 1; }

  // These are constant per game.
  void GetTensorShape(int n, tensorflow::TensorShape* shape) const override {
    *shape = tensorflow::TensorShape({n, kBoardTensorNumLayers, 64});
  }

  // This determines prediction tensor shape.
  int num_possible_moves() const override { return kMoveVectorSize; }

  void ToTensor(tensorflow::Tensor* t, int i) const override {
    BoardToTensor(b_, t->SubSlice(i));
  }

 private:
  chess::Board b_;
  MovegenResult game_state_;
};

}  // namespace

std::unique_ptr<generic::Board> MakeGenericBoard(const Board& b) {
  return std::make_unique<GenericBoard>(b);
}

}  // namespace chess
