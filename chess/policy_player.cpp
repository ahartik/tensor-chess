#include "chess/policy_player.h"

#include <random>

#include "chess/generic_board.h"
#include "chess/tensors.h"

namespace chess {
namespace {

class PolicyPlayer : public Player {
 public:
  explicit PolicyPlayer(generic::PredictionQueue* queue) : queue_(queue) {}

  void Reset(const Board& b) override { b_ = b; }

  void Advance(const Move& m) override { b_ = Board(b_, m); }

  Move GetMove() override {
    auto generic_board = MakeGenericBoard(b_);

    generic::PredictionQueue::Request req;
    req.board = generic_board.get();
    queue_->GetPredictions(&req, 1);

    std::cerr << "Value before making move: " << req.result.value << "\n";
    std::cerr << "Valid moves: ";
    for (const auto [move, p] : req.result.policy) {
      std::cerr << DecodeMove(b_, move) << " " << p << "\n";
    }
    std::cerr << "\n";

    std::uniform_real_distribution<double> rand(0.0, 1.0);
    double r = rand(mt_);
    for (const auto& move : req.result.policy) {
      r -= move.second;
      if (r < 0) {
        return DecodeMove(b_, move.first);
      }
    }
    std::cerr << "This shouldn't happen\n";
    return b_.valid_moves().at(0);
  }

 private:
  generic::PredictionQueue* const queue_;
  std::mt19937_64 mt_;
  Board b_;
};

}  // namespace

std::unique_ptr<Player> MakePolicyPlayer(generic::PredictionQueue* p_queue) {}

}  // namespace chess
