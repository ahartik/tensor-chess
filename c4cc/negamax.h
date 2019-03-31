#ifndef _C4CC_NEGAMAX_H_
#define _C4CC_NEGAMAX_H_

#include "c4cc/board.h"
#include "c4cc/play_game.h"

namespace c4cc {

int32_t StaticEval(const Board& b);

// Returns eval and best move for the current player.
struct NegamaxResult {
  int32_t eval;
  int best_move;
};
NegamaxResult Negamax(const Board& b, int depth);

class NegamaxPlayer : public Player {
 public:
  explicit NegamaxPlayer(int depth) : depth_(depth) {}
  ~NegamaxPlayer() override;

  const Board& board() const override { return current_board_; }
  void SetBoard(const Board& b) override { current_board_ = b; }
  void PlayMove(int move) override { current_board_.MakeMove(move); }
  int GetMove() override { return Negamax(current_board_, depth_).best_move; }

 private:
  // TODO: Implement caching and stuff.
  int depth_;
  Board current_board_;
};

}  // namespace c4cc

#endif
