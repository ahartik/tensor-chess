#ifndef _CHESS_GAME_STATE_H_
#define _CHESS_GAME_STATE_H_

#include "absl/container/node_hash_map.h"
#include "chess/board.h"

namespace chess {

class Game {
 public:
  Game();

  const Board& board() const { return board_; }

  // Whether this game is over or not.
  bool is_over() const { return is_over_; }
  // Winner, or kEmpty if this game was a draw. Requires 'is_over()'.
  Color winner() const { return winner_; }

  void MakeMove(const Move& m);

 private:
  Board board_;
  // Already visited nodes, for detecting threefold repetition.
  absl::node_hash_map<Board, int> visit_count_;
  bool is_over_ = false;
  Color winner_ = Color::kEmpty;
};

}  // namespace chess

#endif
