#ifndef _CHESS_GAME_STATE_H_
#define _CHESS_GAME_STATE_H_

#include "absl/container/flat_hash_map.h"
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

  // Must not be called if this game is already over.
  void Advance(const Move& m);

 private:
  Board board_;
  // Already visited nodes, for detecting threefold repetition.
  absl::flat_hash_map<uint64_t, int> visit_count_;
  bool is_over_ = false;
  Color winner_ = Color::kEmpty;
};

}  // namespace chess

#endif
