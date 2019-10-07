#ifndef _CHESS_GAME_STATE_H_
#define _CHESS_GAME_STATE_H_

#include "absl/container/flat_hash_map.h"

#include "chess/board.h"
#include "chess/player.h"

namespace chess {

class Game {
 public:
  explicit Game(const std::vector<Player*>& players, Board start = Board());

  //
  const Board& board() const { return board_; }
  // Whether this game is over or not.
  bool is_over() const { return is_over_; }
  // Winner, or kEmpty if this game was a draw. Requires 'is_over()'.
  Color winner() const { return winner_; }

  // Must not be called if this game is already over.
  void Advance(const Move& m);

  // Must not be called if this game is already over.
  void Work();

 private:
  // Has either 1 or 2.
  const std::vector<Player*> players_;

  Board board_;
  // Already visited nodes, for detecting threefold repetition.
  absl::flat_hash_map<uint64_t, int> visit_count_;
  bool is_over_ = false;
  Color winner_ = Color::kEmpty;
};

}  // namespace chess

#endif
