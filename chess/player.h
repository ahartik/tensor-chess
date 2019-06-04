#ifndef _CHESS_PLAYER_H_
#define _CHESS_PLAYER_H_

#include <functional>

#include "chess/board.h"
#include "chess/types.h"

namespace chess {

// This class is thread-compatible (but not thread-safe).
class Player {
 public:
  virtual ~Player() {}

  // Returns the move this player wants to make in this position.
  virtual Move GetMove(const Board& b) = 0;
};

}  // namespace chess

#endif
