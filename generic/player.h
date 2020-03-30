#ifndef _GENERIC_PLAYER_H_
#define _GENERIC_PLAYER_H_

#include <memory>

#include "generic/board.h"

namespace generic {

// This class is thread-compatible (but not thread-safe).
class Player {
 public:
  virtual ~Player() {}

  // Current board the player is inspecting.
  virtual const Board& board() const = 0;

  // Switch the board state.
  //
  // This slightly-ugly interface allows caching state between different games.
  virtual void SetBoard(std::unique_ptr<Board> b) = 0;

  // Returns the move this player wants to make in this position.
  // Must not be called if board().is_over() returns true.
  virtual int GetMove() = 0;

  virtual void MakeMove(int move) = 0;
};


}  // namespace generic

#endif
