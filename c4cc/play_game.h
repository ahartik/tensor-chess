#ifndef _C4CC_PLAY_GAME_H_
#define _C4CC_PLAY_GAME_H_

#include <functional>

#include "c4cc/board.h"

namespace c4cc {

// This class is thread-compatible (but not thread-safe).
class Player {
 public:
  virtual ~Player() {}

  // Current board the player is inspecting.
  virtual const Board& board() const = 0;

  // Switch the board state.
  //
  // This slightly-ugly interface allows caching state between different games.
  virtual void SetBoard(const Board& b) = 0;

  // Returns the move this player wants to make in this position.
  // Must not be called if board()->is_over() returns true. 
  virtual int GetMove() = 0;

  virtual void MakeMove(int move) = 0;
};

// Returns end board, and the of moves.
std::pair<Board, std::vector<int>> PlayGame(Player* player1, Player* player2);

// TODO: This function will have to be made asynchronous once we start training
// tensorflow.

}  // namespace c4cc

#endif
