#include "chess/game_state.h"

#include <cstdlib>

#include "chess/movegen.h"

namespace chess {

Game::Game() { ++visit_count_[board_.board_hash()]; }

void Game::Advance(const Move& m) {
  if (is_over_) {
    std::cerr << "Game already over: '" << board_.ToFEN()
              << "' winner: " << int(winner_) << "\n";
    abort();
  }
  assert(!is_over_);
  board_ = Board(board_, m);
  int& rep = visit_count_[board_.board_hash()];
  ++rep;
  if (rep >= 3) {
    is_over_ = true;
    winner_ = Color::kEmpty;
    return;
  }

  // Check for mate / stalemate.
  const MovegenResult res = IterateLegalMoves(board_, [](const Move& m) {});
  if (res == MovegenResult::kCheckmate) {
    is_over_ = true;
    // Player with the previous turn won.
    winner_ = OtherColor(board_.turn());
    return;
  } else if (res == MovegenResult::kStalemate) {
    is_over_ = true;
    winner_ = Color::kEmpty;
    return;
  }
  if (board_.no_progress_count() >= 100) {
    is_over_ = true;
    winner_ = Color::kEmpty;
  }
}

}  // namespace chess
