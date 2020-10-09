#ifndef _CHESS_GENERIC_BOARD_H_
#define _CHESS_GENERIC_BOARD_H_

#include <memory>

#include "chess/board.h"
#include "generic/board.h"

namespace chess {

std::unique_ptr<generic::Board> MakeGenericBoard(const Board& b);

}  // namespace chess

#endif
