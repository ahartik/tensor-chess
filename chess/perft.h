#ifndef _CHESS_PERFT_H_
#define _CHESS_PERFT_H_

#include <cstdint>

#include "chess/board.h"

namespace chess {

int64_t Perft(const Board& board, int d);

}  // namespace chess

#endif
