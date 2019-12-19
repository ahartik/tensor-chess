#ifndef _C4CC_GENERIC_BOARD_H_
#define _C4CC_GENERIC_BOARD_H_

#include <memory>

#include "c4cc/board.h"
#include "generic/board.h"

namespace c4cc {

std::unique_ptr<generic::Board> MakeGenericBoard(const Board& b);

}  // namespace c4cc

#endif
