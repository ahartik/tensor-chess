#ifndef _C4CC_PLAY_GAME_H_
#define _C4CC_PLAY_GAME_H_

#include <functional>

#include "c4cc/board.h"

namespace c4cc {

// Returns end board.
Board PlayGame(const std::function<int(const Board&)>& move_picker_1,
               const std::function<int(const Board&)>& move_picker_2);

// TODO: This function will have to be made asynchronous once we start training
// tensorflow.

}  // namespace c4cc

#endif
