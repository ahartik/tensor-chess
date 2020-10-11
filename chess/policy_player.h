// Simple player implementation plainly using the policy network.
#ifndef _CHESS_POLICY_PLAYER_H_
#define _CHESS_POLICY_PLAYER_H_

#include <memory>

#include "chess/player.h"
#include "generic/prediction_queue.h"

namespace chess {

// Does not take ownership of the prediction queue.
std::unique_ptr<Player> MakePolicyPlayer(generic::PredictionQueue* p_queue);

}  // namespace chess

#endif
