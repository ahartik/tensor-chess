#include "chess/prediction_queue.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chess {
namespace {

// Tests in increasing order of difficulty (for the network):
//
// By default we can run for the "human" network, since we're not only testing
// network sanity here, but the implementation sanity of the prediction queue.
//
// * Policy player beats random bot >80% of the time.
// * Make some obvious test cases, like where queen can be captured from the
//   middle.
// * "value" in some lost/won positions have abs(v) > 0.8
// * "value" in starting position has abs(v) < 0.3
// * Starting position policy for bad moves f3 and g4 are < 1%
//
// These 

}  // namespace
}  // namespace chess
