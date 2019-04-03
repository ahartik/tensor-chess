#include "util/int-set.h"

#include <cstdint>
#include <random>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::testing::Eq;
using ::testing::Pointee;

TEST(PersistentIntMapTest, ConstructDestruct) { IntSet set; }
}  // namespace
