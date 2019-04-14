#include "util/int-set.h"

#include <cstdint>
#include <set>
#include <random>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::testing::Eq;
using ::testing::Pointee;

TEST(IntSetTest, ConstructDestruct) { IntSet set; }

TEST(IntSetTest, InsertMultipleValues) {
  IntSet set;
  std::set<uint64_t> expected;
  for (int i = 0; i < 10; ++i) {
    set = set.Insert(i);
    expected.insert(i);
  }

  // Insert a few random values.
  std::mt19937_64 mt;
  for (int i = 0; i < 1000; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    x |= 0xbeef0000;
    set = set.Insert(x);
    expected.insert(x);
  }
}

}  // namespace
