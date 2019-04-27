#include "chess/bitboard.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chess {
namespace {

TEST(BitboardTest, Iterate) {
  std::vector<int> bits;
  for (int x : BitRange(0x70f)) {
    bits.push_back(x);
  }
  EXPECT_THAT(bits, testing::ElementsAre(0, 1, 2, 3, 8, 9, 10));
}

}
}  // namespace chess
