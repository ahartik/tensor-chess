#include "overmind/persistent-int-map.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace overmind {
namespace {

using ::testing::Pointee;
using ::testing::Eq;

TEST(PersistentIntMapTest, ConstructDestruct) { PersistentIntMap<int, int> map; }

TEST(PersistentIntMapTest, InsertFind) {
  PersistentIntMap<int, int> map;
  map = map.Insert(10, 20);
  const int* val = map.Find(10);
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, 20);
}

TEST(PersistentIntMapTest, InsertMultiple) {
  PersistentIntMap<int, int> map;
  map = map.Insert(10, 2 * 10);
  map = map.Insert(20, 2 * 20);
  map = map.Insert(30, 2 * 30);
  map = map.Insert(40, 2 * 40);
  map = map.Insert(50, 2 * 50);
  map = map.Insert(60, 2 * 60);
  for (int i = 1; i <= 6; ++i) {
    const int* val = map.Find(10 * i);
    ASSERT_NE(val, nullptr) << "i = " << i;
    EXPECT_EQ(*val, 20 * i) << "i = " << i;
  }
}

}  // namespace
}  // namespace overmind
