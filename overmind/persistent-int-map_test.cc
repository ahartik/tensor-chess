#include "gtest/gtest.h"

#include "overmind/persistent-int-map.h"

namespace overmind {
namespace {

TEST(PersistentIntMapTest, ConstructDestruct) { PersistentIntMap<int, int> map; }

TEST(PersistentIntMapTest, InsertFind) {
  PersistentIntMap<int, int> map;
  map = map.Insert(10, 20);
  const int* val = map.Find(10);
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, 20);
}

}  // namespace
}  // namespace overmind
