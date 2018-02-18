#include "overmind/persistent-int-map.h"

#include <random>
#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/time.h"
#include "absl/time/clock.h"

namespace overmind {
namespace {

using ::testing::Eq;
using ::testing::Pointee;

TEST(PersistentIntMapTest, ConstructDestruct) {
  PersistentIntMap<int, int> map;
}

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

TEST(PersistentIntMapTest, Negatives) {
  PersistentIntMap<int, int> map;
  map = map.Insert(10, 20);
  map = map.Insert(-10, 50);
  const int* val = map.Find(10);
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, 20);
  val = map.Find(-10);
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, 50);
}

TEST(PersistentIntMapTest, LookupBenchmark) {
  PersistentIntMap<uint64_t, int> map;
  const int kSize = 1000 * 1000;
  std::vector<uint64_t> keys;
  std::mt19937_64 my_rand;
  for (int i = 0; i < kSize; ++i) {
    uint64_t key = my_rand();
    keys.push_back(key);
    map = map.Insert(key, key & 255);
  }
  std::shuffle(keys.begin(), keys.end(), my_rand);
  const auto start_time = absl::Now();
  for (const uint64_t key : keys) {
    const int* v = map.Find(key);
    ASSERT_NE(v, nullptr);
    EXPECT_EQ(*v, key & 255);
  }
  const auto end_time = absl::Now();
  std::cerr << "Took " << (end_time - start_time) << "\n";
  std::cerr << " = " << (end_time - start_time) / keys.size() << " per item\n";
}


}  // namespace
}  // namespace overmind
