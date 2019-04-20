#include "util/int-set.h"

#include <cstdint>
#include <set>
#include <thread>
#include <random>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

TEST(IntSetTest, ConstructDestruct) { IntSet set; }

TEST(IntSetTest, InsertMultipleValues) {
  IntSet set;
  std::set<uint64_t> expected;
  for (int i = 0; i < 10; ++i) {
    set = set.Insert(i);
    expected.insert(i);
  }

  IntSet only_small = set;

  // Insert a few random values.
  std::mt19937_64 mt;
  for (int i = 0; i < 1000; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    x |= 0xbeef0000;
    set = set.Insert(x);
    expected.insert(x);
  }
  for (uint64_t x : expected) {
    EXPECT_TRUE(set.Contains(x));
    x ^= 0xf0000000;
    EXPECT_FALSE(set.Contains(x));
  }
  for (uint64_t x : expected) {
    if (x < 10) {
      EXPECT_TRUE(only_small.Contains(x));
    } else {
      EXPECT_FALSE(only_small.Contains(x));
    }
  }
  set = IntSet();
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(only_small.Contains(i));
  }

  set = IntSet(expected.begin(), expected.end());
  for (uint64_t x : expected) {
    EXPECT_TRUE(set.Contains(x));
    x ^= 0xf0000000;
    EXPECT_FALSE(set.Contains(x));
  }
}

TEST(IntSetTest, MultiThread) {
  IntSet set;
  std::set<uint64_t> expected;
  // Insert a few random values.
  std::mt19937_64 mt;
  for (int i = 0; i < 1000; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    x |= 0xbeef0000;
    set = set.Insert(x);
    expected.insert(x);
  }

  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      IntSet s = set;
      for (int i = 0; i < 100; ++i) {
        s.Insert(i);
      }
      for (uint64_t x : expected) {
        EXPECT_TRUE(s.Contains(x));
        EXPECT_TRUE(set.Contains(x));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST(IntSetTest, LookupBenchmark) {
  IntSet set;
  std::vector<uint64_t> expected;
  // Insert a few random values.
  std::mt19937_64 mt;
  for (int i = 0; i < 100; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    set = set.Insert(x);
    expected.push_back(x);
  }
  const auto start_time = absl::Now();
  int total = 0;
  for (int i = 0; i < 100; ++i) {
    for (uint64_t x : expected) {
      EXPECT_TRUE(set.Contains(x));
      ++total;
    }
  }
  const auto end_time = absl::Now();
  std::cerr << "Took " << (end_time - start_time) << "\n";
  std::cerr << " = " << (end_time - start_time) / total << " per item\n";
}

TEST(IntSetTest, InsertBenchmark) {
  std::vector<uint64_t> expected;
  std::mt19937_64 mt;
  for (int i = 0; i < 100; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    expected.push_back(x);
  }

  const auto start_time = absl::Now();
  int total = 0;
  std::vector<IntSet> sets;
  for (int i = 0; i < 10000; ++i) {
    sets.clear();
    IntSet set;
    for (uint64_t x : expected) {
      set = set.Insert(x);
      sets.push_back(set);
      ++total;
    }
  }
  const auto end_time = absl::Now();
  std::cerr << "Took " << (end_time - start_time) << "\n";
  std::cerr << " = " << (end_time - start_time) / total << " per item\n";
}

TEST(SmallIntSetTest, ConstructDestruct) { SmallIntSet set; }

TEST(SmallIntSetTest, InsertMultipleValues) {
  SmallIntSet set;
  std::set<uint64_t> expected;
  for (int i = 0; i < 10; ++i) {
    set = set.Insert(i);
    expected.insert(i);
  }

  SmallIntSet only_small = set;

  // Insert a few random values.
  std::mt19937_64 mt;
  for (int i = 0; i < 1000; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    x |= 0xbeef0000;
    set = set.Insert(x);
    expected.insert(x);
  }
  for (uint64_t x : expected) {
    EXPECT_TRUE(set.Contains(x));
    x ^= 0xf0000000;
    EXPECT_FALSE(set.Contains(x));
  }
  for (uint64_t x : expected) {
    if (x < 10) {
      EXPECT_TRUE(only_small.Contains(x));
    } else {
      EXPECT_FALSE(only_small.Contains(x));
    }
  }
  set = SmallIntSet();
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(only_small.Contains(i));
  }
}

TEST(SmallIntSetTest, MultiThread) {
  SmallIntSet set;
  std::set<uint64_t> expected;
  // Insert a few random values.
  std::mt19937_64 mt;
  for (int i = 0; i < 1000; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    x |= 0xbeef0000;
    set = set.Insert(x);
    expected.insert(x);
  }

  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      SmallIntSet s = set;
      for (int i = 0; i < 100; ++i) {
        s.Insert(i);
      }
      for (uint64_t x : expected) {
        EXPECT_TRUE(s.Contains(x));
        EXPECT_TRUE(set.Contains(x));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST(SmallIntSetTest, LookupBenchmark) {
  SmallIntSet set;
  std::vector<uint64_t> expected;
  // Insert a few random values.
  std::mt19937_64 mt;
  for (int i = 0; i < 100; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    set = set.Insert(x);
    expected.push_back(x);
  }
  const auto start_time = absl::Now();
  int total = 0;
  for (int i = 0; i < 100; ++i) {
    for (uint64_t x : expected) {
      EXPECT_TRUE(set.Contains(x));
      ++total;
    }
  }
  const auto end_time = absl::Now();
  std::cerr << "Took " << (end_time - start_time) << "\n";
  std::cerr << " = " << (end_time - start_time) / total << " per item\n";
}

TEST(SmallIntSetTest, InsertBenchmark) {
  std::vector<uint64_t> expected;
  std::mt19937_64 mt;
  for (int i = 0; i < 100; ++i) {
    uint64_t x = mt();
    // Make a bit pattern to make debugging a little easier.
    expected.push_back(x);
  }

  const auto start_time = absl::Now();
  int total = 0;
  std::vector<SmallIntSet> sets;
  for (int i = 0; i < 10000; ++i) {
    sets.clear();
    SmallIntSet set;
    for (uint64_t x : expected) {
      set = set.Insert(x);
      sets.push_back(set);
      ++total;
    }
  }
  const auto end_time = absl::Now();
  std::cerr << "Took " << (end_time - start_time) << "\n";
  std::cerr << " = " << (end_time - start_time) / total << " per item\n";
}

}  // namespace
