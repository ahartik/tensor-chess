#ifndef _CHESS_PREDICTION_CACHE_H_
#define _CHESS_PREDICTION_CACHE_H_

#include <iostream>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "chess/board.h"
#include "chess/types.h"

namespace chess {

class PredictionCache {
 public:
  void Insert(int gen, const Board& b, PredictionResult result) {
    Shard& shard = (*shards_)[ShardForBoard(b)];
    CachedValue val;
    val.gen = gen;
    val.res = std::make_shared<PredictionResult>(std::move(result));
    const auto fp = BoardFingerprint(b);
    absl::MutexLock lock(&shard.mu);
    shard.map[fp] = std::move(val);
  }

  std::shared_ptr<const PredictionResult> Lookup(const Board& b) const {
    const Shard& shard = (*shards_)[ShardForBoard(b)];
    absl::MutexLock lock(&shard.mu);
    auto it = shard.map.find(BoardFingerprint(b));
    if (it != shard.map.end()) {
      return it->second.res;
    }
    return nullptr;
  }

  void ClearOlderThan(int64_t min_gen) {
    for (Shard& shard : *shards_) {
      absl::MutexLock lock(&shard.mu);
      for (auto it = shard.map.begin(); it != shard.map.end();) {
        if (it->second.gen < min_gen) {
          auto to_erase = it;
          ++it;
          shard.map.erase(to_erase);
        } else {
          ++it;
        }
      }
    }
  }

  void Clear() {
    for (Shard& shard : *shards_) {
      decltype(shard.map) newmap;
      absl::MutexLock lock(&shard.mu);
      shard.map.swap(newmap);
    }
  }

 private:
  static constexpr int kNumShards = 64;

  static int ShardForBoard(const Board& b) {
    uint64_t h = b.board_hash();
    return (h >> 30) % kNumShards;
  }

  struct CachedValue {
    int64_t gen = 0;
    std::shared_ptr<const PredictionResult> res;
  };
  struct Shard {
    mutable absl::Mutex mu;
    absl::flat_hash_map<BoardFP, CachedValue> map;
    char pad[64 - sizeof(mu) - sizeof(map)];
  };
  static_assert(sizeof(Shard) == 64);
  const std::unique_ptr<std::array<Shard, kNumShards>> shards_{
      new std::array<Shard, kNumShards>()};
};

}  // namespace chess

#endif
