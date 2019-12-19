#ifndef _C4CC_SHUFFLING_TRAINER_H_
#define _C4CC_SHUFFLING_TRAINER_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <random>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "c4cc/board.h"
#include "generic/model.h"

namespace c4cc {

// This class is thread-safe.
class ShufflingTrainer {
 public:
  explicit ShufflingTrainer(generic::Model* model, int batch_size = 256,
                            int shuffle_size = 10000);
  ~ShufflingTrainer();

  void Train(const Board& b, const Prediction& target);

  int64_t num_trained() const {
    return num_trained_.load(std::memory_order_relaxed);
  };

  void Flush();

 private:
  struct BoardData {
    Board b;
    Prediction target;
  };

  void WorkerThread();

  generic::Model* const model_;
  const int batch_size_;
  const int shuffle_size_;
  // 4 minutes of boards.
  const int max_size_ = 400 * 60 * 4 * 2;

  std::atomic<int64_t> num_trained_;

  absl::Mutex mu_;
  bool stopped_ GUARDED_BY(mu_) = false;
  std::mt19937 rng_ GUARDED_BY(mu_);
  int64_t since_full_flush_ GUARDED_BY(mu_) = 0;
  std::deque<BoardData> data_ GUARDED_BY(mu_);

  std::thread worker_;
};

}  // namespace c4cc

#endif
