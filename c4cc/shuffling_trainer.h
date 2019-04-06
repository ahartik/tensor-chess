#ifndef _C4CC_SHUFFLING_TRAINER_H_
#define _C4CC_SHUFFLING_TRAINER_H_

#include <atomic>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "c4cc/board.h"
#include "c4cc/model.h"

namespace c4cc {

// This class is thread-safe.
class ShufflingTrainer {
 public:
  explicit ShufflingTrainer(Model* model, int batch_size = 256,
                            int shuffle_size = 10000);
  ~ShufflingTrainer();
  // Returns whether training was actually performed.
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

  Model* const model_;
  const int batch_size_;
  const int shuffle_size_;

  std::atomic<int64_t> num_trained_;

  absl::Mutex mu_;
  bool stopped_ = false;
  std::mt19937 rng_;
  int64_t since_full_flush_ = 0;
  std::vector<BoardData> data_;
  std::thread worker_;
};

}  // namespace c4cc

#endif
