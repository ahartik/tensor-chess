#ifndef _CHESS_SHUFFLING_TRAINER_H_
#define _CHESS_SHUFFLING_TRAINER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <deque>
#include <random>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "chess/board.h"
#include "chess/model.h"

namespace chess {

struct TrainingSample {
  Board board;
  // TODO: Change to PredictionResult?
  std::vector<std::pair<Move, float>> moves;
  Color winner;
};

// This class is thread-safe.
class ShufflingTrainer {
 public:
  explicit ShufflingTrainer(Model* model, int batch_size = 256,
                            int shuffle_size = 1000);
  ~ShufflingTrainer();

  void Train(std::unique_ptr<TrainingSample> sample);

  int64_t num_trained() const {
    return num_trained_.load(std::memory_order_relaxed);
  };

  void Flush();

 private:
  void WorkerThread();

  Model* const model_;
  const int batch_size_;
  const int shuffle_size_;
  // TODO: Following comment doesn't make sense for chess yet.
  // 4 minutes of boards.
  const int max_size_ = 400 * 60 * 4 * 2;

  std::atomic<int64_t> num_trained_;

  absl::Mutex mu_;
  bool stopped_ GUARDED_BY(mu_) = false;
  std::mt19937 rng_ GUARDED_BY(mu_);
  int64_t since_full_flush_ GUARDED_BY(mu_) = 0;
  std::deque<std::unique_ptr<TrainingSample>> data_ GUARDED_BY(mu_);

  std::vector<std::thread> workers_;
};

}  // namespace chess

#endif
