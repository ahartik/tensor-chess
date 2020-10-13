#ifndef _GENERIC_SHUFFLING_TRAINER_H_
#define _GENERIC_SHUFFLING_TRAINER_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <random>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "generic/board.h"
#include "generic/model.h"

namespace generic {

// This class is thread-safe.
class ShufflingTrainer {
 public:
  explicit ShufflingTrainer(Model* model, const Board& model_board,
                            int batch_size = 256, int shuffle_size = 10000);
  ~ShufflingTrainer();

  void Train(std::unique_ptr<Board> b, PredictionResult target);

  int64_t num_trained() const {
    return num_trained_.load(std::memory_order_relaxed);
  };

  void Flush();

 private:
  struct BoardData {
    std::unique_ptr<Board> board;
    PredictionResult target;
  };

  void WorkerThread();

  Model* const model_;
  const int batch_size_;
  const int shuffle_size_;
  const int num_moves_;
  tensorflow::TensorShape board_shape_;
  // TODO add better explanation
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

}  // namespace generic

#endif
