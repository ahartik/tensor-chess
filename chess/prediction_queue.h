#ifndef _CHESS_PREDICTION_QUEUE_H_
#define _CHESS_PREDICTION_QUEUE_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "chess/board.h"
#include "chess/model.h"
#include "chess/tensors.h"
#include "tensorflow/core/framework/tensor.h"

namespace chess {


class PredictionQueue {
 public:
  struct Request {
    // Input to the request:
    const Board* board;

    // Output of the request:

    PredictionResult result;
  };
  explicit PredictionQueue(Model* model);
  ~PredictionQueue();

  // Blocks.
  void GetPredictions(Request* requests, int n);

  int64_t num_predictions() const {
    return pred_count_.load(std::memory_order_relaxed);
  }

  double avg_batch_size() const {
    const int preds = num_predictions();
    const int batches = batch_count_.load(std::memory_order_relaxed);
    if (batches == 0) {
      return 0.0;
    }
    return static_cast<double>(preds) / batches;
  }

 private:
  struct WorkBatch {
    WorkBatch(int n) : board_tensor(MakeBoardTensor(n)) {}
    tensorflow::Tensor board_tensor;
    tensorflow::Tensor move_p;
    tensorflow::Tensor value;
    int size = 0;
    bool ready = false;
    int pending_requests = 0;
  };

  void WorkerThread(int worker_id);

  std::shared_ptr<WorkBatch> CreateBatch() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const int max_batch_size_ = 256;
  Model* const model_;

  std::atomic<int64_t> pred_count_{0};
  std::atomic<int64_t> batch_count_{0};

  absl::Mutex mu_;
  std::deque<std::shared_ptr<WorkBatch>> batches_ GUARDED_BY(mu_);
  std::vector<std::shared_ptr<WorkBatch>> freelist_ GUARDED_BY(mu_);
  // TODO: Add freelist for work batch items.
  bool stopped_ = false;
  int num_working_ = 0;

  std::vector<std::thread> workers_;
  // TODO: Possible to put prediction cache here.
};

}  // namespace chess

#endif
