#ifndef _C4CC_PREDICTION_QUEUE_H_
#define _C4CC_PREDICTION_QUEUE_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "generic/board.h"
#include "generic/model.h"
#include "tensorflow/core/framework/tensor.h"

namespace generic {

struct PredictionResult {
  std::vector<std::pair<int, float>> policy;
  double value = 0.0;
};

// This class is thread-safe.
class PredictionQueue {
 public:
  struct Request {
    // Input to the request:
    const Board* board;
    PredictionResult result;
  };

  explicit PredictionQueue(Model* model, int max_batch_size = 64);
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
    explicit WorkBatch(int n, const Board& first_board) {
      tensorflow::TensorShape shape;
      first_board.GetTensorShape(n, &shape);
      board_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
    }
    tensorflow::Tensor board_tensor;
    tensorflow::Tensor move_p;
    tensorflow::Tensor value;
    int size = 0;
    bool ready = false;
    int pending_requests = 0;
  };

  void WorkerThread(int worker_id);

  std::shared_ptr<WorkBatch> CreateBatch(const Board& first_board)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Model* const model_;
  const int max_batch_size_;

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

}  // namespace generic

#endif
