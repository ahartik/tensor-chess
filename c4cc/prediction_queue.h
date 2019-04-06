#ifndef _C4CC_PREDICTION_QUEUE_H_
#define _C4CC_PREDICTION_QUEUE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "c4cc/board.h"
#include "c4cc/model.h"
#include "tensorflow/core/framework/tensor.h"

namespace c4cc {

class PredictionQueue {
 public:
  explicit PredictionQueue(Model* model);
  ~PredictionQueue();
  void GetPredictions(Board* boards, Prediction* preds, int n);

  int64_t num_predictions() const {
    return pred_count_.load(std::memory_order_relaxed);
  }

 private:
  struct Request {
    int offset = 0;
    int n = 0;
    Prediction* output = nullptr;
    bool ready = false;
  };

  void WorkerThread();

  const int max_batch_size_ = 64;
  Model* const model_;

  std::atomic<int64_t> pred_count_{0};

  absl::Mutex mu_;
  struct WorkBatch {
    WorkBatch(int n) : tensor(MakeBoardTensor(n)) {}
    tensorflow::Tensor tensor;
    std::vector<Request*> requests;
    int size() const {
      if (requests.empty()) {
        return 0;
      }
      return requests.back()->offset + requests.back()->n;
    }
  };
  std::unique_ptr<WorkBatch> next_batch_;
  bool stopped_ = false;

  std::thread worker_;
  // TODO: Possible to put prediction cache here.
};

}  // namespace c4cc

#endif
