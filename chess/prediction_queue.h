#ifndef _CHESS_PREDICTION_QUEUE_H_
#define _CHESS_PREDICTION_QUEUE_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "chess/board.h"
#include "chess/model.h"
#include "chess/tensors.h"
#include "chess/prediction_cache.h"
#include "tensorflow/core/framework/tensor.h"
#include "chess/player.h"

namespace chess {

class PredictionQueue {
 public:
  struct Request {
    // Input to the request:
    const Board* board;
    const MoveList* moves;

    PredictionResult result;
  };
  explicit PredictionQueue(Model* model, int max_batch_size = 64);
  ~PredictionQueue();

  // Blocks.
  void GetPredictions(Request* requests, int n);

  // TODO: Consider adding an asynchronous interface too.

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

  // 
  void SetCycleTime(double cycle_time);

  void CacheRealPrediction(const Board& b, const PredictionResult& result);

 private:
  struct WorkBatch {
    explicit WorkBatch(int n) : board_tensor(MakeBoardTensor(n)) {}
    tensorflow::Tensor board_tensor;
    tensorflow::Tensor move_p;
    tensorflow::Tensor value;
    int size = 0;
    bool ready = false;
    int pending_requests = 0;
  };

  void WorkerThread(int worker_id);

  std::shared_ptr<WorkBatch> CreateBatch() EXCLUSIVE_LOCKS_REQUIRED(mu_);

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
  std::atomic<int64_t> gen_{0};

  PredictionCache cache_;
  std::vector<std::thread> workers_;
  // TODO: Possible to put prediction cache here.
};

// Simple player which picks moves (randomly) based on policy network.
//
// This class is thread-compatible.
class PolicyNetworkPlayer : public Player {
 public:
  PolicyNetworkPlayer(PredictionQueue* queue);

  void Reset(const Board& b) override { b_ = b; }
  void Advance(const Move& m) override;
  Move GetMove() override;

 private:
  std::mt19937_64 mt_;
  PredictionQueue* queue_;
  Board b_;
};

}  // namespace chess

#endif
