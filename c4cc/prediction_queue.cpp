#include "c4cc/prediction_queue.h"

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

PredictionQueue::PredictionQueue(Model* model)
    : model_(model), worker_([this] { WorkerThread(); }) {
  next_batch_ = std::make_unique<WorkBatch>(max_batch_size_);
}

PredictionQueue::~PredictionQueue() {
  {
    absl::MutexLock lock(&mu_);
    stopped_ = true;
  }
  worker_.join();
}

void PredictionQueue::WorkerThread() {
  absl::MutexLock lock(&mu_);
  std::vector<Prediction> batch_results(max_batch_size_);
  std::unique_ptr<WorkBatch> current_batch =
      std::make_unique<WorkBatch>(max_batch_size_);
  // LOG(INFO) << "Worker start";
  while (true) {
    const auto stopped_or_have_work = [this] {
      const bool v = stopped_ || (!next_batch_->requests.empty());
      // LOG(INFO) << "Cond check: " << v;
      return v;
    };
    mu_.Await(absl::Condition(&stopped_or_have_work));
    if (stopped_) {
      break;
    }
    std::swap(current_batch, next_batch_);

    CHECK_GT(current_batch->size(), 0);
    // LOG(INFO) << "Worker got " << current_batch->size() << " items";

    pred_count_.fetch_add(current_batch->size(), std::memory_order_relaxed);
    // As there is only one worker thread, we can safely unlock while
    // processing 'current_batch'. During this time, other threads may add
    // requests to the next batch.
    mu_.Unlock();
    // Work on current_batch now.
    auto predict_result = model_->Predict(current_batch->tensor);
    // Convert to prediction objects.
    ReadPredictions(predict_result, batch_results.data());
    // Serve requests now that we got the numbers. That requires a lock again.
    mu_.Lock();
    for (Request* req : current_batch->requests) {
      for (int x = 0; x < req->n; ++x) {
        req->output[x] = batch_results[req->offset + x];
      }
      req->ready = true;
    }

    // At end, clear current batch so next iteration can swap it in nicely.
    current_batch->requests.clear();
  }
}

void PredictionQueue::GetPredictions(Board* boards, Prediction* preds, int n) {
  absl::MutexLock lock(&mu_);
  while (n > 0) {
    // LOG(INFO) << "GetPredictions iter n=" << n;
    // If the next batch is already full, wait for it to be flipped.
    const auto is_not_full = [this] {
      return next_batch_->size() != max_batch_size_;
    };
    mu_.Await(absl::Condition(&is_not_full));

    // Now we know there is some space in the batch.
    Request req;
    req.offset = next_batch_->size();
    // The batch can be most 'max_batch_size_' in size.
    req.n = std::min(n, max_batch_size_ - req.offset);
    req.output = preds;
    next_batch_->requests.push_back(&req);
    // Wait for this request to be done.
    mu_.Await(absl::Condition(&req.ready));
    // Now we can try making a new request.
    n -= req.n;
    boards += req.n;
    preds += req.n;
  }
}

}  // namespace c4cc
