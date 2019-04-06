#include "c4cc/prediction_queue.h"

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

PredictionQueue::PredictionQueue(Model* model)
    : model_(model), worker_([this] { WorkerThread(); }) {}

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
  std::make_unique<WorkBatch>(max_batch_size_);
  while (true) {
    const auto stopped_or_have_work = [this] {
      const bool v = stopped_ || (!batches_.empty());
      return v;
    };
    mu_.Await(absl::Condition(&stopped_or_have_work));
    if (stopped_) {
      break;
    }
    const std::unique_ptr<WorkBatch> current_batch =
        std::move(batches_.front());
    batches_.pop_front();

    CHECK_GT(current_batch->size(), 0);
    // LOG(INFO) << "Worker got " << current_batch->size() << " items";

    // As there is only one worker thread, we can safely unlock while
    // processing 'current_batch'. During this time, other threads may add
    // requests to the next batch.
    mu_.Unlock();
    pred_count_.fetch_add(current_batch->size(), std::memory_order_relaxed);
    batch_count_.fetch_add(1, std::memory_order_relaxed);
    // Work on current_batch now.
    auto predict_result = model_->Predict(current_batch->tensor);
    // TODO: Move this prediction reading part to yet another thread.
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
  }
}

void PredictionQueue::GetPredictions(Board* boards, Prediction* preds, int n) {
  absl::MutexLock lock(&mu_);
  while (n > 0) {
    if (batches_.empty() || batches_.back()->size() == max_batch_size_) {
      // Need a new batch.
      batches_.push_back(std::make_unique<WorkBatch>(max_batch_size_));
    }
    // Now we know there is some space in the last batch.
    WorkBatch* const last_batch = batches_.back().get();
    Request req;
    req.offset = last_batch->size();
    // The batch can be most 'max_batch_size_' in size.
    req.n = std::min(n, max_batch_size_ - req.offset);

    // Write input in the tensor already.
    for (int i = 0; i < req.n; ++i) {
      BoardToTensor(boards[i], &last_batch->tensor, req.offset + i);
    }
    req.output = preds;
    last_batch->requests.push_back(&req);
    // Wait for this request to be done.
    mu_.Await(absl::Condition(&req.ready));
    // Now we can try making a new request.
    n -= req.n;
    boards += req.n;
    preds += req.n;
  }
}

}  // namespace c4cc
