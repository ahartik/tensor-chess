#include "c4cc/prediction_queue.h"

#include "tensorflow/core/platform/logging.h"

namespace c4cc {

PredictionQueue::PredictionQueue(Model* model) : model_(model) {
  const int kNumWorkers = 3;
  for (int i = 0; i < kNumWorkers; ++i) {
    workers_.emplace_back([this, i] { WorkerThread(i); });
  }
}

PredictionQueue::~PredictionQueue() {
  LOG(INFO) << "Destroying queue";
  {
    absl::MutexLock lock(&mu_);
    stopped_ = true;
  }
  for (auto& t : workers_) {
    t.join();
  }
  LOG(INFO) << "Joined";
}

void PredictionQueue::WorkerThread(int worker_id) {
  absl::MutexLock lock(&mu_);
  std::vector<Prediction> batch_results(max_batch_size_);
  while (true) {
    const auto stopped_or_have_work = [this, worker_id] {
      if (stopped_) {
        return true;
      }
      if (num_working_ == 0) {
        // First worker will take anything.
        return !batches_.empty();
      }
      // Rest, only take work if the batch is full.
      return !batches_.empty() && batches_.front()->size == max_batch_size_;
    };
    mu_.Await(absl::Condition(&stopped_or_have_work));
    if (stopped_) {
      break;
    }
    ++num_working_;
    const std::shared_ptr<WorkBatch> current_batch =
        std::move(batches_.front());
    batches_.pop_front();

    CHECK_GT(current_batch->size, 0);
    // LOG(INFO) << "Worker got " << current_batch->size() << " items";

    // As there is only one worker thread, we can safely unlock while
    // processing 'current_batch'. During this time, other threads may add
    // requests to the next batch.
    mu_.Unlock();
    pred_count_.fetch_add(current_batch->size, std::memory_order_relaxed);
    batch_count_.fetch_add(1, std::memory_order_relaxed);
    // Work on current_batch now.
    current_batch->prediction = model_->Predict(current_batch->tensor);
    // Serve requests now that we got the numbers. That requires a lock again.
    mu_.Lock();
    current_batch->ready = true;
    --num_working_;
  }
}

std::shared_ptr<PredictionQueue::WorkBatch> PredictionQueue::CreateBatch() {
  mu_.AssertHeld();
  if (freelist_.empty()) {
    return std::make_shared<WorkBatch>(max_batch_size_);
  }
  auto r = std::move(freelist_.back());
  freelist_.pop_back();

  r->ready = false;
  r->size = 0;
  // Should already be zero.
  CHECK_EQ(r->pending_requests, 0);
  return r;
}

void PredictionQueue::GetPredictions(Board* boards, Prediction* preds, int n) {
  while (n > 0) {
    std::shared_ptr<WorkBatch> last_batch;
    int batch_n = -1;
    int offset = -1;
    {
      absl::MutexLock lock(&mu_);
      if (batches_.empty() || batches_.back()->size == max_batch_size_) {
        // Need a new batch (possibly by taking from freelist_).
        batches_.push_back(CreateBatch());
      }
      // Now we know there is some space in the last batch.
      last_batch = batches_.back();
      offset = last_batch->size;
      // The batch can be most 'max_batch_size_' in size.
      batch_n = std::min(n, max_batch_size_ - offset);
      last_batch->size += batch_n;
      ++last_batch->pending_requests;

      // Write input in the tensor already.
      for (int i = 0; i < batch_n; ++i) {
        BoardToTensor(boards[i], &last_batch->tensor, offset + i);
      }
      // Wait for this batch to be ready.
      mu_.Await(absl::Condition(&last_batch->ready));
    }
    CHECK_GT(batch_n, 0);
    CHECK_GE(offset, 0);
    CHECK_NE(last_batch, nullptr);

    // Batch ready, compute results:
    ReadPredictions(last_batch->prediction, preds, offset, batch_n);

    // Now we can try making a new request.
    n -= batch_n;
    boards += batch_n;
    preds += batch_n;
    {
      absl::MutexLock lock(&mu_);
      --last_batch->pending_requests;
      if (last_batch->pending_requests == 0) {
        // This was the last request to be served from this batch, we can
        // freelist this batch.
        freelist_.push_back(std::move(last_batch));
      }
    }
  }
}

}  // namespace c4cc
