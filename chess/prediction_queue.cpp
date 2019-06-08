#include "chess/prediction_queue.h"

#include "tensorflow/core/platform/logging.h"

namespace chess {

PredictionQueue::PredictionQueue(Model* model, int max_batch_size)
    : model_(model), max_batch_size_(max_batch_size) {
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
    auto prediction = model_->Predict(current_batch->board_tensor);
    current_batch->move_p = std::move(prediction.move_p);
    current_batch->value = std::move(prediction.value);
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

void PredictionQueue::GetPredictions(Request* requests, int n) {
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
        auto slice = last_batch->board_tensor.SubSlice(offset + i);
        BoardToTensor(*requests[i].board, &slice);
      }
      // Wait for this batch to be ready.
      mu_.Await(absl::Condition(&last_batch->ready));
    }
    CHECK_GT(batch_n, 0);
    CHECK_GE(offset, 0);
    CHECK_NE(last_batch, nullptr);

    // Batch ready, dispense results:
    for (int i = 0; i < batch_n; ++i) {
      auto board_policy = last_batch->move_p.SubSlice(offset + i);
      requests[i].result.policy.clear();
      requests[i].result.policy.reserve(requests[i].moves->size());
      double total = 0.0;
      for (const Move& m : *requests[i].moves) {
        const double v =
            MovePriorFromTensor(board_policy, requests[i].board->turn(), m);
        requests[i].result.policy.emplace_back(m, v);
        total += v;
      }
      if (total < 0.2) {
        // Remove this when starting from scratch.
        std::cerr << "Network not good, 4/5 moves are not legal\n";
        std::cerr << requests[i].board->ToPrintString();
      }
      for (auto& move_p : requests[i].result.policy) {
        move_p.second /= total;
      }
      requests[i].result.value = last_batch->value.flat<float>()(i);
    }

    // Now we can try making a new request.
    n -= batch_n;
    requests += batch_n;

    // Potentially freelist last_batch.
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

PolicyNetworkPlayer::PolicyNetworkPlayer(PredictionQueue* queue)
    : queue_(queue) {}

void PolicyNetworkPlayer::Reset() { b_ = Board(); }

void PolicyNetworkPlayer::Advance(const Move& m) { b_ = Board(b_, m); }

Move PolicyNetworkPlayer::GetMove() {
  const auto valid_moves = b_.valid_moves();
  PredictionQueue::Request req;
  req.board = &b_;
  req.moves = &valid_moves;
  queue_->GetPredictions(&req, 1);

  std::uniform_real_distribution<double> rand(0.0, 1.0);
  double r = rand(mt_);
  for (const auto& move : req.result.policy) {
    r -= move.second;
    if (r < 0) {
      return move.first;
    }
  }
  std::cerr << "This shouldn't happen\n";
  return valid_moves[0];
}

}  // namespace chess
