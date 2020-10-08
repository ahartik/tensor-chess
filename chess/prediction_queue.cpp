#include "chess/prediction_queue.h"

#include "tensorflow/core/platform/logging.h"

namespace chess {

namespace {

constexpr int kMaxPendingBatches = 2;
constexpr int kFreelistMaxSize = 2;

constexpr int kGenBuckets = 10;

}  // namespace

PredictionQueue::PredictionQueue(Model* model, int max_batch_size)
    : model_(model), max_batch_size_(max_batch_size) {
  const int kNumWorkers = 2;
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
    // LOG(INFO) << "Worker got " << current_batch->size << " items";

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

void PredictionQueue::GetPredictions(Request* requests_in, int n) {
  std::vector<Request*> not_cached;
  not_cached.reserve(n);

  for (int i = 0; i < n; ++i) {
    auto& r = requests_in[i];
    auto cached = cache_.Lookup(*r.board);
    if (cached != nullptr) {
      r.result = *cached;
    } else {
      not_cached.push_back(&r);
    }
  }
  n = not_cached.size();
  Request** requests = not_cached.data();

  while (n > 0) {
    std::shared_ptr<WorkBatch> last_batch;
    int batch_n = -1;
    int offset = -1;
    {
      absl::MutexLock lock(&mu_);
      if (batches_.empty() || batches_.back()->size == max_batch_size_) {
        auto can_make_batch = [this]() -> bool {
          return batches_.size() < kMaxPendingBatches;
        };
        mu_.Await(absl::Condition(&can_make_batch));
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
        BoardToTensor(*requests[i]->board, &slice);
      }
      // Wait for this batch to be ready.
      mu_.Await(absl::Condition(&last_batch->ready));
    }
    CHECK_GT(batch_n, 0);
    CHECK_GE(offset, 0);
    CHECK_NE(last_batch, nullptr);

    const int64_t gen = gen_;
    // Batch ready, dispense results:
    for (int i = 0; i < batch_n; ++i) {
      auto& request = *requests[i];
      auto board_policy = last_batch->move_p.SubSlice(offset + i);
      request.result.policy.clear();
      request.result.policy.reserve(request.moves->size());
      double total = 0.0;
      for (const Move& m : *request.moves) {
        const double v =
            MovePriorFromTensor(board_policy, request.board->turn(), m);
        request.result.policy.emplace_back(m, v);
        total += v;
      }
      if (total < 0.1) {
        total += 1.0;
        for (auto& move_p : request.result.policy) {
          move_p.second = 1.0 / request.result.policy.size();
        }
      }
      for (auto& move_p : request.result.policy) {
        move_p.second /= total;
      }
      request.result.value = last_batch->value.flat<float>()(i);
      cache_.Insert(gen, *request.board, request.result);
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
        if (freelist_.size() < kFreelistMaxSize) {
          freelist_.push_back(std::move(last_batch));
        }
      }
    }
  }
}

void PredictionQueue::SetCycleTime(double cycle_time) {
  const int64_t gen = kGenBuckets * fmod(cycle_time, 1ll << 20);
  cache_.ClearOlderThan(gen - kGenBuckets);
  gen_ = gen;
}

void PredictionQueue::CacheRealPrediction(const Board& b,
                                          const PredictionResult& result) {
  cache_.Insert(gen_, b, result);
}

PolicyNetworkPlayer::PolicyNetworkPlayer(PredictionQueue* queue)
    : queue_(queue) {}

void PolicyNetworkPlayer::Advance(const Move& m) { b_ = Board(b_, m); }

Move PolicyNetworkPlayer::GetMove() {
  const auto valid_moves = b_.valid_moves();
  PredictionQueue::Request req;
  req.board = &b_;
  req.moves = &valid_moves;
  queue_->GetPredictions(&req, 1);
  std::cerr << "Value before making move: " << req.result.value << "\n";
  std::cerr << "Valid moves: ";
  for (const auto move : valid_moves) {
    std::cerr << move << " ";
  }
  std::cerr << "\n";

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
